from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import nbformat

RUN_UID_ENV_VAR = "LAMIN_INITIATED_BY_RUN_UID"

# ln.track(...) / ln.finish(...) calls including any trailing semicolon.
# Intermediate execute_python steps run in throwaway temp files; letting them
# call ln.track() creates a new Transform/Run per step (registry pollution) and
# triggers lamindb's interactive "renamed (1) or copy (2)?" prompt, which has no
# stdin in a subprocess and crashes. Only the CLI @ln.flow transform is tracked.
_TRACK_CALL_RE = re.compile(r"ln\.track\s*\([^)]*\)\s*;?")
_FINISH_CALL_RE = re.compile(r"ln\.finish\s*\(\s*\)\s*;?")


def strip_tracking_calls(code: str) -> str:
    """Remove ln.track(...) / ln.finish(...) so temp scripts don't create transforms."""
    code = _TRACK_CALL_RE.sub("", code)
    code = _FINISH_CALL_RE.sub("", code)
    return code


# The model frequently rewrites the instance org "laminlabs/" as "lamindb/"
# (confusing it with the package name `import lamindb as ln`). "lamindb" is not
# a real account, so ln.connect("lamindb/...") fails with InstanceNotFoundError.
# This deterministic guard restores the real org before execution.
_BAD_SLUG_RE = re.compile(r"""(connect\s*\(\s*["'])lamindb/""")


def fix_instance_slug(code: str) -> str:
    """Rewrite ln.connect("lamindb/...") to the real org "laminlabs/...".

    Only touches the connect() argument, so a legitimate `import lamindb` and
    `lamindb` references elsewhere are left untouched.
    """
    return _BAD_SLUG_RE.sub(r"\1laminlabs/", code)


def find_plan_file(explicit_plan_file: Path | None = None) -> Path | None:
    """Find an explicit or best candidate markdown plan file."""
    if explicit_plan_file is not None:
        return explicit_plan_file.resolve()

    direct = Path("plan.md")
    if direct.exists():
        return direct.resolve()

    candidates = sorted(
        Path().glob("plan_*.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return None


def extract_runnable_paths(plan_text: str, plan_dir: Path) -> list[Path]:
    """Extract python scripts and notebooks from markdown plan text."""
    candidates: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"`([^`]+\.(?:py|ipynb))`", plan_text):
        candidates.append(match.group(1))

    for line in plan_text.splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        if (
            stripped.endswith(".py") or stripped.endswith(".ipynb")
        ) and " " not in stripped:
            candidates.append(stripped)

    paths: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        if not path.is_absolute():
            path = (plan_dir / path).resolve()
        paths.append(path)
    return paths


def _execute_python(script_path: Path, run_uid: str) -> dict[str, Any]:
    env = os.environ.copy()
    env[RUN_UID_ENV_VAR] = run_uid
    completed = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    return {
        "kind": "python_script",
        "path": str(script_path),
        "status": "success" if completed.returncode == 0 else "error",
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def execute_code_string(
    *, code: str, run_uid: str, strip_tracking: bool = True
) -> dict[str, Any]:
    """Write code string to a temp file, execute it, return stdout/stderr, then clean up.

    By default strips ln.track()/ln.finish() so exploratory steps don't create
    throwaway transforms or hit lamindb's interactive rename prompt.
    """
    import tempfile

    code_to_run = strip_tracking_calls(code) if strip_tracking else code
    code_to_run = fix_instance_slug(code_to_run)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(code_to_run)
        tmp_path = Path(tmp.name)
    try:
        return _execute_python(tmp_path, run_uid)
    finally:
        tmp_path.unlink(missing_ok=True)


def _execute_notebook(notebook_path: Path, run_uid: str) -> dict[str, Any]:
    nb = nbformat.read(notebook_path, as_version=4)
    globals_ns: dict[str, Any] = {}
    outputs: list[str] = []
    errors: list[str] = []
    previous_run_uid = os.environ.get(RUN_UID_ENV_VAR)
    os.environ[RUN_UID_ENV_VAR] = run_uid
    try:
        for idx, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            source = str(cell.source or "")
            try:
                exec(compile(source, str(notebook_path), "exec"), globals_ns)  # noqa: S102
                outputs.append(f"cell_{idx}: ok")
            except Exception as exc:
                errors.append(f"cell_{idx}: {exc}")
                break
    finally:
        if previous_run_uid is None:
            os.environ.pop(RUN_UID_ENV_VAR, None)
        else:
            os.environ[RUN_UID_ENV_VAR] = previous_run_uid
    return {
        "kind": "notebook",
        "path": str(notebook_path),
        "exit_code": 1 if errors else 0,
        "stdout": "\n".join(outputs)[-4000:],
        "stderr": "\n".join(errors)[-4000:],
    }


def execute_plan(*, prompt: str, plan_file: Path, run_uid: str) -> dict[str, Any]:
    plan_text = plan_file.read_text(encoding="utf-8")
    runnable_paths = extract_runnable_paths(plan_text, plan_file.parent)
    payload = execute_runnable_paths(
        prompt=prompt,
        runnable_paths=runnable_paths,
        run_uid=run_uid,
        source=str(plan_file),
    )
    if not runnable_paths:
        payload["final_text"] = "No runnable script/notebook paths found in the plan."
    else:
        failed = [
            event
            for event in payload["trace_events"]
            if event.get("event") in {"script_executed", "notebook_executed"}
            and event.get("exit_code") != 0
        ]
        payload["final_text"] = (
            f"Executed {len(runnable_paths)} runnables from plan; {len(failed)} failed."
        )
    return payload


def execute_runnable_paths(
    *,
    prompt: str,
    runnable_paths: list[Path],
    run_uid: str,
    source: str,
) -> dict[str, Any]:
    """Execute runnable python scripts/notebooks and return trace payload."""
    trace_events: list[dict[str, Any]] = [
        {
            "step": 0,
            "event": "runnables_loaded",
            "source": source,
            "prompt": prompt,
            "runnables_detected": [str(path) for path in runnable_paths],
        }
    ]

    if not runnable_paths:
        return {
            "run_uid": run_uid,
            "trace_events": trace_events,
            "generated_file": None,
            "final_text": "No runnable script/notebook paths to execute.",
        }

    for idx, runnable_path in enumerate(runnable_paths, start=1):
        if not runnable_path.exists():
            trace_events.append(
                {
                    "step": idx,
                    "event": "runnable_missing",
                    "path": str(runnable_path),
                }
            )
            continue

        if runnable_path.suffix == ".ipynb":
            execution = _execute_notebook(runnable_path, run_uid)
            event = "notebook_executed"
        else:
            execution = _execute_python(runnable_path, run_uid)
            event = "script_executed"

        trace_events.append(
            {
                "step": idx,
                "event": event,
                **execution,
            }
        )

    failed = [
        event
        for event in trace_events
        if event.get("event") in {"script_executed", "notebook_executed"}
        and event.get("exit_code") != 0
    ]
    final_text = (
        f"Executed {len(runnable_paths)} runnables; {len(failed)} failed."
        if runnable_paths
        else "No runnables executed."
    )
    return {
        "run_uid": run_uid,
        "trace_events": trace_events,
        "generated_file": None,
        "final_text": final_text,
    }
