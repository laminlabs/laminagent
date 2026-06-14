from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

RUN_UID_ENV_VAR = "LAMIN_INITIATED_BY_RUN_UID"


def find_tool_file(explicit_tool_file: Path | None = None) -> Path | None:
    """Find an explicit or best candidate markdown tool file."""
    if explicit_tool_file is not None:
        return explicit_tool_file.resolve()

    direct = Path("tool.md")
    if direct.exists():
        return direct.resolve()

    candidates = sorted(
        Path().glob("tool_*.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()
    return None


def extract_runnable_paths(tool_text: str, tool_dir: Path) -> list[Path]:
    """Extract python scripts from markdown tool text."""
    candidates: list[str] = []
    seen: set[str] = set()

    for match in re.finditer(r"`([^`]+\.py)`", tool_text):
        candidates.append(match.group(1))

    for line in tool_text.splitlines():
        stripped = line.strip().lstrip("-* ").strip()
        if stripped.endswith(".py") and " " not in stripped:
            candidates.append(stripped)

    paths: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        path = Path(candidate)
        if not path.is_absolute():
            path = (tool_dir / path).resolve()
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
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def execute_tool(*, prompt: str, tool_file: Path, run_uid: str) -> dict[str, Any]:
    tool_text = tool_file.read_text(encoding="utf-8")
    runnable_paths = extract_runnable_paths(tool_text, tool_file.parent)
    payload = execute_runnable_paths(
        prompt=prompt,
        runnable_paths=runnable_paths,
        run_uid=run_uid,
        source=str(tool_file),
    )
    if not runnable_paths:
        payload["final_text"] = "No runnable script paths found in the tool."
    else:
        failed = [
            event
            for event in payload["trace_events"]
            if event.get("event") == "script_executed" and event.get("exit_code") != 0
        ]
        payload["final_text"] = (
            f"Executed {len(runnable_paths)} runnables from tool; {len(failed)} failed."
        )
    return payload


def execute_runnable_paths(
    *,
    prompt: str,
    runnable_paths: list[Path],
    run_uid: str,
    source: str,
) -> dict[str, Any]:
    """Execute runnable python scripts and return trace payload."""
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
            "final_text": "No runnable script paths to execute.",
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
        if event.get("event") == "script_executed" and event.get("exit_code") != 0
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
