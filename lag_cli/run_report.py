from __future__ import annotations

from pathlib import Path
from typing import Any


def pick_agent_source_code(result: dict[str, Any]) -> str | None:
    """Return the final agent-authored Python source from a plan run."""
    generated_files = result.get("generated_files") or []
    py_paths = [
        Path(p) for p in generated_files if isinstance(p, str) and p.endswith(".py")
    ]
    for path in reversed(py_paths):
        if path.is_file():
            return path.read_text(encoding="utf-8")

    trace_events = result.get("trace_events") or []
    for event in reversed(trace_events):
        if not isinstance(event, dict):
            continue
        tool = event.get("tool")
        tool_args = event.get("tool_args") or {}
        if tool == "write_python_script":
            code = tool_args.get("code")
            if isinstance(code, str) and code.strip():
                return code
        if tool == "execute_python":
            tool_result = event.get("tool_result") or {}
            if tool_result.get("exit_code") != 0:
                continue
            code = tool_args.get("code")
            if isinstance(code, str) and code.strip():
                return code
    return None


def sync_transform_source_from_agent(result: dict[str, Any]) -> None:
    """Show the agent script as transform source in lamin.ai, not lag-cli __main__.py."""
    import lamindb as ln
    from lamindb._secret_redaction import redact_secrets_in_source_code

    source = pick_agent_source_code(result)
    if not source:
        return

    run = ln.context.run
    if run is None or run.transform is None:
        return

    source_to_store, _ = redact_secrets_in_source_code(source)
    transform = run.transform
    transform.source_code = source_to_store
    transform.save()

    # also keep a copy on the run for the report UI
    run.params.update({"generated_source": source_to_store})
