from __future__ import annotations

import json
import os
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import lamindb as ln
from dotenv import load_dotenv
from lamin_utils import logger

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Text = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False

from ._agent import run_agent
from ._do_executor import execute_runnable_paths, execute_tool, find_tool_file
from ._output_saver import save_generated_tool_files
from ._run_context import RunContext, create_run_uid
from ._setup import get_task, normalize_task_name, setup

if TYPE_CHECKING:
    from collections.abc import Callable

_STEP_PATTERN = re.compile(r"^step (\d+):\s*(.*)$")
_GEMINI_ATTEMPT_PATTERN = re.compile(r"^gemini request attempt (\d+)/(\d+)$")
_RUNNABLE_KEY_PATTERN = re.compile(r"([A-Za-z0-9_./-]+\.py)")
_COLOR_ENABLED = os.getenv("NO_COLOR") is None
_USAGE_FEATURES_NAMES = (
    "n_call_count",
    "n_prompt_tokens",
    "n_output_tokens",
    "n_total_tokens",
)
_TRACE_REDACT_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
    "x-goog-api-key",
}


def _secho(
    message: str,
    *,
    fg: str | None = None,
    bold: bool = False,
    dim: bool = False,
    nl: bool = True,
) -> None:
    click.secho(message, fg=fg, bold=bold, dim=dim, nl=nl, color=_COLOR_ENABLED)


def _echo_info(message: str) -> None:
    _secho(f"→ {message}", fg="black")


def _echo_success(message: str) -> None:
    _secho(f"✓ {message}", fg="green")


def _echo_warning(message: str) -> None:
    _secho(f"! {message}", fg="yellow")


def _echo_section(title: str) -> None:
    _secho(f"\n[{title}]", fg="bright_cyan", bold=True)


def _echo_key_value(key: str, value: str, *, value_color: str | None = None) -> None:
    _secho("→ ", nl=False, fg="black")
    _secho(f"{key}=", nl=False, fg="black")
    _secho(value, fg=value_color)


def _json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def _parse_json_payload(payload_str: str) -> object | None:
    try:
        return json.loads(payload_str)
    except json.JSONDecodeError:
        return None


def _truncate(value: str, *, max_chars: int = 6000) -> str:
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars]}\n... [truncated {len(value) - max_chars} chars]"


def _redact_payload(payload: object) -> object:
    if isinstance(payload, dict):
        redacted: dict[str, object] = {}
        for key, value in payload.items():
            normalized = str(key).strip().lower()
            if normalized in _TRACE_REDACT_KEYS:
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = _redact_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    return payload


def _console() -> Console | None:
    if not _RICH_AVAILABLE:
        return None
    assert Console is not None
    return Console(color_system="auto", no_color=not _COLOR_ENABLED, soft_wrap=True)


def _print_rich_json(title: str, payload: object) -> None:
    console = _console()
    rendered = _json_dumps(_redact_payload(payload))
    if console is None:
        _echo_section(title)
        _secho(rendered, dim=True)
        return
    assert Panel is not None
    assert Text is not None
    wrapped = Text(rendered, no_wrap=False, overflow="fold")
    console.print(
        Panel(
            wrapped,
            title=title,
            border_style="cyan",
        )
    )


def _collect_tool_key_counts(trace_events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in trace_events:
        if event.get("event") != "tool_call":
            continue
        args = event.get("tool_args", {})
        if not isinstance(args, dict):
            continue
        key = args.get("key")
        if not isinstance(key, str) or not key.strip():
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _summarize_tool_result_payload(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return [str(payload)]
    lines: list[str] = []
    message = payload.get("message")
    if isinstance(message, str) and message.strip():
        lines.append(message.strip())

    status = payload.get("status")
    if isinstance(status, str) and status.strip():
        lines.append(f"status: {status.strip()}")

    if isinstance(payload.get("file"), str):
        lines.append(f"file: {payload['file']}")
    if isinstance(payload.get("key"), str) and payload.get("key"):
        lines.append(f"query key: {payload['key']}")

    searched = payload.get("searched_instances")
    if isinstance(searched, list) and searched:
        lines.append(f"searched instances: {', '.join(str(item) for item in searched)}")

    results = payload.get("results")
    if isinstance(results, list):
        lines.append(f"results: {len(results)}")
        for idx, item in enumerate(results[:3], start=1):
            if not isinstance(item, dict):
                lines.append(f"  {idx}. {str(item)[:120]}")
                continue
            item_type = str(item.get("type", "result"))
            item_key = str(item.get("key", "") or item.get("uid", ""))
            description = str(item.get("description", "")).strip()
            descriptor = f"{item_type}: {item_key}".strip(": ")
            if description:
                lines.append(f"  {idx}. {descriptor} — {description[:120]}")
            else:
                lines.append(f"  {idx}. {descriptor}")
        if len(results) > 3:
            lines.append(f"  ... and {len(results) - 3} more")

    matches = payload.get("matches")
    if isinstance(matches, list):
        lines.append(f"matches: {len(matches)}")
        for idx, item in enumerate(matches[:3], start=1):
            lines.append(f"  {idx}. {str(item)[:120]}")
        if len(matches) > 3:
            lines.append(f"  ... and {len(matches) - 3} more")

    warnings = payload.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.append(f"warnings: {len(warnings)}")
        for warning in warnings[:2]:
            lines.append(f"  - {str(warning)[:180]}")
        if len(warnings) > 2:
            lines.append(f"  ... and {len(warnings) - 2} more")

    if not lines:
        lines.append("tool returned no additional details")
    return lines


def _summarize_tool_call_args(payload: object) -> str:
    if not isinstance(payload, dict) or not payload:
        return "no arguments"
    key = payload.get("key")
    topic = payload.get("topic")
    filename = payload.get("filename")
    template_path = payload.get("template_path")
    skills_root = payload.get("skills_root")
    code = payload.get("code")

    parts: list[str] = []
    if isinstance(key, str) and key:
        parts.append(f"key='{key}'")
    if isinstance(topic, str) and topic:
        parts.append(f"topic='{topic}'")
    if isinstance(filename, str) and filename:
        parts.append(f"filename='{filename}'")
    if isinstance(template_path, str) and template_path:
        parts.append(f"template='{template_path}'")
    if isinstance(skills_root, str) and skills_root:
        parts.append(f"skills_root='{skills_root}'")
    if isinstance(code, str):
        parts.append(f"code_chars={len(code)}")
    if not parts:
        parts = [f"{name}={str(value)[:120]}" for name, value in payload.items()]
    return ", ".join(parts)


def _format_tool_call_detail(detail: str) -> tuple[str, str] | None:
    body = detail.removeprefix("tool call -> ").strip()
    if " args=" not in body:
        return body, "no arguments"
    name, args_str = body.split(" args=", 1)
    parsed_args = _parse_json_payload(args_str.strip())
    if parsed_args is None:
        return name.strip(), args_str.strip()
    return name.strip(), _summarize_tool_call_args(parsed_args)


def _format_progress_message_for_log(message: str) -> str:
    step_match = _STEP_PATTERN.match(message)
    if step_match is None:
        return message
    step, detail = step_match.groups()
    if detail.startswith("tool call -> "):
        formatted = _format_tool_call_detail(detail)
        if formatted is None:
            return message
        name, args_summary = formatted
        return f"step {step}: tool call -> {name} ({args_summary})"
    if not detail.startswith("tool result payload="):
        return message
    payload_str = detail.removeprefix("tool result payload=")
    payload = _parse_json_payload(payload_str)
    if payload is None:
        return message
    summary = " | ".join(_summarize_tool_result_payload(payload))
    return f"step {step}: tool result payload: {summary}"


def _print_verbose_trace(trace_events: list[dict[str, Any]], *, title: str) -> None:
    if not trace_events:
        return
    _echo_section(title)
    for event in [e for e in trace_events if e.get("step") == 0]:
        if event.get("event") == "runnables_loaded":
            _echo_info(f"source={event.get('source')}")
            _echo_info(f"prompt={event.get('prompt')}")
            _print_rich_json("Runnables", event.get("runnables_detected", []))
    steps = sorted(
        {
            int(event["step"])
            for event in trace_events
            if isinstance(event.get("step"), int) and int(event["step"]) > 0
        }
    )
    if not steps:
        return
    for step in steps:
        _secho(f"\nStep {step}", fg="bright_cyan", bold=True)
        for event in [e for e in trace_events if e.get("step") == step]:
            event_type = str(event.get("event", ""))
            if event_type == "llm_request":
                _print_rich_json("Request", event.get("request_payload", {}))
            elif event_type == "llm_response":
                _print_rich_json("Response", event.get("response_payload", {}))
                usage = event.get("usage_metadata")
                if usage:
                    _print_rich_json("Usage Metadata", usage)
            elif event_type == "tool_call":
                _echo_info(f"tool call: {event.get('tool')}")
                _print_rich_json("Tool Call Args", event.get("tool_args", {}))
            elif event_type == "tool_result":
                _echo_info(f"tool result: {event.get('tool')}")
                _print_rich_json("Tool Result", event.get("tool_result", {}))
            elif event_type == "runnable_missing":
                _echo_warning(f"missing runnable: {event.get('path')}")
            elif event_type == "script_executed":
                path = str(event.get("path", ""))
                exit_code = str(event.get("exit_code", ""))
                _echo_info(f"script: {path} exit_code={exit_code}")
                _print_rich_json(
                    "Execution Output",
                    {
                        "stdout": event.get("stdout", ""),
                        "stderr": event.get("stderr", ""),
                    },
                )


def _progress_verbose_live() -> Callable[[str], None]:
    current_step: int | None = None

    def _callback(message: str) -> None:
        nonlocal current_step
        if message.startswith("prompt: "):
            _secho("→ prompt: ", nl=False, fg="black")
            _secho(message.removeprefix("prompt: "), fg="cyan")
            return
        if message.startswith("gemini request attempt"):
            attempt_match = _GEMINI_ATTEMPT_PATTERN.match(message)
            if attempt_match is not None and int(attempt_match.group(1)) > 1:
                _secho(f"→ {message}", fg="magenta")
            return
        if message.startswith("gemini transient status"):
            _secho(f"→ {message}", fg="yellow")
            return
        if message.startswith("gemini request failed"):
            _secho(f"→ {message}", fg="red")
            return
        if message == "model finished without further tool calls":
            _secho(f"→ {message}", fg="green")
            return

        step_match = _STEP_PATTERN.match(message)
        if step_match is None:
            _echo_info(message)
            return

        step, detail = step_match.groups()
        step_num = int(step)
        if detail.startswith("waiting for model response"):
            if current_step != step_num:
                _echo_section(f"Step {step_num}")
                current_step = step_num
            return
        if current_step != step_num:
            _echo_section(f"Step {step_num}")
            current_step = step_num
        if detail.startswith("model text: "):
            _secho("→ model text: ", nl=False, fg="blue")
            _secho(detail.removeprefix("model text: "), dim=True)
        elif detail.startswith("llm request payload="):
            payload_str = detail.removeprefix("llm request payload=")
            parsed_payload = _parse_json_payload(payload_str)
            if parsed_payload is None:
                _secho("→ request payload:", fg="black")
                _secho(f"  {payload_str}", dim=True)
            else:
                _print_rich_json("Request", parsed_payload)
        elif detail.startswith("llm response payload="):
            payload_str = detail.removeprefix("llm response payload=")
            parsed_payload = _parse_json_payload(payload_str)
            if parsed_payload is None:
                _secho("→ response payload:", fg="black")
                _secho(f"  {payload_str}", dim=True)
            else:
                _print_rich_json("Response", parsed_payload)
        elif detail.startswith("tool call -> "):
            body = detail.removeprefix("tool call -> ").strip()
            if " args=" not in body:
                _secho(f"→ tool call -> {body}", fg="magenta")
            else:
                name, _args_str = body.split(" args=", 1)
                _secho(f"→ tool call -> {name.strip()}", fg="magenta")
        elif detail.startswith("wrote file "):
            _secho(f"→ {detail}", fg="green")
        elif detail.startswith("tool result status="):
            return
        elif detail.startswith("tool result payload="):
            return
        else:
            _secho(f"→ {detail}", dim=True)

    return _callback


def _parse_generated_paths(generated_paths_csv: str) -> list[Path]:
    return [
        Path(path_str).resolve()
        for path_str in generated_paths_csv.split(",")
        if path_str.strip()
    ]


def _extract_runnable_keys_from_prompt(prompt: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for key in _RUNNABLE_KEY_PATTERN.findall(prompt):
        normalized = key.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        keys.append(normalized)
    return keys


def _materialize_transform_source(key: str) -> Path | None:
    transform = ln.Transform.filter(key=key).one_or_none()
    if transform is None:
        return None
    source_code = str(getattr(transform, "source_code", "") or "")
    if not source_code:
        artifact = ln.Artifact.filter(
            transform=transform, suffix=Path(key).suffix
        ).first()
        if artifact is not None:
            try:
                source_code = artifact.open().read().decode("utf-8")
            except Exception as exc:
                raise click.ClickException(
                    f"Found transform '{key}' but failed to read source artifact: {exc}"
                ) from exc
    if not source_code:
        raise click.ClickException(
            f"Found transform '{key}' but no executable source code was available."
        )

    output_path = Path(key).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source_code, encoding="utf-8")
    return output_path


def _resolve_existing_runnable_path(key: str) -> Path:
    local_path = Path(key).resolve()
    if local_path.exists():
        return local_path
    materialized = _materialize_transform_source(key)
    if materialized is not None:
        return materialized
    raise click.ClickException(
        f"Runnable tool '{key}' was not found as a local file or transform key in the current instance."
    )


def _resolve_prompt_runnable_paths(prompt: str) -> list[Path]:
    keys = _extract_runnable_keys_from_prompt(prompt)
    return [_resolve_existing_runnable_path(key) for key in keys]


def _set_current_project_env(project: str | None) -> str | None:
    if project:
        os.environ["LAMIN_CURRENT_PROJECT"] = project
    return project


def _project_option_callback(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> str | None:
    return _set_current_project_env(value)


def _warn_if_missing_project(project: str | None) -> None:
    if not project:
        logger.warning("no --project was provided and LAMIN_CURRENT_PROJECT is not set")


def _print_generated_tool_contents(paths: list[Path]) -> None:
    seen: set[Path] = set()
    for path in paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        _echo_section(f"Generated Tool {path.name}")
        _secho(str(path), fg="black")
        content = path.read_text(encoding="utf-8")
        _secho(content, dim=True)
        _secho("--- end generated tool ---", fg="black")


def _normalize_gemini_usage(payload: object) -> dict[str, int]:
    usage = dict.fromkeys(_USAGE_FEATURES_NAMES, 0)
    if not isinstance(payload, dict):
        return usage
    for key in _USAGE_FEATURES_NAMES:
        value = payload.get(key, 0)
        usage[key] = int(value) if isinstance(value, int) else 0
    return usage


def _current_commit_hash16() -> str | None:
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return commit_hash[:16] if commit_hash else None


def _current_runner_env() -> str | None:
    if (
        os.getenv("GITHUB_ACTIONS") == "true"
        and os.getenv("RUNNER_ENVIRONMENT") == "github-hosted"
    ):
        return "github_hosted"
    return None


def _record_usage_task_name(generated_path: str | None) -> str:
    pytest_current_test = os.getenv("PYTEST_CURRENT_TEST", "")
    if pytest_current_test:
        node_id = pytest_current_test.split(" ", 1)[0]
        test_path = node_id.split("::", 1)[0].replace("\\", "/")
        if "/tests/tasks/test_" in f"/{test_path}":
            return normalize_task_name(Path(test_path).name)

    if generated_path:
        return normalize_task_name(Path(generated_path).name)
    return "lag_authoring"


def _log_trace_payload(payload: dict[str, Any]) -> None:
    redacted = _redact_payload(payload)
    if not isinstance(redacted, dict):
        logger.info("lag_trace_summary=unstructured_payload")
        return
    trace_events = redacted.get("trace_events")
    n_trace_events = len(trace_events) if isinstance(trace_events, list) else 0
    summary = {
        "action": redacted.get("action"),
        "run_uid": redacted.get("run_uid"),
        "model": redacted.get("model"),
        "n_trace_events": n_trace_events,
    }
    logger.info(f"lag_trace_summary={_json_dumps(summary)}")


def _log_gemini_usage_record(
    usage: dict[str, int],
    *,
    package_version: str,
    duration_in_sec: float,
    task_name: str,
) -> None:
    if usage["n_call_count"] <= 0:
        return
    task = get_task(task_name=task_name)
    if task is None or task.schema_id is None:
        _echo_warning(
            f"LagEval task registry '{task_name}' is not configured; "
            "skipping LagEval usage record (run `lag setup` to enable it)."
        )
        return
    ln.Record(
        features={
            "package_version": package_version,
            "duration_in_sec": duration_in_sec,
            "commit_hash16": _current_commit_hash16(),
            "runner_env": _current_runner_env(),
            "n_call_count": usage["n_call_count"],
            "n_prompt_tokens": usage["n_prompt_tokens"],
            "n_output_tokens": usage["n_output_tokens"],
            "n_total_tokens": usage["n_total_tokens"],
        },
        type=task,
    ).save()


def _log_gemini_usage_to_run_features(usage: dict[str, int]) -> None:
    if usage["n_call_count"] <= 0:
        return
    ln.context.run.features.add_values(dict(usage))


def _print_gemini_usage_summary(
    usage: dict[str, int], trace_events: list[dict[str, Any]] | None = None
) -> None:
    if usage["n_call_count"] <= 0:
        return
    _echo_section("Usage")
    _echo_key_value("n_call_count", str(usage["n_call_count"]), value_color="yellow")
    _echo_key_value("n_prompt_tokens", str(usage["n_prompt_tokens"]))
    _echo_key_value("n_output_tokens", str(usage["n_output_tokens"]))
    _echo_key_value("n_total_tokens", str(usage["n_total_tokens"]), value_color="cyan")
    calls = usage["n_call_count"]
    avg_per_call = usage["n_total_tokens"] / calls if calls else 0.0
    output_prompt_ratio = (
        usage["n_output_tokens"] / usage["n_prompt_tokens"]
        if usage["n_prompt_tokens"] > 0
        else 0.0
    )
    _echo_key_value("avg_tokens_per_call", f"{avg_per_call:.2f}")
    _echo_key_value("output_prompt_ratio", f"{output_prompt_ratio:.3f}")
    if trace_events:
        repeated = sorted(
            _collect_tool_key_counts(trace_events).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        if repeated:
            _echo_key_value(
                "top_tool_keys",
                ", ".join(f"{key}({count})" for key, count in repeated[:5]),
                value_color="magenta",
            )


def _current_package_version() -> str:
    try:
        return version("laminagent")
    except PackageNotFoundError:
        return "0.0.0"


def run_agent_authoring(
    *,
    prompt: str,
    output_file: Path | None,
    model: str,
    track_outputs: bool,
) -> dict[str, Any]:
    workspace_env_path = Path("~/llms.env").expanduser()
    load_dotenv(dotenv_path=workspace_env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise click.ClickException("GEMINI_API_KEY not found in ~/llms.env")

    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)

    suffix = "py"
    default_name = f"author_{run_uid}.{suffix}"
    output_path = output_file or Path(default_name)

    run_context = RunContext(
        run_uid=run_uid,
        prompt=prompt,
        model=model,
        track_outputs=track_outputs,
    )
    start = time.perf_counter()
    progress_callback: Callable[[str], None] | None = _progress_verbose_live()
    result = run_agent(
        api_key=api_key,
        run_context=run_context,
        output_file=output_path,
        progress_callback=progress_callback,
    )
    elapsed = time.perf_counter() - start

    generated_file = result.get("generated_file")
    generated_files = [
        path_str
        for path_str in result.get("generated_files", [])
        if isinstance(path_str, str) and path_str
    ]
    resolved_runnable_path = result.get("resolved_runnable_path")
    if (
        isinstance(resolved_runnable_path, str)
        and resolved_runnable_path
        and resolved_runnable_path not in generated_files
    ):
        generated_files.append(resolved_runnable_path)
    save_generated_tool_files(generated_files)
    return {
        "run_uid": run_uid,
        "generated_path": generated_file if isinstance(generated_file, str) else None,
        "generated_paths": ",".join(generated_files),
        "final_text": str(result.get("final_text", "") or "").strip(),
        "llm_usage": _normalize_gemini_usage(result.get("llm_usage")),
        "duration_in_sec": elapsed,
        "trace_events": result.get("trace_events", []),
    }


def execute_the_tool(
    prompt: str,
    tool_file: Path,
) -> dict[str, Any]:
    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)

    result = execute_tool(
        prompt=prompt,
        tool_file=tool_file,
        run_uid=run_uid,
    )
    return {
        "run_uid": run_uid,
        "tool_path": str(tool_file),
        "final_text": str(result.get("final_text", "")),
        "trace_events": result.get("trace_events", []),
    }


def execute_existing_from_prompt(prompt: str) -> dict[str, Any]:
    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)
    runnable_paths = _resolve_prompt_runnable_paths(prompt)
    result = execute_runnable_paths(
        prompt=prompt,
        runnable_paths=runnable_paths,
        run_uid=run_uid,
        source="prompt_existing_tools",
    )
    return {
        "run_uid": run_uid,
        "resolved_paths": ",".join(str(path) for path in runnable_paths),
        "final_text": str(result.get("final_text", "")),
        "trace_events": result.get("trace_events", []),
    }


@click.group(invoke_without_command=True)
@click.option("--prompt", required=False, type=str, help="User prompt.")
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional output filename when authoring a new script.",
)
@click.option("--model", type=str, default="gemini-flash-latest", show_default=True)
@click.option(
    "--no-track",
    is_flag=True,
    help="Disable automatic insertion of ln.track()/ln.finish() in generated scripts.",
)
@click.option(
    "--project",
    type=str,
    default=None,
    callback=_project_option_callback,
    help="Project name to set as LAMIN_CURRENT_PROJECT for the initiated run.",
)
@ln.flow("wDJpT3xdqjY8")
def lag(
    prompt: str | None,
    output_file: Path | None,
    model: str,
    no_track: bool,
    project: str | None,
) -> None:
    """LAG CLI."""
    ctx = click.get_current_context()
    if ctx.invoked_subcommand is not None:
        return

    if not prompt:
        raise click.UsageError(
            "`--prompt` is required for lag; use `lag setup` to initialize setup records."
        )
    prompt_text = prompt

    _warn_if_missing_project(project)
    runnable_keys = _extract_runnable_keys_from_prompt(prompt_text)
    if runnable_keys:
        outcome = execute_existing_from_prompt(prompt_text)
        _echo_section("Run")
        _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
        if outcome["resolved_paths"]:
            resolved_paths = _parse_generated_paths(str(outcome["resolved_paths"]))
            for resolved_path in resolved_paths:
                _echo_key_value(
                    "resolved", str(resolved_path), value_color="bright_magenta"
                )
        _log_trace_payload(
            {
                "action": "execute",
                "run_uid": str(outcome["run_uid"]),
                "prompt": prompt_text,
                "resolved_paths": str(outcome.get("resolved_paths", "")),
                "trace_events": list(outcome.get("trace_events", [])),
                "final_text": str(outcome.get("final_text", "")),
            }
        )
        if outcome["final_text"]:
            _echo_section("Model Output")
            _secho(str(outcome["final_text"]), dim=True)
        return

    chosen_tool_file = find_tool_file()
    if chosen_tool_file is not None:
        outcome = execute_the_tool(
            prompt=prompt_text,
            tool_file=chosen_tool_file,
        )
        _echo_section("Run")
        _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
        _echo_key_value("tool", str(outcome["tool_path"]), value_color="magenta")
        _log_trace_payload(
            {
                "action": "execute",
                "run_uid": str(outcome["run_uid"]),
                "prompt": prompt_text,
                "tool_path": str(outcome["tool_path"]),
                "trace_events": list(outcome.get("trace_events", [])),
                "final_text": str(outcome.get("final_text", "")),
            }
        )
        _secho(str(outcome["final_text"]), dim=True)
        return

    outcome = run_agent_authoring(
        prompt=prompt_text,
        output_file=output_file,
        model=model,
        track_outputs=not no_track,
    )
    gemini_usage = _normalize_gemini_usage(outcome.get("llm_usage"))
    _log_gemini_usage_to_run_features(gemini_usage)
    _log_gemini_usage_record(
        gemini_usage,
        package_version=_current_package_version(),
        duration_in_sec=float(outcome.get("duration_in_sec", 0.0) or 0.0),
        task_name=_record_usage_task_name(
            str(outcome.get("generated_path") or "") or None
        ),
    )
    _echo_section("Run")
    _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
    _print_gemini_usage_summary(
        gemini_usage, trace_events=list(outcome.get("trace_events", []))
    )
    if outcome["generated_path"]:
        _echo_key_value(
            "generated",
            str(outcome["generated_path"]),
            value_color="bright_magenta",
        )
    _log_trace_payload(
        {
            "action": "author",
            "run_uid": str(outcome["run_uid"]),
            "prompt": prompt_text,
            "model": model,
            "trace_events": list(outcome.get("trace_events", [])),
            "llm_usage": gemini_usage,
            "final_text": str(outcome.get("final_text", "")),
        }
    )
    if outcome["final_text"]:
        _echo_section("Model Output")
        _secho(str(outcome["final_text"]), dim=True)


@lag.command("setup")
@click.argument(
    "script",
    required=False,
    type=click.Path(path_type=Path, exists=True),
)
def setup_command(script: Path | None) -> None:
    """Set up LagEval registry and schema."""
    setup(script=script)
