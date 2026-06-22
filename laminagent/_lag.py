from __future__ import annotations

import json
import os
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import click
import lamindb as ln
from dotenv import load_dotenv
from lamin_utils import logger

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax

    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional fallback
    Console = None  # type: ignore[assignment]
    Panel = None  # type: ignore[assignment]
    Syntax = None  # type: ignore[assignment]
    _RICH_AVAILABLE = False

from ._agent import run_agent
from ._do_executor import execute_runnable_paths, execute_tool, find_tool_file
from ._output_saver import save_generated_tool_files
from ._run_context import RunContext, create_run_uid
from ._setup import get_task, normalize_task_name, setup

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
    rendered = _truncate(_json_dumps(_redact_payload(payload)))
    if console is None:
        _echo_section(title)
        _secho(rendered, dim=True)
        return
    assert Panel is not None
    assert Syntax is not None
    console.print(
        Panel(
            Syntax(rendered, "json", word_wrap=True),
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


def _progress(message: str) -> None:
    if message.startswith("mode="):
        pretty_message = message.replace("mode=do", "mode=default")
        _echo_info(pretty_message)
        return
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
    if detail.startswith("waiting for model response"):
        return
    _secho(f"→ step {step}: ", nl=False, fg="black")
    if detail.startswith("model text: "):
        _secho("model text: ", nl=False, fg="blue")
        _secho(detail.removeprefix("model text: "), dim=True)
    elif detail.startswith("tool call -> "):
        _secho("tool call -> ", nl=False, fg="magenta")
        _secho(detail.removeprefix("tool call -> "), dim=True)
    elif detail.startswith("wrote file "):
        _secho(detail, fg="green")
    elif detail.startswith("tool result status="):
        status = detail.removeprefix("tool result status=")
        color = "green" if status == "success" else "yellow"
        _secho("tool result status=", nl=False, fg="black")
        _secho(status, fg=color)
    else:
        _secho(detail, dim=True)


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
    if not keys:
        raise click.ClickException(
            "Default mode executes existing tools only. Include at least one .py tool key/path in --prompt, or use --tool to create/update tools."
        )
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
    return "lag_tool_mode"


def _log_trace_payload(payload: dict[str, Any]) -> None:
    redacted = _redact_payload(payload)
    serialized_payload: dict[str, Any]
    if isinstance(redacted, dict):
        serialized_payload = redacted
    else:
        serialized_payload = {"payload": redacted}
    logger.info(f"lag_trace={_json_dumps(serialized_payload)}")


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
        raise click.ClickException(
            f"LagEval task registry '{task_name}' is not configured. "
            "Please run `lag setup` first."
        )
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
    _echo_section("Gemini Usage")
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


def run_agent_mode(
    *,
    mode: str,
    prompt: str,
    output_file: Path | None,
    model: str,
    track_outputs: bool,
    verbose_llm: bool,
) -> dict[str, Any]:
    workspace_env_path = Path("~/llms.env").expanduser()
    load_dotenv(dotenv_path=workspace_env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise click.ClickException("GEMINI_API_KEY not found in ~/llms.env")

    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)

    suffix = "py"
    default_name = f"{mode}_{run_uid}.{suffix}"
    output_path = output_file or Path(default_name)

    run_context = RunContext(
        run_uid=run_uid,
        mode=mode,
        prompt=prompt,
        model=model,
        track_outputs=track_outputs,
    )
    start = time.perf_counter()
    result = run_agent(
        api_key=api_key,
        run_context=run_context,
        output_file=output_path,
        progress_callback=None if verbose_llm else _progress,
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
    if mode == "tool":
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


def execute_generated(
    *,
    prompt: str,
    generated_paths_csv: str,
) -> dict[str, Any]:
    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)
    runnable_paths = _parse_generated_paths(generated_paths_csv)
    result = execute_runnable_paths(
        prompt=prompt,
        runnable_paths=runnable_paths,
        run_uid=run_uid,
        source="generated_outputs",
    )
    return {
        "run_uid": run_uid,
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
@click.pass_context
@click.option("--prompt", required=False, type=str, help="User prompt.")
@click.option(
    "--verbose-llm/--less-verbose",
    "verbose_llm",
    default=True,
    show_default=True,
    help="Show verbose structured execution trace (default). Use --less-verbose for compact logs.",
)
@click.option(
    "--tool",
    "tool_mode",
    is_flag=True,
    help="Switch to toolning mode (tool generation).",
)
@click.option("--output-file", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=str, default="gemini-flash-latest", show_default=True)
@click.option(
    "--tool-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional path to tool file to execute in default mode.",
)
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
    ctx: click.Context,
    prompt: str | None,
    verbose_llm: bool,
    tool_mode: bool,
    output_file: Path | None,
    model: str,
    tool_file: Path | None,
    no_track: bool,
    project: str | None,
) -> None:
    """LAG CLI."""
    if ctx.invoked_subcommand is not None:
        return

    if not prompt:
        raise click.UsageError(
            "`--prompt` is required for default lag mode; use `lag setup` to initialize setup records."
        )
    prompt_text = prompt

    _warn_if_missing_project(project)
    if tool_mode:
        outcome = run_agent_mode(
            mode="tool",
            prompt=prompt_text,
            output_file=output_file,
            model=model,
            track_outputs=not no_track,
            verbose_llm=verbose_llm,
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
        if verbose_llm:
            _print_verbose_trace(
                list(outcome.get("trace_events", [])),
                title="Trace",
            )
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
                "mode": "tool",
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
        return

    chosen_tool_file = find_tool_file(tool_file)
    if chosen_tool_file is not None:
        outcome = execute_the_tool(
            prompt=prompt_text,
            tool_file=chosen_tool_file,
        )
        _echo_section("Run")
        _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
        _echo_key_value("tool", str(outcome["tool_path"]), value_color="magenta")
        if verbose_llm:
            _print_verbose_trace(
                list(outcome.get("trace_events", [])),
                title="Execution Trace",
            )
        _log_trace_payload(
            {
                "mode": "tool_file",
                "run_uid": str(outcome["run_uid"]),
                "prompt": prompt_text,
                "tool_path": str(outcome["tool_path"]),
                "trace_events": list(outcome.get("trace_events", [])),
                "final_text": str(outcome.get("final_text", "")),
            }
        )
        _secho(str(outcome["final_text"]), dim=True)
        return

    outcome = execute_existing_from_prompt(prompt_text)
    _echo_section("Run")
    _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
    if verbose_llm:
        _print_verbose_trace(
            list(outcome.get("trace_events", [])),
            title="Execution Trace",
        )
    if outcome["resolved_paths"]:
        resolved_paths = _parse_generated_paths(str(outcome["resolved_paths"]))
        for resolved_path in resolved_paths:
            _echo_key_value(
                "resolved", str(resolved_path), value_color="bright_magenta"
            )
    _log_trace_payload(
        {
            "mode": "default",
            "run_uid": str(outcome["run_uid"]),
            "prompt": prompt_text,
            "resolved_paths": str(outcome.get("resolved_paths", "")),
            "trace_events": list(outcome.get("trace_events", [])),
            "final_text": str(outcome.get("final_text", "")),
        }
    )
    if outcome.get("generated_path"):
        _echo_key_value(
            "generated",
            str(outcome["generated_path"]),
            value_color="bright_magenta",
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
