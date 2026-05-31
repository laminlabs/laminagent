from __future__ import annotations

import os
import re
from pathlib import Path

import click
import lamindb as ln
from dotenv import load_dotenv
from lamin_utils import logger

from .agent import run_agent
from .do_executor import execute_plan, execute_runnable_paths, find_plan_file
from .output_saver import save_generated_tool_files
from .run_context import RunContext, create_run_uid

_STEP_PATTERN = re.compile(r"^step (\d+):\s*(.*)$")
_GEMINI_ATTEMPT_PATTERN = re.compile(r"^gemini request attempt (\d+)/(\d+)$")
_RUNNABLE_KEY_PATTERN = re.compile(r"([A-Za-z0-9_./-]+\.(?:py|ipynb))")
_COLOR_ENABLED = os.getenv("NO_COLOR") is None


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
            "Default mode executes existing tools only. Include at least one .py/.ipynb tool key/path in --prompt, or use --plan to create/update tools."
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


def run_agent_mode(
    *,
    mode: str,
    prompt: str,
    output_file: Path | None,
    model: str,
    track_outputs: bool,
) -> dict[str, str | None]:
    workspace_env_path = Path("~/llms.env").expanduser()
    load_dotenv(dotenv_path=workspace_env_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise click.ClickException("GEMINI_API_KEY not found in ~/llms.env")

    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)

    suffix = "md" if mode == "plan" else "py"
    default_name = f"{mode}_{run_uid}.{suffix}"
    output_path = output_file or Path(default_name)

    run_context = RunContext(
        run_uid=run_uid,
        mode=mode,
        prompt=prompt,
        model=model,
        track_outputs=track_outputs,
    )
    result = run_agent(
        api_key=api_key,
        run_context=run_context,
        output_file=output_path,
        progress_callback=_progress,
    )

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
    if mode == "plan":
        save_generated_tool_files(generated_files)
    return {
        "run_uid": run_uid,
        "generated_path": generated_file if isinstance(generated_file, str) else None,
        "generated_paths": ",".join(generated_files),
        "final_text": str(result.get("final_text", "") or "").strip(),
    }


def execute_the_plan(
    prompt: str,
    plan_file: Path,
) -> dict[str, str | None]:
    lamindb_run_uid = str(getattr(ln.context.run, "uid", "") or "") or None
    run_uid = create_run_uid(lamindb_run_uid)

    result = execute_plan(
        prompt=prompt,
        plan_file=plan_file,
        run_uid=run_uid,
    )
    return {
        "run_uid": run_uid,
        "plan_path": str(plan_file),
        "final_text": str(result.get("final_text", "")),
    }


def execute_generated(
    *,
    prompt: str,
    generated_paths_csv: str,
) -> dict[str, str | None]:
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
    }


def execute_existing_from_prompt(prompt: str) -> dict[str, str | None]:
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
    }


@click.command()
@click.option("--prompt", required=True, type=str, help="User prompt.")
@click.option(
    "--plan",
    "plan_mode",
    is_flag=True,
    help="Switch to planning mode (plan generation).",
)
@click.option("--output-file", type=click.Path(path_type=Path), default=None)
@click.option("--model", type=str, default="gemini-flash-latest", show_default=True)
@click.option(
    "--plan-file",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Optional path to plan file to execute in default mode.",
)
@click.option(
    "--no-track",
    is_flag=True,
    help="Disable automatic insertion of ln.track()/ln.finish() in generated scripts/notebooks.",
)
@click.option(
    "--project",
    type=str,
    default=None,
    callback=_project_option_callback,
    help="Project name to set as LAMIN_CURRENT_PROJECT for the initiated run.",
)
@ln.flow("wDJpT3xdqjY8")
def main(
    prompt: str,
    plan_mode: bool,
    output_file: Path | None,
    model: str,
    plan_file: Path | None,
    no_track: bool,
    project: str | None,
) -> None:
    """LAG CLI."""
    _warn_if_missing_project(project)
    if plan_mode:
        outcome = run_agent_mode(
            mode="plan",
            prompt=prompt,
            output_file=output_file,
            model=model,
            track_outputs=not no_track,
        )
        _echo_section("Run")
        _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
        if outcome["generated_path"]:
            _echo_key_value(
                "generated",
                str(outcome["generated_path"]),
                value_color="bright_magenta",
            )
        if outcome["final_text"]:
            _echo_section("Model Output")
            _secho(str(outcome["final_text"]), dim=True)
        return

    chosen_plan_file = find_plan_file(plan_file)
    if chosen_plan_file is not None:
        outcome = execute_the_plan(
            prompt=prompt,
            plan_file=chosen_plan_file,
        )
        _echo_section("Run")
        _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
        _echo_key_value("plan", str(outcome["plan_path"]), value_color="magenta")
        _secho(str(outcome["final_text"]), dim=True)
        return

    outcome = execute_existing_from_prompt(prompt)
    _echo_section("Run")
    _echo_key_value("run_uid", str(outcome["run_uid"]), value_color="green")
    if outcome["resolved_paths"]:
        resolved_paths = _parse_generated_paths(str(outcome["resolved_paths"]))
        for resolved_path in resolved_paths:
            _echo_key_value(
                "resolved", str(resolved_path), value_color="bright_magenta"
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


if __name__ == "__main__":
    main()
