from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner
from laminagent._lag import (
    _extract_runnable_keys_from_prompt,
    _log_gemini_usage_record,
    _parse_generated_paths,
    _print_generated_tool_contents,
    _resolve_prompt_runnable_paths,
    _set_current_project_env,
    _warn_if_missing_project,
    lag,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _clear_lamin_current_project(monkeypatch) -> None:
    monkeypatch.delenv("LAMIN_CURRENT_PROJECT", raising=False)


@pytest.fixture(autouse=True)
def _bypass_lag_flow_wrapper(monkeypatch) -> None:
    callback = lag.callback
    unwrapped_callback = callback
    while hasattr(unwrapped_callback, "__wrapped__"):
        unwrapped_callback = unwrapped_callback.__wrapped__

    def _callback_without_flow(*args, **kwargs):
        ctx = click.get_current_context()
        return unwrapped_callback(ctx, *args, **kwargs)

    monkeypatch.setattr(lag, "callback", _callback_without_flow)


def test_parse_generated_paths_filters_empty_entries(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    paths = _parse_generated_paths(f"{a},,{b},")
    assert paths == [a.resolve(), b.resolve()]


def test_print_generated_tool_contents_prints_each_file_once(
    tmp_path: Path, capsys
) -> None:
    a = tmp_path / "a.py"
    a.write_text("print('a')\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("print('b')\n", encoding="utf-8")

    _print_generated_tool_contents([a, b, a])
    output = capsys.readouterr().out

    assert output.count("[Generated Tool ") == 2
    assert "print('a')" in output
    assert "print('b')" in output


def test_set_current_project_env_sets_current_project(monkeypatch) -> None:
    monkeypatch.delenv("LAMIN_CURRENT_PROJECT", raising=False)
    project = _set_current_project_env("demo-project")
    assert project == "demo-project"
    assert os.environ["LAMIN_CURRENT_PROJECT"] == "demo-project"


def test_warn_if_missing_project_logs_warning(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_warning(message: str) -> None:
        calls.append(message)

    monkeypatch.setattr("laminagent._lag.logger.warning", _fake_warning)
    _warn_if_missing_project(None)
    assert len(calls) == 1


def test_extract_runnable_keys_from_prompt_deduplicates() -> None:
    prompt = "rerun test-lag/create_fasta.py and test-lag/create_fasta.py plus x.py"
    keys = _extract_runnable_keys_from_prompt(prompt)
    assert keys == ["test-lag/create_fasta.py", "x.py"]


def test_resolve_prompt_runnable_paths_requires_explicit_key() -> None:
    with pytest.raises(
        click.ClickException, match="Default mode executes existing tools only"
    ):
        _resolve_prompt_runnable_paths("please rerun the tool")


def test_lag_eval_setup_routes_to_setup_handler(monkeypatch) -> None:
    called: dict[str, Path | None] = {}

    def _fake_setup(script: Path | None) -> None:
        called["script"] = script

    monkeypatch.setattr("laminagent._lag.setup_from_script_or_cwd", _fake_setup)
    runner = CliRunner()
    result = runner.invoke(lag, ["eval", "setup"])

    assert result.exit_code == 0
    assert called["script"] is None


def test_lag_eval_setup_accepts_script_argument(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "tests" / "tasks" / "test_01.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    called: dict[str, Path | None] = {}

    def _fake_setup(received_script: Path | None) -> None:
        called["script"] = received_script

    monkeypatch.setattr("laminagent._lag.setup_from_script_or_cwd", _fake_setup)
    runner = CliRunner()
    result = runner.invoke(lag, ["eval", "setup", str(script)])

    assert result.exit_code == 0
    assert called["script"] == script


def test_lag_default_mode_still_requires_prompt() -> None:
    runner = CliRunner()
    result = runner.invoke(lag, [])
    assert result.exit_code != 0
    assert "--prompt" in result.output


def test_lag_default_mode_executes_prompt_path(monkeypatch) -> None:
    def _fake_execute(prompt: str) -> dict[str, str | None]:
        assert prompt == "run test-lag/create_fasta.py"
        return {
            "run_uid": "run-1",
            "resolved_paths": "",
            "final_text": "done",
        }

    monkeypatch.setattr("laminagent._lag.execute_existing_from_prompt", _fake_execute)
    runner = CliRunner()
    result = runner.invoke(lag, ["--prompt", "run test-lag/create_fasta.py"])

    assert result.exit_code == 0
    assert "run_uid=run-1" in result.output


def test_log_gemini_usage_record_writes_record(monkeypatch) -> None:
    class FakeRecord:
        payload = None

        def __init__(self, *, features, type):
            self.features = features
            self.type = type

        def save(self):
            FakeRecord.payload = {"features": self.features, "type": self.type}
            return self

    monkeypatch.setattr("laminagent._lag.ensure_eval_task", lambda **_kwargs: "task")
    monkeypatch.setattr("laminagent._lag.ln.Record", FakeRecord)
    monkeypatch.setattr("laminagent._lag._current_commit_hash16", lambda: "abc123")
    monkeypatch.setattr("laminagent._lag._current_runner_env", lambda: "github_hosted")

    _log_gemini_usage_record(
        {
            "n_call_count": 1,
            "n_prompt_tokens": 2,
            "n_output_tokens": 3,
            "n_total_tokens": 5,
        },
        package_version="0.1.0",
        duration_in_sec=0.2,
        task_name="tool.py",
    )

    assert FakeRecord.payload is not None
    assert FakeRecord.payload["type"] == "task"
    assert FakeRecord.payload["features"]["package_version"] == "0.1.0"
    assert FakeRecord.payload["features"]["n_total_tokens"] == 5
