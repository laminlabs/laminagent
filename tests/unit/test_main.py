from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner
from laminagent._lag import (
    _extract_runnable_keys_from_prompt,
    _format_progress_message_for_log,
    _log_gemini_usage_record,
    _parse_generated_paths,
    _print_generated_tool_contents,
    _redact_payload,
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
def _stub_setup(monkeypatch) -> None:
    monkeypatch.setattr("laminagent._lag.setup", lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def _bypass_lag_flow_wrapper(monkeypatch) -> None:
    callback = lag.callback
    unwrapped_callback = callback
    while hasattr(unwrapped_callback, "__wrapped__"):
        unwrapped_callback = unwrapped_callback.__wrapped__

    def _callback_without_flow(*args, **kwargs):
        return unwrapped_callback(*args, **kwargs)

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


def test_resolve_prompt_runnable_paths_returns_empty_without_keys() -> None:
    assert _resolve_prompt_runnable_paths("please rerun the tool") == []


def test_lag_setup_routes_to_setup_handler(monkeypatch) -> None:
    called: dict[str, Path | None] = {}

    def _fake_setup(script: Path | None) -> None:
        called["script"] = script

    monkeypatch.setattr("laminagent._lag.setup", _fake_setup)
    runner = CliRunner()
    result = runner.invoke(lag, ["setup"])

    assert result.exit_code == 0
    assert called["script"] is None


def test_lag_setup_accepts_script_argument(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "tests" / "tasks" / "test_01.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    called: dict[str, Path | None] = {}

    def _fake_setup(*, script: Path | None = None, **_kwargs) -> None:
        called["script"] = script

    monkeypatch.setattr("laminagent._lag.setup", _fake_setup)
    runner = CliRunner()
    result = runner.invoke(lag, ["setup", str(script)])

    assert result.exit_code == 0
    assert called["script"] == script


def test_lag_still_requires_prompt() -> None:
    runner = CliRunner()
    result = runner.invoke(lag, [])
    assert result.exit_code != 0
    assert "--prompt" in result.output


def test_lag_default_mode_executes_prompt_path(monkeypatch) -> None:
    monkeypatch.setattr("laminagent._lag.setup", lambda *args, **kwargs: None)

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
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "run_uid=run-1" in clean_output


def test_lag_auto_authoring_is_verbose_by_default(monkeypatch) -> None:
    captured: dict[str, bool] = {}

    def _fake_run_agent_authoring(**kwargs):
        captured["verbose_llm"] = bool(kwargs["verbose_llm"])
        return {
            "run_uid": "run-1",
            "generated_path": None,
            "generated_paths": "",
            "final_text": "ok",
            "llm_usage": {
                "n_call_count": 1,
                "n_prompt_tokens": 1,
                "n_output_tokens": 1,
                "n_total_tokens": 2,
            },
            "duration_in_sec": 0.1,
            "trace_events": [],
        }

    monkeypatch.setattr("laminagent._lag.find_tool_file", lambda: None)
    monkeypatch.setattr(
        "laminagent._lag.run_agent_authoring", _fake_run_agent_authoring
    )
    monkeypatch.setattr(
        "laminagent._lag._log_gemini_usage_to_run_features", lambda *_: None
    )
    monkeypatch.setattr(
        "laminagent._lag._log_gemini_usage_record", lambda *_, **__: None
    )
    runner = CliRunner()
    result = runner.invoke(lag, ["--prompt", "build tool"])

    assert result.exit_code == 0
    assert captured["verbose_llm"] is True


def test_lag_auto_authoring_allows_less_verbose_flag(monkeypatch) -> None:
    captured: dict[str, bool] = {}

    def _fake_run_agent_authoring(**kwargs):
        captured["verbose_llm"] = bool(kwargs["verbose_llm"])
        return {
            "run_uid": "run-1",
            "generated_path": None,
            "generated_paths": "",
            "final_text": "ok",
            "llm_usage": {
                "n_call_count": 1,
                "n_prompt_tokens": 1,
                "n_output_tokens": 1,
                "n_total_tokens": 2,
            },
            "duration_in_sec": 0.1,
            "trace_events": [],
        }

    monkeypatch.setattr("laminagent._lag.find_tool_file", lambda: None)
    monkeypatch.setattr(
        "laminagent._lag.run_agent_authoring", _fake_run_agent_authoring
    )
    monkeypatch.setattr(
        "laminagent._lag._log_gemini_usage_to_run_features", lambda *_: None
    )
    monkeypatch.setattr(
        "laminagent._lag._log_gemini_usage_record", lambda *_, **__: None
    )
    runner = CliRunner()
    result = runner.invoke(lag, ["--less-verbose", "--prompt", "build tool"])

    assert result.exit_code == 0
    assert captured["verbose_llm"] is False


def test_lag_auto_executes_discovered_tool_file(monkeypatch, tmp_path: Path) -> None:
    tool_file = tmp_path / "tool.md"
    tool_file.write_text("- run `a.py`\n", encoding="utf-8")
    called: dict[str, str] = {}

    monkeypatch.setattr("laminagent._lag.find_tool_file", lambda: tool_file)

    def _fake_execute_the_tool(prompt: str, tool_file: Path) -> dict[str, str | list]:
        called["prompt"] = prompt
        called["tool"] = str(tool_file)
        return {
            "run_uid": "run-1",
            "tool_path": str(tool_file),
            "final_text": "done",
            "trace_events": [],
        }

    monkeypatch.setattr("laminagent._lag.execute_the_tool", _fake_execute_the_tool)
    runner = CliRunner()
    result = runner.invoke(lag, ["--prompt", "please run latest tool"])

    assert result.exit_code == 0
    assert called["prompt"] == "please run latest tool"
    assert called["tool"] == str(tool_file)


def test_trace_is_logged_with_redaction(monkeypatch) -> None:
    logged: list[str] = []

    monkeypatch.setattr(
        "laminagent._lag.logger.info", lambda message: logged.append(message)
    )

    def _fake_execute(prompt: str) -> dict[str, object]:
        assert prompt == "run test-lag/create_fasta.py"
        return {
            "run_uid": "run-1",
            "resolved_paths": "",
            "final_text": "done",
            "trace_events": [
                {
                    "step": 1,
                    "event": "tool_call",
                    "tool": "x",
                    "tool_args": {"api_key": "super-secret"},
                }
            ],
        }

    monkeypatch.setattr("laminagent._lag.execute_existing_from_prompt", _fake_execute)
    runner = CliRunner()
    result = runner.invoke(lag, ["--prompt", "run test-lag/create_fasta.py"])

    assert result.exit_code == 0
    assert logged
    assert "lag_trace_summary=" in logged[-1]
    assert "super-secret" not in "".join(logged)


def test_redact_payload_masks_known_secret_keys() -> None:
    payload = {"api_key": "abc", "nested": {"authorization": "Bearer token", "ok": 1}}
    redacted = _redact_payload(payload)
    assert isinstance(redacted, dict)
    assert redacted["api_key"] == "***REDACTED***"
    assert isinstance(redacted["nested"], dict)
    assert redacted["nested"]["authorization"] == "***REDACTED***"


def test_format_progress_message_for_log_summarizes_tool_payload() -> None:
    message = (
        'step 1: tool result payload={"message":"Found 5 LaminDB matches for '
        '\\"artifact\\".","results":[{"type":"artifact","key":"a"},{"type":"artifact","key":"b"}]}'
    )
    formatted = _format_progress_message_for_log(message)
    assert "step 1: tool result payload:" in formatted
    assert 'Found 5 LaminDB matches for "artifact".' in formatted
    assert "results: 2" in formatted


def test_format_progress_message_for_log_summarizes_tool_call_args() -> None:
    message = 'step 2: tool call -> get_lamindb_skill args={"key":"artifact"}'
    formatted = _format_progress_message_for_log(message)
    assert formatted == "step 2: tool call -> get_lamindb_skill (key='artifact')"


def test_log_gemini_usage_record_writes_record(monkeypatch) -> None:
    class FakeTask:
        schema_id = 1

    class FakeRecord:
        payload = None

        def __init__(self, *, features, type):
            self.features = features
            self.type = type

        def save(self):
            FakeRecord.payload = {"features": self.features, "type": self.type}
            return self

    monkeypatch.setattr("laminagent._lag.get_task", lambda **_kwargs: FakeTask())
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
    assert FakeRecord.payload["type"].schema_id == 1
    assert FakeRecord.payload["features"]["package_version"] == "0.1.0"
    assert FakeRecord.payload["features"]["n_total_tokens"] == 5


def test_log_gemini_usage_record_skips_when_task_not_configured(monkeypatch) -> None:
    monkeypatch.setattr("laminagent._lag.get_task", lambda **_kwargs: None)
    warnings: list[str] = []
    monkeypatch.setattr("laminagent._lag._echo_warning", warnings.append)

    _log_gemini_usage_record(
        {
            "n_call_count": 1,
            "n_prompt_tokens": 2,
            "n_output_tokens": 3,
            "n_total_tokens": 5,
        },
        package_version="0.1.0",
        duration_in_sec=0.2,
        task_name="tool",
    )
    assert warnings
    assert "not configured" in warnings[0]


def test_record_usage_task_name_prefers_pytest_task_context(monkeypatch) -> None:
    from laminagent._lag import _record_usage_task_name

    monkeypatch.setenv(
        "PYTEST_CURRENT_TEST",
        "tests/tasks/test_01_create_fasta_for_favorite_protein.py::test_create_favorite_protein_sequence (call)",
    )

    task_name = _record_usage_task_name("save_protein.py")
    assert task_name == "01_create_fasta_for_favorite_protein"


def test_record_usage_task_name_falls_back_to_generated_script(monkeypatch) -> None:
    from laminagent._lag import _record_usage_task_name

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    task_name = _record_usage_task_name("save_protein.py")
    assert task_name == "save_protein"
