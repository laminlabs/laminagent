from pathlib import Path

from laminagent._agent import _dispatch_tool, _function_declarations, run_agent
from laminagent._run_context import RunContext


def test_defaults_python_extension_by_tool_type(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="p",
        model="m",
    )
    captured: dict[str, str] = {}

    def _fake_write_python_script(**kwargs):
        captured["filename"] = str(kwargs["filename"])
        return {"status": "success", "file": str(kwargs["filename"])}

    monkeypatch.setattr(
        "laminagent._agent.write_python_script", _fake_write_python_script
    )
    _dispatch_tool(
        name="write_python_script",
        args={"code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("tool_run.md"),
        existing_generated_files=[],
    )
    assert captured["filename"].endswith(".py")
    assert captured["filename"] == "tool_run.py"


def test_function_declarations_include_authoring_tools() -> None:
    names = {entry["name"] for entry in _function_declarations()}
    assert "write_python_script" in names
    assert "read_skill_from_lamindb_instance" in names


def test_enforces_explicit_key_filename_reuse() -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="make new version of test-lag/create_fasta.py",
        model="m",
    )
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "create_fasta_albumin.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("analysis.py"),
        existing_generated_files=[],
    )
    assert result["status"] == "error"
    assert "Update that exact file" in str(result["message"])


def test_rejects_second_runnable_filename_in_same_run() -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="write a script",
        model="m",
    )
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "second.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("analysis.py"),
        existing_generated_files=["first.py"],
    )
    assert result["status"] == "error"
    assert "already created" in str(result["message"])


def test_dispatch_read_skill_from_lamindb_instance(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="read skill",
        model="m",
    )

    monkeypatch.setattr(
        "laminagent._agent.read_skill_from_lamindb_instance",
        lambda **kwargs: {
            "status": "success",
            "skill_uid": kwargs["uid"],
            "source_instance": kwargs["instance_slug"],
            "content": "abc",
            "run_uid": kwargs["run_uid"],
            "warnings": [],
            "message": "ok",
        },
    )

    result = _dispatch_tool(
        name="read_skill_from_lamindb_instance",
        args={"uid": "u5muNUOPnWPBuZ8z", "instance_slug": "laminlabs/biomed-skills"},
        run_context=run_context,
        default_output_file=Path("analysis.py"),
        existing_generated_files=[],
    )
    assert result["status"] == "success"
    assert result["skill_uid"] == "u5muNUOPnWPBuZ8z"
    assert result["source_instance"] == "laminlabs/biomed-skills"


def test_dispatch_read_skill_passes_empty_uid_through(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="read skill",
        model="m",
    )
    monkeypatch.setattr(
        "laminagent._agent.read_skill_from_lamindb_instance",
        lambda **kwargs: {
            "status": "success",
            "skill_uid": kwargs["uid"],
            "source_instance": kwargs["instance_slug"],
            "content": "abc",
            "run_uid": kwargs["run_uid"],
            "warnings": [],
            "message": "ok",
        },
    )
    result = _dispatch_tool(
        name="read_skill_from_lamindb_instance",
        args={},
        run_context=run_context,
        default_output_file=Path("analysis.py"),
        existing_generated_files=[],
    )
    assert result["status"] == "success"
    assert result["skill_uid"] == ""
    assert result["source_instance"] == ""


def test_run_agent_aggregates_usage_metadata(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="make a tool",
        model="m",
    )
    model_response = {
        "usageMetadata": {
            "promptTokenCount": 11,
            "candidatesTokenCount": 7,
            "totalTokenCount": 18,
        },
        "candidates": [{"content": {"parts": [{"text": "done"}]}}],
    }
    monkeypatch.setattr(
        "laminagent._agent._post_generate_content", lambda **_kwargs: model_response
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=1,
    )

    assert result["llm_usage"] == {
        "n_call_count": 1,
        "n_prompt_tokens": 11,
        "n_output_tokens": 7,
        "n_total_tokens": 18,
    }
    assert any(
        event.get("event") == "llm_request" and "request_payload" in event
        for event in result["trace_events"]
    )
    assert any(
        event.get("event") == "llm_response"
        and event.get("usage_metadata", {}).get("totalTokenCount") == 18
        for event in result["trace_events"]
    )


def test_run_agent_handles_missing_usage_metadata(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="make a tool",
        model="m",
    )
    model_response = {
        "candidates": [{"content": {"parts": [{"text": "done"}]}}],
    }
    monkeypatch.setattr(
        "laminagent._agent._post_generate_content", lambda **_kwargs: model_response
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=1,
    )

    assert result["llm_usage"] == {
        "n_call_count": 1,
        "n_prompt_tokens": 0,
        "n_output_tokens": 0,
        "n_total_tokens": 0,
    }


def test_run_agent_stops_after_successful_write_python_script(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="write a script",
        model="m",
    )
    call_count = {"n": 0}
    tool_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "write_python_script",
                                "args": {
                                    "filename": "save_protein.py",
                                    "code": "print('ok')",
                                },
                            }
                        }
                    ]
                }
            }
        ]
    }

    def _fake_post_generate_content(**_kwargs):
        call_count["n"] += 1
        return tool_response

    monkeypatch.setattr(
        "laminagent._agent._post_generate_content", _fake_post_generate_content
    )
    monkeypatch.setattr(
        "laminagent._agent._dispatch_tool",
        lambda **_kwargs: {
            "status": "success",
            "file": "save_protein.py",
            "run_uid": "run-1",
        },
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=5,
    )

    assert call_count["n"] == 1
    assert result["generated_file"] == "save_protein.py"
    assert result["final_text"] == "Wrote runnable script 'save_protein.py'."


def test_run_agent_hard_fails_on_tool_error(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        prompt="write a script",
        model="m",
    )
    tool_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "read_skill_from_lamindb_instance",
                                "args": {"uid": "u5muNUOPnWPBuZ8z"},
                            }
                        }
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(
        "laminagent._agent._post_generate_content", lambda **_kwargs: tool_response
    )
    monkeypatch.setattr(
        "laminagent._agent._dispatch_tool",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Could not read skill 'u5muNUOPnWPBuZ8z' from 'laminlabs/biomed-skills'."
            )
        ),
    )

    try:
        run_agent(
            api_key="dummy",
            run_context=run_context,
            output_file=Path("out.py"),
            max_steps=5,
        )
    except RuntimeError as exc:
        assert "Could not read skill" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected RuntimeError for tool error")
