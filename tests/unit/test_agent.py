from pathlib import Path

from lag_cli.agent import (
    _dispatch_tool,
    _function_declarations,
    _looks_like_wrapper_runner,
    run_agent,
)
from lag_cli.run_context import RunContext


def test_detects_subprocess_wrapper_runner() -> None:
    code = """
import subprocess
result = subprocess.run(["python", "write_hello.py"], capture_output=True, text=True)
print(result.stdout)
"""
    assert _looks_like_wrapper_runner(code, ["write_hello.py"])


def test_allows_regular_task_script() -> None:
    code = """
import lamindb as ln
with open("hello.txt", "w") as f:
    f.write("Hello agent!")
ln.Artifact("hello.txt").save()
"""
    assert not _looks_like_wrapper_runner(code, [])


def test_rejects_additional_runnable_filename_in_do_mode() -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="exec",
        prompt="p",
        model="m",
    )
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "create_hello_file.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=["hello_agent.py"],
    )
    assert result["status"] == "error"
    assert "Rejected additional runnable tool file in do mode" in str(result["message"])


def test_allows_overwriting_existing_runnable_filename_in_do_mode(
    monkeypatch,
) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="exec",
        prompt="p",
        model="m",
    )

    def _fake_write_python_script(**kwargs):
        return {"status": "success", "file": str(kwargs["filename"])}

    monkeypatch.setattr("lag_cli.agent.write_python_script", _fake_write_python_script)
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "hello_agent.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=["hello_agent.py"],
    )
    assert result["status"] == "success"
    assert result["file"] == "hello_agent.py"


def test_defaults_python_extension_by_tool_type(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="tool",
        prompt="p",
        model="m",
    )
    captured: dict[str, str] = {}

    def _fake_write_python_script(**kwargs):
        captured["filename"] = str(kwargs["filename"])
        return {"status": "success", "file": str(kwargs["filename"])}

    monkeypatch.setattr("lag_cli.agent.write_python_script", _fake_write_python_script)
    _dispatch_tool(
        name="write_python_script",
        args={"code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("tool_run.md"),
        existing_generated_files=[],
    )
    assert captured["filename"].endswith(".py")
    assert captured["filename"] == "tool_run.py"


def test_tool_mode_function_declarations() -> None:
    names = {entry["name"] for entry in _function_declarations("tool")}
    assert "get_local_skill" in names
    assert "get_lamindb_skill" in names
    assert "write_from_template" in names
    assert "write_python_script" in names


def test_tool_mode_enforces_explicit_key_filename_reuse() -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="tool",
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


def test_fails_fast_when_explicit_tool_key_not_found_in_do_mode(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="exec",
        prompt="rerun tool",
        model="m",
    )

    monkeypatch.setattr(
        "lag_cli.agent.get_lamindb_skill",
        lambda **_kwargs: {
            "run_uid": "run-1",
            "results": [],
            "searched_instances": ["laminlabs/lamindata"],
        },
    )

    result = _dispatch_tool(
        name="get_lamindb_skill",
        args={"key": "test-lag/create_fasta.py"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=[],
    )

    assert result["status"] == "error"
    assert result["fatal"] is True
    assert "Aborting without generating a new tool." in str(result["message"])


def test_run_agent_stops_after_fatal_tool_error(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="exec",
        prompt="rerun",
        model="m",
    )

    tool_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_lamindb_skill",
                                "args": {"key": "test-lag/create_fasta.py"},
                            }
                        }
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        "lag_cli.agent._post_generate_content", lambda **_kwargs: tool_response
    )
    monkeypatch.setattr(
        "lag_cli.agent._dispatch_tool",
        lambda **_kwargs: {
            "status": "error",
            "fatal": True,
            "message": "Tool key 'test-lag/create_fasta.py' was not found.",
        },
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=5,
    )

    assert result["final_text"] == "Tool key 'test-lag/create_fasta.py' was not found."


def test_short_circuits_when_explicit_tool_key_found(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="exec",
        prompt="rerun tool",
        model="m",
    )

    monkeypatch.setattr(
        "lag_cli.agent.get_lamindb_skill",
        lambda **_kwargs: {
            "run_uid": "run-1",
            "results": [
                {
                    "type": "transform",
                    "uid": "u1",
                    "key": "test-lag/create_fasta.py",
                }
            ],
            "searched_instances": ["laminlabs/lamindata"],
        },
    )
    result = _dispatch_tool(
        name="get_lamindb_skill",
        args={"key": "test-lag/create_fasta.py"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=[],
    )

    assert result["status"] == "success"
    assert result["short_circuit_execute"] is True
    assert result["resolved_runnable_path"] == "test-lag/create_fasta.py"


def test_run_agent_stops_after_short_circuit_lookup(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="exec",
        prompt="rerun",
        model="m",
    )
    tool_response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_lamindb_skill",
                                "args": {"key": "test-lag/create_fasta.py"},
                            }
                        }
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        "lag_cli.agent._post_generate_content", lambda **_kwargs: tool_response
    )
    monkeypatch.setattr(
        "lag_cli.agent._dispatch_tool",
        lambda **_kwargs: {
            "status": "success",
            "short_circuit_execute": True,
            "resolved_runnable_path": "test-lag/create_fasta.py",
            "message": "Found existing runnable tool.",
        },
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=5,
    )

    assert result["final_text"] == "Found existing runnable tool."
    assert result["resolved_runnable_path"] == "test-lag/create_fasta.py"


def test_run_agent_aggregates_usage_metadata(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="tool",
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
        "lag_cli.agent._post_generate_content", lambda **_kwargs: model_response
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=1,
    )

    assert result["gemini_usage"] == {
        "n_call_count": 1,
        "n_prompt_tokens": 11,
        "n_output_tokens": 7,
        "n_total_tokens": 18,
    }


def test_run_agent_handles_missing_usage_metadata(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="tool",
        prompt="make a tool",
        model="m",
    )
    model_response = {
        "candidates": [{"content": {"parts": [{"text": "done"}]}}],
    }
    monkeypatch.setattr(
        "lag_cli.agent._post_generate_content", lambda **_kwargs: model_response
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=1,
    )

    assert result["gemini_usage"] == {
        "n_call_count": 1,
        "n_prompt_tokens": 0,
        "n_output_tokens": 0,
        "n_total_tokens": 0,
    }
