from lag_cli.run_report import pick_agent_source_code


def test_pick_agent_source_code_prefers_written_file(tmp_path) -> None:
    script = tmp_path / "curate.py"
    script.write_text("print('from file')\n", encoding="utf-8")
    result = {
        "generated_files": [str(script)],
        "trace_events": [
            {
                "tool": "execute_python",
                "tool_args": {"code": "print('from exec')"},
                "tool_result": {"exit_code": 0},
            }
        ],
    }
    assert pick_agent_source_code(result) == "print('from file')\n"


def test_pick_agent_source_code_falls_back_to_last_successful_execute() -> None:
    result = {
        "generated_files": [],
        "trace_events": [
            {
                "tool": "execute_python",
                "tool_args": {"code": "raise Error()"},
                "tool_result": {"exit_code": 1},
            },
            {
                "tool": "execute_python",
                "tool_args": {"code": "print('ok')\n"},
                "tool_result": {"exit_code": 0},
            },
        ],
    }
    assert pick_agent_source_code(result) == "print('ok')\n"
