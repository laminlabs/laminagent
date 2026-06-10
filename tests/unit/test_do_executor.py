from pathlib import Path

from lag_cli.do_executor import execute_tool, extract_runnable_paths, find_tool_file


def test_extract_runnable_paths(tmp_path: Path) -> None:
    tool_text = """
    - run `scripts/a.py`
    - scripts/b.py
    - notebooks/c.ipynb
    """
    paths = extract_runnable_paths(tool_text, tmp_path)
    assert len(paths) == 3
    assert paths[0].name == "a.py"
    assert paths[1].name == "b.py"
    assert paths[2].name == "c.ipynb"


def test_execute_tool_runs_python_scripts(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('hello from script')\n", encoding="utf-8")
    tool = tmp_path / "tool.md"
    tool.write_text(f"- run `{script.name}`\n", encoding="utf-8")

    result = execute_tool(
        prompt="execute this tool",
        tool_file=tool,
        run_uid="test-run",
    )

    assert result["run_uid"] == "test-run"
    assert "Executed 1 runnables" in str(result["final_text"])
    script_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "script_executed"
    ]
    assert len(script_events) == 1
    assert script_events[0]["exit_code"] == 0


def test_find_tool_file_prefers_tool_md(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tool_older.md").write_text("old", encoding="utf-8")
    (tmp_path / "tool.md").write_text("main", encoding="utf-8")
    found = find_tool_file()
    assert found is not None
    assert found.name == "tool.md"


def test_execute_tool_runs_notebook_cells(tmp_path: Path) -> None:
    nb = tmp_path / "n.ipynb"
    nb.write_text(
        """
{
 "cells": [
  {"cell_type": "code", "source": "x=1\\n", "metadata": {}, "outputs": []},
  {"cell_type": "code", "source": "y=x+1\\n", "metadata": {}, "outputs": []}
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
        """.strip(),
        encoding="utf-8",
    )
    tool = tmp_path / "tool.md"
    tool.write_text(f"- `{nb.name}`", encoding="utf-8")
    result = execute_tool(prompt="execute", tool_file=tool, run_uid="run-notebook")
    notebook_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "notebook_executed"
    ]
    assert len(notebook_events) == 1
    assert notebook_events[0]["exit_code"] == 0


def test_execute_tool_passes_master_run_uid_env_to_python_script(
    tmp_path: Path,
) -> None:
    script = tmp_path / "env_check.py"
    script.write_text(
        "import os\nprint(os.getenv('LAMIN_INITIATED_BY_RUN_UID', ''))\n",
        encoding="utf-8",
    )
    tool = tmp_path / "tool.md"
    tool.write_text(f"- run `{script.name}`\n", encoding="utf-8")

    result = execute_tool(
        prompt="execute this tool",
        tool_file=tool,
        run_uid="master-run-uid",
    )
    script_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "script_executed"
    ]
    assert len(script_events) == 1
    assert "master-run-uid" in str(script_events[0]["stdout"])


def test_execute_tool_sets_master_run_uid_env_for_notebook(tmp_path: Path) -> None:
    nb = tmp_path / "n.ipynb"
    nb.write_text(
        """
{
 "cells": [
  {"cell_type": "code", "source": "import os\\nassert os.getenv('LAMIN_INITIATED_BY_RUN_UID') == 'run-notebook'\\n", "metadata": {}, "outputs": []}
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
        """.strip(),
        encoding="utf-8",
    )
    tool = tmp_path / "tool.md"
    tool.write_text(f"- `{nb.name}`", encoding="utf-8")
    result = execute_tool(prompt="execute", tool_file=tool, run_uid="run-notebook")
    notebook_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "notebook_executed"
    ]
    assert len(notebook_events) == 1
    assert notebook_events[0]["exit_code"] == 0
