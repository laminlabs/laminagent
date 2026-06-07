from pathlib import Path

from lag_cli.do_executor import execute_plan, extract_runnable_paths, find_plan_file


def test_extract_runnable_paths(tmp_path: Path) -> None:
    plan_text = """
    - run `scripts/a.py`
    - scripts/b.py
    - notebooks/c.ipynb
    """
    paths = extract_runnable_paths(plan_text, tmp_path)
    assert len(paths) == 3
    assert paths[0].name == "a.py"
    assert paths[1].name == "b.py"
    assert paths[2].name == "c.ipynb"


def test_execute_plan_runs_python_scripts(tmp_path: Path) -> None:
    script = tmp_path / "hello.py"
    script.write_text("print('hello from script')\n", encoding="utf-8")
    plan = tmp_path / "plan.md"
    plan.write_text(f"- run `{script.name}`\n", encoding="utf-8")

    result = execute_plan(
        prompt="execute this plan",
        plan_file=plan,
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


def test_find_plan_file_prefers_plan_md(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "plan_older.md").write_text("old", encoding="utf-8")
    (tmp_path / "plan.md").write_text("main", encoding="utf-8")
    found = find_plan_file()
    assert found is not None
    assert found.name == "plan.md"


def test_execute_plan_runs_notebook_cells(tmp_path: Path) -> None:
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
    plan = tmp_path / "plan.md"
    plan.write_text(f"- `{nb.name}`", encoding="utf-8")
    result = execute_plan(prompt="execute", plan_file=plan, run_uid="run-notebook")
    notebook_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "notebook_executed"
    ]
    assert len(notebook_events) == 1
    assert notebook_events[0]["exit_code"] == 0


def test_execute_plan_passes_master_run_uid_env_to_python_script(
    tmp_path: Path,
) -> None:
    script = tmp_path / "env_check.py"
    script.write_text(
        "import os\nprint(os.getenv('LAMIN_INITIATED_BY_RUN_UID', ''))\n",
        encoding="utf-8",
    )
    plan = tmp_path / "plan.md"
    plan.write_text(f"- run `{script.name}`\n", encoding="utf-8")

    result = execute_plan(
        prompt="execute this plan",
        plan_file=plan,
        run_uid="master-run-uid",
    )
    script_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "script_executed"
    ]
    assert len(script_events) == 1
    assert "master-run-uid" in str(script_events[0]["stdout"])


def test_execute_plan_sets_master_run_uid_env_for_notebook(tmp_path: Path) -> None:
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
    plan = tmp_path / "plan.md"
    plan.write_text(f"- `{nb.name}`", encoding="utf-8")
    result = execute_plan(prompt="execute", plan_file=plan, run_uid="run-notebook")
    notebook_events = [
        event
        for event in result["trace_events"]
        if event.get("event") == "notebook_executed"
    ]
    assert len(notebook_events) == 1
    assert notebook_events[0]["exit_code"] == 0
