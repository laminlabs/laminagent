import ast
import shutil
import subprocess
import sys
from pathlib import Path

from testutils import TESTDB1_DEV_DIR, run_laminagent

PROMPT = (
    "Curate `ln.examples.datasets.mini_immuno.get_dataset1()` using "
    "`ln.examples.datasets.mini_immuno.define_mini_immuno_schema_flexible()` "
    "and save as a lamindb artifact. "
    "Write a Python script, not a notebook."
)

RUN_DIR = f"{TESTDB1_DEV_DIR}/test_02"


def test_curate_mini_immuno(setup_testdb1) -> None:
    # each test gets its own subdirectory so runs don't interfere
    run_dir = Path(RUN_DIR)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    # step 1: write the curation script
    result = run_laminagent(RUN_DIR, "--tool", "--prompt", PROMPT)
    print(f"\n--- agent stdout ---\n{result.stdout}")
    assert result.returncode == 0, (
        f"lag_cli failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runnable_files = list(run_dir.rglob("*.py"))
    assert runnable_files, (
        f"agent wrote no files\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert len(runnable_files) == 1, "agent should write exactly one .py file"

    script = runnable_files[0]
    code = script.read_text()
    ast.parse(code)
    assert "ln.curators.DataFrameCurator" in code, (
        f"{script.name} uses wrong curator class — must use ln.curators.DataFrameCurator"
    )

    # step 2: execute the script
    script_result = subprocess.run(
        [sys.executable, script.name],
        cwd=RUN_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert script_result.returncode == 0, (
        f"{script.name} failed\nSTDOUT:\n{script_result.stdout}\nSTDERR:\n{script_result.stderr}"
    )
