import ast
import subprocess
import sys
from pathlib import Path

from testutils import TESTDB1_DEV_DIR, run_lag_cli

PROMPT = (
    "Curate `ln.examples.datasets.mini_immuno.get_dataset1()` using "
    "`ln.examples.datasets.mini_immuno.define_mini_immuno_schema_flexible()` "
    "Save as a lamindb artifact. The instance is already connected. "
    "Write a Python script, not a notebook."
)


def test_curate_mini_immuno(setup_testdb1) -> None:
    # step 1: write the curation script
    result = run_lag_cli(TESTDB1_DEV_DIR, "--tool", "--prompt", PROMPT)
    print(f"\n--- agent stdout ---\n{result.stdout}")
    assert result.returncode == 0, (
        f"lag_cli failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runnable_files = list(Path(TESTDB1_DEV_DIR).rglob("*.py"))
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
        cwd=TESTDB1_DEV_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    assert script_result.returncode == 0, (
        f"{script.name} failed\nSTDOUT:\n{script_result.stdout}\nSTDERR:\n{script_result.stderr}"
    )
