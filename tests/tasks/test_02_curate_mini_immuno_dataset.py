import ast
import shutil
import subprocess
import sys
from pathlib import Path

from testutils import TESTDB1_DEV_DIR, run_laminagent

PROMPT = (
    "Create a script that curates artifact MdvQpu992LjdLxNz0000 from the laminlabs/lamindata LaminDB instance using "
    "schema pnQvQVcQ417bfmVq and skill u5muNUOPnWPBuZ8z from instance laminlabs/biomed-skills."
)

RUN_DIR = Path(f"{TESTDB1_DEV_DIR}/test_02")


def test_curate_mini_immuno() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    RUN_DIR.mkdir(parents=True)
    # step 1: write the curation script
    result = run_laminagent(RUN_DIR, "--prompt", PROMPT)
    print(f"\n--- agent stdout ---\n{result.stdout}")
    assert result.returncode == 0, (
        f"lag failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    runnable_files = list(RUN_DIR.rglob("*.py"))
    assert runnable_files, (
        f"agent wrote no files\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert len(runnable_files) == 1, "agent should write exactly one .py file"

    script = runnable_files[0]
    code = script.read_text()
    ast.parse(code)
    print(code)
    assert "ln.Artifact(df" in code, "ln.Artifact(df not in code"

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
