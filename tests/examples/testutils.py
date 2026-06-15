import subprocess
import sys

TESTDB1_NAME = "testdb1"
TESTDB1_STORAGE = f"./{TESTDB1_NAME}-storage"
TESTDB1_DEV_DIR = f"./{TESTDB1_NAME}-dev-dir"


def run_laminagent(run_dir: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "laminagent", *args],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=True,
    )
