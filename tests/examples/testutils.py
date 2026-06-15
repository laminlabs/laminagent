import subprocess
import sys

TESTDB1_NAME = "testdb1"
TESTDB1_STORAGE = f"./{TESTDB1_NAME}-storage"
TESTDB1_DEV_DIR = f"./{TESTDB1_NAME}-dev-dir"


def run_laminagent(run_dir: str, *args: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "laminagent", *args]
    try:
        return subprocess.run(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise AssertionError(
            "laminagent CLI failed.\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {run_dir}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        ) from exc
