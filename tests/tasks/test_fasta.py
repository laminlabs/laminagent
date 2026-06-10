import subprocess
import sys

PROMPT = (
    "Write your favorite protein sequence in a fasta file and save it as an artifact"
)


def run_lag_cli(run_dir: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lag_cli", *args],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )


def test_create_favorite_protein_sequence(setup_lamindb) -> None:
    run_lag_cli(
        "./testdb1-runs",
        "--tool",
        "--prompt",
        PROMPT,
    )
