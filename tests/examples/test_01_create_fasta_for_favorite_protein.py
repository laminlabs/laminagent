import ast
import subprocess
import sys
from pathlib import Path

from testutils import TESTDB1_DEV_DIR, run_lag_cli

PROMPT = (
    "Write a Python script that writes your favorite protein sequence to a file called protein.fasta "
    "and saves it as a LaminDB artifact."
)

_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBZXJUO*-")


def is_valid_fasta(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith(">"):
        return False
    seq = "".join(l for l in lines if not l.startswith(">"))
    return bool(seq) and all(c.upper() in _VALID_AMINO_ACIDS for c in seq)


def test_create_favorite_protein_sequence() -> None:
    # step 1: write the script
    result = run_lag_cli(TESTDB1_DEV_DIR, "--tool", "--prompt", PROMPT)
    assert result.returncode == 0

    runnable_files = list(Path(TESTDB1_DEV_DIR).rglob("*.py")) + list(
        Path(TESTDB1_DEV_DIR).rglob("*.ipynb")
    )
    assert runnable_files
    assert not any(p.suffix == ".ipynb" for p in runnable_files), (
        "agent wrote a notebook instead of a Python script"
    )
    assert len([p for p in runnable_files if p.suffix == ".py"]) == 1, (
        "agent should write exactly one .py file"
    )

    script = next(p for p in runnable_files if p.suffix == ".py")
    code = script.read_text()
    ast.parse(code)

    # step 2: execute the script directly
    subprocess.run(
        [sys.executable, script.name],
        cwd=TESTDB1_DEV_DIR,
        check=True,
    )

    # step 3: check .fasta was produced and is valid
    fasta_files = list(Path(TESTDB1_DEV_DIR).rglob("*.fasta"))
    assert fasta_files, "script ran but produced no .fasta file"
    for fasta in fasta_files:
        assert is_valid_fasta(fasta.read_text()), f"{fasta.name} is not valid FASTA"
