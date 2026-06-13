import ast
import subprocess
import sys
from pathlib import Path

PROMPT = (
    "Write a Python script that writes a protein sequence to a file called protein.fasta "
    "and saves it as a LaminDB artifact."
)

_VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWYBZXJUO*-")


def run_lag_cli(run_dir: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lag_cli", *args],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=True,
    )


def _is_valid_fasta(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines or not lines[0].startswith(">"):
        return False
    seq = "".join(l for l in lines if not l.startswith(">"))
    return bool(seq) and all(c.upper() in _VALID_AMINO_ACIDS for c in seq)


def test_create_favorite_protein_sequence(setup_lamindb) -> None:
    # step 1: write the script
    result = run_lag_cli("./testdb1-runs", "--tool", "--prompt", PROMPT)
    assert result.returncode == 0

    runnable_files = list(Path("./testdb1-runs").rglob("*.py")) + list(
        Path("./testdb1-runs").rglob("*.ipynb")
    )
    assert runnable_files

    for path in runnable_files:
        if path.suffix == ".py":
            code = path.read_text()
            try:
                ast.parse(code)
            except SyntaxError as e:
                raise AssertionError(f"{path.name} is not valid Python: {e}") from e
            assert "import lamindb" in code, f"{path.name} does not import lamindb"
            assert ".save(" in code, f"{path.name} does not call .save()"

    # step 2: execute the script (pass the filename so the CLI knows what to run)
    script_name = runnable_files[0].name
    result = run_lag_cli("./testdb1-runs", "--prompt", script_name)
    assert result.returncode == 0

    # step 3: check .fasta was produced and is valid
    fasta_files = list(Path("./testdb1-runs").rglob("*.fasta"))
    assert fasta_files, "script ran but produced no .fasta file"
    for fasta in fasta_files:
        assert _is_valid_fasta(fasta.read_text()), f"{fasta.name} is not valid FASTA"
