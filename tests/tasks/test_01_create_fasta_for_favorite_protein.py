import ast
import subprocess
import sys
from pathlib import Path

import lamindb as ln
from testutils import TESTDB1_DEV_DIR, run_laminagent

EXAMPLE_UID = "RL6ZsKnHZvlhDg1d"

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
    result = run_laminagent(TESTDB1_DEV_DIR, "--prompt", PROMPT)
    assert result.returncode == 0
    run = ln.Run.filter().order_by("-created_at").first()
    assert run is not None, "No run found after laminagent invocation"
    example_record = ln.Record.filter(uid=EXAMPLE_UID).one_or_none()
    if example_record is None:
        example_record = ln.Record(name="example_fasta_run")
        example_record.uid = EXAMPLE_UID
        example_record.save()
    run.records.add(example_record)
    feature_values = run.features.get_values()
    for key in ("n_call_count", "n_prompt_tokens", "n_output_tokens", "n_total_tokens"):
        assert key in feature_values, f"Missing usage feature: {key}"

    runnable_files = list(Path(TESTDB1_DEV_DIR).rglob("*.py"))
    assert runnable_files
    assert len(runnable_files) == 1, "agent should write exactly one .py file"
    script = runnable_files[0]
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
