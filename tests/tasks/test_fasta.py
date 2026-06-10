import os
from pathlib import Path

import pytest
from lag_cli.agent import run_agent
from lag_cli.do_executor import execute_runnable_paths
from lag_cli.run_context import RunContext, create_run_uid

PROMPT = "Write a protein sequence in a fasta file and save as an artifact"
MODEL = "gemini-flash-latest"

# Task-based / outcome-based evaluation knobs (AgentBench-style): run the agent
# N times and require the success rate to clear a threshold, instead of a single
# pass/fail, because agent output is stochastic.
N_RUNS = int(os.getenv("FASTA_TEST_RUNS", "3"))
MIN_SUCCESS_RATE = float(os.getenv("FASTA_TEST_MIN_SUCCESS_RATE", "0.67"))

# 20 standard amino acids + ambiguity codes (B, Z, X, J), selenocysteine (U),
# pyrrolysine (O), stop (*) and gap (-).
_AMINO_ACIDS = set("ABCDEFGHIKLMNPQRSTVWYZXJUO*-")


def _runnable_paths(generated_files: list[str], root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in generated_files:
        if not isinstance(raw, str) or not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.suffix.lower() in {".py", ".ipynb"}:
            paths.append(path)
    return paths


def _is_valid_fasta(text: str) -> bool:
    """Outcome checker: text parses as FASTA with >=1 record of valid residues."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or not lines[0].startswith(">"):
        return False
    records: list[list[str]] = []
    for line in lines:
        if line.startswith(">"):
            records.append([])
        else:
            records[-1].append(line)
    if not records:
        return False
    for seq_lines in records:
        seq = "".join(seq_lines)
        if not seq or any(ch.upper() not in _AMINO_ACIDS for ch in seq):
            return False
    return True


def _find_valid_fasta(root: Path) -> Path | None:
    for path in root.rglob("*.fasta"):
        try:
            if _is_valid_fasta(path.read_text()):
                return path
        except OSError:
            continue
    return None


def _run_agent_once(work_dir: Path, api_key: str) -> tuple[bool, str]:
    """One episode: run the agent, execute its script, score the outcome."""
    run_context = RunContext(
        run_uid=create_run_uid(),
        mode="exec",
        prompt=PROMPT,
        model=MODEL,
        track_outputs=False,  # no LaminDB tracking; we only score the FASTA on disk
    )
    result = run_agent(
        api_key=api_key,
        run_context=run_context,
        output_file=work_dir / "do.py",
        max_steps=8,
        progress_callback=lambda m: print(f"[agent] {m}"),
    )

    runnable = _runnable_paths(result.get("generated_files", []), work_dir)
    if not runnable:
        return (
            False,
            f"agent wrote no runnable script (final_text={result.get('final_text')!r})",
        )

    execute_runnable_paths(
        prompt=PROMPT,
        runnable_paths=runnable,
        run_uid=run_context.run_uid,
        source="test_fasta",
    )

    valid = _find_valid_fasta(work_dir)
    if valid is None:
        produced = [p.name for p in work_dir.rglob("*.fasta")]
        return False, f"no valid FASTA produced (fasta files seen: {produced})"
    return True, f"valid FASTA: {valid.name}"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="needs a real GEMINI_API_KEY in the environment",
)
def test_agent_protein_fasta_success_rate(tmp_path, monkeypatch) -> None:
    """Outcome-based, multi-run evaluation of the agent on a single example task.

    Following task-based agent benchmarks (e.g. AgentBench, Liu et al. ICLR'24;
    HumanEval's pass@k), success is decided by a programmatic checker on the goal
    state (a valid FASTA file), and we report the success rate across N stochastic
    runs rather than a single pass/fail.
    """
    api_key = os.environ["GEMINI_API_KEY"]
    outcomes: list[tuple[int, bool, str]] = []
    for i in range(N_RUNS):
        work_dir = tmp_path / f"run_{i}"
        work_dir.mkdir()
        monkeypatch.chdir(work_dir)
        ok, detail = _run_agent_once(work_dir, api_key)
        outcomes.append((i, ok, detail))
        print(f"[run {i}] {'PASS' if ok else 'FAIL'} - {detail}")

    successes = sum(1 for _, ok, _ in outcomes if ok)
    rate = successes / N_RUNS
    report = "\n".join(
        f"  run {i}: {'PASS' if ok else 'FAIL'} - {detail}"
        for i, ok, detail in outcomes
    )
    print(f"\nsuccess rate: {successes}/{N_RUNS} = {rate:.0%}\n{report}")

    assert rate >= MIN_SUCCESS_RATE, (
        f"success rate {successes}/{N_RUNS} = {rate:.0%} is below the required "
        f"{MIN_SUCCESS_RATE:.0%}\n{report}"
    )
