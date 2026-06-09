import os
from pathlib import Path

import pytest
from lag_cli.agent import run_agent
from lag_cli.do_executor import execute_runnable_paths
from lag_cli.run_context import RunContext, create_run_uid

PROMPT = "Write a protein sequence in a fasta file and save as an artifact"
MODEL = "gemini-flash-latest"


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


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="needs a real GEMINI_API_KEY in the environment",
)
def test_agent_writes_protein_fasta(tmp_path, monkeypatch) -> None:
    """Given the prompt, the agent should write a script that produces a FASTA file."""
    monkeypatch.chdir(tmp_path)

    run_context = RunContext(
        run_uid=create_run_uid(),
        mode="exec",
        prompt=PROMPT,
        model=MODEL,
        track_outputs=False,  # no LaminDB tracking; we only verify the FASTA on disk
    )
    result = run_agent(
        api_key=os.environ["GEMINI_API_KEY"],
        run_context=run_context,
        output_file=tmp_path / "do.py",
        max_steps=8,
        progress_callback=lambda m: print(f"[agent] {m}"),
    )

    runnable = _runnable_paths(result.get("generated_files", []), tmp_path)
    assert runnable, (
        f"agent wrote no runnable script. final_text={result.get('final_text')!r}"
    )

    execute_runnable_paths(
        prompt=PROMPT,
        runnable_paths=runnable,
        run_uid=run_context.run_uid,
        source="test_fasta",
    )

    fasta_files = list(tmp_path.rglob("*.fasta"))
    assert fasta_files, "agent ran but produced no .fasta file"
