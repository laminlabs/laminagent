import os
from pathlib import Path

import lamindb as ln
import pytest
from dotenv import load_dotenv
from lag_cli.agent import run_agent
from lag_cli.do_executor import execute_runnable_paths
from lag_cli.run_context import RunContext, create_run_uid

load_dotenv(Path("~/llms.env").expanduser())

INSTANCE_SLUG = "laminlabs/lamindata"
# Fixed transform uid so the test reuses one Transform identity (no interactive
# versioning prompt), exactly like the CLI's @ln.flow("wDJpT3xdqjY8").
FLOW_UID = "fastatest001"
PROMPT = (
    "Write a single runnable Python script that does exactly this, IN THIS ORDER:\n"
    "1. import lamindb as ln\n"
    f"2. ln.connect('{INSTANCE_SLUG}')\n"
    "3. ln.track()  # MUST be here, BEFORE writing/saving anything\n"
    "4. write a short protein sequence in FASTA format (a '>' header line followed "
    "by amino-acid letters) to a file named protein.fasta\n"
    "5. save it as an artifact with: "
    "ln.Artifact('protein.fasta', key='protein.fasta', description='favorite protein').save()\n"
    "6. ln.finish()\n"
    "Constraints: do NOT download anything; do NOT call ln.setup.init or ln.setup.load. "
    "ln.track() MUST come before ln.Artifact(...).save() so the artifact links to the run."
)
MODEL = "gemini-flash-latest"


def _artifact_uids() -> set[str]:
    """uids of every artifact currently in the connected instance."""
    return {artifact.uid for artifact in ln.Artifact.filter().all()}


def _runnable_paths(generated_files: list[str], root: Path) -> list[Path]:
    paths: list[Path] = []
    for raw in generated_files:
        path = Path(raw)
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.suffix.lower() in {".py", ".ipynb"}:
            paths.append(path)
    return paths


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY"),
    reason="needs a real GEMINI_API_KEY in ~/llms.env",
)
def test_agent_saves_artifact(tmp_path, monkeypatch, setup_lamindb) -> None:
    """Real end-to-end run against ishitajain9717/mutation-registry, WITH tracking.

    A @ln.flow-wrapped inner helper creates a real parent Run (like the CLI). Its
    real uid is passed to the executed script via LAMIN_INITIATED_BY_RUN_UID, so the
    child script's ln.track() links to a real run instead of failing on a fake uid.
    The script writes protein.fasta and saves a tracked artifact; we assert a NEW
    artifact appears AND that it is linked to a run (i.e. tracking actually worked).
    """
    monkeypatch.chdir(tmp_path)

    before = _artifact_uids()
    status: dict = {"ok": False, "last_error": "run did not complete"}

    @ln.flow(FLOW_UID)
    def _run_tracked() -> None:
        # Real parent run created by @ln.flow; hand its uid to the child script.
        parent_run_uid = create_run_uid(ln.context.run.uid)
        run_context = RunContext(
            run_uid=parent_run_uid,
            mode="exec",
            prompt=PROMPT,
            model=MODEL,
            # Script owns ln.track()/ln.finish() (per prompt); no writer injection.
            track_outputs=True,
        )

        agent_result = run_agent(
            api_key=os.environ["GEMINI_API_KEY"],
            run_context=run_context,
            output_file=tmp_path / "do.py",
            max_steps=8,
            progress_callback=lambda m: print(f"[agent] {m}"),
        )

        generated_files = [
            f
            for f in agent_result.get("generated_files", [])
            if isinstance(f, str) and f
        ]
        runnable_paths = _runnable_paths(generated_files, tmp_path)
        if not runnable_paths:
            status["last_error"] = (
                "agent wrote no runnable script. "
                f"final_text={agent_result.get('final_text')!r}"
            )
            return

        exec_result = execute_runnable_paths(
            prompt=PROMPT,
            runnable_paths=runnable_paths,
            run_uid=parent_run_uid,  # becomes LAMIN_INITIATED_BY_RUN_UID
            source="test_fasta",
        )
        failures = [
            event
            for event in exec_result["trace_events"]
            if event.get("event") == "script_executed" and event.get("exit_code") != 0
        ]
        if failures:
            status["last_error"] = f"script failed: {failures}"
            return

        status["ok"] = True

    try:
        _run_tracked()
    except Exception as exc:
        # Known lamindb + Django 5.2 flow-teardown crash (same one the CLI's
        # _safe_main swallows). The actual work already ran before teardown.
        if "Unsupported lookup" not in str(exc) and "BigAutoField" not in str(exc):
            raise

    if not status["ok"]:
        pytest.fail(f"agent could not produce a working script. {status['last_error']}")

    after = _artifact_uids()
    new_artifacts = after - before
    assert new_artifacts, (
        "script ran but no new artifact was saved in the instance. "
        f"last_error={status['last_error']!r}"
    )

    # Prove tracking actually happened: at least one new artifact must link to a run.
    tracked = [
        a
        for a in ln.Artifact.filter(uid__in=list(new_artifacts)).all()
        if a.run is not None
    ]
    assert tracked, (
        "a new artifact was saved but none are linked to a run (tracking did not "
        f"work). new_artifacts={sorted(new_artifacts)}"
    )
