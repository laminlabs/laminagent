import json
from pathlib import Path
from types import SimpleNamespace

from lag_cli.agent import (
    _dispatch_tool,
    _looks_like_wrapper_runner,
    resolve_skill_key,
    run_agent,
)
from lag_cli.run_context import RunContext


def test_resolve_skill_key_query_instance() -> None:
    assert (
        resolve_skill_key(
            "Count total artifacts and transforms in ishitajain9717/mutation-registry."
        )
        == "query-instance"
    )


def test_resolve_skill_key_analysis_not_query_instance() -> None:
    prompt = (
        "Connect to ishitajain9717/mutation-registry. Run cell type annotation "
        "with celltypist and pathway enrichment with gseapy on "
        "ln.core.datasets.anndata_seurat_ifnb(preprocess=False, populate_registries=True). "
        "Link pathways via schema.pathways.set(). Done when annotated artifact is saved."
    )
    assert resolve_skill_key(prompt) == "analysis-registries"


def test_resolve_skill_key_bulkrna_not_query_instance() -> None:
    prompt = (
        "Curate ln.core.datasets.file_tsv_rnaseq_nfcore_salmon_merged_gene_counts(). "
        "Save the curated artifact when validation passes."
    )
    assert resolve_skill_key(prompt) == "curate-bulkrna"


def test_resolve_skill_key_count_matrix_phrase_routes_to_bulkrna() -> None:
    # regression: "gene count matrix ... curated artifact" must not route to
    # query-instance just because it contains "count" and "artifact"
    prompt = (
        "Ingest the nf-core salmon merged bulk RNA-seq gene count matrix, reshape it "
        "into a tidy AnnData, validate it, and save the curated artifact with labels."
    )
    assert resolve_skill_key(prompt) == "curate-bulkrna"


def test_resolve_skill_key_how_many_routes_to_query_instance() -> None:
    assert (
        resolve_skill_key("How many transforms and artifacts are in this instance?")
        == "query-instance"
    )


def test_resolve_skill_key_scrnaseq_not_bulkrna() -> None:
    # regression: "scRNA-seq" contains the substring "rna-seq" and must NOT
    # route to bulk RNA curation
    prompt = (
        "Curate the Conde 2022 human immune cells scRNA-seq dataset with "
        "anndata_human_immune_cells() and seed a collection scrna/collection1."
    )
    assert resolve_skill_key(prompt) == "curate-scrna"


def test_resolve_skill_key_standardize_append_beats_scrna() -> None:
    assert (
        resolve_skill_key(
            "Standardize this scRNA dataset and append it to the collection."
        )
        == "standardize-append-scrna"
    )


def test_detects_subprocess_wrapper_runner() -> None:
    code = """
import subprocess
result = subprocess.run(["python", "write_hello.py"], capture_output=True, text=True)
print(result.stdout)
"""
    assert _looks_like_wrapper_runner(code, ["write_hello.py"])


def test_allows_regular_task_script() -> None:
    code = """
import lamindb as ln
with open("hello.txt", "w") as f:
    f.write("Hello agent!")
ln.Artifact("hello.txt").save()
"""
    assert not _looks_like_wrapper_runner(code, [])


def test_rejects_additional_runnable_filename_in_do_mode() -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="do",
        prompt="p",
        model="m",
    )
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "create_hello_file.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=["hello_agent.py"],
    )
    assert result["status"] == "error"
    assert "Rejected additional runnable tool file in do mode" in str(result["message"])


def test_allows_overwriting_existing_runnable_filename_in_do_mode(
    monkeypatch,
) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="do",
        prompt="p",
        model="m",
    )

    def _fake_write_python_script(**kwargs):
        return {"status": "success", "file": str(kwargs["filename"])}

    monkeypatch.setattr("lag_cli.agent.write_python_script", _fake_write_python_script)
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "hello_agent.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=["hello_agent.py"],
    )
    assert result["status"] == "success"
    assert result["file"] == "hello_agent.py"


def test_defaults_python_extension_by_tool_type(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="plan",
        prompt="p",
        model="m",
    )
    captured: dict[str, str] = {}

    def _fake_write_python_script(**kwargs):
        captured["filename"] = str(kwargs["filename"])
        return {"status": "success", "file": str(kwargs["filename"])}

    monkeypatch.setattr("lag_cli.agent.write_python_script", _fake_write_python_script)
    _dispatch_tool(
        name="write_python_script",
        args={"code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("plan_run.md"),
        existing_generated_files=[],
    )
    assert captured["filename"].endswith(".py")
    assert captured["filename"] == "plan_run.py"


def test_defaults_notebook_extension_by_tool_type(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="plan",
        prompt="p",
        model="m",
    )
    captured: dict[str, str] = {}

    def _fake_write_jupyter_notebook(**kwargs):
        captured["filename"] = str(kwargs["filename"])
        return {"status": "success", "file": str(kwargs["filename"])}

    monkeypatch.setattr(
        "lag_cli.agent.write_jupyter_notebook", _fake_write_jupyter_notebook
    )
    _dispatch_tool(
        name="write_jupyter_notebook",
        args={"cells": [{"type": "code", "content": "x=1"}]},
        run_context=run_context,
        default_output_file=Path("plan_run.md"),
        existing_generated_files=[],
    )
    assert captured["filename"].endswith(".ipynb")
    assert captured["filename"] == "plan_run.ipynb"


def test_plan_mode_enforces_explicit_key_filename_reuse() -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="plan",
        prompt="make new version of test-lag/create_fasta.py",
        model="m",
    )
    result = _dispatch_tool(
        name="write_python_script",
        args={"filename": "create_fasta_albumin.py", "code": "print('x')"},
        run_context=run_context,
        default_output_file=Path("analysis.py"),
        existing_generated_files=[],
    )
    assert result["status"] == "error"
    assert "Update that exact file" in str(result["message"])


def test_fails_fast_when_explicit_tool_key_not_found_in_do_mode(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="do",
        prompt="rerun tool",
        model="m",
    )

    monkeypatch.setattr(
        "lag_cli.agent.get_lamindb_skill",
        lambda **_kwargs: {
            "run_uid": "run-1",
            "results": [],
            "searched_instances": ["laminlabs/lamindata"],
        },
    )

    result = _dispatch_tool(
        name="get_lamindb_skill",
        args={"key": "test-lag/create_fasta.py"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=[],
    )

    assert result["status"] == "error"
    assert result["fatal"] is True
    assert "Aborting without generating a new tool." in str(result["message"])


def _fake_tool_call_message(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                function=SimpleNamespace(
                    name=name,
                    arguments=json.dumps(args),
                ),
            )
        ],
    )


def test_run_agent_stops_after_fatal_tool_error(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="do",
        prompt="rerun",
        model="m",
    )

    monkeypatch.setattr(
        "lag_cli.agent.get_lamindb_skill",
        lambda **_kwargs: {"results": [], "skill_content": "", "warnings": []},
    )
    monkeypatch.setattr(
        "lag_cli.agent._call_llm",
        lambda **_kwargs: _fake_tool_call_message(
            "get_lamindb_skill", {"key": "test-lag/create_fasta.py"}
        ),
    )
    monkeypatch.setattr(
        "lag_cli.agent._dispatch_tool",
        lambda **_kwargs: {
            "status": "error",
            "fatal": True,
            "message": "Tool key 'test-lag/create_fasta.py' was not found.",
        },
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=5,
    )

    assert result["final_text"] == "Tool key 'test-lag/create_fasta.py' was not found."


def test_short_circuits_when_explicit_tool_key_found(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="do",
        prompt="rerun tool",
        model="m",
    )

    monkeypatch.setattr(
        "lag_cli.agent.get_lamindb_skill",
        lambda **_kwargs: {
            "run_uid": "run-1",
            "results": [
                {
                    "type": "transform",
                    "uid": "u1",
                    "key": "test-lag/create_fasta.py",
                }
            ],
            "searched_instances": ["laminlabs/lamindata"],
        },
    )
    result = _dispatch_tool(
        name="get_lamindb_skill",
        args={"key": "test-lag/create_fasta.py"},
        run_context=run_context,
        default_output_file=Path("out.py"),
        existing_generated_files=[],
    )

    assert result["status"] == "success"
    assert result["short_circuit_execute"] is True
    assert result["resolved_runnable_path"] == "test-lag/create_fasta.py"


def test_run_agent_stops_after_short_circuit_lookup(monkeypatch) -> None:
    run_context = RunContext(
        run_uid="run-1",
        mode="do",
        prompt="rerun",
        model="m",
    )
    monkeypatch.setattr(
        "lag_cli.agent.get_lamindb_skill",
        lambda **_kwargs: {"results": [], "skill_content": "", "warnings": []},
    )
    monkeypatch.setattr(
        "lag_cli.agent._call_llm",
        lambda **_kwargs: _fake_tool_call_message(
            "get_lamindb_skill", {"key": "test-lag/create_fasta.py"}
        ),
    )
    monkeypatch.setattr(
        "lag_cli.agent._dispatch_tool",
        lambda **_kwargs: {
            "status": "success",
            "short_circuit_execute": True,
            "resolved_runnable_path": "test-lag/create_fasta.py",
            "message": "Found existing runnable tool.",
        },
    )

    result = run_agent(
        api_key="dummy",
        run_context=run_context,
        output_file=Path("out.py"),
        max_steps=5,
    )

    assert result["final_text"] == "Found existing runnable tool."
    assert result["resolved_runnable_path"] == "test-lag/create_fasta.py"
