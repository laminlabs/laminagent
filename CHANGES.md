# LAG CLI — Architecture & Change Notes

This document describes the current behavior of the LAG CLI, the LaminDB
biomedical curation agent. It is organized by subsystem. Each section states
what the code does and the reason the behavior exists.

---

## 1. Run tracking model

There are two distinct execution paths in the CLI, and only one of them runs:

- `write_python_script` writes a human-readable script to disk as
  `plan_<run_uid>.py`. This file is never executed; it exists as a readable
  record of what the agent produced.
- `execute_python` is the path that actually runs code. It routes through
  `execute_code_string()` in `do_executor.py`, which writes the code to a
  temporary file (`tmp*.py`), runs it in a subprocess, and deletes it.

LaminDB tracking is triggered by `ln.track()` being called at runtime, and the
resulting `Transform` is named after the file that is running. Because the
executed file is the temporary file (not `plan_*.py`), any tracking would be
attributed to the temp file rather than the plan file.

### Tracking is stripped from executed code

`strip_tracking_calls()` in `do_executor.py` removes `ln.track()` and
`ln.finish()` from code before execution. `execute_code_string()` defaults to
`strip_tracking=True`.

Reason: the ReAct loop executes code repeatedly, including failed attempts. If
each execution called `ln.track()`, every attempt would create a throwaway
`Transform`/`Run` in the registry, and changing temp-file names would trigger
LaminDB's interactive rename prompt, which blocks the agent. Stripping tracking
keeps the registry clean and the loop non-interactive.

### The CLI session itself is tracked

`main()` in `__main__.py` is decorated with `@ln.flow()`. This produces one
`Transform`/`Run` per CLI invocation, recorded under
`pypackages/lag_cli/__main__.py`.

Consequence: a run exists per CLI invocation, but artifacts produced inside the
agent's subprocess are saved with `run_id=None` (the subprocess runs untracked,
so `ln.context.run` is `None` there). The session run and the produced artifact
both exist, but there is no lineage edge linking them.

### `--no-track` flag

`--no-track` disables automatic insertion of `ln.track()`/`ln.finish()` in
generated scripts and notebooks.

---

## 2. Skill loading

`get_lamindb_skill()` in `context.py` connects to `laminlabs/biomed-skills` to
fetch skill markdown files from the central registry, with a fallback to local
`biomed-skills/*.md` files when the remote lookup fails. Artifact filtering uses
`is_latest=True` and de-duplicates by key so a single latest copy of each skill
is loaded.

Reason: skills are maintained centrally in `laminlabs/biomed-skills`; the local
fallback keeps the CLI usable offline or when the remote is unavailable.

### Skill content budget

`MAX_SKILL_CHARS` is set to 12000.

Reason: skill files contain full copy-paste scripts and step-by-step
instructions. A smaller budget truncated the instructions before they reached
the model, leading to reconstructed/incorrect code.

---

## 3. Skill routing

`resolve_skill_key()` in `agent.py` selects a single skill from the user prompt.
Matching order prioritizes specific curation skills over generic terms:

1. analysis / annotation / pathway enrichment → `analysis-registries`
2. standardize-and-append → `standardize-append-scrna`
3. single-cell keywords (scRNA, single-cell) → `curate-scrna`
4. bulk keywords (bulk, salmon, nf-core, gene count matrix) → `curate-bulkrna`
5. count/query phrasing (handled by `_COUNT_QUERY`) → `query-instance`
6. default → `curate-scrna`

`_COUNT_QUERY` is a proximity-based regex so "count" only routes to the query
skill when near "artifact"/"transform", and "how many" phrasing is included.

Reason: earlier routing was keyword-substring based. "scRNA-seq" matched the
substring "RNA-seq" and routed to the bulk skill; a bare "count" routed analysis
tasks to the query skill. The ordered, proximity-aware logic prevents these
mis-routes. Regression cases are covered in `tests/test_agent.py`.

---

## 4. LLM integration

LLM calls go through LiteLLM (`litellm.completion()`) in `agent.py`, replacing
direct provider HTTP. The default model is `groq/llama-3.3-70b-versatile`.
Rate-limit handling distinguishes Groq daily-token-limit responses and surfaces
a hint to switch models or wait for reset.

Reason: ReAct loops are token-intensive (prompt, code, error, revised code per
step), which exhausted free-tier quotas quickly. LiteLLM provides a uniform
interface for swapping providers; Groq was selected as the default to avoid the
quota ceilings hit on other providers.

---

## 5. Skill files (`biomed-skills/*.md`)

All five skills were aligned to the official LaminDB documentation and the
agent's untracked-execution model.

- `ln.track()`/`ln.finish()` calls were removed from all skill code blocks, with
  a note that the CLI handles run tracking.
- `curate-bulkrna.md`: reshaping logic for the nf-core salmon merged gene count
  matrix uses `artifact.load()` and the documented tidy-AnnData transform. The
  organism is prompt-driven. A complete copy-paste script is included at the top.
- `curate-scrna.md`: follows the `docs.lamin.ai/scrna` workflow using
  `ln.core.datasets.anndata_human_immune_cells()`, defines the
  donor/tissue/cell_type/assay features, catches `ln.errors.ValidationError`,
  standardizes cell types, adds the single valid public ontology term
  (`animal cell`), saves with `key="datasets/conde22.h5ad"`, and seeds a
  collection. The first validation attempt passes `key=` (see below). A complete
  copy-paste script is included at the top.
- `analysis-registries.md`: the `from lamin_usecases import ...` import was
  replaced with a self-contained loader that downloads `ifnb.h5ad` directly from
  S3, because `lamin_usecases` is a repository, not a pip-installable package.
- `query-instance.md`: counts artifacts and transforms in an instance.

### `key=` required in untracked `from_anndata`

`ln.Artifact.from_anndata(...)` requires at least one of `key`, `run`, or
`description`. When code runs untracked, `run` is `None`, so the first
validation attempt in `curate-scrna.md` passes `key="datasets/conde22.h5ad"`.
Without it, the call raises `ValueError: Pass one of key, run or description as a
parameter` before validation runs.

Reason: with `ln.track()` active, `run` was populated automatically and `key`
could be omitted. In the untracked execution model, `key` must be explicit.

---

## 6. Django 5.2 / BigAutoField

`_patch_bigautofield_lookup()` in `__main__.py` runs at import time and registers
lookups (`Exact`, `IExact`, `In`, `GreaterThan`, `GreaterThanOrEqual`,
`LessThan`, `LessThanOrEqual`, `IsNull`) on `BigAutoField`, `AutoField`, and
`SmallAutoField`.

Context: Django 5.2 changed lookup inheritance for auto-increment fields. At the
end of a tracked run, LaminDB's `@ln.flow` cleanup calls `run.save()`, which
issues `base_qs.filter(pk=pk_val)` and can raise
`FieldError: Unsupported lookup 'exact' for BigAutoField`.

Current status: the patch executes without error, but it does not fully suppress
the cleanup traceback in all cases on this environment. When the traceback
appears, it occurs at the very end of a run, after the artifact has already been
saved; it does not affect the saved data.

---

## 7. Known behaviors

- Artifacts produced by the agent are saved with `run_id=None` because the
  agent's code runs untracked in a subprocess. The session run (from `@ln.flow`)
  and the artifact exist independently, without a lineage link.
- `could not update run params: 'NoneType' object has no attribute 'params'`
  appears when `ln.context.run` is `None` at the point params are logged; params
  logging then no-ops.
- Historical `tmp*.py` transforms in the registry originate from a period before
  tracking was stripped from executed code: the executed temp file contained
  `ln.track()`, so the resulting `Transform` was named after the temp file.
