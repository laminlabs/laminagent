⚠️ This repo is in development right now. ⚠️

# Lag CLI: Lamin Agent CLI

`lag` uses a single auto flow to execute existing runnable tools or author/update runnable scripts (`.py`).

## Setup

```bash
pip install laminagent
```

Create `~/llms.env` with:

```bash
GEMINI_API_KEY=your_api_key_here
```

Make sure LaminDB is initialized and connected.

## Quick Start

Run with auto flow:

```bash
lag --prompt "Write a text file with 'Hello agent!' in it, please"
```

You can explore runnable example scenarios in `tests/tasks`.

Before running `lag`, you can initialize LagEval registry types manually:

```bash
lag setup
```

If these records are missing, `lag` still runs; run `lag setup` when you also want LagEval usage records.

## Single Auto Flow

`lag` decides behavior from prompt and local context:

- If `--prompt` includes explicit runnable `.py` keys/paths, `lag` executes those scripts.
- Otherwise, if `tool.md` (or latest `tool_*.md`) exists, `lag` executes scripts referenced there.
- Otherwise, `lag` invokes LLM authoring to create/update a runnable `.py` script and saves it via `ln.Transform`.

### Setup mode (`setup`)

Create or refresh LagEval record types used by `lag`:

```bash
lag setup
```

When run from a repository root, this command:

- creates or reuses schema `lag_eval`
- creates or reuses top-level eval type `LagEval`
- creates or reuses task types for `tests/tasks/*.py` (excluding `conftest.py` and `testutils.py`)

You can also set up a single task script:

```bash
lag setup tests/tasks/test_01_create_fasta_for_favorite_protein.py
```

## Common Flags

- `--project <name>` sets `LAMIN_CURRENT_PROJECT`.
- `--model <model-name>` selects the Gemini model during authoring.
- `--output-file <path>` sets output filename for generated content during authoring.
- `--no-track` disables `ln.track()` / `ln.finish()` injection in generated scripts.

## Run Context Propagation

When `lag` executes scripts, it propagates:

- `LAMIN_INITIATED_BY_RUN_UID` (master run uid)
- `LAMIN_CURRENT_PROJECT` (if `--project` is provided)

`lag` itself does not directly create output artifacts; produced outputs are tracked by executed tool code.

## Eval Telemetry Persistence

When authoring runs, laminagent stores telemetry as records and also annotates run features. Logged record features include:

- `package_version`
- `duration_in_sec`
- `commit_hash16`
- `runner_env`
- `n_call_count`
- `n_prompt_tokens`
- `n_output_tokens`
- `n_total_tokens`
