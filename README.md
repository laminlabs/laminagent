⚠️ This repo is in development right now. ⚠️

# Lag CLI: Lamin Agent CLI

`lag` can execute existing runnable tools and author/update runnable scripts (`.py`).

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

Run in default mode:

```bash
lag --prompt "Write a text file with 'Hello agent!' in it, please"
```

You can explore runnable example scenarios in `tests/tasks`.

Before running `lag`, you can initialize LagEval registry types manually:

```bash
lag setup
```

If these records are missing, run `lag setup` before using `lag --tool`.

## Modes

### Default mode (execute only)

- If a tool exists (`tool.md` or latest `tool_*.md`, or `--tool-file`), `lag` executes the referenced runnable tools.
- Otherwise, `lag` executes existing runnable tools referenced in `--prompt` (explicit `.py` key or path).
- Default mode does not create or update tools. If a referenced tool is missing, it fails with a clear error.

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

### Planning mode (`--tool`)

Generate or update runnable tools (without executing them):

```bash
lag --tool --prompt "Update test-lag/create_fasta.py with another protein"
```

Generated/updated tool files are saved via `lamin save` in tool mode.

## Common Flags

- `--project <name>` sets `LAMIN_CURRENT_PROJECT`.
- `--model <model-name>` selects the Gemini model (`--tool` mode).
- `--output-file <path>` sets output filename for generated content (`--tool` mode).
- `--tool-file <path>` executes a specific tool in default mode.
- `--no-track` disables `ln.track()` / `ln.finish()` injection in generated scripts (`--tool` mode).

## Run Context Propagation

When `lag` executes scripts, it propagates:

- `LAMIN_INITIATED_BY_RUN_UID` (master run uid)
- `LAMIN_CURRENT_PROJECT` (if `--project` is provided)

`lag` itself does not directly create output artifacts; produced outputs are tracked by executed tool code.

## Eval Telemetry Persistence

In `--tool` mode, laminagent stores telemetry as records and also annotates run features. Logged record features include:

- `package_version`
- `duration_in_sec`
- `commit_hash16`
- `runner_env`
- `n_call_count`
- `n_prompt_tokens`
- `n_output_tokens`
- `n_total_tokens`
