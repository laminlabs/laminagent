⚠️ This repo is in development right now. ⚠️

# Lag: Lamin's Agent CLI

`lag` can execute existing runnable tools and author/update runnable scripts (`.py`) and notebooks (`.ipynb`).

## Setup

```bash
pip install lag-cli
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

## Modes

### Default mode (execute only)

- If a tool exists (`tool.md` or latest `tool_*.md`, or `--tool-file`), `lag` executes the referenced runnable tools.
- Otherwise, `lag` executes existing runnable tools referenced in `--prompt` (explicit `.py` / `.ipynb` key or path).
- Default mode does not create or update tools. If a referenced tool is missing, it fails with a clear error.

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
- `--no-track` disables `ln.track()` / `ln.finish()` injection in generated scripts/notebooks (`--tool` mode).

## Run Context Propagation

When `lag` executes scripts/notebooks, it propagates:

- `LAMIN_INITIATED_BY_RUN_UID` (master run uid)
- `LAMIN_CURRENT_PROJECT` (if `--project` is provided)

`lag` itself does not directly create output artifacts; produced outputs are tracked by executed tool code.
