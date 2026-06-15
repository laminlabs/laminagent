from __future__ import annotations

import subprocess
from pathlib import Path

import click


def save_generated_tool_files(paths: list[str]) -> None:
    allowed_suffixes = {".py"}
    seen: set[str] = set()
    for path_str in paths:
        if not path_str or path_str in seen:
            continue
        seen.add(path_str)
        path = Path(path_str)
        if path.suffix.lower() not in allowed_suffixes:
            continue
        if not path.exists():
            continue
        completed = subprocess.run(
            ["lamin", "save", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise click.ClickException(
                f"Failed to save generated tool file via lamin save: {path}. {stderr}"
            )
