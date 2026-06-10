from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click
import pytest
from lag_cli.__main__ import (
    _extract_runnable_keys_from_prompt,
    _parse_generated_paths,
    _print_generated_tool_contents,
    _resolve_prompt_runnable_paths,
    _set_current_project_env,
    _warn_if_missing_project,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_generated_paths_filters_empty_entries(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    paths = _parse_generated_paths(f"{a},,{b},")
    assert paths == [a.resolve(), b.resolve()]


def test_print_generated_tool_contents_prints_each_file_once(
    tmp_path: Path, capsys
) -> None:
    a = tmp_path / "a.py"
    a.write_text("print('a')\n", encoding="utf-8")
    b = tmp_path / "b.py"
    b.write_text("print('b')\n", encoding="utf-8")

    _print_generated_tool_contents([a, b, a])
    output = capsys.readouterr().out

    assert output.count("[Generated Tool ") == 2
    assert "print('a')" in output
    assert "print('b')" in output


def test_set_current_project_env_sets_current_project(monkeypatch) -> None:
    monkeypatch.delenv("LAMIN_CURRENT_PROJECT", raising=False)
    project = _set_current_project_env("demo-project")
    assert project == "demo-project"
    assert os.environ["LAMIN_CURRENT_PROJECT"] == "demo-project"


def test_warn_if_missing_project_logs_warning(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_warning(message: str) -> None:
        calls.append(message)

    monkeypatch.setattr("lag_cli.__main__.logger.warning", _fake_warning)
    _warn_if_missing_project(None)
    assert len(calls) == 1


def test_extract_runnable_keys_from_prompt_deduplicates() -> None:
    prompt = "rerun test-lag/create_fasta.py and test-lag/create_fasta.py plus x.ipynb"
    keys = _extract_runnable_keys_from_prompt(prompt)
    assert keys == ["test-lag/create_fasta.py", "x.ipynb"]


def test_resolve_prompt_runnable_paths_requires_explicit_key() -> None:
    with pytest.raises(
        click.ClickException, match="Default mode executes existing tools only"
    ):
        _resolve_prompt_runnable_paths("please rerun the tool")
