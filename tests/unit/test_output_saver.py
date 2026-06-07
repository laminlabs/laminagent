from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import click
import pytest
from lag_cli import output_saver

if TYPE_CHECKING:
    from pathlib import Path


def test_save_generated_tool_files_uses_lamin_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.md"
    script = tmp_path / "task.py"
    plan.write_text("# plan\n", encoding="utf-8")
    script.write_text("print('ok')\n", encoding="utf-8")

    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(output_saver.subprocess, "run", _fake_run)
    output_saver.save_generated_tool_files([str(plan), str(script), str(plan)])

    assert calls == [["lamin", "save", str(script)]]


def test_save_generated_tool_files_raises_on_failed_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "task.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stderr="boom")

    monkeypatch.setattr(output_saver.subprocess, "run", _fake_run)

    with pytest.raises(
        click.ClickException, match="Failed to save generated tool file"
    ):
        output_saver.save_generated_tool_files([str(script)])


def test_save_generated_tool_files_skips_non_runnable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n", encoding="utf-8")
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(output_saver.subprocess, "run", _fake_run)
    output_saver.save_generated_tool_files([str(plan)])

    assert calls == []
