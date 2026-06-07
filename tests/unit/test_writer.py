from pathlib import Path

import nbformat
from lag_cli.writer import write_jupyter_notebook, write_python_script


def test_write_python_script(tmp_path: Path) -> None:
    out = tmp_path / "generated.py"
    result = write_python_script(
        code="print('hello')\n",
        filename=str(out),
        run_uid="test-run",
    )
    assert result["status"] == "success"
    assert out.exists()
    assert "hello" in out.read_text(encoding="utf-8")
    text = out.read_text(encoding="utf-8")
    assert "ln.track()" in text
    assert "ln.finish()" in text
    assert "_lag_before_files" not in text


def test_write_jupyter_notebook(tmp_path: Path) -> None:
    out = tmp_path / "generated.ipynb"
    result = write_jupyter_notebook(
        cells=[
            {"type": "markdown", "content": "# Title"},
            {"type": "code", "content": "x = 1"},
        ],
        filename=str(out),
        run_uid="test-run",
    )
    assert result["status"] == "success"
    assert out.exists()
    nb = nbformat.read(out, as_version=4)
    code_sources = [cell.source for cell in nb.cells if cell.cell_type == "code"]
    assert any("ln.track()" in src for src in code_sources)
    assert any("ln.finish()" in src for src in code_sources)


def test_write_python_script_no_track(tmp_path: Path) -> None:
    out = tmp_path / "plain.py"
    result = write_python_script(
        code="print('hello')\n",
        filename=str(out),
        run_uid="test-run",
        track_outputs=False,
    )
    assert result["status"] == "success"
    text = out.read_text(encoding="utf-8")
    assert "ln.track()" not in text
    assert "ln.finish()" not in text


def test_write_python_script_places_track_after_import(tmp_path: Path) -> None:
    out = tmp_path / "ordered.py"
    result = write_python_script(
        code="import lamindb as ln\n\nprint('hello')\n",
        filename=str(out),
        run_uid="test-run",
    )
    assert result["status"] == "success"
    text = out.read_text(encoding="utf-8")
    assert text.startswith("import lamindb as ln\nln.track()\n")
