from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any


def _ensure_tracked_python_code(code: str) -> str:
    text = code.rstrip() + "\n"
    has_track = "ln.track(" in text
    has_finish = "ln.finish(" in text

    if has_track and has_finish:
        return text

    lines = text.rstrip("\n").splitlines()
    import_idx = next(
        (
            idx
            for idx, line in enumerate(lines)
            if line.startswith("import lamindb as ln")
        ),
        None,
    )
    connect_idx = next(
        (idx for idx, line in enumerate(lines) if line.startswith("ln.connect")),
        None,
    )
    if import_idx is None:
        lines.insert(0, "import lamindb as ln")
        import_idx = 0

    if not has_track:
        connect_idx = next(
            (idx for idx, line in enumerate(lines) if line.startswith("ln.connect")),
            None,
        )
        insert_at = (connect_idx if connect_idx is not None else import_idx) + 1
        lines.insert(insert_at, "ln.track()")
    if not has_finish:
        lines.append("")
        lines.append("ln.finish()")
    return "\n".join(lines) + "\n"


def write_python_script(
    *,
    code: str,
    filename: str,
    run_uid: str,
    track_outputs: bool = True,
) -> dict[str, Any]:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    code_to_write = _ensure_tracked_python_code(code) if track_outputs else code
    path.write_text(code_to_write, encoding="utf-8")
    return {
        "status": "success",
        "file": str(path),
        "run_uid": run_uid,
        "tracking_enabled": track_outputs,
    }


def write_markdown_tool(
    *,
    markdown: str,
    filename: str,
    run_uid: str,
) -> dict[str, Any]:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return {
        "status": "success",
        "file": str(path),
        "run_uid": run_uid,
    }


def write_from_template(
    *,
    template_path: str,
    filename: str,
    run_uid: str,
) -> dict[str, Any]:
    src = Path(template_path)
    if not src.exists():
        return {
            "status": "error",
            "message": f"Template not found: {src}",
            "run_uid": run_uid,
        }
    dst = Path(filename)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "status": "success",
        "file": str(dst),
        "template": str(src),
        "run_uid": run_uid,
    }
