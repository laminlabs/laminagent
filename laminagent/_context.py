from __future__ import annotations

from pathlib import Path
from typing import Any

import lamindb as ln


def get_local_skill(
    *, topic: str, run_uid: str, skills_root: str | None = None, limit: int = 3
) -> dict[str, Any]:
    root = Path(skills_root or "~/work/repos/scientific-agent-skills").expanduser()
    if not root.exists():
        return {
            "run_uid": run_uid,
            "matches": [],
            "message": f"Skills root not found: {root}",
        }

    matches: list[dict[str, str]] = []
    topic_lower = topic.lower()
    for skill_file in root.rglob("SKILL.md"):
        if len(matches) >= limit:
            break
        rel = str(skill_file.relative_to(root))
        if topic_lower in rel.lower():
            content = skill_file.read_text(encoding="utf-8")
            matches.append({"path": str(skill_file), "content": content[:8000]})
            continue
        content = skill_file.read_text(encoding="utf-8")
        if topic_lower in content.lower():
            matches.append({"path": str(skill_file), "content": content[:8000]})

    return {
        "run_uid": run_uid,
        "matches": matches,
        "message": f"Found {len(matches)} local skill matches for '{topic}'.",
    }


def read_skill_from_lamindb_instance(
    *,
    uid: str,
    run_uid: str,
    instance_slug: str,
) -> dict[str, Any]:
    db = ln.DB(instance_slug)
    record = db.Record.get(uid)
    content = record.notes

    return {
        "run_uid": run_uid,
        "status": "success",
        "skill_uid": uid,
        "source_instance": instance_slug,
        "content": content,
        "warnings": [],
        "message": (
            f"Read skill '{uid}' from '{instance_slug}' ({len(content)} chars)."
        ),
    }
