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


def _collect_db_matches(
    db: Any, key: str, limit: int
) -> tuple[list[dict[str, str]], list[str]]:
    query_lower = key.lower()
    results: list[dict[str, str]] = []
    warnings: list[str] = []

    try:
        transform = db.Transform.get(key=key)
        description = str(getattr(transform, "description", "") or "")
        results.append(
            {
                "type": "transform",
                "uid": str(transform.uid),
                "key": str(getattr(transform, "key", "") or ""),
                "description": description[:1000],
            }
        )
    except Exception as exc:
        message = str(exc)
        if "does not exist" not in message.lower():
            warnings.append(f"Transform lookup failed: {exc}")

    try:
        if len(results) < limit:
            artifacts = db.Artifact.filter().all()
            for artifact in artifacts:
                desc = str(getattr(artifact, "description", "") or "")
                key = str(getattr(artifact, "key", "") or "")
                haystack = f"{key}\n{desc}".lower()
                if query_lower in haystack:
                    results.append(
                        {
                            "type": "artifact",
                            "uid": str(artifact.uid),
                            "key": key,
                            "description": desc[:1000],
                        }
                    )
                    if len(results) >= limit:
                        break
    except Exception as exc:
        warnings.append(f"Artifact lookup failed: {exc}")

    return results, warnings


def get_lamindb_skill(*, key: str, run_uid: str, limit: int = 5) -> dict[str, Any]:
    """Best-effort lookup from the current instance, then biomed-skills fallback."""
    current_slug: str | None = None
    warnings: list[str] = []

    try:
        current_slug = str(ln.setup.settings.instance.slug)
    except Exception as exc:
        current_slug = None
        warnings.append(f"Could not read current LaminDB instance before lookup: {exc}")

    slugs_to_search: list[str] = []
    if current_slug:
        slugs_to_search.append(current_slug)
    if "laminlabs/biomed-skills" not in slugs_to_search:
        slugs_to_search.append("laminlabs/biomed-skills")

    results: list[dict[str, str]] = []
    searched_instances: list[str] = []

    for slug in slugs_to_search:
        try:
            db = ln.DB(slug)
            searched_instances.append(slug)
        except Exception as exc:
            warnings.append(f"Could not open DB('{slug}'): {exc}")
            continue

        instance_results, instance_warnings = _collect_db_matches(
            db=db, key=key, limit=limit
        )
        warnings.extend(instance_warnings)
        results.extend(instance_results)
        if results:
            break

    payload = {
        "run_uid": run_uid,
        "key": key,
        "results": results,
        "searched_instances": searched_instances,
        "message": f"Found {len(results)} LaminDB matches for '{key}'.",
        "warnings": warnings,
    }
    return payload
