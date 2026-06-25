from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lamindb as ln

if TYPE_CHECKING:
    from collections.abc import Callable


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


def read_skill_from_lamindb_instance(
    *,
    uid: str,
    run_uid: str,
    instance_slug: str = "laminlabs/biomed-skills",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    normalized_uid = uid.strip()
    normalized_instance = instance_slug.strip() or "laminlabs/biomed-skills"
    if not normalized_uid:
        raise ValueError("Missing required skill uid.")

    db_start = time.perf_counter()
    db = ln.DB(normalized_instance)
    db_elapsed_ms = (time.perf_counter() - db_start) * 1000
    if progress_callback is not None:
        progress_callback(
            f"skill lookup: opened DB('{normalized_instance}') in {db_elapsed_ms:.1f}ms"
        )

    record_start = time.perf_counter()
    record = db.Record.get(normalized_uid)
    record_elapsed_ms = (time.perf_counter() - record_start) * 1000
    if progress_callback is not None:
        progress_callback(
            f"skill lookup: loaded Record('{normalized_uid}') in {record_elapsed_ms:.1f}ms"
        )

    readme_start = time.perf_counter()
    readme_block = record.ablocks.get(kind="readme", is_latest=True)
    content = str(getattr(readme_block, "content", "") or "")
    readme_elapsed_ms = (time.perf_counter() - readme_start) * 1000
    if progress_callback is not None:
        progress_callback(
            f"skill lookup: loaded README block in {readme_elapsed_ms:.1f}ms"
        )

    if not content.strip():
        raise ValueError(
            f"Skill '{normalized_uid}' was found in '{normalized_instance}', "
            "but README content is empty."
        )
    total_elapsed_ms = (time.perf_counter() - total_start) * 1000
    if progress_callback is not None:
        progress_callback(
            f"skill lookup: completed in {total_elapsed_ms:.1f}ms "
            f"(content_chars={len(content)})"
        )

    return {
        "run_uid": run_uid,
        "status": "success",
        "skill_uid": normalized_uid,
        "source_instance": normalized_instance,
        "content": content,
        "warnings": [],
        "message": (
            f"Read skill '{normalized_uid}' from '{normalized_instance}' "
            f"({len(content)} chars)."
        ),
    }
