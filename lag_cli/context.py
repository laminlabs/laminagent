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


def _load_local_biomed_skills(*, key_lower: str, limit: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    _local_root = Path(__file__).resolve().parent.parent.parent / "biomed-skills"
    if not _local_root.is_dir():
        return results
    for skill_path in sorted(_local_root.glob("*.md")):
        if key_lower not in skill_path.name.lower():
            continue
        try:
            content = skill_path.read_text(encoding="utf-8")
            results.append(
                {
                    "type": "local_file",
                    "uid": "",
                    "key": f"biomed-skills/{skill_path.name}",
                    "description": "Local skill (biomed-skills/)",
                    "content": content,
                }
            )
            if len(results) >= limit:
                break
        except Exception as exc:
            # caller collects warnings
            raise OSError(f"Could not read local '{skill_path}': {exc}") from exc
    return results


def get_skill_local_first(
    *, key: str, run_uid: str | None = None, limit: int = 3
) -> dict[str, Any]:
    """Load skills from local biomed-skills/*.md first, falling back to remote.

    Reading local files needs NO ``ln.connect()`` switch, so in the normal case
    Django is never re-initialized mid-process — which is what destabilizes the
    tracked run's cleanup (the BigAutoField crash). Only when a skill isn't found
    locally do we fall back to the remote laminlabs/biomed-skills (which does
    switch instances).
    """
    key_lower = key.lower()
    try:
        local_results = _load_local_biomed_skills(key_lower=key_lower, limit=limit)
    except OSError:
        local_results = []

    if local_results:
        skill_content = "\n\n---\n\n".join(
            r["content"] for r in local_results if r.get("content")
        )
        return {
            "run_uid": run_uid,
            "key": key,
            "results": local_results,
            "searched_instances": ["local/biomed-skills"],
            "message": f"Found {len(local_results)} skill(s) locally.",
            "skill_content": skill_content,
            "warnings": [],
        }

    # not found locally — fall back to the remote registry (this switches instances)
    return get_lamindb_skill(key=key, run_uid=run_uid, limit=limit)


def get_lamindb_skill(
    *, key: str, run_uid: str | None = None, limit: int = 3
) -> dict[str, Any]:
    """Load .md skills from laminlabs/biomed-skills whose key contains the search term.

    Uses ln.DB() to query the remote instance without calling ln.connect(), which
    prevents Django re-initialization and keeps the tracked run stable. Falls back
    to local ``biomed-skills/*.md`` if remote fails.
    """
    warnings: list[str] = []
    results: list[dict[str, str]] = []
    key_lower = key.lower()
    searched_instances: list[str] = []

    try:
        db = ln.DB("laminlabs/biomed-skills")
        searched_instances.append("laminlabs/biomed-skills")
        # only the latest version of each skill — re-uploads create new versions
        all_artifacts = db.Artifact.filter(suffix=".md", is_latest=True).all()
        seen_keys: set[str] = set()
        for artifact in all_artifacts:
            artifact_key = str(getattr(artifact, "key", "") or "")
            desc = str(getattr(artifact, "description", "") or "")
            if key_lower not in artifact_key.lower() and key_lower not in desc.lower():
                continue
            if artifact_key in seen_keys:
                continue
            seen_keys.add(artifact_key)
            content = ""
            try:
                local_path = artifact.cache()
                content = Path(local_path).read_text(encoding="utf-8")
            except Exception as exc:
                warnings.append(f"Could not read '{artifact_key}': {exc}")
            results.append(
                {
                    "type": "artifact",
                    "uid": str(getattr(artifact, "uid", "") or ""),
                    "key": artifact_key,
                    "description": desc[:1000],
                    "content": content,
                }
            )
            if len(results) >= limit:
                break
    except Exception as exc:
        warnings.append(f"Could not load skills from laminlabs/biomed-skills: {exc}")

    if not results:
        try:
            local_results = _load_local_biomed_skills(key_lower=key_lower, limit=limit)
            if local_results:
                searched_instances.append("local/biomed-skills")
                results = local_results
        except OSError as exc:
            warnings.append(str(exc))

    skill_content = "\n\n---\n\n".join(
        r["content"] for r in results if r.get("content")
    )

    return {
        "run_uid": run_uid,
        "key": key,
        "results": results,
        "searched_instances": searched_instances,
        "message": f"Found {len(results)} skill(s).",
        "skill_content": skill_content,
        "warnings": warnings,
    }
