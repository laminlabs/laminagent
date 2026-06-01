from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import lamindb as ln
import requests


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


def _collect_skill_artifacts(query: str, limit: int) -> tuple[list[dict[str, str]], list[str]]:
    """Fetch .md skill artifacts from the currently connected instance and read their content from cache."""
    query_lower = query.lower()
    results: list[dict[str, str]] = []
    warnings: list[str] = []

    try:
        artifacts = ln.Artifact.filter(suffix=".md").all()
        for artifact in artifacts:
            desc = str(getattr(artifact, "description", "") or "")
            artifact_key = str(getattr(artifact, "key", "") or "")
            haystack = f"{artifact_key}\n{desc}".lower()
            if query_lower in haystack:
                content = ""
                try:
                    local_path = artifact.cache()
                    content = Path(local_path).read_text(encoding="utf-8")
                except Exception as content_exc:
                    warnings.append(f"Could not read content for '{artifact_key}': {content_exc}")
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
        warnings.append(f"Artifact lookup failed: {exc}")

    return results, warnings


def get_lamindb_skill(*, key: str, run_uid: str | None = None, limit: int = 3) -> dict[str, Any]:
    """Load .md skills from laminlabs/biomed-skills whose key contains the search term."""
    warnings: list[str] = []
    results: list[dict[str, str]] = []
    key_lower = key.lower()

    try:
        ln.connect("laminlabs/biomed-skills")
        all_artifacts = ln.Artifact.filter(suffix=".md").all()
        for artifact in all_artifacts:
            artifact_key = str(getattr(artifact, "key", "") or "")
            desc = str(getattr(artifact, "description", "") or "")
            # only include artifacts whose key or description contains the search term
            if key_lower not in artifact_key.lower() and key_lower not in desc.lower():
                continue
            content = ""
            try:
                local_path = artifact.cache()
                content = Path(local_path).read_text(encoding="utf-8")
            except Exception as exc:
                warnings.append(f"Could not read '{artifact_key}': {exc}")
            results.append({
                "type": "artifact",
                "uid": str(getattr(artifact, "uid", "") or ""),
                "key": artifact_key,
                "description": desc[:1000],
                "content": content,
            })
            if len(results) >= limit:
                break
    except Exception as exc:
        warnings.append(f"Could not load skills from laminlabs/biomed-skills: {exc}")

    skill_content = "\n\n---\n\n".join(
        r["content"] for r in results if r.get("content")
    )

    return {
        "run_uid": run_uid,
        "key": key,
        "results": results,
        "searched_instances": ["laminlabs/biomed-skills"],
        "message": f"Found {len(results)} skill(s).",
        "skill_content": skill_content,
        "warnings": warnings,
    }


def _fetch_all_skill_metadata() -> tuple[list[dict[str, str]], list[str]]:
    """Connect to laminlabs/biomed-skills and return metadata (key + description) for all .md artifacts."""
    warnings: list[str] = []
    skills_meta: list[dict[str, str]] = []

    try:
        ln.connect("laminlabs/biomed-skills")
        artifacts = ln.Artifact.filter(suffix=".md").all()
        for artifact in artifacts:
            skills_meta.append({
                "key": str(getattr(artifact, "key", "") or ""),
                "description": str(getattr(artifact, "description", "") or ""),
                "uid": str(getattr(artifact, "uid", "") or ""),
            })
    except Exception as exc:
        warnings.append(f"Could not fetch skill metadata: {exc}")

    return skills_meta, warnings


def _load_skill_content(uid: str) -> str:
    """Load full content of a skill artifact by uid. Assumes laminlabs/biomed-skills is already connected."""
    content = ""
    try:
        artifact = ln.Artifact.get(uid)
        local_path = artifact.cache()
        content = Path(local_path).read_text(encoding="utf-8")
    except Exception:
        pass
    return content


def retrieve_relevant_skills(*, prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
    """Biomni-style skill retrieval: one LLM call selects relevant skills, then loads their content.

    Returns the full markdown content of relevant skills joined as a single string,
    ready to inject into the agent's system prompt.
    """
    skills_meta, _ = _fetch_all_skill_metadata()
    if not skills_meta:
        return ""

    # build selection prompt — just titles and descriptions, no full content
    skills_list = "\n".join(
        f"{i}. {s['key']}: {s['description']}" for i, s in enumerate(skills_meta)
    )
    selection_prompt = (
        f"You are selecting which skill documents are relevant to a task.\n\n"
        f"TASK: {prompt}\n\n"
        f"AVAILABLE SKILLS:\n{skills_list}\n\n"
        f"Return ONLY the indices of relevant skills as a JSON array, e.g.: [0, 1]\n"
        f"Return [] if none are relevant. No explanation, just the array."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "X-goog-api-key": api_key}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": selection_prompt}]}],
        "generationConfig": {"temperature": 0.0},
    }

    # one call with simple retry
    for attempt in range(5):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=60)
            if response.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
            response.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 2:
                return ""
            time.sleep(30)

    try:
        text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        indices = json.loads(match.group()) if match else []
    except Exception:
        return ""

    # load full content only for selected skills
    selected_contents = []
    for idx in indices:
        if 0 <= idx < len(skills_meta):
            content = _load_skill_content(skills_meta[idx]["uid"])
            if content:
                selected_contents.append(content)

    return "\n\n---\n\n".join(selected_contents)
