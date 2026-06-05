from types import SimpleNamespace

import lag_cli.context as context


def test_get_lamindb_skill_connects_to_biomed_skills(monkeypatch) -> None:
    db_calls: list[str] = []

    class _FakeDB:
        def __init__(self, slug):
            db_calls.append(slug)
            self.Artifact = _FakeArtifactQuery()

    monkeypatch.setattr(context.ln, "DB", _FakeDB)

    artifact = SimpleNamespace(
        uid="ar1",
        key="biomed-skills/query-instance.md",
        description="Count artifacts",
    )
    artifact.cache = lambda: __import__("pathlib").Path("skill.md")

    class _FakeArtifactQuery:
        def filter(self, **_kwargs):
            return self

        def all(self):
            return [artifact]

    import pathlib

    original_read_text = pathlib.Path.read_text

    def _read_text(self, encoding="utf-8"):
        if str(self) == "skill.md":
            return "# Query instance"
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(pathlib.Path, "read_text", _read_text)

    result = context.get_lamindb_skill(key="query-instance", run_uid="run-1")

    assert db_calls == ["laminlabs/biomed-skills"]
    assert result["searched_instances"] == ["laminlabs/biomed-skills"]
    assert result["results"][0]["key"] == "biomed-skills/query-instance.md"


def test_get_lamindb_skill_dedupes_same_key(monkeypatch) -> None:
    monkeypatch.setattr(context.ln, "connect", lambda slug: None)
    monkeypatch.setattr(
        context.ln,
        "setup",
        SimpleNamespace(
            settings=SimpleNamespace(
                instance=SimpleNamespace(slug="test-owner/test-instance")
            )
        ),
    )

    def _make_artifact(uid: str):
        a = SimpleNamespace(
            uid=uid,
            key="biomed-skills/curate-bulkrna.md",
            description="bulk rna",
        )
        a.cache = lambda: __import__("pathlib").Path("skill.md")
        return a

    class _FakeArtifactQuery:
        def filter(self, **_kwargs):
            return self

        def all(self):
            # two versions of the same key
            return [_make_artifact("v1"), _make_artifact("v2")]

    monkeypatch.setattr(context.ln, "Artifact", _FakeArtifactQuery())

    import pathlib

    original_read_text = pathlib.Path.read_text

    def _read_text(self, encoding="utf-8"):
        if str(self) == "skill.md":
            return "# Bulk RNA"
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(pathlib.Path, "read_text", _read_text)

    result = context.get_lamindb_skill(key="curate-bulkrna", run_uid="run-x")

    keys = [r["key"] for r in result["results"]]
    assert keys == ["biomed-skills/curate-bulkrna.md"]


def test_get_lamindb_skill_falls_back_to_local(monkeypatch) -> None:
    db_calls: list[str] = []

    class _FakeDB:
        def __init__(self, slug):
            db_calls.append(slug)
            self.Artifact = _FakeArtifactQuery()

    monkeypatch.setattr(context.ln, "DB", _FakeDB)

    class _FakeArtifactQuery:
        def filter(self, **_kwargs):
            return self

        def all(self):
            return []

    monkeypatch.setattr(
        context,
        "_load_local_biomed_skills",
        lambda **kwargs: [
            {
                "type": "local_file",
                "uid": "",
                "key": "biomed-skills/curate-scrna.md",
                "description": "Local skill (biomed-skills/)",
                "content": "# Curate scRNA",
            }
        ],
    )

    result = context.get_lamindb_skill(key="curate-scrna", run_uid="run-2")

    assert db_calls == ["laminlabs/biomed-skills"]
    assert result["searched_instances"] == [
        "laminlabs/biomed-skills",
        "local/biomed-skills",
    ]
    assert result["results"][0]["key"] == "biomed-skills/curate-scrna.md"
