from types import SimpleNamespace

import lag_cli.context as context

DatasetMap = dict[str, dict[str, list[SimpleNamespace]]]


class _FakeQuery:
    def __init__(self, records):
        self._records = records

    def filter(self):
        return self

    def all(self):
        return self._records

    def get(self, **kwargs):
        key = kwargs.get("key")
        for record in self._records:
            if getattr(record, "key", None) == key:
                return record
        raise LookupError("Does not exist")


class _FakeDB:
    def __init__(
        self, slug: str, datasets: dict[str, dict[str, list[SimpleNamespace]]]
    ):
        data = datasets[slug]
        self.Transform = _FakeQuery(data["transforms"])
        self.Artifact = _FakeQuery(data["artifacts"])


def test_get_lamindb_skill_searches_current_instance_first(monkeypatch) -> None:
    db_calls: list[str] = []
    datasets: DatasetMap = {
        "laminlabs/lamindata": {
            "transforms": [
                SimpleNamespace(
                    uid="tr1",
                    key="test-lag/create_fasta.py",
                    description="FASTA tool",
                )
            ],
            "artifacts": [],
        },
        "laminlabs/biomed-skills": {"transforms": [], "artifacts": []},
    }

    def _fake_db(slug: str):
        db_calls.append(slug)
        return _FakeDB(slug, datasets)

    monkeypatch.setattr(context.ln, "DB", _fake_db)
    monkeypatch.setattr(
        context.ln,
        "setup",
        SimpleNamespace(
            settings=SimpleNamespace(
                instance=SimpleNamespace(slug="laminlabs/lamindata")
            )
        ),
    )
    monkeypatch.setattr(
        context.ln,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ln.connect() should not be used for read-only lookup")
        ),
    )

    result = context.get_lamindb_skill(key="test-lag/create_fasta.py", run_uid="run-1")
    assert db_calls == ["laminlabs/lamindata"]
    assert result["searched_instances"] == ["laminlabs/lamindata"]
    assert len(result["results"]) == 1
    assert result["results"][0]["type"] == "transform"
    assert result["results"][0]["key"] == "test-lag/create_fasta.py"


def test_get_lamindb_skill_falls_back_to_biomed_skills(monkeypatch) -> None:
    db_calls: list[str] = []
    datasets: DatasetMap = {
        "laminlabs/lamindata": {"transforms": [], "artifacts": []},
        "laminlabs/biomed-skills": {
            "transforms": [],
            "artifacts": [
                SimpleNamespace(
                    uid="ar1", key="create_fasta.py", description="script artifact"
                )
            ],
        },
    }

    def _fake_db(slug: str):
        db_calls.append(slug)
        return _FakeDB(slug, datasets)

    monkeypatch.setattr(context.ln, "DB", _fake_db)
    monkeypatch.setattr(
        context.ln,
        "setup",
        SimpleNamespace(
            settings=SimpleNamespace(
                instance=SimpleNamespace(slug="laminlabs/lamindata")
            )
        ),
    )
    monkeypatch.setattr(
        context.ln,
        "connect",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("ln.connect() should not be used for read-only lookup")
        ),
    )

    result = context.get_lamindb_skill(key="fasta", run_uid="run-2")
    assert db_calls == ["laminlabs/lamindata", "laminlabs/biomed-skills"]
    assert result["searched_instances"] == [
        "laminlabs/lamindata",
        "laminlabs/biomed-skills",
    ]
    assert len(result["results"]) == 1
    assert result["results"][0]["type"] == "artifact"
