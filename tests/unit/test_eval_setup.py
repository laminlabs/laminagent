import importlib
import sys
import types
from pathlib import Path


def test_setup_creates_eval_registry_with_expected_schema(monkeypatch) -> None:
    class Query:
        def __init__(self, record):
            self._record = record

        def one_or_none(self):
            return self._record

    class FakeFeature:
        _store: dict[str, object] = {}

        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype

        @classmethod
        def filter(cls, *, name):
            return Query(cls._store.get(name))

        def save(self):
            self.__class__._store[self.name] = self
            return self

    class FakeSchema:
        _store: dict[str, object] = {}

        def __init__(self, *, features, name):
            self.features = features
            self.name = name

        @classmethod
        def filter(cls, *, name):
            return Query(cls._store.get(name))

        def save(self):
            self.__class__._store[self.name] = self
            return self

    class FakeRecord:
        _store: list[object] = []

        def __init__(self, *, name, is_type, schema=None, type=None):
            self.name = name
            self.is_type = is_type
            self.schema = schema
            self.type = type
            self.schema_id = 1 if schema is not None else None

        @classmethod
        def filter(cls, *, name, is_type, type=None):
            record = next(
                (
                    item
                    for item in cls._store
                    if item.name == name
                    and item.is_type == is_type
                    and item.type == type
                ),
                None,
            )
            return Query(record)

        def save(self):
            self.schema_id = 1 if self.schema is not None else None
            if self not in self.__class__._store:
                self.__class__._store.append(self)
            return self

    fake_ln = types.SimpleNamespace(
        Feature=FakeFeature,
        Schema=FakeSchema,
        Record=FakeRecord,
    )

    monkeypatch.setitem(sys.modules, "lamindb", fake_ln)
    sys.modules.pop("laminagent._eval_setup", None)
    module = importlib.import_module("laminagent._eval_setup")

    module.setup(
        script_basenames=["test_01_create_fasta_for_favorite_protein.py"],
        verbose=False,
    )
    module.setup(
        script_basenames=["test_01_create_fasta_for_favorite_protein.py"],
        verbose=False,
    )

    schema = FakeSchema.filter(name=module.EVAL_SCHEMA_NAME).one_or_none()
    registry = FakeRecord.filter(
        name=module.EVAL_REGISTRY_NAME, is_type=True
    ).one_or_none()
    task = FakeRecord.filter(
        name="test_01_create_fasta_for_favorite_protein.py", is_type=True, type=registry
    ).one_or_none()

    assert schema is not None
    assert schema.name == module.EVAL_SCHEMA_NAME
    assert {feature.name for feature in schema.features} == {
        "package_version",
        "duration_in_sec",
        "commit_hash16",
        "runner_env",
        "n_call_count",
        "n_prompt_tokens",
        "n_output_tokens",
        "n_total_tokens",
    }
    assert registry is not None
    assert registry.schema is schema
    assert task is not None
    assert task.schema is schema


def test_setup_from_script_or_cwd_collects_task_scripts(
    tmp_path: Path, monkeypatch
) -> None:
    package_dir = tmp_path / "laminagent"
    tasks_dir = package_dir / "tests" / "tasks"
    tasks_dir.mkdir(parents=True)
    (tasks_dir / "test_01.py").write_text("print('a')\n", encoding="utf-8")
    (tasks_dir / "test_02.py").write_text("print('b')\n", encoding="utf-8")
    (tasks_dir / "conftest.py").write_text("", encoding="utf-8")
    (tasks_dir / "testutils.py").write_text("", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_setup(*, script_basenames, verbose=True):
        captured["script_basenames"] = script_basenames
        captured["verbose"] = verbose

    monkeypatch.chdir(package_dir)
    monkeypatch.setattr("laminagent._eval_setup.setup", _fake_setup)

    from laminagent._eval_setup import setup_from_script_or_cwd

    setup_from_script_or_cwd(None)

    assert captured["script_basenames"] == ["test_01.py", "test_02.py"]
