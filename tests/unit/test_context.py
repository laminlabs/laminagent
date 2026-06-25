import laminagent._context as context


def test_read_skill_from_lamindb_instance_success(monkeypatch) -> None:
    class _FakeRecord:
        notes = "Use DataFrameCurator."

    class _FakeRecordRegistry:
        @staticmethod
        def get(uid: str):
            assert uid == "u5muNUOPnWPBuZ8z"
            return _FakeRecord()

    class _FakeSkillDB:
        Record = _FakeRecordRegistry()

    monkeypatch.setattr(context.ln, "DB", lambda _slug: _FakeSkillDB())

    result = context.read_skill_from_lamindb_instance(
        uid="u5muNUOPnWPBuZ8z",
        run_uid="run-1",
        instance_slug="laminlabs/biomed-skills",
    )
    assert result["status"] == "success"
    assert result["skill_uid"] == "u5muNUOPnWPBuZ8z"
    assert result["source_instance"] == "laminlabs/biomed-skills"
    assert "DataFrameCurator" in result["content"]


def test_read_skill_from_lamindb_instance_raises_on_db_error(monkeypatch) -> None:
    def _raise(*_args, **_kwargs):
        raise RuntimeError("db unavailable")

    monkeypatch.setattr(context.ln, "DB", _raise)

    try:
        context.read_skill_from_lamindb_instance(
            uid="u5muNUOPnWPBuZ8z",
            run_uid="run-1",
            instance_slug="laminlabs/biomed-skills",
        )
    except RuntimeError as exc:
        assert "db unavailable" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected RuntimeError from ln.DB failure")
