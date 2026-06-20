from __future__ import annotations

from pathlib import Path

import lamindb as ln

SETUP_SCHEMA_NAME = "laminagent_eval"
SETUP_REGISTRY_NAME = "LaminAgent"

_USAGE_FEATURE_DTYPES: dict[str, type | str] = {
    "package_version": str,
    "duration_in_sec": float,
    "commit_hash16": str,
    "runner_env": str,
    "n_call_count": int,
    "n_prompt_tokens": int,
    "n_output_tokens": int,
    "n_total_tokens": int,
}


def _get_or_create_feature(name: str, dtype: type | str) -> ln.Feature:
    feature = ln.Feature.filter(name=name).one_or_none()
    if feature is None:
        feature = ln.Feature(name=name, dtype=dtype).save()
    return feature


def get_or_create_schema() -> ln.Schema:
    features = [
        _get_or_create_feature(name, dtype)
        for name, dtype in _USAGE_FEATURE_DTYPES.items()
    ]
    schema = ln.Schema.filter(name=SETUP_SCHEMA_NAME).one_or_none()
    if schema is None:
        schema = ln.Schema(features=features, name=SETUP_SCHEMA_NAME).save()
    return schema


def get_or_create_registry(schema: ln.Schema) -> ln.Record:
    registry = ln.Record.filter(name=SETUP_REGISTRY_NAME, is_type=True).one_or_none()
    if registry is None:
        registry = ln.Record(
            name=SETUP_REGISTRY_NAME, is_type=True, schema=schema
        ).save()
    elif registry.schema_id is None:
        registry.schema = schema
        registry.save()
    return registry


def get_or_create_task(
    task_name: str, registry: ln.Record, schema: ln.Schema
) -> ln.Record:
    task = ln.Record.filter(name=task_name, type=registry, is_type=True).one_or_none()
    if task is None:
        task = ln.Record(
            name=task_name, type=registry, is_type=True, schema=schema
        ).save()
    elif task.schema_id is None:
        task.schema = schema
        task.save()
    return task


def ensure_task(task_name: str) -> ln.Record:
    schema = get_or_create_schema()
    registry = get_or_create_registry(schema)
    return get_or_create_task(task_name, registry, schema)


def parse_task_name_from_script(script: Path) -> str:
    script = script.resolve()
    assert script.parent.name == "tasks"
    assert script.parent.parent.name == "tests"
    return script.name


def _discover_task_basenames() -> list[str]:
    tasks_dir = Path.cwd() / "tests" / "tasks"
    return sorted(
        path.name
        for path in tasks_dir.glob("*.py")
        if path.is_file() and path.name not in {"conftest.py", "testutils.py"}
    )


def setup(
    script: Path | None = None,
    *,
    script_basenames: list[str] | None = None,
    verbose: bool = True,
) -> None:
    schema = get_or_create_schema()
    registry = get_or_create_registry(schema)

    if script is not None and script_basenames is not None:
        raise ValueError("Pass either script or script_basenames, not both.")

    if script_basenames is not None:
        task_basenames = list(script_basenames)
    elif script is not None:
        task_basenames = [parse_task_name_from_script(script)]
    else:
        task_basenames = _discover_task_basenames()

    for task_basename in task_basenames:
        get_or_create_task(task_basename, registry=registry, schema=schema)

    if verbose:
        print(
            "Configured LaminAgent registry "
            f"'{registry.name}' with schema '{schema.name}'."
        )
        print(f"Configured {len(task_basenames)} task registries.")
