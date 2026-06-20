from __future__ import annotations

from pathlib import Path

import lamindb as ln
from lamin_utils import logger

SETUP_SCHEMA_NAME = "lag_eval"
SETUP_REGISTRY_NAME = "LagEval"
USAGE_FEATURE_TYPE_NAME = "LagEval"

_BASE_FEATURE_DTYPES: dict[str, type | str] = {
    "package_version": str,
    "duration_in_sec": float,
    "commit_hash16": str,
    "runner_env": str,
}

_USAGE_FEATURE_DTYPES: dict[str, type | str] = {
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


def _get_or_create_usage_feature_type() -> ln.Feature:
    feature_type = ln.Feature.filter(
        name=USAGE_FEATURE_TYPE_NAME, is_type=True
    ).one_or_none()
    if feature_type is None:
        feature_type = ln.Feature(
            name=USAGE_FEATURE_TYPE_NAME,
            description="Auto-generated features tracking LagEval usage",
            is_type=True,
        ).save()
    return feature_type


def _get_or_create_usage_feature(
    name: str, dtype: type | str, feature_type: ln.Feature
) -> ln.Feature:
    feature = ln.Feature.filter(name=name, type=feature_type).one_or_none()
    if feature is None:
        feature = ln.Feature(name=name, dtype=dtype, type=feature_type).save()
    return feature


def get_or_create_schema() -> ln.Schema:
    features = [
        _get_or_create_feature(name, dtype)
        for name, dtype in _BASE_FEATURE_DTYPES.items()
    ]
    usage_feature_type = _get_or_create_usage_feature_type()
    features.extend(
        _get_or_create_usage_feature(name, dtype, usage_feature_type)
        for name, dtype in _USAGE_FEATURE_DTYPES.items()
    )
    schema = ln.Schema.filter(name=SETUP_SCHEMA_NAME).one_or_none()
    if schema is None:
        schema = ln.Schema(features=features, name=SETUP_SCHEMA_NAME).save()
    return schema


def get_or_create_registry(schema: ln.Schema) -> ln.Record:
    registry = ln.Record.filter(name=SETUP_REGISTRY_NAME, is_type=True).one_or_none()
    if registry is None:
        registry = ln.Record(name=SETUP_REGISTRY_NAME, is_type=True).save()
    elif registry.schema_id is not None:
        registry.schema = None
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


def get_registry() -> ln.Record | None:
    return ln.Record.filter(name=SETUP_REGISTRY_NAME, is_type=True).one_or_none()


def get_task(task_name: str) -> ln.Record | None:
    registry = get_registry()
    if registry is None:
        return None
    return ln.Record.filter(name=task_name, type=registry, is_type=True).one_or_none()


def parse_task_name_from_script(script: Path) -> str:
    script = script.resolve()
    assert script.parent.name == "tasks"
    assert script.parent.parent.name == "tests"
    return normalize_task_name(script.name)


def normalize_task_name(filename: str) -> str:
    task_name = Path(filename).name
    if task_name.endswith(".py"):
        task_name = task_name[: -len(".py")]
    if task_name.startswith("test_"):
        task_name = task_name[len("test_") :]
    return task_name


def _discover_task_basenames() -> list[str]:
    tasks_dir = Path.cwd() / "tests" / "tasks"
    return sorted(
        normalize_task_name(path.name)
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
        task_basenames = [normalize_task_name(name) for name in script_basenames]
    elif script is not None:
        task_basenames = [parse_task_name_from_script(script)]
    else:
        task_basenames = _discover_task_basenames()

    for task_basename in task_basenames:
        get_or_create_task(task_basename, registry=registry, schema=schema)

    if verbose:
        logger.important(
            "Configured LagEval registry without a schema and "
            f"{len(task_basenames)} task registries in schema '{schema.name}'."
        )
