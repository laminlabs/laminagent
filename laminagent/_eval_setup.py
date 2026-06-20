from __future__ import annotations

from pathlib import Path

import lamindb as ln

EVAL_SCHEMA_NAME = "laminagent_eval"
EVAL_REGISTRY_NAME = "LaminAgent"

_FEATURE_DTYPES: dict[str, type | str] = {
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
        _get_or_create_feature(name, dtype) for name, dtype in _FEATURE_DTYPES.items()
    ]
    schema = ln.Schema.filter(name=EVAL_SCHEMA_NAME).one_or_none()
    if schema is None:
        schema = ln.Schema(features=features, name=EVAL_SCHEMA_NAME).save()
    return schema


def get_or_create_eval_registry(schema: ln.Schema) -> ln.Record:
    registry = ln.Record.filter(name=EVAL_REGISTRY_NAME, is_type=True).one_or_none()
    if registry is None:
        registry = ln.Record(
            name=EVAL_REGISTRY_NAME, is_type=True, schema=schema
        ).save()
    elif registry.schema_id is None:
        registry.schema = schema
        registry.save()
    return registry


def get_or_create_task(
    task_name: str, eval_registry: ln.Record, schema: ln.Schema
) -> ln.Record:
    task = ln.Record.filter(
        name=task_name, type=eval_registry, is_type=True
    ).one_or_none()
    if task is None:
        task = ln.Record(
            name=task_name, type=eval_registry, is_type=True, schema=schema
        ).save()
    elif task.schema_id is None:
        task.schema = schema
        task.save()
    return task


def ensure_eval_task(task_name: str) -> ln.Record:
    schema = get_or_create_schema()
    registry = get_or_create_eval_registry(schema)
    return get_or_create_task(task_name, registry, schema)


def parse_task_name_from_script(script: Path) -> str:
    script = script.resolve()
    assert script.parent.name == "tasks"
    assert script.parent.parent.name == "tests"
    return script.name


def setup(script_basenames: list[str] | None = None, verbose: bool = True) -> None:
    schema = get_or_create_schema()
    eval_registry = get_or_create_eval_registry(schema)
    created_scripts = 0
    for script_basename in script_basenames or []:
        get_or_create_task(script_basename, eval_registry=eval_registry, schema=schema)
        created_scripts += 1
    if verbose:
        print(
            "Configured LaminAgent eval registry "
            f"'{eval_registry.name}' with schema '{schema.name}'."
        )
        print(f"Configured {created_scripts} eval task registries.")


def setup_from_script_or_cwd(script: Path | None) -> None:
    if script is not None:
        script_basename = parse_task_name_from_script(script)
        setup(script_basenames=[script_basename])
        return

    tasks_dir = Path.cwd() / "tests" / "tasks"
    script_basenames = sorted(
        path.name
        for path in tasks_dir.glob("*.py")
        if path.is_file() and path.name != "conftest.py" and path.name != "testutils.py"
    )
    setup(script_basenames=script_basenames)
