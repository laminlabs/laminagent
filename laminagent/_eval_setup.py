from __future__ import annotations

from pathlib import Path

import lamindb as ln

EVAL_SCHEMA_NAME = "laminprofiler"
EVAL_REGISTRY_NAME = "LaminAgentEval"

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


def get_or_create_package(
    package_name: str, eval_registry: ln.Record, schema: ln.Schema
) -> ln.Record:
    package = ln.Record.filter(
        name=package_name, type=eval_registry, is_type=True
    ).one_or_none()
    if package is None:
        package = ln.Record(
            name=package_name, type=eval_registry, is_type=True, schema=schema
        ).save()
    elif package.schema_id is None:
        package.schema = schema
        package.save()
    return package


def get_or_create_task(
    task_name: str, package: ln.Record, schema: ln.Schema
) -> ln.Record:
    task = ln.Record.filter(name=task_name, type=package, is_type=True).one_or_none()
    if task is None:
        task = ln.Record(
            name=task_name, type=package, is_type=True, schema=schema
        ).save()
    elif task.schema_id is None:
        task.schema = schema
        task.save()
    return task


def ensure_eval_task(package_name: str, task_name: str) -> ln.Record:
    schema = get_or_create_schema()
    registry = get_or_create_eval_registry(schema)
    package = get_or_create_package(package_name, registry, schema)
    return get_or_create_task(task_name, package, schema)


def parse_registry_names_from_script(script: Path) -> tuple[str, str]:
    script = script.resolve()
    assert script.parent.name == "examples"
    assert script.parent.parent.name == "tests"
    package_name = script.parent.parent.parent.name.replace("-", "_")
    script_basename = script.name
    return package_name, script_basename


def setup(
    package_name: str | None = None,
    script_basenames: list[str] | None = None,
    verbose: bool = True,
) -> None:
    schema = get_or_create_schema()
    eval_registry = get_or_create_eval_registry(schema)
    created_scripts = 0
    if package_name is not None:
        package = get_or_create_package(
            package_name=package_name, eval_registry=eval_registry, schema=schema
        )
        for script_basename in script_basenames or []:
            get_or_create_task(script_basename, package=package, schema=schema)
            created_scripts += 1
    if verbose:
        print(
            "Configured LaminAgent eval registry "
            f"'{eval_registry.name}' with schema '{schema.name}'."
        )
        if package_name is not None:
            print(
                f"Configured package '{package_name}' with {created_scripts} eval task "
                "registries."
            )


def setup_from_script_or_cwd(script: Path | None) -> None:
    if script is not None:
        package_name, script_basename = parse_registry_names_from_script(script)
        setup(package_name=package_name, script_basenames=[script_basename])
        return

    package_name = Path.cwd().name.replace("-", "_")
    examples_dir = Path.cwd() / "tests" / "examples"
    script_basenames = sorted(
        path.name
        for path in examples_dir.glob("*.py")
        if path.is_file() and path.name != "conftest.py" and path.name != "testutils.py"
    )
    setup(package_name=package_name, script_basenames=script_basenames)
