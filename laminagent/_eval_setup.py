"""Backward-compatible imports for setup helpers.

Use ``laminagent.setup`` for public access.
"""

from laminagent.setup import (
    SETUP_REGISTRY_NAME as EVAL_REGISTRY_NAME,
)
from laminagent.setup import (
    SETUP_SCHEMA_NAME as EVAL_SCHEMA_NAME,
)
from laminagent.setup import (
    ensure_task as ensure_eval_task,
)
from laminagent.setup import (
    get_or_create_registry as get_or_create_eval_registry,
)
from laminagent.setup import (
    get_or_create_schema,
    get_or_create_task,
    parse_task_name_from_script,
    setup,
)

__all__ = [
    "EVAL_REGISTRY_NAME",
    "EVAL_SCHEMA_NAME",
    "ensure_eval_task",
    "get_or_create_eval_registry",
    "get_or_create_schema",
    "get_or_create_task",
    "parse_task_name_from_script",
    "setup",
]
