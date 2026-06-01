from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

from .context import get_lamindb_skill, get_local_skill
from .writer import (
    write_from_template,
    write_jupyter_notebook,
    write_python_script,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .run_context import RunContext

PLAN_SYSTEM_INSTRUCTION = (
    "You are a tool authoring agent. In --plan mode, create or update runnable "
    "tool files (.py/.ipynb) that satisfy the prompt. If the prompt references an "
    "explicit tool key/path, update that exact file instead of creating a new name. "
    "You may read skills/query LaminDB for context, but do not write markdown plans."
)

DO_SYSTEM_INSTRUCTION = (
    "You are a scientific coding agent. First retrieve relevant context when useful, "
    "then write runnable analysis code. For every output file your script/notebook writes, "
    "explicitly call ln.Artifact('<output_path>').save() in the generated code. "
    "Do not create helper runner scripts that only execute other generated scripts via subprocess; "
    "write the task directly in the produced runnable tool file(s)."
)


def _function_declarations(mode: str) -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = [
        {
            "name": "get_local_skill",
            "description": "Find relevant local SKILL.md docs for a topic.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "topic": {"type": "STRING"},
                    "skills_root": {"type": "STRING"},
                },
                "required": ["topic"],
            },
        },
        {
            "name": "get_lamindb_skill",
            "description": "Query laminlabs/biomed-skills for relevant transforms/artifacts.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "key": {"type": "STRING"},
                    "limit": {"type": "NUMBER"},
                },
                "required": ["key"],
            },
        },
    ]
    if mode == "plan":
        declarations.append(
            {
                "name": "write_from_template",
                "description": "Create a file from an existing template path.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "template_path": {"type": "STRING"},
                        "filename": {"type": "STRING"},
                    },
                    "required": ["template_path", "filename"],
                },
            }
        )
    if mode == "plan":
        declarations.append(
            {
                "name": "write_jupyter_notebook",
                "description": "Write an ipynb notebook file with markdown/code cells.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "filename": {"type": "STRING"},
                        "cells": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "type": {"type": "STRING"},
                                    "content": {"type": "STRING"},
                                },
                                "required": ["type", "content"],
                            },
                        },
                    },
                    "required": ["filename", "cells"],
                },
            }
        )
    if mode in {"plan", "do"}:
        declarations.append(
            {
                "name": "write_python_script",
                "description": "Write a runnable Python script file.",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "filename": {"type": "STRING"},
                        "code": {"type": "STRING"},
                    },
                    "required": ["filename", "code"],
                },
            }
        )
    return declarations


def _tool_payload(mode: str) -> list[dict[str, Any]]:
    return [{"functionDeclarations": _function_declarations(mode)}]


def _extract_text(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


def _looks_like_wrapper_runner(code: str, existing_generated_files: list[str]) -> bool:
    text = code.lower()
    if "subprocess.run" not in text:
        return False
    if "python" not in text and "sys.executable" not in text:
        return False
    if "artifact(" in text:
        return False

    py_target_match = re.search(r"""["'][^"']+\.py["']""", code)
    if not py_target_match:
        return False

    existing_names = {
        Path(path_str).name
        for path_str in existing_generated_files
        if path_str.endswith(".py")
    }
    if not existing_names:
        return True
    return any(name in code for name in existing_names)


def _is_runnable_tool_path(path_str: str) -> bool:
    suffix = Path(path_str).suffix.lower()
    return suffix in {".py", ".ipynb"}


def _is_explicit_tool_key(key: str) -> bool:
    stripped = key.strip().lower()
    return stripped.endswith(".py") or stripped.endswith(".ipynb")


def _extract_explicit_tool_keys(text: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"([A-Za-z0-9_./-]+\.(?:py|ipynb))", text):
        key = match.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _default_filename_for_tool(tool_name: str, default_output_file: Path) -> str:
    suffix_by_tool = {
        "write_python_script": ".py",
        "write_jupyter_notebook": ".ipynb",
    }
    expected_suffix = suffix_by_tool.get(tool_name)
    if expected_suffix is None:
        return str(default_output_file)
    return str(default_output_file.with_suffix(expected_suffix))


def _post_generate_content(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: int = 120,
    max_attempts: int = 4,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }
    backoff_seconds = 1.0
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            if progress_callback is not None:
                progress_callback(f"gemini request attempt {attempt}/{max_attempts}")
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            status = response.status_code
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts:
                if progress_callback is not None:
                    progress_callback(
                        f"gemini transient status {status}, retrying in {backoff_seconds:.1f}s"
                    )
                time.sleep(backoff_seconds)
                backoff_seconds *= 2
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt >= max_attempts:
                break
            if progress_callback is not None:
                progress_callback(
                    f"gemini request failed ({exc.__class__.__name__}), retrying in {backoff_seconds:.1f}s"
                )
            time.sleep(backoff_seconds)
            backoff_seconds *= 2

    if isinstance(last_error, requests.HTTPError) and last_error.response is not None:
        status = last_error.response.status_code
        body_preview = last_error.response.text[:1000]
        raise RuntimeError(
            f"Gemini API request failed after retries (status={status}). "
            f"Response preview: {body_preview}"
        ) from last_error
    raise RuntimeError("Gemini API request failed after retries.") from last_error


def _dispatch_tool(
    *,
    name: str,
    args: dict[str, Any],
    run_context: RunContext,
    default_output_file: Path,
    existing_generated_files: list[str],
) -> dict[str, Any]:
    if name == "get_local_skill":
        return get_local_skill(
            topic=str(args.get("topic", "")),
            skills_root=args.get("skills_root"),
            run_uid=run_context.run_uid,
        )
    if name == "get_lamindb_skill":
        key = str(args.get("key", ""))
        result = get_lamindb_skill(
            key=key,
            limit=int(args.get("limit", 5)),
            run_uid=run_context.run_uid,
        )
        if (
            run_context.mode == "do"
            and _is_explicit_tool_key(key)
            and not result.get("results")
        ):
            searched = result.get("searched_instances", [])
            searched_str = ", ".join(searched) if isinstance(searched, list) else "none"
            return {
                "status": "error",
                "fatal": True,
                "run_uid": run_context.run_uid,
                "message": (
                    f"Tool key '{key}' was not found in searched instances ({searched_str}). "
                    "Aborting without generating a new tool."
                ),
            }
        if (
            run_context.mode == "do"
            and _is_explicit_tool_key(key)
            and result.get("results")
        ):
            matched_key = key
            first_result = (
                result.get("results", [])[0]
                if isinstance(result.get("results"), list) and result.get("results")
                else None
            )
            if isinstance(first_result, dict):
                candidate_key = first_result.get("key")
                if isinstance(candidate_key, str) and candidate_key.strip():
                    matched_key = candidate_key.strip()
            return {
                "status": "success",
                "run_uid": run_context.run_uid,
                "message": (
                    f"Found existing runnable tool '{matched_key}'. "
                    "Skipping generation and proceeding to execution."
                ),
                "resolved_runnable_path": matched_key,
                "short_circuit_execute": True,
            }
        return result
    if name == "write_python_script":
        filename = str(
            args.get("filename") or ""
        ).strip() or _default_filename_for_tool(name, default_output_file)
        code = str(args.get("code", ""))
        if run_context.mode == "plan":
            explicit_keys = _extract_explicit_tool_keys(run_context.prompt)
            if len(explicit_keys) == 1 and filename != explicit_keys[0]:
                return {
                    "status": "error",
                    "message": (
                        "Prompt references explicit tool key "
                        f"'{explicit_keys[0]}'. Update that exact file instead of "
                        f"creating '{filename}'."
                    ),
                    "run_uid": run_context.run_uid,
                }
        if run_context.mode == "do":
            existing_runnables = [
                path_str
                for path_str in existing_generated_files
                if _is_runnable_tool_path(path_str)
            ]
            if existing_runnables and filename not in existing_runnables:
                existing_name = Path(existing_runnables[0]).name
                return {
                    "status": "error",
                    "message": (
                        "Rejected additional runnable tool file in do mode. "
                        f"Reuse the existing file '{existing_name}' instead of creating "
                        f"'{Path(filename).name}'."
                    ),
                    "run_uid": run_context.run_uid,
                }
        if run_context.mode == "do" and _looks_like_wrapper_runner(
            code, existing_generated_files
        ):
            return {
                "status": "error",
                "message": (
                    "Rejected wrapper runner script. In do mode, write the task directly "
                    "instead of invoking another generated script via subprocess."
                ),
                "run_uid": run_context.run_uid,
            }
        return write_python_script(
            code=code,
            filename=filename,
            run_uid=run_context.run_uid,
            track_outputs=run_context.track_outputs,
        )
    if name == "write_jupyter_notebook":
        filename = str(
            args.get("filename") or ""
        ).strip() or _default_filename_for_tool(name, default_output_file)
        if run_context.mode == "plan":
            explicit_keys = _extract_explicit_tool_keys(run_context.prompt)
            if len(explicit_keys) == 1 and filename != explicit_keys[0]:
                return {
                    "status": "error",
                    "message": (
                        "Prompt references explicit tool key "
                        f"'{explicit_keys[0]}'. Update that exact file instead of "
                        f"creating '{filename}'."
                    ),
                    "run_uid": run_context.run_uid,
                }
        cells = args.get("cells")
        if not isinstance(cells, list):
            cells = [{"type": "code", "content": ""}]
        return write_jupyter_notebook(
            cells=cells,
            filename=filename,
            run_uid=run_context.run_uid,
            track_outputs=run_context.track_outputs,
        )
    if name == "write_from_template":
        filename = str(args.get("filename") or default_output_file)
        return write_from_template(
            template_path=str(args.get("template_path", "")),
            filename=filename,
            run_uid=run_context.run_uid,
        )
    return {
        "status": "error",
        "message": f"Unknown tool: {name}",
        "run_uid": run_context.run_uid,
    }


def run_agent(
    *,
    api_key: str,
    run_context: RunContext,
    output_file: Path,
    max_steps: int = 20,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    system_instruction = (
        PLAN_SYSTEM_INSTRUCTION if run_context.mode == "plan" else DO_SYSTEM_INSTRUCTION
    )
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{run_context.model}:generateContent"
    )
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {"text": f"{system_instruction}\n\nPrompt: {run_context.prompt}"},
            ],
        }
    ]
    trace_events: list[dict[str, Any]] = []
    generated_file: str | None = None
    generated_files: list[str] = []
    final_text = ""
    fatal_error: str | None = None
    resolved_runnable_path: str | None = None
    short_circuit_execute = False
    if progress_callback is not None:
        progress_callback(f"mode={run_context.mode} model={run_context.model}")
        progress_callback(f"prompt: {run_context.prompt}")

    for step in range(1, max_steps + 1):
        if progress_callback is not None:
            progress_callback(f"step {step}: waiting for model response")
        payload = {
            "contents": contents,
            "tools": _tool_payload(run_context.mode),
            "generationConfig": {"temperature": 0.2},
        }
        data = _post_generate_content(
            url=url,
            api_key=api_key,
            payload=payload,
            progress_callback=progress_callback,
        )
        candidate = data.get("candidates", [{}])[0]
        response_message = candidate.get("content", {})
        contents.append(response_message)
        parts = response_message.get("parts", [])
        text_preview = _extract_text(parts)
        if progress_callback is not None and text_preview:
            preview = (
                text_preview if len(text_preview) <= 300 else f"{text_preview[:300]}..."
            )
            progress_callback(f"step {step}: model text: {preview}")

        trace_events.append({"step": step, "model_response": response_message})
        tool_calls = [p.get("functionCall") for p in parts if "functionCall" in p]
        if not tool_calls:
            final_text = _extract_text(parts)
            if progress_callback is not None:
                progress_callback("model finished without further tool calls")
            break

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = str(tool_call.get("name", ""))
            args = tool_call.get("args", {})
            if not isinstance(args, dict):
                args = {}
            if progress_callback is not None:
                progress_callback(
                    f"step {step}: tool call -> {name} args={json.dumps(args)}"
                )

            result = _dispatch_tool(
                name=name,
                args=args,
                run_context=run_context,
                default_output_file=output_file,
                existing_generated_files=generated_files,
            )
            generated = result.get("file")
            if isinstance(generated, str) and generated:
                generated_file = generated
                if generated not in generated_files:
                    generated_files.append(generated)
                if progress_callback is not None:
                    progress_callback(f"step {step}: wrote file {generated}")

            trace_events.append(
                {
                    "step": step,
                    "tool": name,
                    "tool_args": args,
                    "tool_result": result,
                }
            )
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": result,
                            }
                        }
                    ],
                }
            )
            if progress_callback is not None:
                status = result.get("status", "ok")
                progress_callback(f"step {step}: tool result status={status}")
                if status == "error" and result.get("message"):
                    progress_callback(f"step {step}: tool error: {result['message']}")
                if result.get("short_circuit_execute") and result.get("message"):
                    progress_callback(f"step {step}: {result['message']}")

            resolved_path = result.get("resolved_runnable_path")
            if isinstance(resolved_path, str) and resolved_path.strip():
                resolved_runnable_path = resolved_path.strip()
            if result.get("short_circuit_execute"):
                short_circuit_execute = True
                final_text = str(
                    result.get(
                        "message",
                        f"Resolved runnable '{resolved_runnable_path}' for execution.",
                    )
                )
                break

            if result.get("fatal"):
                fatal_error = str(result.get("message", "Fatal tool error."))
                break

        if fatal_error is not None:
            final_text = fatal_error
            break
        if short_circuit_execute:
            break

    return {
        "run_uid": run_context.run_uid,
        "contents": contents,
        "trace_events": trace_events,
        "generated_file": generated_file,
        "generated_files": generated_files,
        "resolved_runnable_path": resolved_runnable_path,
        "final_text": final_text,
    }


def write_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
