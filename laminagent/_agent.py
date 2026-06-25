from __future__ import annotations

import copy
import json
import re
import time
from typing import TYPE_CHECKING, Any

import requests

from ._context import read_skill_from_lamindb_instance
from ._writer import write_python_script

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ._run_context import RunContext

SYSTEM_INSTRUCTION = (
    "You are a scientific coding agent. Write runnable Python code in one script whenever possible. "
    "For simple requests, call write_python_script exactly once and then finish. "
    "Do not create helper runner scripts that only execute other generated scripts via subprocess; "
    "write the task directly in the produced runnable script. "
    "If the prompt references an existing script, update that script instead of creating a new one. "
    "If the prompt references a biomed skill UID and instance, call read_skill_from_lamindb_instance "
    "before writing code and follow the returned README instructions. "
    "Do not write defensive code but write concise cosde that assumes the latest version of lamindb."
)


def _function_declarations() -> list[dict[str, Any]]:
    declarations: list[dict[str, Any]] = []
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
    declarations.append(
        {
            "name": "read_skill_from_lamindb_instance",
            "description": (
                "Read a skill README content by uid from a LaminDB instance."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "uid": {"type": "STRING"},
                    "instance_slug": {"type": "STRING"},
                },
                "required": ["uid"],
            },
        }
    )
    return declarations


def _tool_payload() -> list[dict[str, Any]]:
    return [{"functionDeclarations": _function_declarations()}]


def _extract_text(parts: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for part in parts:
        text = part.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "\n".join(chunks).strip()


def _extract_explicit_tool_keys(text: str) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for match in re.findall(r"([A-Za-z0-9_./-]+\.py)", text):
        key = match.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _default_filename_for_tool(tool_name: str, default_output_file: Path) -> str:
    suffix_by_tool = {
        "write_python_script": ".py",
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
    if name == "read_skill_from_lamindb_instance":
        uid = str(args.get("uid") or "").strip()
        instance_slug = str(args.get("instance_slug") or "").strip()
        return read_skill_from_lamindb_instance(
            uid=uid,
            run_uid=run_context.run_uid,
            instance_slug=instance_slug or "laminlabs/biomed-skills",
        )

    if name == "write_python_script":
        filename = str(
            args.get("filename") or ""
        ).strip() or _default_filename_for_tool(name, default_output_file)
        code = str(args.get("code", ""))
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
        if existing_generated_files and filename not in existing_generated_files:
            existing_name = existing_generated_files[0]
            return {
                "status": "error",
                "message": (
                    "A runnable script was already created in this run. "
                    f"Update '{existing_name}' instead of creating '{filename}'."
                ),
                "run_uid": run_context.run_uid,
            }
        return write_python_script(
            code=code,
            filename=filename,
            run_uid=run_context.run_uid,
            track_outputs=run_context.track_outputs,
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
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{run_context.model}:generateContent"
    )
    contents: list[dict[str, Any]] = [
        {
            "role": "user",
            "parts": [
                {"text": f"{SYSTEM_INSTRUCTION}\n\nPrompt: {run_context.prompt}"},
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
        progress_callback(f"model={run_context.model}")
        progress_callback(f"prompt: {run_context.prompt}")

    for step in range(1, max_steps + 1):
        if progress_callback is not None:
            progress_callback(f"step {step}: waiting for model response")
        payload = {
            "contents": contents,
            "tools": _tool_payload(),
            "generationConfig": {"temperature": 0.2},
        }
        if progress_callback is not None:
            progress_callback(
                "step "
                f"{step}: llm request payload="
                f"{json.dumps(payload, ensure_ascii=False, default=str)}"
            )
        trace_events.append(
            {
                "step": step,
                "event": "llm_request",
                "request_payload": copy.deepcopy(payload),
            }
        )
        data = _post_generate_content(
            url=url,
            api_key=api_key,
            payload=payload,
            progress_callback=progress_callback,
        )
        if progress_callback is not None:
            progress_callback(
                "step "
                f"{step}: llm response payload="
                f"{json.dumps(data, ensure_ascii=False, default=str)}"
            )
        usage_metadata = data.get("usageMetadata")
        if isinstance(usage_metadata, dict):
            run_context.llm_usage.add_usage_metadata(usage_metadata)
        else:
            run_context.llm_usage.add_usage_metadata(None)
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

        trace_events.append(
            {
                "step": step,
                "event": "llm_response",
                "response_payload": copy.deepcopy(data),
                "model_response": copy.deepcopy(response_message),
                "usage_metadata": (
                    copy.deepcopy(usage_metadata)
                    if isinstance(usage_metadata, dict)
                    else None
                ),
                "text": _extract_text(parts),
            }
        )
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
            trace_events.append(
                {
                    "step": step,
                    "event": "tool_call",
                    "tool": name,
                    "tool_args": copy.deepcopy(args),
                }
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
                    "event": "tool_result",
                    "tool": name,
                    "tool_args": copy.deepcopy(args),
                    "tool_result": copy.deepcopy(result),
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
            if name == "write_python_script" and result.get("status") == "success":
                final_text = f"Wrote runnable script '{generated_file or 'script.py'}'."
                short_circuit_execute = True
                break
            if progress_callback is not None:
                status = result.get("status", "ok")
                progress_callback(f"step {step}: tool result status={status}")
                progress_callback(
                    "step "
                    f"{step}: tool result payload={json.dumps(result, ensure_ascii=False, default=str)}"
                )
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
            raise RuntimeError(fatal_error)
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
        "llm_usage": run_context.llm_usage.to_dict(),
    }


def write_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
