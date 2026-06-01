from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm
from dotenv import load_dotenv

from .context import get_lamindb_skill, get_local_skill
from .do_executor import execute_code_string
from .writer import (
    write_from_template,
    write_jupyter_notebook,
    write_python_script,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .run_context import RunContext

# load API keys from ~/llms.env so both GEMINI_API_KEY and ANTHROPIC_API_KEY are available
load_dotenv(dotenv_path=Path("~/llms.env").expanduser())

PLAN_SYSTEM_INSTRUCTION = (
    "You are a scientific curation agent that works in an iterative loop. "
    "Follow the provided skills exactly. Your workflow is:\n"
    "1. Write a runnable Python script with write_python_script.\n"
    "2. EXECUTE it immediately with execute_python.\n"
    "3. Read the tool result carefully: check exit_code, stdout, and stderr.\n"
    "   - exit_code=0 and artifact saved → you are done.\n"
    "   - exit_code!=0 or any error in stderr → identify the SPECIFIC error, fix ONLY that part, and execute again.\n"
    "   - Do NOT retry the exact same code after an error — you must change something.\n"
    "4. Repeat until exit_code=0 and the artifact is saved.\n"
    "You are NOT done after merely writing the script. "
    "Strictly obey registry rules: never add new ontology terms autonomously; "
    "if a label cannot be mapped, stop and report it to the user."
)

DO_SYSTEM_INSTRUCTION = (
    "You are a scientific coding agent. First retrieve relevant context when useful, "
    "then write runnable analysis code. For every output file your script/notebook writes, "
    "explicitly call ln.Artifact('<output_path>').save() in the generated code. "
    "Do not create helper runner scripts that only execute other generated scripts via subprocess; "
    "write the task directly in the produced runnable tool file(s)."
)


def _tool_definitions(mode: str) -> list[dict[str, Any]]:
    """Return tools in OpenAI/LiteLLM format."""
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": (
                    "Execute Python code and return stdout and stderr. "
                    "Use this to run curation code, observe the result, and fix errors. "
                    "Call this repeatedly until lamindb validation passes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                    },
                    "required": ["code"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_local_skill",
                "description": "Find relevant local SKILL.md docs for a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "skills_root": {"type": "string"},
                    },
                    "required": ["topic"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_lamindb_skill",
                "description": "Query laminlabs/biomed-skills for relevant transforms/artifacts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string"},
                        "limit": {"type": "number"},
                    },
                    "required": ["key"],
                },
            },
        },
    ]
    if mode == "plan":
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "write_from_template",
                    "description": "Create a file from an existing template path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "template_path": {"type": "string"},
                            "filename": {"type": "string", "description": "Output filename (optional, auto-generated if omitted)."},
                        },
                        "required": ["template_path"],
                    },
                },
            }
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "write_jupyter_notebook",
                    "description": "Write an ipynb notebook file with markdown/code cells.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Output filename (optional, auto-generated if omitted)."},
                            "cells": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "content": {"type": "string"},
                                    },
                                    "required": ["type", "content"],
                                },
                            },
                        },
                        "required": ["cells"],
                    },
                },
            }
        )
    if mode in {"plan", "do"}:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "write_python_script",
                    "description": "Write a runnable Python script file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string", "description": "Output filename (optional, auto-generated if omitted)."},
                            "code": {"type": "string"},
                        },
                        "required": ["code"],
                    },
                },
            }
        )
    return tools


def _extract_text_from_content(content: Any) -> str:
    """Extract plain text from an OpenAI-style message content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = [block.get("text", "") for block in content if isinstance(block, dict)]
        return "\n".join(c for c in chunks if c).strip()
    return ""


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



def _dispatch_tool(
    *,
    name: str,
    args: dict[str, Any],
    run_context: RunContext,
    default_output_file: Path,
    existing_generated_files: list[str],
) -> dict[str, Any]:
    if name == "execute_python":
        code = str(args.get("code", ""))
        return execute_code_string(code=code, run_uid=run_context.run_uid)
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


def _trim_messages(messages: list[dict[str, Any]], keep_tool_results: int = 2) -> list[dict[str, Any]]:
    """Keep system + user + all assistant messages, but only the last N tool results.

    Old tool results bloat the context window fast (stdout/stderr can be 4k chars each).
    We summarise older ones to a one-liner so the model still sees the history shape
    but doesn't hit token limits.
    """
    result: list[dict[str, Any]] = []
    tool_result_indices: list[int] = []

    for i, msg in enumerate(messages):
        result.append(msg)
        if msg.get("role") == "tool":
            tool_result_indices.append(i)

    # replace older tool results with a short summary
    to_summarise = tool_result_indices[:-keep_tool_results] if len(tool_result_indices) > keep_tool_results else []
    for i in to_summarise:
        try:
            payload = json.loads(result[i].get("content", "{}"))
            exit_code = payload.get("exit_code", "?")
            stdout_snippet = (payload.get("stdout") or "")[:80].strip()
            result[i] = {
                "role": "tool",
                "tool_call_id": result[i].get("tool_call_id", ""),
                "content": json.dumps({"exit_code": exit_code, "summary": stdout_snippet or "(no output)"}),
            }
        except (json.JSONDecodeError, AttributeError):
            pass

    return result


def _call_llm(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    progress_callback: Callable[[str], None] | None = None,
    max_attempts: int = 5,
) -> Any:
    """Call the LLM via LiteLLM with retry/backoff for rate limits."""
    import time

    backoff = 30.0
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        if progress_callback is not None:
            progress_callback(f"calling {model} (attempt {attempt}/{max_attempts}) ...")
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
            return response.choices[0].message
        except litellm.RateLimitError as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            if progress_callback is not None:
                progress_callback(f"rate limit hit, retrying in {backoff:.0f}s ...")
            time.sleep(backoff)
            backoff *= 2
        except Exception:
            raise
    raise RuntimeError(f"LLM rate limit persisted after {max_attempts} attempts") from last_exc


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

    # set API key in env so LiteLLM can pick it up for any provider
    model = run_context.model
    if model.startswith("claude"):
        os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
    elif model.startswith("groq/"):
        os.environ.setdefault("GROQ_API_KEY", api_key)
    elif "/" not in model:
        # bare Gemini model name like "gemini-2.5-flash" — add provider prefix
        os.environ.setdefault("GEMINI_API_KEY", api_key)
        model = f"gemini/{model}"
    else:
        # already has a provider prefix like "gemini/...", "openai/...", etc.
        os.environ.setdefault("GEMINI_API_KEY", api_key)

    # pick skill key from prompt keywords — fall back to "curate" for scRNA
    _prompt_lower = run_context.prompt.lower()
    if "bulkrna" in _prompt_lower or "bulk rna" in _prompt_lower or "bulk_rna" in _prompt_lower or "rnaseq" in _prompt_lower:
        _skill_key = "bulkrna"
    elif "standardize" in _prompt_lower and "append" in _prompt_lower:
        _skill_key = "standardize-append"
    else:
        _skill_key = "curate"

    # retrieve relevant skills upfront and inject into context before the loop
    skill_result = get_lamindb_skill(key=_skill_key)
    skill_content = skill_result.get("skill_content", "")
    if progress_callback is not None:
        if skill_content:
            progress_callback(f"skills loaded: {[r['key'] for r in skill_result.get('results', [])]}")
        else:
            progress_callback(f"no skills found — warnings: {skill_result.get('warnings', [])}")

    # trim skill content to fit within smaller model context windows
    MAX_SKILL_CHARS = 4000
    if len(skill_content) > MAX_SKILL_CHARS:
        skill_content = skill_content[:MAX_SKILL_CHARS] + "\n\n[skill truncated for brevity]"

    system_text = system_instruction
    if skill_content:
        system_text = f"{system_instruction}\n\n## Relevant Skills\n\n{skill_content}"

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": run_context.prompt},
    ]
    tools = _tool_definitions(run_context.mode)

    trace_events: list[dict[str, Any]] = []
    generated_file: str | None = None
    generated_files: list[str] = []
    final_text = ""
    fatal_error: str | None = None
    resolved_runnable_path: str | None = None
    short_circuit_execute = False

    if progress_callback is not None:
        progress_callback(f"mode={run_context.mode} model={model}")
        progress_callback(f"prompt: {run_context.prompt}")

    for step in range(1, max_steps + 1):
        if progress_callback is not None:
            progress_callback(f"step {step}: waiting for model response")

        response_msg = _call_llm(
            model=model,
            messages=_trim_messages(messages),
            tools=tools,
            progress_callback=progress_callback,
        )

        # add assistant message to history
        messages.append(response_msg)

        text_content = _extract_text_from_content(response_msg.content)
        if progress_callback is not None and text_content:
            preview = text_content if len(text_content) <= 300 else f"{text_content[:300]}..."
            progress_callback(f"step {step}: model text: {preview}")

        trace_events.append({"step": step, "model_response": {"content": text_content}})

        tool_calls = response_msg.tool_calls or []
        if not tool_calls:
            final_text = text_content
            if progress_callback is not None:
                progress_callback("model finished without further tool calls")
            break

        for tool_call in tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
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

            # add tool result back to messages in OpenAI format
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

            if progress_callback is not None:
                status = result.get("status", "ok")
                progress_callback(f"step {step}: tool result status={status}")
                if status == "error" and result.get("message"):
                    progress_callback(f"step {step}: tool error: {result['message']}")
                # show execution output so we can follow what the script actually did
                if name == "execute_python":
                    stdout = (result.get("stdout") or "").strip()
                    stderr = (result.get("stderr") or "").strip()
                    exit_code = result.get("exit_code", "?")
                    progress_callback(f"step {step}: exit_code={exit_code}")
                    if stdout:
                        snippet = stdout if len(stdout) <= 600 else stdout[-600:]
                        progress_callback(f"step {step}: stdout: {snippet}")
                    if stderr:
                        snippet = stderr if len(stderr) <= 600 else stderr[-600:]
                        progress_callback(f"step {step}: stderr: {snippet}")
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
        "contents": messages,
        "trace_events": trace_events,
        "generated_file": generated_file,
        "generated_files": generated_files,
        "resolved_runnable_path": resolved_runnable_path,
        "final_text": final_text,
    }


def write_trace_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
