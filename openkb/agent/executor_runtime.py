"""Executor-only tool-loop runtime for OpenKB agents."""
from __future__ import annotations

import asyncio
import codecs
import json
import re
from dataclasses import dataclass, field
from inspect import signature
from typing import Any, Callable

from json_repair import repair_json

from openkb.executor import build_executor_config, run_llm_with_system, run_llm_with_system_streaming
from openkb.json_utils import extract_json


ToolHandler = Callable[..., Any]


@dataclass
class ExecutorTool:
    """A local tool exposed to the executor runtime."""

    name: str
    description: str
    handler: ToolHandler
    arguments: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.arguments:
            self.arguments = tuple(signature(self.handler).parameters)


@dataclass
class ExecutorAgent:
    """Provider-agnostic agent description for the executor runtime."""

    name: str
    instructions: str
    tools: list[ExecutorTool]
    model: str
    provider: str = ""
    effort: str = "medium"
    working_dir: str = ""
    timeout: int = 300
    max_turns: int = 50


@dataclass
class ExecutorRunResult:
    """Result of an executor-driven tool loop."""

    final_output: str
    history: list[dict[str, Any]]
    turns: int


def _format_tools(tools: list[ExecutorTool]) -> str:
    if not tools:
        return "- No tools available."

    lines: list[str] = []
    for tool in tools:
        args = ", ".join(tool.arguments)
        sig = f"{tool.name}({args})" if args else f"{tool.name}()"
        lines.append(f"- {sig}: {tool.description}")
    return "\n".join(lines)


def _format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(empty)"

    rendered: list[str] = []
    for item in history:
        role = item.get("role")
        item_type = item.get("type")
        if role in {"user", "assistant"}:
            rendered.append(f"[{role.upper()}]\n{item.get('content', '')}")
            continue
        if item_type == "tool_call":
            rendered.append(
                "[TOOL_CALL]\n"
                f"name: {item.get('name', '')}\n"
                f"reason: {item.get('reason', '')}\n"
                f"arguments: {json.dumps(item.get('arguments', {}), ensure_ascii=False, sort_keys=True)}"
            )
            continue
        if item_type == "tool_result":
            rendered.append(
                "[TOOL_RESULT]\n"
                f"name: {item.get('name', '')}\n"
                f"output:\n{item.get('output', '')}"
            )
            continue
        rendered.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
    return "\n\n".join(rendered)


def _build_user_prompt(agent: ExecutorAgent, history: list[dict[str, Any]]) -> str:
    tool_section = _format_tools(agent.tools)
    history_section = _format_history(history)
    return (
        "Return exactly one JSON object and nothing else.\n\n"
        "Valid actions:\n"
        '1. {"type":"tool_call","reason":"short reason","tool":"tool_name","args":{...}}\n'
        '2. {"type":"final","content":"final answer"}\n\n'
        "Rules:\n"
        "- Use only the listed tools.\n"
        "- Keep `reason` to one short sentence.\n"
        "- If a tool result is insufficient, call another tool.\n"
        "- When you have enough evidence, return `final`.\n"
        "- Do not emit markdown fences or prose outside the JSON object.\n\n"
        f"Available tools:\n{tool_section}\n\n"
        f"Conversation and tool transcript:\n{history_section}\n"
    )


def parse_executor_action(text: str) -> dict[str, Any]:
    """Parse a JSON action from executor output, repairing malformed JSON when needed."""
    action = extract_json(text)
    if action is None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_nl = cleaned.find("\n")
            cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        action = json.loads(repair_json(cleaned))

    if not isinstance(action, dict):
        raise ValueError("Executor action must be a JSON object.")

    action_type = action.get("type")
    if action_type not in {"tool_call", "final"}:
        raise ValueError(f"Unsupported executor action type: {action_type!r}")

    if action_type == "tool_call":
        if not isinstance(action.get("tool"), str) or not action["tool"]:
            raise ValueError("tool_call action requires a tool name.")
        args = action.get("args", {})
        if args is None:
            action["args"] = {}
        elif not isinstance(args, dict):
            raise ValueError("tool_call args must be an object.")
        action.setdefault("reason", "")
    else:
        if not isinstance(action.get("content"), str):
            raise ValueError("final action requires string content.")

    return action


def _tool_result_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _truncate_tool_output(text: str, limit: int = 12000) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit] + f"\n\n[truncated {omitted} chars]"


class _JSONContentStreamExtractor:
    """Extract a specific JSON string field incrementally from streamed text."""

    def __init__(self, field_name: str, on_text_delta: Callable[[str], None]) -> None:
        self.field_name = field_name
        self.on_text_delta = on_text_delta
        self._search_buffer = ""
        self._in_field = False
        self._escaped = False
        self._closed = False
        self._unicode_buffer = ""

    def feed(self, chunk: str) -> None:
        if self._closed or not chunk:
            return

        if not self._in_field:
            self._search_buffer += chunk
            pattern = rf'"{re.escape(self.field_name)}"\s*:\s*"'
            match = re.search(pattern, self._search_buffer)
            if not match:
                self._search_buffer = self._search_buffer[-64:]
                return
            self._in_field = True
            chunk = self._search_buffer[match.end():]
            self._search_buffer = ""

        emitted: list[str] = []
        index = 0
        while index < len(chunk):
            char = chunk[index]
            if self._escaped:
                if char == "u":
                    self._unicode_buffer = "\\u"
                    self._escaped = False
                    index += 1
                    continue
                emitted.append({
                    "n": "\n",
                    "r": "\r",
                    "t": "\t",
                    '"': '"',
                    "\\": "\\",
                    "/": "/",
                }.get(char, char))
                self._escaped = False
                index += 1
                continue
            if self._unicode_buffer:
                self._unicode_buffer += char
                index += 1
                if len(self._unicode_buffer) < 6:
                    continue
                try:
                    emitted.append(codecs.decode(self._unicode_buffer, "unicode_escape"))
                except Exception:
                    emitted.append(self._unicode_buffer)
                self._unicode_buffer = ""
                continue
            if char == "\\":
                self._escaped = True
                index += 1
                continue
            if char == '"':
                self._closed = True
                break
            emitted.append(char)
            index += 1

        text = "".join(emitted)
        if text:
            self.on_text_delta(text)


async def run_executor_agent(
    agent: ExecutorAgent,
    message: str,
    *,
    history: list[dict[str, Any]] | None = None,
    on_tool_call: Callable[[str, dict[str, Any], str], None] | None = None,
    on_text_delta: Callable[[str], None] | None = None,
) -> ExecutorRunResult:
    """Run the executor loop until the agent returns a final answer."""
    transcript = list(history or [])
    transcript.append({"role": "user", "content": message})

    tools_by_name = {tool.name: tool for tool in agent.tools}

    for turn_index in range(1, agent.max_turns + 1):
        prompt = _build_user_prompt(agent, transcript)
        cfg = build_executor_config(
            model=agent.model,
            provider=agent.provider,
            effort=agent.effort,
            working_dir=agent.working_dir,
            timeout=agent.timeout,
        )
        if on_text_delta is not None:
            extractor = _JSONContentStreamExtractor("content", on_text_delta)
            result = await asyncio.to_thread(
                run_llm_with_system_streaming,
                agent.instructions,
                prompt,
                cfg,
                extractor.feed,
            )
        else:
            result = await asyncio.to_thread(
                run_llm_with_system,
                agent.instructions,
                prompt,
                cfg,
            )
        if result.error:
            raise RuntimeError(result.error)

        action = parse_executor_action(result.text)
        if action["type"] == "final":
            final_output = action["content"].strip()
            transcript.append({"role": "assistant", "content": final_output})
            return ExecutorRunResult(
                final_output=final_output,
                history=transcript,
                turns=turn_index,
            )

        tool_name = action["tool"]
        args = action.get("args", {})
        reason = action.get("reason", "")
        transcript.append(
            {
                "type": "tool_call",
                "name": tool_name,
                "arguments": args,
                "reason": reason,
            }
        )
        if on_tool_call is not None:
            on_tool_call(tool_name, args, reason)

        tool = tools_by_name.get(tool_name)
        if tool is None:
            output = f"Unknown tool: {tool_name}"
        else:
            try:
                output = _tool_result_to_text(tool.handler(**args))
            except Exception as exc:  # pragma: no cover - defensive path
                output = f"Tool {tool_name} failed: {exc}"

        transcript.append(
            {
                "type": "tool_result",
                "name": tool_name,
                "output": _truncate_tool_output(output),
            }
        )

    raise RuntimeError(f"Executor agent exceeded max turns ({agent.max_turns}).")
