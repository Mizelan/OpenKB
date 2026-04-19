"""Tests for the executor-only agent runtime."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from openkb.agent.executor_runtime import (
    ExecutorAgent,
    ExecutorRunResult,
    ExecutorTool,
    parse_executor_action,
    run_executor_agent,
)
from openkb.executor import LLMResult


def test_parse_executor_action_accepts_fenced_final_json():
    action = parse_executor_action(
        '```json\n{"type":"final","content":"done"}\n```'
    )
    assert action["type"] == "final"
    assert action["content"] == "done"


def test_parse_executor_action_repairs_malformed_json():
    action = parse_executor_action(
        '{"type":"tool_call","tool":"read_file","args":{"path":"index.md",},}'
    )
    assert action["type"] == "tool_call"
    assert action["tool"] == "read_file"
    assert action["args"]["path"] == "index.md"


@pytest.mark.asyncio
async def test_run_executor_agent_executes_tool_before_final():
    calls: list[tuple[str, dict]] = []

    def read_file(path: str) -> str:
        calls.append(("read_file", {"path": path}))
        return "# Index"

    responses = iter(
        [
            LLMResult(
                text='{"type":"tool_call","reason":"Need the index first.","tool":"read_file","args":{"path":"index.md"}}',
                provider="claude",
                model="sonnet",
            ),
            LLMResult(
                text='{"type":"final","content":"The answer is in the index."}',
                provider="claude",
                model="sonnet",
            ),
        ]
    )

    with patch(
        "openkb.agent.executor_runtime.run_llm_with_system",
        side_effect=lambda system_prompt, user_prompt, cfg: next(responses),
    ):
        result = await run_executor_agent(
            ExecutorAgent(
                name="wiki-query",
                instructions="Use tools.",
                tools=[ExecutorTool(name="read_file", description="Read a file.", handler=read_file)],
                model="sonnet",
            ),
            "What is in the KB?",
        )

    assert isinstance(result, ExecutorRunResult)
    assert result.final_output == "The answer is in the index."
    assert calls == [("read_file", {"path": "index.md"})]
    assert any(item.get("type") == "tool_call" for item in result.history)
    assert any(item.get("type") == "tool_result" for item in result.history)


@pytest.mark.asyncio
async def test_run_executor_agent_records_unknown_tool_error_and_recovers():
    responses = iter(
        [
            LLMResult(
                text='{"type":"tool_call","reason":"Try a missing tool.","tool":"missing_tool","args":{}}',
                provider="claude",
                model="sonnet",
            ),
            LLMResult(
                text='{"type":"final","content":"Recovered after tool error."}',
                provider="claude",
                model="sonnet",
            ),
        ]
    )

    with patch(
        "openkb.agent.executor_runtime.run_llm_with_system",
        side_effect=lambda system_prompt, user_prompt, cfg: next(responses),
    ):
        result = await run_executor_agent(
            ExecutorAgent(
                name="wiki-query",
                instructions="Use tools.",
                tools=[],
                model="sonnet",
            ),
            "Recover from tool failure.",
        )

    assert result.final_output == "Recovered after tool error."
    tool_results = [item for item in result.history if item.get("type") == "tool_result"]
    assert tool_results
    assert "Unknown tool" in tool_results[0]["output"]


@pytest.mark.asyncio
async def test_run_executor_agent_streams_final_content_field_only():
    streamed: list[str] = []

    def fake_stream(system_prompt, user_prompt, cfg, on_text_delta=None):
        assert on_text_delta is not None
        for chunk in ['{"type":"final","content":"Hello ', 'streaming', ' world"}']:
            on_text_delta(chunk)
        return LLMResult(
            text='{"type":"final","content":"Hello streaming world"}',
            provider="claude",
            model="sonnet",
        )

    with patch(
        "openkb.agent.executor_runtime.run_llm_with_system_streaming",
        side_effect=fake_stream,
    ):
        result = await run_executor_agent(
            ExecutorAgent(
                name="wiki-query",
                instructions="Use tools.",
                tools=[],
                model="sonnet",
            ),
            "Stream the answer.",
            on_text_delta=streamed.append,
        )

    assert result.final_output == "Hello streaming world"
    assert "".join(streamed) == "Hello streaming world"


@pytest.mark.asyncio
async def test_run_executor_agent_streams_unicode_escape_across_chunks():
    streamed: list[str] = []

    def fake_stream(system_prompt, user_prompt, cfg, on_text_delta=None):
        assert on_text_delta is not None
        for chunk in ['{"type":"final","content":"\\uD55C', '\\uAE00"}']:
            on_text_delta(chunk)
        return LLMResult(
            text='{"type":"final","content":"한글"}',
            provider="claude",
            model="sonnet",
        )

    with patch(
        "openkb.agent.executor_runtime.run_llm_with_system_streaming",
        side_effect=fake_stream,
    ):
        result = await run_executor_agent(
            ExecutorAgent(
                name="wiki-query",
                instructions="Use tools.",
                tools=[],
                model="sonnet",
            ),
            "Stream unicode.",
            on_text_delta=streamed.append,
        )

    assert result.final_output == "한글"
    assert "".join(streamed) == "한글"
