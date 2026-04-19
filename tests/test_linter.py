"""Tests for openkb.agent.linter (Task 14)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openkb.agent.executor_runtime import ExecutorRunResult
from openkb.agent.linter import build_lint_agent, run_knowledge_lint
from openkb.schema import SCHEMA_MD


class TestBuildLintAgent:
    def test_agent_name(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "gpt-4o-mini")
        assert agent.name == "wiki-linter"

    def test_agent_has_two_tools(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "gpt-4o-mini")
        assert len(agent.tools) == 2

    def test_agent_tool_names(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "gpt-4o-mini")
        names = {t.name for t in agent.tools}
        assert "list_files" in names
        assert "read_file" in names

    def test_schema_in_instructions(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "gpt-4o-mini")
        assert SCHEMA_MD in agent.instructions

    def test_agent_model(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "custom-model")
        assert agent.model == "custom-model"

    def test_instructions_mention_contradictions(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "gpt-4o-mini")
        assert "Contradictions" in agent.instructions or "contradictions" in agent.instructions

    def test_instructions_mention_gaps(self, tmp_path):
        agent = build_lint_agent(str(tmp_path), "gpt-4o-mini")
        assert "Gaps" in agent.instructions or "gaps" in agent.instructions


class TestRunKnowledgeLint:
    @pytest.mark.asyncio
    async def test_returns_final_output(self, tmp_path):
        (tmp_path / "wiki").mkdir()

        with patch("openkb.agent.linter.run_executor_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ExecutorRunResult(
                final_output="## Lint Report\n\nNo issues found.",
                history=[],
                turns=1,
            )
            result = await run_knowledge_lint(tmp_path, "gpt-4o-mini")

        assert "No issues found" in result

    @pytest.mark.asyncio
    async def test_calls_runner_with_correct_agent(self, tmp_path):
        (tmp_path / "wiki").mkdir()

        captured = {}

        async def fake_run(agent, message, **kwargs):
            captured["agent"] = agent
            return ExecutorRunResult(final_output="report", history=[], turns=1)

        with patch("openkb.agent.linter.run_executor_agent", side_effect=fake_run):
            await run_knowledge_lint(tmp_path, "gpt-4o-mini")

        assert captured["agent"].name == "wiki-linter"

    @pytest.mark.asyncio
    async def test_handles_empty_final_output(self, tmp_path):
        (tmp_path / "wiki").mkdir()

        with patch("openkb.agent.linter.run_executor_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ExecutorRunResult(final_output="", history=[], turns=1)
            result = await run_knowledge_lint(tmp_path, "gpt-4o-mini")

        assert "completed" in result.lower() or result != ""
