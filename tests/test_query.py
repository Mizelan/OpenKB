"""Tests for openkb.agent.query (Task 11)."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from openkb.agent.query import build_query_agent, run_query
from openkb.agent.executor_runtime import ExecutorRunResult
from openkb.frontmatter import parse_fm
from openkb.promotion import latest_exploration_path, promote_exploration
from openkb.review import ReviewQueue
from openkb.schema import SCHEMA_MD


class TestBuildQueryAgent:
    def test_agent_name(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        assert agent.name == "wiki-query"

    def test_agent_has_four_tools(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        assert len(agent.tools) == 4

    def test_agent_tool_names(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        names = {t.name for t in agent.tools}
        assert "read_file" in names
        assert "get_page_content" in names
        assert "get_image" in names
        assert "search_related" in names

    def test_instructions_mention_get_page_content(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        assert "get_page_content" in agent.instructions
        assert "pageindex_retrieve" not in agent.instructions

    def test_instructions_mention_search_related(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        assert "search_related" in agent.instructions


class TestSearchRelatedTool:
    def test_agent_has_four_tools(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        assert len(agent.tools) == 4

    def test_agent_tool_names_include_search_related(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        names = {t.name for t in agent.tools}
        assert "search_related" in names

    def test_schema_in_instructions(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "gpt-4o-mini")
        assert SCHEMA_MD in agent.instructions

    def test_agent_model(self, tmp_path):
        agent = build_query_agent(str(tmp_path), "my-model")
        assert agent.model == "my-model"


class TestRunQuery:
    @pytest.mark.asyncio
    async def test_run_query_returns_final_output(self, tmp_path):
        (tmp_path / "wiki").mkdir()
        (tmp_path / ".openkb").mkdir()

        with patch("openkb.agent.query.run_executor_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = ExecutorRunResult(
                final_output="The answer is 42.",
                history=[],
                turns=1,
            )
            answer = await run_query("What is the answer?", tmp_path, "gpt-4o-mini")

        assert answer == "The answer is 42."

    @pytest.mark.asyncio
    async def test_run_query_passes_question_to_agent(self, tmp_path):
        (tmp_path / "wiki").mkdir()
        (tmp_path / ".openkb").mkdir()

        captured = {}

        async def fake_run(agent, message, **kwargs):
            captured["message"] = message
            return ExecutorRunResult(final_output="answer", history=[], turns=1)

        with patch("openkb.agent.query.run_executor_agent", side_effect=fake_run):
            await run_query("How does attention work?", tmp_path, "gpt-4o-mini")

        assert "How does attention work?" in captured["message"]

    @pytest.mark.asyncio
    async def test_run_query_stream_prints_streamed_text_without_duplicate_final_output(self, tmp_path, capsys):
        (tmp_path / "wiki").mkdir()
        (tmp_path / ".openkb").mkdir()

        async def fake_run(agent, message, **kwargs):
            on_text_delta = kwargs["on_text_delta"]
            on_text_delta("Hello ")
            on_text_delta("world")
            return ExecutorRunResult(final_output="Hello world", history=[], turns=1)

        with patch("openkb.agent.query.run_executor_agent", side_effect=fake_run):
            answer = await run_query("Stream this.", tmp_path, "gpt-4o-mini", stream=True, raw=True)

        assert answer == "Hello world"
        assert capsys.readouterr().out == "Hello world\n"

    @pytest.mark.asyncio
    async def test_run_query_keeps_stream_callback_for_codex_provider(self, tmp_path):
        (tmp_path / "wiki").mkdir()
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text(
            "provider: codex\nmodel: gpt-5.4-mini\neffort: low\n",
            encoding="utf-8",
        )

        captured = {}

        async def fake_run(agent, message, **kwargs):
            captured["on_tool_call"] = kwargs.get("on_tool_call")
            captured["on_text_delta"] = kwargs.get("on_text_delta")
            return ExecutorRunResult(final_output="Done.", history=[], turns=1)

        with patch("openkb.agent.query.run_executor_agent", side_effect=fake_run):
            answer = await run_query("Use codex.", tmp_path, "gpt-5.4-mini", stream=True, raw=True)

        assert answer == "Done."
        assert captured["on_tool_call"] is not None
        assert captured["on_text_delta"] is not None


class TestExplorationPromotion:
    def _make_kb(self, tmp_path: Path) -> Path:
        kb_dir = tmp_path
        (kb_dir / "wiki" / "explorations").mkdir(parents=True)
        (kb_dir / "wiki" / "queries").mkdir(parents=True)
        (kb_dir / "wiki" / "concepts").mkdir(parents=True)
        (kb_dir / ".openkb").mkdir()
        return kb_dir

    def test_promotes_query_save_exploration_to_query_page(self, tmp_path):
        kb_dir = self._make_kb(tmp_path)
        exploration = kb_dir / "wiki" / "explorations" / "attention.md"
        exploration.write_text(
            (
                "---\n"
                'query: "How does attention work?"\n'
                "---\n\n"
                "# Attention\n\n"
                "Answer body.\n"
            ),
            encoding="utf-8",
        )

        result = promote_exploration(kb_dir, "wiki/explorations/attention.md", mode="query_page")

        assert result["mode"] == "query_page"
        assert result["status"] == "created"
        target = kb_dir / result["target_path"]
        assert target.exists()
        meta, body = parse_fm(target.read_text(encoding="utf-8"))
        assert meta["type"] == "wiki-query"
        assert meta["query"] == "How does attention work?"
        assert meta["promoted_from"] == "wiki/explorations/attention.md"
        assert "# Attention" in body
        assert "Answer body." in body

    def test_query_page_refresh_returns_updated_status(self, tmp_path):
        kb_dir = self._make_kb(tmp_path)
        exploration = kb_dir / "wiki" / "explorations" / "attention.md"
        target = kb_dir / "wiki" / "queries" / "attention.md"
        exploration.write_text(
            (
                "---\n"
                'query: "Updated question"\n'
                "---\n\n"
                "# Attention\n\n"
                "New answer body.\n"
            ),
            encoding="utf-8",
        )
        target.write_text("# Old Query\n", encoding="utf-8")

        result = promote_exploration(kb_dir, "wiki/explorations/attention.md", mode="query_page")

        assert result["status"] == "updated"
        meta, body = parse_fm(target.read_text(encoding="utf-8"))
        assert meta["query"] == "Updated question"
        assert "New answer body." in body

    def test_promotes_chat_transcript_to_actionable_concept_seed(self, tmp_path):
        kb_dir = self._make_kb(tmp_path)
        exploration = kb_dir / "wiki" / "explorations" / "agent-loop-20260419.md"
        exploration.write_text(
            (
                "---\n"
                'session: "sess-1"\n'
                "---\n\n"
                "# Chat transcript  agent-loop\n\n"
                "## [1] Tell me about agent loops\n\n"
                "Agent loops help orchestrate work.\n"
            ),
            encoding="utf-8",
        )

        result = promote_exploration(kb_dir, "explorations/agent-loop-20260419.md", mode="concept_seed")

        assert result["mode"] == "concept_seed"
        assert result["status"] == "queued"
        items = ReviewQueue(kb_dir / ".openkb").list()
        assert len(items) == 1
        item = items[0]
        assert item.action_type == "create_placeholder"
        assert item.payload["path"] == "concepts/agent-loop-20260419.md"
        assert item.source_path == "wiki/explorations/agent-loop-20260419.md"

    @pytest.mark.parametrize(
        ("path", "mode", "error_text"),
        [
            ("", "query_page", "Exploration path is required."),
            ("wiki/concepts/not-allowed.md", "query_page", "Promotion source must live under wiki/explorations/."),
            ("wiki/explorations/missing.md", "query_page", "Exploration not found"),
            ("wiki/explorations/attention.md", "bad_mode", "Unsupported promotion mode"),
        ],
    )
    def test_rejects_invalid_promotion_inputs(self, tmp_path, path, mode, error_text):
        kb_dir = self._make_kb(tmp_path)
        exploration = kb_dir / "wiki" / "explorations" / "attention.md"
        exploration.write_text("# Attention\n", encoding="utf-8")
        (kb_dir / "wiki" / "concepts" / "not-allowed.md").write_text("# Not Allowed\n", encoding="utf-8")

        with pytest.raises((FileNotFoundError, ValueError), match=error_text):
            promote_exploration(kb_dir, path, mode=mode)

    def test_latest_exploration_path_finds_newest_nested_file(self, tmp_path):
        kb_dir = self._make_kb(tmp_path)
        older = kb_dir / "wiki" / "explorations" / "attention.md"
        newer = kb_dir / "wiki" / "explorations" / "sessions" / "attention.md"
        older.write_text("# Older\n", encoding="utf-8")
        newer.parent.mkdir(parents=True, exist_ok=True)
        newer.write_text("# Newer\n", encoding="utf-8")
        os.utime(older, (1, 1))
        os.utime(newer, (2, 2))

        latest = latest_exploration_path(kb_dir)

        assert latest == "wiki/explorations/sessions/attention.md"

    def test_nested_query_page_promotion_preserves_relative_path(self, tmp_path):
        kb_dir = self._make_kb(tmp_path)
        first = kb_dir / "wiki" / "explorations" / "alpha" / "attention.md"
        second = kb_dir / "wiki" / "explorations" / "beta" / "attention.md"
        first.parent.mkdir(parents=True, exist_ok=True)
        second.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("# Alpha Attention\n", encoding="utf-8")
        second.write_text("# Beta Attention\n", encoding="utf-8")

        first_result = promote_exploration(kb_dir, "wiki/explorations/alpha/attention.md", mode="query_page")
        second_result = promote_exploration(kb_dir, "wiki/explorations/beta/attention.md", mode="query_page")

        assert first_result["target_path"] == "wiki/queries/alpha/attention.md"
        assert second_result["target_path"] == "wiki/queries/beta/attention.md"
        assert (kb_dir / "wiki" / "queries" / "alpha" / "attention.md").exists()
        assert (kb_dir / "wiki" / "queries" / "beta" / "attention.md").exists()

    def test_nested_concept_seed_promotion_preserves_relative_path(self, tmp_path):
        kb_dir = self._make_kb(tmp_path)
        first = kb_dir / "wiki" / "explorations" / "alpha" / "attention.md"
        second = kb_dir / "wiki" / "explorations" / "beta" / "attention.md"
        first.parent.mkdir(parents=True, exist_ok=True)
        second.parent.mkdir(parents=True, exist_ok=True)
        first.write_text("# Alpha Attention\n", encoding="utf-8")
        second.write_text("# Beta Attention\n", encoding="utf-8")

        first_result = promote_exploration(kb_dir, "wiki/explorations/alpha/attention.md", mode="concept_seed")
        second_result = promote_exploration(kb_dir, "wiki/explorations/beta/attention.md", mode="concept_seed")

        assert first_result["target_path"] == "concepts/alpha/attention.md"
        assert second_result["target_path"] == "concepts/beta/attention.md"
        items = ReviewQueue(kb_dir / ".openkb").list()
        assert [item.payload["path"] for item in items] == [
            "concepts/alpha/attention.md",
            "concepts/beta/attention.md",
        ]
