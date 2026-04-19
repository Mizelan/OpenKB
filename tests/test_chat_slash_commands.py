"""Tests for slash commands in the chat REPL."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from prompt_toolkit.styles import Style

from openkb.agent.chat import _handle_slash, _run_add, _run_turn, run_chat
from openkb.frontmatter import parse_fm
from openkb.review import ReviewQueue
from openkb.agent.chat_session import ChatSession
from openkb.agent.executor_runtime import ExecutorRunResult


def _setup_kb(tmp_path: Path) -> Path:
    """Create a minimal KB structure and return kb_dir."""
    kb_dir = tmp_path
    (kb_dir / "raw").mkdir()
    (kb_dir / "wiki" / "sources" / "images").mkdir(parents=True)
    (kb_dir / "wiki" / "summaries").mkdir(parents=True)
    (kb_dir / "wiki" / "concepts").mkdir(parents=True)
    (kb_dir / "wiki" / "explorations").mkdir(parents=True)
    (kb_dir / "wiki" / "queries").mkdir(parents=True)
    (kb_dir / "wiki" / "reports").mkdir(parents=True)
    openkb_dir = kb_dir / ".openkb"
    openkb_dir.mkdir()
    (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
    (openkb_dir / "hashes.json").write_text(json.dumps({}))
    return kb_dir


def _make_session(kb_dir: Path) -> ChatSession:
    return ChatSession.new(kb_dir, "gpt-4o-mini", "en")


_STYLE = Style.from_dict({})


def _collect_fmt():
    """Return (patch, collected) where collected is a list of printed strings."""
    collected: list[str] = []

    def _fake_fmt(_style, *fragments):
        for _cls, text in fragments:
            collected.append(text)

    return patch("openkb.agent.chat._fmt", _fake_fmt), collected


# --- /status and /list use click.echo, captured by capsys ---


@pytest.mark.asyncio
async def test_slash_status(tmp_path, capsys):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    result = await _handle_slash("/status", kb_dir, session, _STYLE)
    assert result is None
    output = capsys.readouterr().out
    assert "Knowledge Base Status" in output


@pytest.mark.asyncio
async def test_slash_list_empty(tmp_path, capsys):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    result = await _handle_slash("/list", kb_dir, session, _STYLE)
    assert result is None
    output = capsys.readouterr().out
    assert "No documents indexed yet" in output


@pytest.mark.asyncio
async def test_slash_list_with_docs(tmp_path, capsys):
    kb_dir = _setup_kb(tmp_path)
    hashes = {"abc": {"name": "paper.pdf", "type": "pdf"}}
    (kb_dir / ".openkb" / "hashes.json").write_text(json.dumps(hashes))
    session = _make_session(kb_dir)
    result = await _handle_slash("/list", kb_dir, session, _STYLE)
    assert result is None
    output = capsys.readouterr().out
    assert "paper.pdf" in output


# --- /add, /exit, /clear, /help, /unknown use _fmt → need patching ---


@pytest.mark.asyncio
async def test_slash_add_missing_arg(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()
    with p:
        result = await _handle_slash("/add", kb_dir, session, _STYLE)
    assert result is None
    assert any("Usage: /add <path>" in s for s in collected)


@pytest.mark.asyncio
async def test_slash_add_nonexistent_path(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()
    with p:
        result = await _handle_slash("/add /no/such/path", kb_dir, session, _STYLE)
    assert result is None
    assert any("Path does not exist" in s for s in collected)


@pytest.mark.asyncio
async def test_slash_add_unsupported_type(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    bad_file = tmp_path / "file.xyz"
    bad_file.write_text("data")
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()
    with p:
        result = await _handle_slash(f"/add {bad_file}", kb_dir, session, _STYLE)
    assert result is None
    assert any("Unsupported file type" in s for s in collected)


@pytest.mark.asyncio
async def test_slash_add_single_file(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    doc = tmp_path / "test.md"
    doc.write_text("# Hello")
    p, _collected = _collect_fmt()
    with p, patch("openkb.cli.add_single_file") as mock_add:
        await _run_add(str(doc), kb_dir, _STYLE)
        mock_add.assert_called_once_with(doc, kb_dir)


@pytest.mark.asyncio
async def test_slash_add_directory_with_progress(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.md").write_text("# A")
    (docs_dir / "b.txt").write_text("B")
    (docs_dir / "skip.xyz").write_text("skip")
    p, collected = _collect_fmt()
    with p, patch("openkb.cli.add_single_file") as mock_add:
        await _run_add(str(docs_dir), kb_dir, _STYLE)
        assert mock_add.call_count == 2
    output = "".join(collected)
    assert "Found 2 supported file(s)" in output
    assert "[1/2]" in output
    assert "[2/2]" in output


@pytest.mark.asyncio
async def test_slash_lint(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    with patch("openkb.cli.run_lint", new_callable=AsyncMock, return_value=tmp_path / "report.md"):
        result = await _handle_slash("/lint", kb_dir, session, _STYLE)
    assert result is None


@pytest.mark.asyncio
async def test_run_chat_handles_ctrl_c_during_slash_command(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)

    class _FakePromptSession:
        def __init__(self) -> None:
            self.calls = 0

        async def prompt_async(self) -> str:
            self.calls += 1
            if self.calls == 1:
                return "/lint"
            raise EOFError

    prompt = _FakePromptSession()
    p, collected = _collect_fmt()
    with (
        p,
        patch("openkb.agent.chat.build_query_agent", return_value=object()),
        patch("openkb.agent.chat._print_header"),
        patch("openkb.agent.chat._make_prompt_session", return_value=prompt),
        patch("openkb.agent.chat._handle_slash", new_callable=AsyncMock, side_effect=KeyboardInterrupt),
    ):
        await run_chat(kb_dir, session, no_color=True)

    assert prompt.calls == 2
    assert any("[aborted]" in s for s in collected)


@pytest.mark.asyncio
async def test_slash_unknown(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()
    with p:
        result = await _handle_slash("/foobar", kb_dir, session, _STYLE)
    assert result is None
    assert any("Unknown command" in s for s in collected)


@pytest.mark.asyncio
async def test_slash_exit(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, _collected = _collect_fmt()
    with p:
        result = await _handle_slash("/exit", kb_dir, session, _STYLE)
    assert result == "exit"


@pytest.mark.asyncio
async def test_slash_clear(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, _collected = _collect_fmt()
    with p:
        result = await _handle_slash("/clear", kb_dir, session, _STYLE)
    assert result == "new_session"


@pytest.mark.asyncio
async def test_run_turn_streams_response_and_persists_history(tmp_path, capsys):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)

    async def fake_run(agent, message, **kwargs):
        kwargs["on_text_delta"]("Hello ")
        kwargs["on_text_delta"]("chat")
        return ExecutorRunResult(
            final_output="Hello chat",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello chat"},
            ],
            turns=1,
        )

    with patch("openkb.agent.executor_runtime.run_executor_agent", side_effect=fake_run):
        await _run_turn(object(), session, "Hi", _STYLE, use_color=False, raw=True)

    assert capsys.readouterr().out == "\nHello chat\n\n"
    assert session.assistant_texts == ["Hello chat"]


@pytest.mark.asyncio
async def test_run_turn_keeps_text_streaming_callback_for_codex_provider(tmp_path, capsys):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)

    class _Agent:
        provider = "codex"

    captured = {}

    async def fake_run(agent, message, **kwargs):
        captured["on_text_delta"] = kwargs.get("on_text_delta")
        return ExecutorRunResult(
            final_output="Hello codex chat",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello codex chat"},
            ],
            turns=1,
        )

    with patch("openkb.agent.executor_runtime.run_executor_agent", side_effect=fake_run):
        await _run_turn(_Agent(), session, "Hi", _STYLE, use_color=False, raw=True)

    assert captured["on_text_delta"] is not None
    assert capsys.readouterr().out == "\nHello codex chat\n\n"
    assert session.assistant_texts == ["Hello codex chat"]


@pytest.mark.asyncio
async def test_slash_save_writes_transcript_page(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    session.record_turn(
        "What is an agent loop?",
        "An agent loop coordinates repeated planning and execution.",
        [
            {"role": "user", "content": "What is an agent loop?"},
            {"role": "assistant", "content": "An agent loop coordinates repeated planning and execution."},
        ],
    )
    p, collected = _collect_fmt()

    with p:
        result = await _handle_slash("/save loop-notes", kb_dir, session, _STYLE)

    assert result is None
    assert any("Saved to" in s for s in collected)
    saved = next((kb_dir / "wiki" / "explorations").glob("loop-notes-*.md"))
    meta, body = parse_fm(saved.read_text(encoding="utf-8"))
    assert meta["session"] == session.id
    assert "What is an agent loop?" in body


@pytest.mark.asyncio
async def test_slash_promote_latest_query_page(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    older = kb_dir / "wiki" / "explorations" / "attention-20260419.md"
    latest = kb_dir / "wiki" / "explorations" / "sessions" / "attention-20260420.md"
    older.write_text(
        (
            "---\n"
            'query: "Older attention?"\n'
            "---\n\n"
            "# Attention\n\n"
            "Older answer.\n"
        ),
        encoding="utf-8",
    )
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(
        (
            "---\n"
            'query: "What is attention?"\n'
            "---\n\n"
            "# Attention\n\n"
            "Saved answer.\n"
        ),
        encoding="utf-8",
    )
    p, collected = _collect_fmt()

    with p:
        result = await _handle_slash("/promote latest query_page", kb_dir, session, _STYLE)

    assert result is None
    assert any("Promoted latest exploration to query page" in s for s in collected)
    promoted = kb_dir / "wiki" / "queries" / "sessions" / "attention-20260420.md"
    assert promoted.exists()


@pytest.mark.asyncio
async def test_slash_promote_latest_concept_seed(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    latest = kb_dir / "wiki" / "explorations" / "context-window-20260419.md"
    latest.write_text("# Context Window\n\nSaved answer.\n", encoding="utf-8")
    p, collected = _collect_fmt()

    with p:
        result = await _handle_slash("/promote latest concept_seed", kb_dir, session, _STYLE)

    assert result is None
    assert any("Queued concept seed from latest exploration" in s for s in collected)
    items = ReviewQueue(kb_dir / ".openkb").list()
    assert len(items) == 1
    assert items[0].payload["path"] == "concepts/context-window-20260419.md"


@pytest.mark.asyncio
async def test_slash_promote_latest_requires_valid_arguments(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()

    with p:
        result = await _handle_slash("/promote", kb_dir, session, _STYLE)

    assert result is None
    assert any("Usage: /promote latest <query_page|concept_seed>" in s for s in collected)


@pytest.mark.asyncio
async def test_slash_promote_latest_rejects_invalid_mode(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()

    with p:
        result = await _handle_slash("/promote latest bad_mode", kb_dir, session, _STYLE)

    assert result is None
    assert any("Mode must be query_page or concept_seed." in s for s in collected)


@pytest.mark.asyncio
async def test_slash_promote_latest_requires_saved_exploration(tmp_path):
    kb_dir = _setup_kb(tmp_path)
    session = _make_session(kb_dir)
    p, collected = _collect_fmt()

    with p:
        result = await _handle_slash("/promote latest query_page", kb_dir, session, _STYLE)

    assert result is None
    assert any("Promotion failed: No saved explorations found." in s for s in collected)
