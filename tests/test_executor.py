"""Tests for openkb.executor — provider inference, ABC executors, stream parsers, batch flow."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from openkb.executor import (
    ExecutorConfig,
    LLMResult,
    infer_provider_from_model,
    build_executor_config,
    normalize_model_for_provider,
    run_llm_with_system_streaming,
    _parse_claude_stream,
    _parse_codex_app_stream,
    ClaudeExecutor,
    CodexExecutor,
    CodexAppExecutor,
    OllamaExecutor,
    run_llm_with_system,
)


# ---------------------------------------------------------------------------
# infer_provider_from_model
# ---------------------------------------------------------------------------

class TestInferProvider:
    @pytest.mark.parametrize("model,expected", [
        # Anthropic / Claude family
        ("anthropic/claude-sonnet-4-6", "claude"),
        ("claude-sonnet-4-6", "claude"),
        ("sonnet", "claude"),
        ("opus", "claude"),
        ("haiku", "claude"),
        # OpenAI / GPT family
        ("openai/gpt-5.4-mini", "codex_app"),
        ("gpt-5.4-mini", "codex_app"),
        ("gpt-5.4", "codex_app"),
        ("gpt-4o", "codex_app"),
        ("o3-mini", "codex_app"),
        ("o4-mini", "codex_app"),
        # Ollama family
        ("ollama/llama3", "ollama"),
        ("llama3", "ollama"),
        ("mistral", "ollama"),
        ("phi3", "ollama"),
        ("qwen2", "ollama"),
        # Gemini
        ("gemini/gemini-3.1-pro-preview", "codex_app"),
    ])
    def test_model_to_provider(self, model, expected):
        assert infer_provider_from_model(model) == expected

    def test_empty_string_defaults_claude(self):
        assert infer_provider_from_model("") == "claude"

    def test_unknown_model_defaults_claude(self):
        assert infer_provider_from_model("some-unknown-model") == "claude"


class TestNormalizeModelForProvider:
    @pytest.mark.parametrize(
        ("provider", "model", "expected"),
        [
            ("claude", "anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"),
            ("codex_app", "openai/gpt-5.4-mini", "gpt-5.4-mini"),
            ("codex", "openai/gpt-5.4-mini", "gpt-5.4-mini"),
            ("ollama", "ollama/llama3", "llama3"),
            ("claude", "sonnet", "sonnet"),
        ],
    )
    def test_normalizes_known_prefixes(self, provider, model, expected):
        assert normalize_model_for_provider(model, provider) == expected

    def test_build_executor_config_prefers_explicit_provider_and_normalizes_model(self):
        cfg = build_executor_config(
            model="openai/gpt-5.4-mini",
            provider="codex_app",
            effort="high",
        )
        assert cfg.provider == "codex_app"
        assert cfg.model == "gpt-5.4-mini"
        assert cfg.effort == "high"


# ---------------------------------------------------------------------------
# ExecutorConfig.effective_model
# ---------------------------------------------------------------------------

class TestExecutorConfigEffectiveModel:
    def test_explicit_model_takes_priority(self):
        cfg = ExecutorConfig(provider="codex_app", model="gpt-5.4")
        assert cfg.effective_model == "gpt-5.4"

    def test_default_model_per_provider(self):
        assert ExecutorConfig(provider="claude").effective_model == "sonnet"
        assert ExecutorConfig(provider="codex_app").effective_model == "gpt-5.4-mini"
        assert ExecutorConfig(provider="codex").effective_model == "gpt-5.4-mini"
        assert ExecutorConfig(provider="ollama").effective_model == "sonnet"


# ---------------------------------------------------------------------------
# Executor ABC — build_args
# ---------------------------------------------------------------------------

class TestClaudeExecutorBuildArgs:
    def test_includes_stream_json(self):
        cfg = ExecutorConfig(provider="claude", model="sonnet")
        ex = ClaudeExecutor(cfg)
        args = ex.build_args("hello")
        assert "--output-format" in args
        assert "stream-json" in args
        assert "-p" in args
        assert "hello" in args

    def test_model_and_effort_flags(self):
        cfg = ExecutorConfig(provider="claude", model="opus", effort="high")
        ex = ClaudeExecutor(cfg)
        args = ex.build_args("test")
        i_model = args.index("--model")
        assert args[i_model + 1] == "opus"
        i_effort = args.index("--effort")
        assert args[i_effort + 1] == "high"


class TestCodexAppExecutorBuildArgs:
    def test_exec_json_subcommand(self):
        cfg = ExecutorConfig(provider="codex_app", model="gpt-5.4-mini")
        ex = CodexAppExecutor(cfg)
        args = ex.build_args("summarize this")
        assert args[0] == "exec"
        assert "--json" in args
        assert "-m" in args
        i_m = args.index("-m")
        assert args[i_m + 1] == "gpt-5.4-mini"
        assert args[-1] == "summarize this"


class TestCodexExecutorBuildArgs:
    def test_codex_uses_exec_json_without_effort_flag(self):
        cfg = ExecutorConfig(provider="codex", model="gpt-5.4-mini")
        ex = CodexExecutor(cfg)
        args = ex.build_args("prompt text")
        assert args[0] == "exec"
        assert "--json" in args
        assert "--ephemeral" in args
        assert "-m" in args
        i_m = args.index("-m")
        assert args[i_m + 1] == "gpt-5.4-mini"
        assert "--effort" not in args
        assert args[-1] == "prompt text"

    def test_codex_maps_effort_to_model_reasoning_effort_config(self):
        cfg = ExecutorConfig(provider="codex", model="gpt-5.4-mini", effort="low")
        ex = CodexExecutor(cfg)
        args = ex.build_args("prompt text")
        i_c = args.index("-c")
        assert args[i_c + 1] == 'model_reasoning_effort="low"'


class TestOllamaExecutor:
    def test_provider_name(self):
        cfg = ExecutorConfig(provider="ollama")
        ex = OllamaExecutor(cfg)
        assert ex.provider_name == "ollama"
        assert ex.binary_name == "claude"

    def test_build_env_sets_ollama_endpoint(self):
        cfg = ExecutorConfig(provider="ollama", ollama_base_url="http://localhost:11434")
        ex = OllamaExecutor(cfg)
        env = ex.build_env()
        assert env is not None
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:11434"
        assert env["ANTHROPIC_API_KEY"] == ""

    def test_build_args_inherits_claude_format(self):
        cfg = ExecutorConfig(provider="ollama", model="llama3")
        ex = OllamaExecutor(cfg)
        args = ex.build_args("test prompt")
        assert "--output-format" in args
        assert "stream-json" in args
        assert "-p" in args

    def test_parse_output_uses_claude_stream(self):
        cfg = ExecutorConfig(provider="ollama", model="llama3")
        ex = OllamaExecutor(cfg)
        event = {"type": "result", "result": "hello", "usage": {"input_tokens": 5, "output_tokens": 3}}
        result = ex.parse_output(json.dumps(event))
        assert result.text == "hello"
        assert result.provider == "claude"  # parser returns claude as provider


# ---------------------------------------------------------------------------
# Stream parsers
# ---------------------------------------------------------------------------

class TestParseClaudeStream:
    def test_result_event(self):
        event = {"type": "result", "result": "Hello world", "usage": {"input_tokens": 10, "output_tokens": 5}}
        stdout = json.dumps(event)
        result = _parse_claude_stream(stdout, "sonnet")
        assert result.text == "Hello world"
        assert result.input_tokens == 10
        assert result.output_tokens == 5

    def test_content_block_delta(self):
        lines = [
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi "}}),
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "there"}}),
        ]
        stdout = "\n".join(lines)
        result = _parse_claude_stream(stdout, "sonnet")
        assert result.text == "Hi there"

    def test_empty_stdout(self):
        result = _parse_claude_stream("", "sonnet")
        assert result.text == ""


class TestParseCodexAppStream:
    def test_item_completed_agent_message(self):
        events = [
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Summary text."}},
            {"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 30}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        result = _parse_codex_app_stream(stdout, "gpt-5.4-mini")
        assert result.text == "Summary text."
        assert result.input_tokens == 50
        assert result.output_tokens == 30
        assert result.provider == "codex_app"

    def test_multiple_items_concatenated(self):
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Part 1. "}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Part 2."}},
            {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 10}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        result = _parse_codex_app_stream(stdout, "gpt-5.4-mini")
        assert result.text == "Part 1. Part 2."

    def test_error_event_no_text(self):
        events = [
            {"type": "error", "message": "Rate limit exceeded"},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        result = _parse_codex_app_stream(stdout, "gpt-5.4-mini")
        assert result.error == "Rate limit exceeded"
        assert result.text == ""

    def test_error_after_text_keeps_text(self):
        events = [
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Good data."}},
            {"type": "turn.failed", "message": "Something went wrong"},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        result = _parse_codex_app_stream(stdout, "gpt-5.4-mini")
        assert result.text == "Good data."
        assert result.error is None

    def test_empty_stdout(self):
        result = _parse_codex_app_stream("", "gpt-5.4-mini")
        assert result.text == ""

    def test_non_agent_message_items_ignored(self):
        events = [
            {"type": "item.completed", "item": {"type": "tool_call", "text": "ignored"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "kept."}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        result = _parse_codex_app_stream(stdout, "gpt-5.4-mini")
        assert result.text == "kept."

    def test_nested_content_blocks_are_extracted(self):
        events = [
            {
                "type": "item.delta",
                "delta": {
                    "content": [
                        {"type": "output_text", "text": "Nested "},
                        {"type": "output_text", "text": "delta"},
                    ]
                },
            },
            {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}},
        ]
        stdout = "\n".join(json.dumps(e) for e in events)
        result = _parse_codex_app_stream(stdout, "gpt-5.4-mini")
        assert result.text == "Nested delta"


class TestStreamingExecutors:
    def test_claude_streaming_emits_text_delta_and_final_result(self):
        streamed: list[str] = []
        lines = [
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello "}}) + "\n",
            json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Claude"}}) + "\n",
            json.dumps({"type": "result", "usage": {"input_tokens": 4, "output_tokens": 2}}) + "\n",
        ]

        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0

        with patch("openkb.executor.subprocess.Popen", return_value=mock_proc):
            result = run_llm_with_system_streaming(
                "system",
                "user",
                ExecutorConfig(provider="claude", model="sonnet"),
                streamed.append,
            )

        assert "".join(streamed) == "Hello Claude"
        assert result.text == "Hello Claude"

    def test_codex_app_streaming_emits_item_delta_and_final_result(self):
        streamed: list[str] = []
        lines = [
            json.dumps({"type": "item.delta", "delta": {"text": "Hello "}}) + "\n",
            json.dumps({"type": "item.delta", "delta": {"text": "Codex"}}) + "\n",
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}) + "\n",
        ]

        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read.return_value = ""
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0

        with patch("openkb.executor.subprocess.Popen", return_value=mock_proc):
            result = run_llm_with_system_streaming(
                "system",
                "user",
                ExecutorConfig(provider="codex_app", model="gpt-5.4-mini"),
                streamed.append,
            )

        assert "".join(streamed) == "Hello Codex"
        assert result.text == "Hello Codex"

    def test_codex_app_streaming_handles_nested_content_blocks(self):
        streamed: list[str] = []
        lines = [
            json.dumps(
                {
                    "type": "item.delta",
                    "delta": {
                        "content": [
                            {"type": "output_text", "text": "Nested "},
                            {"type": "output_text", "text": "stream"},
                        ]
                    },
                }
            ) + "\n",
            json.dumps({"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}}) + "\n",
        ]

        mock_proc = MagicMock()
        mock_proc.stdout = iter(lines)
        mock_proc.stderr = MagicMock()
        mock_proc.communicate.return_value = ("", "")
        mock_proc.returncode = 0

        with patch("openkb.executor.subprocess.Popen", return_value=mock_proc):
            result = run_llm_with_system_streaming(
                "system",
                "user",
                ExecutorConfig(provider="codex_app", model="gpt-5.4-mini"),
                streamed.append,
            )

        assert "".join(streamed) == "Nested stream"
        assert result.text == "Nested stream"

    def test_codex_streaming_uses_buffered_fallback_and_emits_final_text(self):
        streamed: list[str] = []
        events = [
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": '{"type":"final","content":"Hello Codex"}'}},
            {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 2}},
        ]
        mock_stdout = "\n".join(json.dumps(e) for e in events)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = mock_stdout
        mock_proc.stderr = ""

        with patch("openkb.executor.subprocess.run", return_value=mock_proc):
            result = run_llm_with_system_streaming(
                "system",
                "user",
                ExecutorConfig(provider="codex", model="gpt-5.4-mini"),
                streamed.append,
            )

        assert "".join(streamed) == '{"type":"final","content":"Hello Codex"}'
        assert result.text == '{"type":"final","content":"Hello Codex"}'


# ---------------------------------------------------------------------------
# _llm_call integration: provider auto-detection
# ---------------------------------------------------------------------------

class TestLlmCallProviderDetection:
    def test_gpt_model_uses_codex_app(self):
        from unittest.mock import patch
        from openkb.agent.compiler import _llm_call

        captured_cfg = {}

        def fake_run(system_prompt, user_prompt, cfg):
            captured_cfg.update(provider=cfg.provider, model=cfg.model)
            return LLMResult(text="ok", provider=cfg.provider, model=cfg.model)

        with patch("openkb.executor.run_llm_with_system", side_effect=fake_run):
            _llm_call("gpt-5.4-mini", [{"role": "user", "content": "test"}], "step")

        assert captured_cfg["provider"] == "codex_app"
        assert captured_cfg["model"] == "gpt-5.4-mini"

    def test_claude_model_uses_claude(self):
        from unittest.mock import patch
        from openkb.agent.compiler import _llm_call

        captured_cfg = {}

        def fake_run(system_prompt, user_prompt, cfg):
            captured_cfg.update(provider=cfg.provider, model=cfg.model)
            return LLMResult(text="ok", provider=cfg.provider, model=cfg.model)

        with patch("openkb.executor.run_llm_with_system", side_effect=fake_run):
            _llm_call("anthropic/claude-sonnet-4-6", [{"role": "user", "content": "test"}], "step")

        assert captured_cfg["provider"] == "claude"
        assert captured_cfg["model"] == "claude-sonnet-4-6"

    def test_explicit_config_overrides_inference(self):
        from unittest.mock import patch
        from openkb.agent.compiler import _llm_call

        explicit_cfg = ExecutorConfig(provider="claude", model="sonnet")
        captured_cfg = {}

        def fake_run(system_prompt, user_prompt, cfg):
            captured_cfg.update(provider=cfg.provider, model=cfg.model)
            return LLMResult(text="ok", provider=cfg.provider, model=cfg.model)

        with patch("openkb.executor.run_llm_with_system", side_effect=fake_run):
            _llm_call("gpt-5.4-mini", [{"role": "user", "content": "test"}], "step",
                       executor_config=explicit_cfg)

        assert captured_cfg["provider"] == "claude"
        assert captured_cfg["model"] == "sonnet"


# ---------------------------------------------------------------------------
# CodexAppExecutor.run with mock subprocess
# ---------------------------------------------------------------------------

class TestCodexAppExecutorRun:
    def test_mock_subprocess_returns_parsed_result(self):
        """CodexAppExecutor.run → mock subprocess → _parse_codex_app_stream."""
        events = [
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Generated summary."}},
            {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]
        mock_stdout = "\n".join(json.dumps(e) for e in events)

        cfg = ExecutorConfig(provider="codex_app", model="gpt-5.4-mini")
        ex = CodexAppExecutor(cfg)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = mock_stdout

        with patch("openkb.executor.subprocess.run", return_value=mock_proc):
            result = ex.run("write a summary")

        assert result.text == "Generated summary."
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.provider == "codex_app"
        assert result.error is None

    def test_mock_subprocess_error(self):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stderr = "model not found"

        cfg = ExecutorConfig(provider="codex_app", model="gpt-5.4-mini")
        ex = CodexAppExecutor(cfg)

        with patch("openkb.executor.subprocess.run", return_value=mock_proc):
            result = ex.run("test")

        assert result.error is not None
        assert "Exit code 1" in result.error


class TestCodexExecutorRun:
    def test_mock_subprocess_returns_parsed_result(self):
        events = [
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "Generated answer."}},
            {"type": "turn.completed", "usage": {"input_tokens": 80, "output_tokens": 20}},
        ]
        mock_stdout = "\n".join(json.dumps(e) for e in events)

        cfg = ExecutorConfig(provider="codex", model="gpt-5.4-mini")
        ex = CodexExecutor(cfg)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = mock_stdout

        with patch("openkb.executor.subprocess.run", return_value=mock_proc):
            result = ex.run("write an answer")

        assert result.text == "Generated answer."
        assert result.input_tokens == 80
        assert result.output_tokens == 20
        assert result.provider == "codex"
        assert result.error is None


# ---------------------------------------------------------------------------
# Batch compile with codex_app — integration
# ---------------------------------------------------------------------------

def _make_codex_app_response(text: str, in_tok: int = 50, out_tok: int = 20) -> LLMResult:
    """Build a realistic CodexAppExecutor LLMResult."""
    return LLMResult(text=text, provider="codex_app", model="gpt-5.4-mini",
                     input_tokens=in_tok, output_tokens=out_tok,
                     total_tokens=in_tok + out_tok)


def _setup_kb(tmp_path: Path, model: str = "gpt-5.4-mini") -> Path:
    """Create a minimal KB directory structure for compile tests."""
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "summaries").mkdir(parents=True)
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "index.md").write_text(
        "# Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
        encoding="utf-8",
    )
    (tmp_path / ".openkb").mkdir()
    return wiki


class TestBatchCompileWithCodexApp:
    """Verify batch compilation (multiple docs) routes through codex_app."""

    @pytest.mark.asyncio
    async def test_single_doc_gpt_model_routes_codex_app(self, tmp_path):
        from openkb.agent.compiler import compile_short_doc

        wiki = _setup_kb(tmp_path)
        source_path = wiki / "sources" / "doc1.md"
        source_path.write_text("# Doc1\n\nContent about neural networks.", encoding="utf-8")

        providers_seen: list[str] = []

        def fake_run(system_prompt, user_prompt, cfg):
            providers_seen.append(cfg.provider)
            # Return step-appropriate JSON
            step = _infer_step(system_prompt, user_prompt)
            return _make_codex_app_response(step)

        with patch("openkb.executor.run_llm_with_system", side_effect=fake_run):
            await compile_short_doc("doc1", source_path, tmp_path, "gpt-5.4-mini")

        assert all(p == "codex_app" for p in providers_seen), f"Expected all codex_app, got {providers_seen}"
        assert (wiki / "summaries" / "doc1.md").exists()
        assert (wiki / "concepts").exists()

    @pytest.mark.asyncio
    async def test_two_docs_batch_both_use_codex_app(self, tmp_path):
        """Simulate batch: add two docs sequentially, both route to codex_app."""
        from openkb.agent.compiler import compile_short_doc

        wiki = _setup_kb(tmp_path)

        # Create two source docs
        for name in ("doc1", "doc2"):
            (wiki / "sources" / f"{name}.md").write_text(
                f"# {name}\n\nContent about {name}.", encoding="utf-8"
            )

        providers_seen: list[str] = []

        def fake_run(system_prompt, user_prompt, cfg):
            providers_seen.append(cfg.provider)
            step = _infer_step(system_prompt, user_prompt)
            return _make_codex_app_response(step)

        with patch("openkb.executor.run_llm_with_system", side_effect=fake_run):
            for name in ("doc1", "doc2"):
                src = wiki / "sources" / f"{name}.md"
                await compile_short_doc(name, src, tmp_path, "gpt-5.4-mini")

        assert all(p == "codex_app" for p in providers_seen), f"Expected all codex_app, got {providers_seen}"
        assert len(providers_seen) >= 4  # analysis + summary + plan + concept per doc
        # Both summaries should exist
        assert (wiki / "summaries" / "doc1.md").exists()
        assert (wiki / "summaries" / "doc2.md").exists()

    @pytest.mark.asyncio
    async def test_codex_app_output_parsing_in_pipeline(self, tmp_path):
        """Verify codex_app JSON output is correctly parsed by the pipeline."""
        from openkb.agent.compiler import compile_short_doc

        wiki = _setup_kb(tmp_path)
        source_path = wiki / "sources" / "doc1.md"
        source_path.write_text("# Doc1\n\nContent.", encoding="utf-8")

        # Simulate real codex_app JSON responses (already parsed to LLMResult)
        analysis_json = json.dumps({"entities": [{"name": "AI", "type": "technology"}], "concept_actions": [], "review_items": []})
        summary_json = json.dumps({"brief": "About AI", "content": "# Summary\n\nArtificial intelligence overview."})
        plan_json = json.dumps({"create": [], "update": [], "related": []})

        step_responses = {
            "analysis": _make_codex_app_response(analysis_json),
            "summary": _make_codex_app_response(summary_json),
            "concepts-plan": _make_codex_app_response(plan_json),
        }

        def fake_run(system_prompt, user_prompt, cfg):
            step = _infer_step(system_prompt, user_prompt)
            return step_responses.get(step, _make_codex_app_response("{}"))

        with patch("openkb.executor.run_llm_with_system", side_effect=fake_run):
            await compile_short_doc("doc1", source_path, tmp_path, "gpt-5.4-mini")

        summary_text = (wiki / "summaries" / "doc1.md").read_text()
        assert "AI" in summary_text or "About" in summary_text or "artificial" in summary_text.lower()


def _infer_step(system_prompt: str, user_prompt: str) -> str:
    """Heuristic to determine which compile step a call corresponds to."""
    if "concept_actions" in user_prompt or "entities" in user_prompt:
        return "analysis"
    if "brief" in user_prompt and "content" in user_prompt and "concept" not in user_prompt.lower():
        return "summary"
    if "create" in user_prompt and "update" in user_prompt:
        return "concepts-plan"
    return "concept-page"
