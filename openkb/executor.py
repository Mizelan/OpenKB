"""Subprocess-based LLM executor for OpenKB.

Calls claude, codex, or ollama CLI as subprocesses — no API keys needed.
Supports streaming JSON output from claude CLI and codex exec --json.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

logger = logging.getLogger(__name__)

TextDeltaCallback = Callable[[str], None]


@dataclass
class LLMResult:
    """Result from an LLM subprocess call."""
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class ExecutorConfig:
    """Configuration for the LLM executor."""
    provider: str = "claude"       # claude | codex | codex_app | ollama
    model: str = ""                # empty = provider default
    effort: str = "medium"         # low | medium | high
    working_dir: str = ""          # working directory for subprocess
    timeout: int = 300             # seconds
    # Ollama-specific
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_auth_token: str = "ollama"

    @property
    def effective_model(self) -> str:
        if self.model:
            return self.model
        defaults = {
            "claude": "sonnet",
            "codex": "gpt-5.4-mini",
            "codex_app": "gpt-5.4-mini",
            "ollama": "sonnet",
        }
        return defaults.get(self.provider, "sonnet")


def infer_provider_from_model(model: str) -> str:
    """Infer the executor provider from a LiteLLM-style model string.

    Recognised prefixes / patterns:
      - ``anthropic/`` or starts with ``claude``  → ``claude``
      - ``openai/`` or starts with ``gpt``, ``o3``, ``o4`` → ``codex_app``
      - ``ollama/`` → ``ollama``
      - bare name matching known model families → best-guess provider
      - fallback → ``claude``
    """
    if not model:
        return "claude"

    m = model.lower()

    if m.startswith("anthropic/"):
        return "claude"
    if m.startswith("openai/"):
        return "codex_app"
    if m.startswith("ollama/"):
        return "ollama"
    if m.startswith("gemini/"):
        return "codex_app"

    if m.startswith(("claude", "sonnet", "opus", "haiku")):
        return "claude"
    if m.startswith(("gpt", "o3", "o4", "chatgpt")):
        return "codex_app"
    if m.startswith(("llama", "mistral", "phi", "qwen", "gemma", "deepseek", "codellama")):
        return "ollama"

    return "claude"


def normalize_model_for_provider(model: str, provider: str) -> str:
    """Normalize legacy provider-prefixed model strings for executor CLIs.

    Config may keep LiteLLM-style values such as ``anthropic/claude-sonnet-4-6``
    for backward compatibility. Executor CLIs should receive the native model
    token they expect, so prefixes are stripped only at invocation time.
    """
    if not model:
        return model

    normalized = model.strip()
    provider_prefixes = {
        "claude": ("anthropic/",),
        "codex": ("openai/", "gemini/"),
        "codex_app": ("openai/", "gemini/"),
        "ollama": ("ollama/",),
    }

    for prefix in provider_prefixes.get(provider, ()):
        if normalized.lower().startswith(prefix):
            return normalized[len(prefix):]

    return normalized


def build_executor_config(
    *,
    model: str,
    provider: str = "",
    effort: str = "medium",
    working_dir: str = "",
    timeout: int = 300,
) -> ExecutorConfig:
    """Build a normalized executor config shared by all OpenKB LLM paths."""
    resolved_provider = provider or infer_provider_from_model(model)
    normalized_model = normalize_model_for_provider(model, resolved_provider)
    return ExecutorConfig(
        provider=resolved_provider,
        model=normalized_model,
        effort=effort,
        working_dir=working_dir,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Stream parsers (used by executors)
# ---------------------------------------------------------------------------

def _collect_claude_stream(
    lines: Iterable[str],
    model: str,
    on_text_delta: TextDeltaCallback | None = None,
) -> LLMResult:
    """Collect Claude stream-json events into an LLMResult."""
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    saw_delta = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "assistant":
            if not saw_delta:
                content = event.get("message", {}).get("content", [])
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
                        text_parts.append(text)
                        if text and on_text_delta is not None:
                            on_text_delta(text)
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        pass

        elif event_type == "result":
            result_text = event.get("result", "")
            if result_text and not text_parts:
                text_parts.append(result_text)
                if on_text_delta is not None:
                    on_text_delta(result_text)
            usage = event.get("usage", {})
            input_tokens = usage.get("input_tokens", input_tokens)
            output_tokens = usage.get("output_tokens", output_tokens)

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text = delta.get("text", "")
                saw_delta = True
                text_parts.append(text)
                if text and on_text_delta is not None:
                    on_text_delta(text)

    full_text = "".join(text_parts).strip()
    total_tokens = input_tokens + output_tokens

    return LLMResult(
        text=full_text,
        provider="claude",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _parse_claude_stream(stdout: str, model: str) -> LLMResult:
    """Parse claude CLI stream-json output into an LLMResult."""
    return _collect_claude_stream(stdout.splitlines(), model)


def _extract_codex_text_delta(event: dict[str, object]) -> str:
    """Best-effort extraction of text deltas from Codex JSON events."""
    event_type = str(event.get("type", ""))
    if event_type in {"response.output_text.delta", "response.output_text_delta"}:
        delta = event.get("delta", "")
        return delta if isinstance(delta, str) else ""

    if event_type == "item.delta":
        delta_obj = event.get("delta", {})
        if isinstance(delta_obj, dict):
            for key in ("text", "content", "delta"):
                value = delta_obj.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    text = _extract_text_from_blocks(value)
                    if text:
                        return text
        if isinstance(delta_obj, list):
            text = _extract_text_from_blocks(delta_obj)
            if text:
                return text

    if event_type in {"response.output_item.delta", "response.completed"}:
        item = event.get("item", {})
        if isinstance(item, dict):
            for key in ("text", "content"):
                value = item.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, list):
                    text = _extract_text_from_blocks(value)
                    if text:
                        return text
    return ""


def _extract_text_from_blocks(blocks: list[object]) -> str:
    """Extract text from nested event block lists."""
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
            continue
        if not isinstance(block, dict):
            continue
        for key in ("text", "content"):
            value = block.get(key)
            if isinstance(value, str):
                parts.append(value)
    return "".join(parts)


def _collect_codex_app_stream(
    lines: Iterable[str],
    model: str,
    on_text_delta: TextDeltaCallback | None = None,
) -> LLMResult:
    """Collect codex exec --json events into an LLMResult.

    Expected event types:
      - item.completed  → item.type == "agent_message", item.text
      - turn.completed → usage (input_tokens, output_tokens)
      - error / turn.failed → error message
    """
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    saw_delta = False

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")
        delta_text = _extract_codex_text_delta(event)
        if delta_text:
            saw_delta = True
            text_parts.append(delta_text)
            if on_text_delta is not None:
                on_text_delta(delta_text)
            continue

        if event_type == "item.completed":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") == "agent_message" and not saw_delta:
                text = item.get("text", "")
                text_parts.append(text)
                if text and on_text_delta is not None:
                    on_text_delta(text)

        elif event_type == "turn.completed":
            usage = event.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

        elif event_type in ("error", "turn.failed"):
            msg = event.get("message", "") or json.dumps(event.get("error", {}))
            if not text_parts:
                return LLMResult(
                    text="", provider="codex_app", model=model,
                    error=msg,
                )

    full_text = "".join(text_parts).strip()
    total_tokens = input_tokens + output_tokens

    return LLMResult(
        text=full_text,
        provider="codex_app",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _parse_codex_app_stream(stdout: str, model: str) -> LLMResult:
    """Parse codex exec --json output into an LLMResult."""
    return _collect_codex_app_stream(stdout.splitlines(), model)


def _with_provider(result: LLMResult, provider: str) -> LLMResult:
    """Return *result* with a provider label suited to the caller."""
    result.provider = provider
    return result


def _find_binary(name: str) -> str:
    """Find a binary on PATH, or return the name as-is."""
    import shutil
    found = shutil.which(name)
    return found or name


# ---------------------------------------------------------------------------
# ABC interface
# ---------------------------------------------------------------------------

class BaseExecutor(ABC):
    """Abstract base class for LLM executors."""

    def __init__(self, cfg: ExecutorConfig) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def binary_name(self) -> str: ...

    @abstractmethod
    def build_args(self, prompt: str) -> list[str]: ...

    @abstractmethod
    def parse_output(self, stdout: str) -> LLMResult: ...

    def parse_stream(self, lines: Iterable[str], on_text_delta: TextDeltaCallback | None = None) -> LLMResult:
        """Parse streaming stdout lines. Defaults to full buffered parsing."""
        buffered = "".join(lines)
        result = self.parse_output(buffered)
        if result.text and on_text_delta is not None:
            on_text_delta(result.text)
        return result

    def run(self, prompt: str) -> LLMResult:
        """Execute the LLM call: build args → subprocess → parse output."""
        binary = _find_binary(self.binary_name)
        args = self.build_args(prompt)
        env = self.build_env()

        try:
            proc = subprocess.run(
                [binary] + args,
                capture_output=True,
                text=True,
                timeout=self.cfg.timeout,
                cwd=self.cfg.working_dir or None,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return self._timeout_result()

        if proc.returncode != 0:
            return LLMResult(
                text="", provider=self.provider_name,
                model=self.cfg.effective_model,
                error=f"Exit code {proc.returncode}: {proc.stderr[:500]}",
            )

        return self.parse_output(proc.stdout)

    def run_streaming(
        self,
        prompt: str,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> LLMResult:
        """Execute the LLM call while surfacing text deltas when available."""
        binary = _find_binary(self.binary_name)
        args = self.build_args(prompt)
        env = self.build_env()

        try:
            proc = subprocess.Popen(
                [binary] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.cfg.working_dir or None,
                env=env,
                bufsize=1,
            )
        except OSError as exc:
            return LLMResult(
                text="",
                provider=self.provider_name,
                model=self.cfg.effective_model,
                error=str(exc),
            )

        try:
            stdout_iter = proc.stdout if proc.stdout is not None else ()
            result = self.parse_stream(stdout_iter, on_text_delta)
            _, stderr = proc.communicate(timeout=self.cfg.timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return self._timeout_result()
        finally:
            if proc.stdout is not None and hasattr(proc.stdout, "close"):
                proc.stdout.close()
            if proc.stderr is not None and hasattr(proc.stderr, "close"):
                proc.stderr.close()

        if proc.returncode != 0:
            return LLMResult(
                text="",
                provider=self.provider_name,
                model=self.cfg.effective_model,
                error=f"Exit code {proc.returncode}: {stderr[:500]}",
            )

        return result

    def build_env(self) -> dict[str, str] | None:
        """Build environment variables. Override for provider-specific env."""
        return None

    def _timeout_result(self) -> LLMResult:
        return LLMResult(
            text="", provider=self.provider_name,
            model=self.cfg.effective_model,
            error=f"Timeout after {self.cfg.timeout}s",
        )


# ---------------------------------------------------------------------------
# Concrete executors
# ---------------------------------------------------------------------------

class ClaudeExecutor(BaseExecutor):
    """Claude CLI with stream-json output."""
    provider_name = "claude"
    binary_name = "claude"

    def build_args(self, prompt: str) -> list[str]:
        return [
            "--verbose",
            "--output-format", "stream-json",
            "--dangerously-skip-permissions",
            "--model", self.cfg.effective_model,
            "--effort", self.cfg.effort,
            "-p", prompt,
        ]

    def parse_output(self, stdout: str) -> LLMResult:
        return _parse_claude_stream(stdout, self.cfg.effective_model)

    def parse_stream(self, lines: Iterable[str], on_text_delta: TextDeltaCallback | None = None) -> LLMResult:
        return _collect_claude_stream(lines, self.cfg.effective_model, on_text_delta)


class CodexExecutor(BaseExecutor):
    """Codex executor using `codex exec --json` with provider-local labeling."""
    provider_name = "codex"
    binary_name = "codex"

    def build_args(self, prompt: str) -> list[str]:
        return [
            "exec",
            "--json",
            "--ephemeral",
            "-c", f'model_reasoning_effort="{self.cfg.effort}"',
            "-m", self.cfg.effective_model,
            prompt,
        ]

    def parse_output(self, stdout: str) -> LLMResult:
        return _with_provider(_parse_codex_app_stream(stdout, self.cfg.effective_model), self.provider_name)

    def parse_stream(self, lines: Iterable[str], on_text_delta: TextDeltaCallback | None = None) -> LLMResult:
        return _with_provider(
            _collect_codex_app_stream(lines, self.cfg.effective_model, on_text_delta),
            self.provider_name,
        )

    def run_streaming(
        self,
        prompt: str,
        on_text_delta: TextDeltaCallback | None = None,
    ) -> LLMResult:
        """Buffered fallback for Codex CLI to avoid stdout/stderr deadlocks."""
        result = self.run(prompt)
        if result.text and on_text_delta is not None:
            on_text_delta(result.text)
        return result


class CodexAppExecutor(BaseExecutor):
    """Codex App executor using `codex exec --json` with structured output."""
    provider_name = "codex_app"
    binary_name = "codex"

    def build_args(self, prompt: str) -> list[str]:
        return [
            "exec", "--json",
            "-m", self.cfg.effective_model,
            prompt,
        ]

    def parse_output(self, stdout: str) -> LLMResult:
        return _parse_codex_app_stream(stdout, self.cfg.effective_model)

    def parse_stream(self, lines: Iterable[str], on_text_delta: TextDeltaCallback | None = None) -> LLMResult:
        return _collect_codex_app_stream(lines, self.cfg.effective_model, on_text_delta)


class OllamaExecutor(ClaudeExecutor):
    """Ollama uses claude CLI binary with endpoint redirection.

    Inherits build_args/parse_output from ClaudeExecutor (same stream-json
    protocol) but overrides build_env to redirect to the ollama endpoint.
    """
    provider_name = "ollama"
    binary_name = "claude"

    def build_env(self) -> dict[str, str] | None:
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = self.cfg.ollama_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = self.cfg.ollama_auth_token
        env["ANTHROPIC_API_KEY"] = ""
        return env


# ---------------------------------------------------------------------------
# Executor registry
# ---------------------------------------------------------------------------

EXECUTORS: dict[str, type[BaseExecutor]] = {
    "claude": ClaudeExecutor,
    "codex": CodexExecutor,
    "codex_app": CodexAppExecutor,
    "ollama": OllamaExecutor,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_llm(prompt: str, cfg: ExecutorConfig | None = None) -> LLMResult:
    """Run an LLM call via subprocess and return the result.

    For claude CLI: uses stream-json output format and parses JSON events.
    For codex_app: uses `codex exec --json` and parses JSON events.
    For other providers: returns raw stdout.
    """
    if cfg is None:
        cfg = ExecutorConfig()

    executor_cls = EXECUTORS.get(cfg.provider)
    if not executor_cls:
        return LLMResult(
            text="", provider=cfg.provider, model=cfg.effective_model,
            error=f"Unknown provider: {cfg.provider}",
        )

    executor = executor_cls(cfg)
    t0 = time.time()
    result = executor.run(prompt)
    result.elapsed_seconds = time.time() - t0
    return result


def run_llm_streaming(
    prompt: str,
    cfg: ExecutorConfig | None = None,
    on_text_delta: TextDeltaCallback | None = None,
) -> LLMResult:
    """Run an LLM call via subprocess while surfacing incremental text when possible."""
    if cfg is None:
        cfg = ExecutorConfig()

    executor_cls = EXECUTORS.get(cfg.provider)
    if not executor_cls:
        return LLMResult(
            text="", provider=cfg.provider, model=cfg.effective_model,
            error=f"Unknown provider: {cfg.provider}",
        )

    executor = executor_cls(cfg)
    t0 = time.time()
    result = executor.run_streaming(prompt, on_text_delta)
    result.elapsed_seconds = time.time() - t0
    return result


def run_llm_with_system(system_prompt: str, user_prompt: str, cfg: ExecutorConfig | None = None) -> LLMResult:
    """Run an LLM call with separate system and user prompts."""
    combined = f"<system>\n{system_prompt}\n</system>\n\n{user_prompt}"
    return run_llm(combined, cfg)


def run_llm_with_system_streaming(
    system_prompt: str,
    user_prompt: str,
    cfg: ExecutorConfig | None = None,
    on_text_delta: TextDeltaCallback | None = None,
) -> LLMResult:
    """Run an LLM call with separate system/user prompts and stream text deltas."""
    combined = f"<system>\n{system_prompt}\n</system>\n\n{user_prompt}"
    return run_llm_streaming(combined, cfg, on_text_delta)
