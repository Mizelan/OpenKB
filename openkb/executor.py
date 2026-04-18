"""Subprocess-based LLM executor for pkm2.

Calls claude, codex, or ollama CLI as subprocesses — no API keys needed.
Supports streaming JSON output from claude CLI and plain text from others.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


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
    provider: str = "claude"       # claude | codex | ollama
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
            "ollama": "sonnet",
        }
        return defaults.get(self.provider, "sonnet")


def _find_binary(name: str) -> str:
    """Find a binary on PATH, or return the name as-is."""
    import shutil
    found = shutil.which(name)
    return found or name


def _build_claude_args(prompt: str, cfg: ExecutorConfig, session_id: str = "") -> list[str]:
    """Build command arguments for claude CLI."""
    args = [
        "--verbose",
        "--output-format", "stream-json",
        "--dangerously-skip-permissions",
        "--model", cfg.effective_model,
        "--effort", cfg.effort,
    ]
    if session_id:
        args.extend(["--resume", session_id])
    args.extend(["-p", prompt])
    return args


def _build_codex_args(prompt: str, cfg: ExecutorConfig) -> list[str]:
    """Build command arguments for codex CLI."""
    args = [
        "--model", cfg.effective_model or "gpt-5.4-mini",
        "--effort", cfg.effort,
    ]
    args.extend(["-p", prompt])
    return args


def run_llm(prompt: str, cfg: ExecutorConfig | None = None) -> LLMResult:
    """Run an LLM call via subprocess and return the result.

    For claude CLI: uses stream-json output format and parses JSON events.
    For other providers: returns raw stdout.
    """
    if cfg is None:
        cfg = ExecutorConfig()

    t0 = time.time()
    provider = cfg.provider

    if provider == "claude":
        result = _run_claude(prompt, cfg)
    elif provider == "codex":
        result = _run_codex(prompt, cfg)
    elif provider == "ollama":
        result = _run_ollama(prompt, cfg)
    else:
        return LLMResult(
            text="", provider=provider, model=cfg.effective_model,
            error=f"Unknown provider: {provider}",
        )

    result.elapsed_seconds = time.time() - t0
    return result


def _run_claude(prompt: str, cfg: ExecutorConfig) -> LLMResult:
    """Execute claude CLI as a subprocess and parse stream-json output."""
    binary = _find_binary("claude")
    args = _build_claude_args(prompt, cfg)

    env = os.environ.copy()
    # Ollama passthrough: redirect claude CLI to ollama endpoint
    if cfg.provider == "ollama":
        env["ANTHROPIC_BASE_URL"] = cfg.ollama_base_url
        env["ANTHROPIC_AUTH_TOKEN"] = cfg.ollama_auth_token
        env["ANTHROPIC_API_KEY"] = ""

    cwd = cfg.working_dir or None

    try:
        proc = subprocess.run(
            [binary] + args,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            cwd=cwd,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return LLMResult(
            text="", provider="claude", model=cfg.effective_model,
            error=f"Timeout after {cfg.timeout}s",
        )

    if proc.returncode != 0:
        return LLMResult(
            text="", provider="claude", model=cfg.effective_model,
            error=f"Exit code {proc.returncode}: {proc.stderr[:500]}",
        )

    return _parse_claude_stream(proc.stdout, cfg.effective_model)


def _parse_claude_stream(stdout: str, model: str) -> LLMResult:
    """Parse claude CLI stream-json output into an LLMResult."""
    text_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue

        event_type = event.get("type", "")

        if event_type == "assistant":
            content = event.get("message", {}).get("content", [])
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    pass  # skip tool calls in compilation

        elif event_type == "result":
            # Final result event
            result_text = event.get("result", "")
            if result_text and not text_parts:
                text_parts.append(result_text)
            usage = event.get("usage", {})
            input_tokens = usage.get("input_tokens", input_tokens)
            output_tokens = usage.get("output_tokens", output_tokens)

        elif event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))

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


def _run_codex(prompt: str, cfg: ExecutorConfig) -> LLMResult:
    """Execute codex CLI as a subprocess."""
    binary = _find_binary("codex")
    args = _build_codex_args(prompt, cfg)

    try:
        proc = subprocess.run(
            [binary] + args,
            capture_output=True,
            text=True,
            timeout=cfg.timeout,
            cwd=cfg.working_dir or None,
        )
    except subprocess.TimeoutExpired:
        return LLMResult(
            text="", provider="codex", model=cfg.effective_model,
            error=f"Timeout after {cfg.timeout}s",
        )

    if proc.returncode != 0:
        return LLMResult(
            text="", provider="codex", model=cfg.effective_model,
            error=f"Exit code {proc.returncode}: {proc.stderr[:500]}",
        )

    return LLMResult(
        text=proc.stdout.strip(),
        provider="codex",
        model=cfg.effective_model,
    )


def _run_ollama(prompt: str, cfg: ExecutorConfig) -> LLMResult:
    """Execute ollama via claude CLI with Ollama endpoint redirection."""
    # Ollama runs through the claude CLI with env overrides
    return _run_claude(prompt, cfg)


def run_llm_with_system(system_prompt: str, user_prompt: str, cfg: ExecutorConfig | None = None) -> LLMResult:
    """Run an LLM call with separate system and user prompts.

    Combines system and user prompts into a single prompt string with
    clear delimiters, since subprocess-based CLIs typically accept
    a single prompt argument.
    """
    combined = f"<system>\n{system_prompt}\n</system>\n\n{user_prompt}"
    return run_llm(combined, cfg)