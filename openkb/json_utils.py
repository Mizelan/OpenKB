"""Shared JSON extraction utilities."""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


def extract_json(text: str, start_char: str = "") -> list | dict | None:
    """Extract the first JSON structure (object or array) from text.

    Strips conversational wrapper text before/after the JSON payload
    and truncates trailing content after the closing bracket.

    Args:
        text: Raw text potentially containing JSON.
        start_char: If set, only look for this opening character ("{" or "[").
            If empty, auto-detect from the first "{" or "[" found.

    Returns:
        Parsed JSON value (dict or list), or None if no valid JSON found.
    """
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Find the first JSON structure
    if start_char:
        start = cleaned.find(start_char)
        if start == -1:
            return None
    else:
        if cleaned and not cleaned.startswith(("{", "[")):
            brace_idx = cleaned.find("{")
            bracket_idx = cleaned.find("[")
            indices = [i for i in (brace_idx, bracket_idx) if i != -1]
            if not indices:
                return None
            start = min(indices)
        else:
            start = 0

    substring = cleaned[start:] if start > 0 else cleaned
    open_ch = substring[0]
    close_ch = "}" if open_ch == "{" else "]"

    # Bracket-depth tracking to find the matching close
    depth = 0
    in_string = False
    escape = False
    end = len(substring)
    for i, c in enumerate(substring):
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    json_str = substring[:end].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None