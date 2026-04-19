"""PyYAML-based frontmatter utilities for OpenKB.

Provides parse → dict modify → serialize flow, replacing manual string
manipulation and regex-based field injection.
"""
from __future__ import annotations

import yaml


class _NoTimestampSafeLoader(yaml.SafeLoader):
    """SafeLoader variant that keeps YAML timestamps as plain strings."""


_NoTimestampSafeLoader.yaml_implicit_resolvers = {
    key: [
        (tag, regexp)
        for tag, regexp in value
        if tag != "tag:yaml.org,2002:timestamp"
    ]
    for key, value in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None

    for idx in range(1, len(lines)):
        if lines[idx].strip() != "---":
            continue
        fm_block = "".join(lines[1:idx]).strip()
        body = "".join(lines[idx + 1 :]).lstrip("\n")
        return fm_block, body
    return None


def parse_fm(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Returns (metadata_dict, body_string).
    If no valid frontmatter, returns ({}, original_text).
    """
    parsed = _split_frontmatter(text)
    if not parsed:
        return {}, text

    fm_block, body = parsed

    try:
        meta = yaml.load(fm_block, Loader=_NoTimestampSafeLoader)
    except yaml.YAMLError:
        return {}, text

    if meta is None:
        meta = {}

    if not isinstance(meta, dict):
        return {}, text

    return meta, body


def serialize_fm(meta: dict, body: str) -> str:
    """Serialize metadata dict + body back to markdown with YAML frontmatter.

    Empty dict produces body-only output (no frontmatter block).
    """
    if not meta:
        return body

    fm = yaml.dump(
        meta,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    ).strip()
    return f"---\n{fm}\n---\n\n{body}"


def update_fm(text: str, **fields) -> str:
    """Parse frontmatter, update specified fields, re-serialize.

    Single-pass field injection — no regex, no re-parsing.
    """
    meta, body = parse_fm(text)
    for key, value in fields.items():
        meta[key] = value
    return serialize_fm(meta, body)
