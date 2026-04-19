"""Action executors for actionable review items."""
from __future__ import annotations

import os
from pathlib import Path

from openkb.review.models import ReviewItem


def apply_review_action(kb_dir: Path, item: ReviewItem) -> Path:
    """Apply a review item's wiki mutation and return the changed file path."""
    if item.action_type is None:
        raise ValueError("Review item does not define an action_type.")

    wiki_dir = kb_dir / "wiki"
    if item.action_type == "create_placeholder":
        return _create_placeholder(wiki_dir, item.payload)
    if item.action_type == "alias_concept":
        return _create_alias_page(wiki_dir, item.payload)
    if item.action_type == "mark_stale":
        return _mark_page_stale(wiki_dir, item.payload)

    raise ValueError(f"Unsupported review action: {item.action_type}")


def _create_placeholder(wiki_dir: Path, payload: dict) -> Path:
    target = _resolve_wiki_path(wiki_dir, _require_string(payload, "path"))
    title = payload.get("title") or target.stem.replace("-", " ").replace("_", " ").title()
    if target.exists():
        raise FileExistsError(f"Placeholder already exists: {target.relative_to(wiki_dir)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"# {title}\n\nPlaceholder page created from review item.\n",
        encoding="utf-8",
    )
    return target


def _create_alias_page(wiki_dir: Path, payload: dict) -> Path:
    target = _resolve_wiki_path(wiki_dir, _require_string(payload, "path"))
    alias = _require_string(payload, "alias")
    concept_target = _require_string(payload, "target")
    if target.exists():
        raise FileExistsError(f"Alias page already exists: {target.relative_to(wiki_dir)}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"# {alias}\n\nAlias of [[{concept_target}]].\n",
        encoding="utf-8",
    )
    return target


def _mark_page_stale(wiki_dir: Path, payload: dict) -> Path:
    target = _resolve_wiki_path(wiki_dir, _require_string(payload, "path"))
    reason = _require_string(payload, "reason")
    if not target.exists():
        raise FileNotFoundError(f"Cannot mark missing page as stale: {target.relative_to(wiki_dir)}")

    existing = target.read_text(encoding="utf-8").rstrip()
    stale_notice = f"\n\n> [!warning] Stale\n> {reason}\n"
    if stale_notice.strip() not in existing:
        target.write_text(f"{existing}{stale_notice}", encoding="utf-8")
    return target


def _require_string(payload: dict, key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Review action payload requires a non-empty {key!r} string.")
    return value.strip()


def _resolve_wiki_path(wiki_dir: Path, relative_path: str) -> Path:
    target = (wiki_dir / relative_path).resolve()
    wiki_root = wiki_dir.resolve()
    if target != wiki_root and os.path.commonpath([str(wiki_root), str(target)]) != str(wiki_root):
        raise ValueError("Review action path must stay within the wiki directory.")
    return target
