"""Promotion helpers for turning saved explorations into durable KB artifacts."""
from __future__ import annotations

import re
from pathlib import Path

from openkb.frontmatter import parse_fm, serialize_fm
from openkb.review import ReviewItem, ReviewQueue

_VALID_PROMOTION_MODES = {"query_page", "concept_seed"}


def promote_exploration(kb_dir: Path, rel_path: str, *, mode: str) -> dict[str, object]:
    """Promote a saved exploration into a durable query page or concept seed."""
    if mode not in _VALID_PROMOTION_MODES:
        raise ValueError(f"Unsupported promotion mode: {mode}")

    source_path, source_rel = _resolve_exploration_path(kb_dir, rel_path)
    meta, body = parse_fm(source_path.read_text(encoding="utf-8"))
    if mode == "query_page":
        return _promote_to_query_page(kb_dir, source_rel, meta, body)
    return _promote_to_concept_seed(kb_dir, source_rel, meta, body)


def latest_exploration_path(kb_dir: Path) -> str:
    """Return the most recently modified exploration path relative to the KB root."""
    explore_dir = kb_dir / "wiki" / "explorations"
    if not explore_dir.exists():
        raise FileNotFoundError("No saved explorations found.")
    files = [path for path in explore_dir.rglob("*.md") if path.is_file()]
    if not files:
        raise FileNotFoundError("No saved explorations found.")
    latest = max(files, key=lambda path: path.stat().st_mtime)
    return latest.relative_to(kb_dir).as_posix()


def _resolve_exploration_path(kb_dir: Path, rel_path: str) -> tuple[Path, str]:
    normalized = rel_path.strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Exploration path is required.")

    if normalized.startswith("wiki/explorations/"):
        candidate = kb_dir / normalized
    elif normalized.startswith("explorations/"):
        candidate = kb_dir / "wiki" / normalized
    elif normalized.startswith("wiki/"):
        candidate = kb_dir / normalized
    else:
        candidate = kb_dir / "wiki" / normalized.lstrip("/")

    resolved = candidate.resolve()
    wiki_root = (kb_dir / "wiki").resolve()
    if resolved != wiki_root and wiki_root not in resolved.parents:
        raise ValueError("Promotion path must stay within the wiki directory.")
    if not resolved.exists():
        raise FileNotFoundError(f"Exploration not found: {rel_path}")

    source_rel = resolved.relative_to(kb_dir).as_posix()
    if not source_rel.startswith("wiki/explorations/"):
        raise ValueError("Promotion source must live under wiki/explorations/.")
    return resolved, source_rel


def _promote_to_query_page(
    kb_dir: Path,
    source_rel: str,
    meta: dict[str, object],
    body: str,
) -> dict[str, object]:
    query_dir = kb_dir / "wiki" / "queries"
    query_dir.mkdir(parents=True, exist_ok=True)
    source_subpath = Path(source_rel).relative_to("wiki/explorations")
    target = query_dir / source_subpath
    target.parent.mkdir(parents=True, exist_ok=True)
    status = "updated" if target.exists() else "created"

    query_text = _derive_query_text(meta, body, source_rel)
    page_meta: dict[str, object] = {
        "type": "wiki-query",
        "query": query_text,
        "promoted_from": source_rel,
    }
    for key in ("session", "model", "created"):
        if key in meta:
            page_meta[key] = meta[key]
    page_body = body.strip() or f"# {target.stem.replace('-', ' ').title()}\n"
    target.write_text(serialize_fm(page_meta, page_body), encoding="utf-8")
    return {
        "mode": "query_page",
        "status": status,
        "source_path": source_rel,
        "target_path": target.relative_to(kb_dir).as_posix(),
    }


def _promote_to_concept_seed(
    kb_dir: Path,
    source_rel: str,
    meta: dict[str, object],
    body: str,
) -> dict[str, object]:
    openkb_dir = kb_dir / ".openkb"
    queue = ReviewQueue(openkb_dir)
    concept_rel = Path(source_rel).relative_to("wiki/explorations")
    concept_slug = concept_rel.stem
    concept_path = Path("concepts") / concept_rel
    title = _derive_title(meta, body, concept_slug)
    concept_path_str = concept_path.as_posix()

    for existing in queue.list():
        if (
            existing.source_path == source_rel
            and existing.action_type == "create_placeholder"
            and existing.payload.get("path") == concept_path_str
        ):
            return {
                "mode": "concept_seed",
                "status": "existing",
                "source_path": source_rel,
                "target_path": concept_path_str,
                "queue_size": len(queue.list()),
            }

    description = f"Promote saved exploration {source_rel} into a concept seed placeholder."
    item = ReviewItem(
        type="missing_page",
        title=f"Promote exploration: {title}",
        description=description,
        source_path=source_rel,
        affected_pages=[concept_path_str],
        action_type="create_placeholder",
        payload={"path": concept_path_str, "title": title},
    )
    queue.add([item])
    return {
        "mode": "concept_seed",
        "status": "queued",
        "source_path": source_rel,
        "target_path": concept_path_str,
        "queue_size": len(queue.list()),
    }


def _derive_query_text(meta: dict[str, object], body: str, source_rel: str) -> str:
    query_value = meta.get("query")
    if isinstance(query_value, str) and query_value.strip():
        return query_value.strip()

    first_prompt = re.search(r"^## \[\d+\] (.+)$", body, flags=re.MULTILINE)
    if first_prompt:
        return first_prompt.group(1).strip()
    return f"Promoted exploration from {source_rel}"


def _derive_title(meta: dict[str, object], body: str, fallback_slug: str) -> str:
    query_value = meta.get("query")
    if isinstance(query_value, str) and query_value.strip():
        return _normalize_title(query_value)

    heading = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    if heading:
        return _normalize_title(heading.group(1))

    first_prompt = re.search(r"^## \[\d+\] (.+)$", body, flags=re.MULTILINE)
    if first_prompt:
        return _normalize_title(first_prompt.group(1))

    return fallback_slug.replace("-", " ").replace("_", " ").title()


def _normalize_title(value: str) -> str:
    text = " ".join(value.strip().split())
    text = re.sub(r"[?!.]+$", "", text)
    return text[:120] or "Promoted Exploration"
