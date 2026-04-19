"""Planning-only stale-page refresh helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from openkb.frontmatter import parse_fm
from openkb.review import ReviewQueue

_REFRESH_EXCLUDED = {"AGENTS.md", "SCHEMA.md", "log.md", "index.md"}


def collect_stale_pages(kb_dir: Path) -> list[dict[str, object]]:
    """Collect stale wiki pages and the reasons they need refresh attention."""
    review_reasons = _collect_review_reasons(kb_dir)
    stale_pages: list[dict[str, object]] = []

    for path in sorted((kb_dir / "wiki").rglob("*.md")):
        if path.name in _REFRESH_EXCLUDED or "reports" in path.relative_to(kb_dir / "wiki").parts:
            continue
        page_rel = path.relative_to(kb_dir / "wiki").as_posix()
        meta, _ = parse_fm(path.read_text(encoding="utf-8"))
        reasons: list[str] = []
        reasons.extend(_supporting_source_reasons(kb_dir, path, meta))
        reasons.extend(review_reasons.get(page_rel, []))
        if reasons:
            stale_pages.append({"path": page_rel, "reasons": reasons})

    seen_paths = {entry["path"] for entry in stale_pages}
    for page_rel, reasons in sorted(review_reasons.items()):
        if page_rel in seen_paths:
            continue
        stale_pages.append({"path": page_rel, "reasons": reasons})

    return stale_pages


def render_refresh_plan(stale_pages: list[dict[str, object]]) -> str:
    count = len(stale_pages)
    lines = [f"{count} stale page(s) found"]
    for entry in stale_pages:
        reasons = "; ".join(entry["reasons"])
        lines.append(f"- {entry['path']} ({reasons})")
    return "\n".join(lines)


def _supporting_source_reasons(kb_dir: Path, page_path: Path, meta: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    updated_at = _parse_updated_at(meta.get("updated_at"))
    supporting_sources = meta.get("supporting_sources")
    if isinstance(supporting_sources, str):
        supporting_sources = [supporting_sources]
    if not isinstance(supporting_sources, list):
        supporting_sources = []

    for source in supporting_sources:
        source_rel = str(source)
        source_path = kb_dir / source_rel
        if not source_path.exists():
            reasons.append(f"supporting source missing: {source_rel}")
            continue
        if updated_at is not None and source_path.stat().st_mtime > updated_at:
            reasons.append(f"supporting source newer than page: {source_rel}")
    return reasons


def _collect_review_reasons(kb_dir: Path) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    queue = ReviewQueue(kb_dir / ".openkb")
    for item in queue.list():
        if item.action_type != "mark_stale":
            continue
        target = item.payload.get("path")
        if not isinstance(target, str) or not target:
            continue
        reason = item.payload.get("reason")
        label = f"pending mark_stale review item: {reason}" if isinstance(reason, str) and reason else "pending mark_stale review item"
        reasons.setdefault(target, []).append(label)
    return reasons


def _parse_updated_at(value: object) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None
