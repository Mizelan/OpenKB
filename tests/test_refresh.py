from __future__ import annotations

from pathlib import Path

from openkb.refresh import collect_stale_pages
from openkb.review import ReviewItem, ReviewQueue


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_kb(tmp_path: Path) -> Path:
    kb_dir = tmp_path / "kb"
    (kb_dir / ".openkb").mkdir(parents=True)
    (kb_dir / "wiki" / "concepts").mkdir(parents=True)
    (kb_dir / "wiki" / "summaries").mkdir(parents=True)
    return kb_dir


def test_collects_stale_page_when_supporting_source_is_missing(tmp_path):
    kb_dir = _make_kb(tmp_path)
    _write(
        kb_dir / "wiki" / "concepts" / "attention.md",
        (
            "---\n"
            "updated_at: 2026-04-19T00:00:00Z\n"
            "supporting_sources:\n"
            "  - sources/missing.md\n"
            "---\n\n"
            "# Attention\n"
        ),
    )

    stale = collect_stale_pages(kb_dir)

    assert stale == [
        {
            "path": "concepts/attention.md",
            "reasons": ["supporting source missing: sources/missing.md"],
        }
    ]


def test_collects_stale_page_when_supporting_source_is_newer_than_page(tmp_path):
    kb_dir = _make_kb(tmp_path)
    _write(kb_dir / "sources" / "paper.md", "# Paper\n")
    _write(
        kb_dir / "wiki" / "concepts" / "attention.md",
        (
            "---\n"
            "updated_at: 2020-01-01T00:00:00Z\n"
            "supporting_sources:\n"
            "  - sources/paper.md\n"
            "---\n\n"
            "# Attention\n"
        ),
    )

    stale = collect_stale_pages(kb_dir)

    assert stale == [
        {
            "path": "concepts/attention.md",
            "reasons": ["supporting source newer than page: sources/paper.md"],
        }
    ]


def test_collects_stale_page_from_pending_mark_stale_review_item(tmp_path):
    kb_dir = _make_kb(tmp_path)
    _write(kb_dir / "wiki" / "concepts" / "attention.md", "# Attention\n")
    queue = ReviewQueue(kb_dir / ".openkb")
    queue.add([
        ReviewItem(
            type="suggestion",
            title="Mark attention stale",
            description="Needs refresh",
            source_path="summaries/attention.md",
            action_type="mark_stale",
            payload={"path": "concepts/attention.md", "reason": "Needs refresh"},
        )
    ])

    stale = collect_stale_pages(kb_dir)

    assert stale == [
        {
            "path": "concepts/attention.md",
            "reasons": ["pending mark_stale review item: Needs refresh"],
        }
    ]


def test_collects_pending_mark_stale_even_when_target_file_is_missing(tmp_path):
    kb_dir = _make_kb(tmp_path)
    queue = ReviewQueue(kb_dir / ".openkb")
    queue.add([
        ReviewItem(
            type="suggestion",
            title="Mark missing stale",
            description="Needs refresh",
            source_path="summaries/attention.md",
            action_type="mark_stale",
            payload={"path": "concepts/missing.md", "reason": "Needs refresh"},
        )
    ])

    stale = collect_stale_pages(kb_dir)

    assert stale == [
        {
            "path": "concepts/missing.md",
            "reasons": ["pending mark_stale review item: Needs refresh"],
        }
    ]


def test_ignores_newer_source_check_when_updated_at_is_malformed(tmp_path):
    kb_dir = _make_kb(tmp_path)
    _write(kb_dir / "sources" / "paper.md", "# Paper\n")
    _write(
        kb_dir / "wiki" / "concepts" / "attention.md",
        (
            "---\n"
            "updated_at: not-a-timestamp\n"
            "supporting_sources:\n"
            "  - sources/paper.md\n"
            "---\n\n"
            "# Attention\n"
        ),
    )

    stale = collect_stale_pages(kb_dir)

    assert stale == []


def test_handles_empty_supporting_sources_without_marking_stale(tmp_path):
    kb_dir = _make_kb(tmp_path)
    _write(
        kb_dir / "wiki" / "concepts" / "attention.md",
        (
            "---\n"
            "updated_at: 2026-04-19T00:00:00Z\n"
            "supporting_sources: []\n"
            "---\n\n"
            "# Attention\n"
        ),
    )

    stale = collect_stale_pages(kb_dir)

    assert stale == []
