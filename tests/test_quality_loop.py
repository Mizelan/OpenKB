from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from openkb.quality_loop import run_quality_convergence
from openkb.review import ReviewItem, ReviewQueue


def _make_quality_kb(tmp_path: Path) -> Path:
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text("model: gpt-5.4-mini\n", encoding="utf-8")
    (kb_dir / "wiki" / "reports").mkdir(parents=True)
    return kb_dir


def test_run_quality_convergence_writes_latest_reports_and_counts_pending_reviews(tmp_path):
    kb_dir = _make_quality_kb(tmp_path)
    queue = ReviewQueue(kb_dir / ".openkb")
    queue.add([
        ReviewItem(
            type="suggestion",
            title="Review me",
            description="Pending item",
            source_path="summaries/paper.md",
        )
    ])

    structural_path = kb_dir / "wiki" / "reports" / "structural_latest.md"
    structural_path.write_text("Structural report", encoding="utf-8")

    with (
        patch(
            "openkb.quality_loop.run_internal_maintenance",
            return_value={"issues": ["one", "two"], "report_path": structural_path},
        ),
        patch(
            "openkb.quality_loop.run_knowledge_lint",
            new=AsyncMock(return_value="# Semantic\n\nAll good."),
        ),
        patch(
            "openkb.quality_loop.inspect_background_insights_state",
            return_value={
                "status": "ready",
                "summary": "Last insights run: 2026-04-19T10:00:00Z",
                "last_run": "2026-04-19T10:00:00Z",
                "report_path": ".openkb/insights.md",
            },
        ),
    ):
        result = run_quality_convergence(kb_dir, "gpt-5.4-mini")

    assert result["structural_report"] == "wiki/reports/structural_latest.md"
    assert result["semantic_report"] == "wiki/reports/semantic_latest.md"
    assert result["quality_report"] == "wiki/reports/quality_latest.md"
    assert result["structural_issue_count"] == 2
    assert result["pending_review_count"] == 1
    assert (kb_dir / "wiki" / "reports" / "semantic_latest.md").exists()
    quality_text = (kb_dir / "wiki" / "reports" / "quality_latest.md").read_text(encoding="utf-8")
    assert "wiki/reports/structural_latest.md" in quality_text
    assert "wiki/reports/semantic_latest.md" in quality_text
    assert "Pending review items: 1" in quality_text


def test_run_quality_convergence_marks_missing_insights_explicitly(tmp_path):
    kb_dir = _make_quality_kb(tmp_path)
    structural_path = kb_dir / "wiki" / "reports" / "structural_latest.md"
    structural_path.write_text("Structural report", encoding="utf-8")

    with (
        patch(
            "openkb.quality_loop.run_internal_maintenance",
            return_value={"issues": [], "report_path": structural_path},
        ),
        patch(
            "openkb.quality_loop.run_knowledge_lint",
            new=AsyncMock(return_value="# Semantic\n\nNothing notable."),
        ),
        patch(
            "openkb.quality_loop.inspect_background_insights_state",
            return_value={
                "status": "missing",
                "summary": "Last insights run: missing",
                "last_run": None,
                "report_path": None,
            },
        ),
    ):
        result = run_quality_convergence(kb_dir, "gpt-5.4-mini")

    assert result["insights"]["status"] == "missing"
    quality_text = (kb_dir / "wiki" / "reports" / "quality_latest.md").read_text(encoding="utf-8")
    assert "Last insights run: missing" in quality_text


def test_run_quality_convergence_counts_real_structural_issue_entries(tmp_path):
    kb_dir = _make_quality_kb(tmp_path)
    structural_path = kb_dir / "wiki" / "reports" / "structural_latest.md"
    structural_path.write_text("Structural report", encoding="utf-8")

    with (
        patch(
            "openkb.quality_loop.run_internal_maintenance",
            return_value={
                "issues": {
                    "broken_source_links": ["a", "b"],
                    "broken_wiki_links": [],
                    "broken_concept_links": ["c"],
                    "duplicate_slug_groups": [],
                },
                "report_path": structural_path,
            },
        ),
        patch(
            "openkb.quality_loop.run_knowledge_lint",
            new=AsyncMock(return_value="# Semantic\n\nAll good."),
        ),
        patch(
            "openkb.quality_loop.inspect_background_insights_state",
            return_value={
                "status": "ready",
                "summary": "Last insights run: 2026-04-19T10:00:00Z",
                "last_run": "2026-04-19T10:00:00Z",
                "report_path": ".openkb/insights.md",
            },
        ),
    ):
        result = run_quality_convergence(kb_dir, "gpt-5.4-mini")

    assert result["structural_issue_count"] == 3
    quality_text = (kb_dir / "wiki" / "reports" / "quality_latest.md").read_text(encoding="utf-8")
    assert "Structural issues: 3" in quality_text


def test_run_quality_convergence_surfaces_semantic_lint_failure_without_crashing(tmp_path):
    kb_dir = _make_quality_kb(tmp_path)
    structural_path = kb_dir / "wiki" / "reports" / "structural_latest.md"
    structural_path.write_text("Structural report", encoding="utf-8")

    with (
        patch(
            "openkb.quality_loop.run_internal_maintenance",
            return_value={"issues": [], "report_path": structural_path},
        ),
        patch(
            "openkb.quality_loop.run_knowledge_lint",
            new=AsyncMock(side_effect=RuntimeError("semantic boom")),
        ),
        patch(
            "openkb.quality_loop.inspect_background_insights_state",
            return_value={
                "status": "ready",
                "summary": "Last insights run: 2026-04-19T10:00:00Z",
                "last_run": "2026-04-19T10:00:00Z",
                "report_path": ".openkb/insights.md",
            },
        ),
    ):
        result = run_quality_convergence(kb_dir, "gpt-5.4-mini")

    assert result["semantic_report"] == "wiki/reports/semantic_latest.md"
    semantic_text = (kb_dir / "wiki" / "reports" / "semantic_latest.md").read_text(encoding="utf-8")
    assert "Knowledge lint failed: semantic boom" in semantic_text
