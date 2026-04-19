"""Quality convergence command support."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from openkb.agent.linter import run_knowledge_lint
from openkb.graph.insights_bg import inspect_background_insights_state
from openkb.maintenance import run_internal_maintenance
from openkb.review import ReviewQueue


def run_quality_convergence(kb_dir: Path, model: str) -> dict[str, Any]:
    """Aggregate current KB quality signals and refresh stable latest reports."""
    reports_dir = kb_dir / "wiki" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    structural = run_internal_maintenance(kb_dir)
    structural_report = _relative_to_kb(kb_dir, structural["report_path"])
    structural_issue_count = _count_structural_issues(structural.get("issues", []))

    try:
        semantic_text = asyncio.run(run_knowledge_lint(kb_dir, model))
    except Exception as exc:
        semantic_text = f"Knowledge lint failed: {exc}"
    semantic_path = reports_dir / "semantic_latest.md"
    semantic_path.write_text(semantic_text, encoding="utf-8")
    semantic_report = _relative_to_kb(kb_dir, semantic_path)

    insights = inspect_background_insights_state(kb_dir)
    pending_review_count = len(ReviewQueue(kb_dir / ".openkb").list())

    quality_path = reports_dir / "quality_latest.md"
    quality_path.write_text(
        _render_quality_report(
            structural_issue_count=structural_issue_count,
            structural_report=structural_report,
            semantic_report=semantic_report,
            pending_review_count=pending_review_count,
            insights=insights,
        ),
        encoding="utf-8",
    )

    return {
        "structural_issue_count": structural_issue_count,
        "structural_report": structural_report,
        "semantic_report": semantic_report,
        "quality_report": _relative_to_kb(kb_dir, quality_path),
        "pending_review_count": pending_review_count,
        "insights": insights,
    }


def _count_structural_issues(issues: object) -> int:
    if isinstance(issues, dict):
        count = 0
        for value in issues.values():
            if isinstance(value, list):
                count += len(value)
            elif value:
                count += 1
        return count
    if isinstance(issues, list):
        return len(issues)
    return 0


def _render_quality_report(
    *,
    structural_issue_count: int,
    structural_report: str,
    semantic_report: str,
    pending_review_count: int,
    insights: dict[str, Any],
) -> str:
    lines = [
        "# Quality Latest",
        "",
        f"Structural issues: {structural_issue_count}",
        f"Structural report: {structural_report}",
        f"Semantic report: {semantic_report}",
        f"Pending review items: {pending_review_count}",
        insights["summary"],
    ]
    if insights.get("report_path"):
        lines.append(f"Insights report: {insights['report_path']}")
    return "\n".join(lines) + "\n"


def _relative_to_kb(kb_dir: Path, path: Path) -> str:
    return path.relative_to(kb_dir).as_posix()
