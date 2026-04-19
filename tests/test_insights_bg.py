from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import networkx as nx

from openkb.graph.insights_bg import _bg_insights, inspect_background_insights_state, maybe_trigger_insights


def _make_kb(tmp_path: Path) -> Path:
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / "wiki").mkdir()
    return kb_dir


def _insight_result() -> dict:
    return {
        "orphans": [("concepts/orphan", 0)],
        "sparse_communities": [],
        "bridge_nodes": [],
        "surprising_connections": [("a", "b", "cross-community", 0.8)],
        "communities_summary": {
            0: {
                "size": 3,
                "cohesion": 0.5,
                "representative": "doc2",
            }
        },
    }


def test_first_run_triggers_insights(tmp_path):
    kb_dir = _make_kb(tmp_path)
    thread = Mock()

    with patch("openkb.graph.insights_bg._bg_insights") as mock_bg, \
         patch("openkb.graph.insights_bg.threading.Thread", return_value=thread) as mock_thread, \
         patch("openkb.graph.insights_bg.atexit.register") as mock_register:
        result = maybe_trigger_insights(kb_dir)

    assert result == "triggered"
    mock_thread.assert_called_once_with(target=mock_bg, args=(kb_dir,), daemon=True)
    thread.start.assert_called_once_with()
    mock_register.assert_called_once()


def test_cooldown_not_expired_skips(tmp_path):
    kb_dir = _make_kb(tmp_path)
    state_path = kb_dir / ".openkb" / "last_insights.json"
    state_path.write_text(
        json.dumps({"last_run": "2026-04-19T10:30:00Z", "cooldown_seconds": 3600}),
        encoding="utf-8",
    )

    with patch("openkb.graph.insights_bg._utc_now_timestamp", return_value=1_744_500_001.0), \
         patch("openkb.graph.insights_bg._bg_insights") as mock_bg:
        result = maybe_trigger_insights(kb_dir)

    assert result == "cached"
    mock_bg.assert_not_called()


def test_cooldown_expired_triggers(tmp_path):
    kb_dir = _make_kb(tmp_path)
    state_path = kb_dir / ".openkb" / "last_insights.json"
    state_path.write_text(
        json.dumps({"last_run": "2020-01-01T00:00:00Z", "cooldown_seconds": 3600}),
        encoding="utf-8",
    )
    thread = Mock()

    with patch("openkb.graph.insights_bg.threading.Thread", return_value=thread) as mock_thread:
        result = maybe_trigger_insights(kb_dir)

    assert result == "triggered"
    mock_thread.assert_called_once()
    thread.start.assert_called_once_with()


def test_generates_insights_md(tmp_path):
    kb_dir = _make_kb(tmp_path)
    openkb_dir = kb_dir / ".openkb"
    wiki_dir = kb_dir / "wiki"

    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 5
    mock_graph.number_of_edges.return_value = 3

    with patch("openkb.graph.insights_bg.build_and_save_graph", return_value=(mock_graph, openkb_dir / "graph.json")), \
         patch("openkb.graph.insights_bg.generate_insights", return_value=_insight_result()), \
         patch("openkb.graph.insights_bg.click.echo"):
        _bg_insights(kb_dir)

    insights_path = openkb_dir / "insights.md"
    assert insights_path.exists()
    content = insights_path.read_text(encoding="utf-8")
    assert "Communities" in content or "communities" in content.lower()


def test_updates_last_insights_json(tmp_path):
    kb_dir = _make_kb(tmp_path)
    openkb_dir = kb_dir / ".openkb"

    mock_graph = MagicMock()
    mock_graph.number_of_nodes.return_value = 2
    mock_graph.number_of_edges.return_value = 1

    with patch("openkb.graph.insights_bg.build_and_save_graph", return_value=(mock_graph, openkb_dir / "graph.json")), \
         patch("openkb.graph.insights_bg.generate_insights", return_value=_insight_result()), \
         patch("openkb.graph.insights_bg.click.echo"):
        _bg_insights(kb_dir)

    state_path = openkb_dir / "last_insights.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert "last_run" in state
    assert "cooldown_seconds" in state


def test_graph_load_error_handled(tmp_path):
    from openkb.graph.build import GraphLoadError

    kb_dir = _make_kb(tmp_path)

    with patch("openkb.graph.insights_bg.build_and_save_graph", side_effect=GraphLoadError("corrupted")), \
         patch("openkb.graph.insights_bg.click.echo"):
        _bg_insights(kb_dir)


def test_inspect_background_insights_state_reports_missing(tmp_path):
    kb_dir = _make_kb(tmp_path)

    result = inspect_background_insights_state(kb_dir)

    assert result["status"] == "missing"
    assert result["summary"] == "Last insights run: missing"
    assert result["report_path"] is None


def test_inspect_background_insights_state_reports_unreadable(tmp_path):
    kb_dir = _make_kb(tmp_path)
    state_path = kb_dir / ".openkb" / "last_insights.json"
    state_path.write_text("{invalid", encoding="utf-8")

    result = inspect_background_insights_state(kb_dir)

    assert result["status"] == "unreadable"
    assert "unreadable" in result["summary"]


def test_inspect_background_insights_state_reports_ready(tmp_path):
    kb_dir = _make_kb(tmp_path)
    state_path = kb_dir / ".openkb" / "last_insights.json"
    state_path.write_text(
        json.dumps({"last_run": "2026-04-19T10:30:00Z", "cooldown_seconds": 3600}),
        encoding="utf-8",
    )
    report_path = kb_dir / ".openkb" / "insights.md"
    report_path.write_text("# Insights\n", encoding="utf-8")

    result = inspect_background_insights_state(kb_dir)

    assert result["status"] == "ready"
    assert result["last_run"] == "2026-04-19T10:30:00Z"
    assert result["report_path"] == ".openkb/insights.md"
