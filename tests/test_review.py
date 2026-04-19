"""Tests for openkb.review — ReviewItem, parse_review_blocks, ReviewQueue."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from openkb.review import ReviewItem, parse_review_blocks, ReviewQueue


# ---------------------------------------------------------------------------
# ReviewItem
# ---------------------------------------------------------------------------


class TestReviewItem:
    def test_construction_required_fields(self):
        item = ReviewItem(
            type="contradiction",
            title="Conflict in definitions",
            description="Two pages define X differently",
            source_path="summaries/doc.md",
        )
        assert item.type == "contradiction"
        assert item.title == "Conflict in definitions"
        assert item.description == "Two pages define X differently"
        assert item.source_path == "summaries/doc.md"
        assert item.affected_pages == []
        assert item.search_queries == []
        assert item.options == []

    def test_construction_all_fields(self):
        item = ReviewItem(
            type="missing_page",
            title="Missing page for Y",
            description="Concept Y has no page yet",
            source_path="summaries/doc2.md",
            affected_pages=["concepts/y.md"],
            search_queries=["Y definition"],
            options=[{"action": "create", "slug": "y"}],
        )
        assert item.affected_pages == ["concepts/y.md"]
        assert item.search_queries == ["Y definition"]
        assert len(item.options) == 1

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            ReviewItem(type="invalid_type", title="T", description="D", source_path="s.md")

    def test_to_dict_roundtrip(self):
        item = ReviewItem(
            type="duplicate",
            title="Duplicate concept",
            description="Same as existing",
            source_path="summaries/a.md",
            affected_pages=["concepts/b.md"],
            search_queries=["b concept"],
            options=[{"action": "merge"}],
        )
        d = item.to_dict()
        assert d["type"] == "duplicate"
        assert d["options"] == [{"action": "merge"}]

        restored = ReviewItem.from_dict(d)
        assert restored.type == item.type
        assert restored.title == item.title
        assert restored.affected_pages == item.affected_pages
        assert restored.search_queries == item.search_queries
        assert restored.options == item.options

    def test_actionable_fields_roundtrip(self):
        item = ReviewItem(
            type="missing_page",
            title="Missing Graph page",
            description="Graph concept needs a placeholder",
            source_path="summaries/graph.md",
            action_type="create_placeholder",
            payload={"path": "concepts/graph.md", "title": "Graph"},
            status="pending",
        )

        data = item.to_dict()
        assert data["action_type"] == "create_placeholder"
        assert data["payload"] == {"path": "concepts/graph.md", "title": "Graph"}
        assert data["status"] == "pending"

        restored = ReviewItem.from_dict(data)
        assert restored.action_type == "create_placeholder"
        assert restored.payload == {"path": "concepts/graph.md", "title": "Graph"}
        assert restored.status == "pending"

    def test_from_dict_minimal(self):
        d = {"type": "suggestion", "title": "T", "description": "D", "source_path": "s.md"}
        item = ReviewItem.from_dict(d)
        assert item.type == "suggestion"
        assert item.affected_pages == []

    def test_from_dict_invalid_type_raises(self):
        d = {"type": "bad", "title": "T", "description": "D", "source_path": "s.md"}
        with pytest.raises(ValueError):
            ReviewItem.from_dict(d)

    def test_invalid_action_type_raises(self):
        with pytest.raises(ValueError):
            ReviewItem(
                type="suggestion",
                title="T",
                description="D",
                source_path="s.md",
                action_type="promote_exploration",
            )

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            ReviewItem(
                type="suggestion",
                title="T",
                description="D",
                source_path="s.md",
                status="done",
            )


# ---------------------------------------------------------------------------
# parse_review_blocks
# ---------------------------------------------------------------------------


class TestParseReviewBlocks:
    def test_single_block_with_items(self):
        text = (
            "Here is the analysis.\n"
            "---REVIEW---\n"
            '[{"type": "contradiction", "title": "Conflict", '
            '"description": "Desc", "source_path": "s.md"}]'
        )
        items = parse_review_blocks(text)
        assert len(items) == 1
        assert items[0].type == "contradiction"
        assert items[0].title == "Conflict"

    def test_no_review_block(self):
        text = "Just regular text, no review block here."
        items = parse_review_blocks(text)
        assert items == []

    def test_malformed_json_returns_empty(self):
        text = "---REVIEW---\nthis is not json"
        items = parse_review_blocks(text)
        assert items == []

    def test_multiple_items_in_block(self):
        items_data = [
            {"type": "contradiction", "title": "A", "description": "DA", "source_path": "a.md"},
            {"type": "missing_page", "title": "B", "description": "DB", "source_path": "b.md"},
        ]
        text = f"Analysis text\n---REVIEW---\n{json.dumps(items_data)}"
        items = parse_review_blocks(text)
        assert len(items) == 2
        assert items[0].title == "A"
        assert items[1].title == "B"

    def test_review_block_with_extra_text(self):
        """Review block content after the JSON array should be ignored."""
        text = (
            "---REVIEW---\n"
            '[{"type": "suggestion", "title": "S", "description": "DS", "source_path": "s.md"}]\n'
            "Some trailing text"
        )
        items = parse_review_blocks(text)
        assert len(items) == 1
        assert items[0].type == "suggestion"


# ---------------------------------------------------------------------------
# ReviewQueue
# ---------------------------------------------------------------------------


class TestReviewQueue:
    def test_save_and_load_roundtrip(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        item = ReviewItem(
            type="confirm",
            title="Verify X",
            description="Please verify",
            source_path="summaries/x.md",
        )
        q.add([item])
        # Reload from disk
        q2 = ReviewQueue(openkb_dir)
        loaded = q2.list()
        assert len(loaded) == 1
        assert loaded[0].title == "Verify X"

    def test_list_returns_insertion_order(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(type="confirm", title="First", description="D", source_path="a.md"),
            ReviewItem(type="confirm", title="Second", description="D", source_path="b.md"),
        ])
        items = q.list()
        assert len(items) == 2
        assert items[0].title == "First"
        assert items[1].title == "Second"

    def test_accept_removes_and_returns_item(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(type="confirm", title="A", description="D", source_path="a.md"),
            ReviewItem(type="confirm", title="B", description="D", source_path="b.md"),
        ])
        accepted = q.accept(0)
        assert accepted.title == "A"
        assert len(q.list()) == 1
        assert q.list()[0].title == "B"
        assert accepted.status == "accepted"

    def test_skip_removes_without_return(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(type="confirm", title="A", description="D", source_path="a.md"),
            ReviewItem(type="confirm", title="B", description="D", source_path="b.md"),
        ])
        result = q.skip(0)
        assert result is None
        assert len(q.list()) == 1
        assert q.list()[0].title == "B"

    def test_save_empty_queue(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        q.save()
        queue_file = openkb_dir / "review_queue.json"
        assert queue_file.exists()
        data = json.loads(queue_file.read_text())
        assert data == []

    def test_load_from_preexisting_file(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        queue_file = openkb_dir / "review_queue.json"
        items_data = [
            {"type": "confirm", "title": "Pre", "description": "D", "source_path": "p.md"},
        ]
        queue_file.write_text(json.dumps(items_data), encoding="utf-8")
        q = ReviewQueue(openkb_dir)
        loaded = q.list()
        assert len(loaded) == 1
        assert loaded[0].title == "Pre"

    def test_accept_out_of_range_raises(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        with pytest.raises(IndexError):
            q.accept(0)

    def test_skip_out_of_range_raises(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        with pytest.raises(IndexError):
            q.skip(0)

    def test_load_no_file_returns_empty(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        assert q.list() == []

    def test_queue_roundtrip_preserves_action_metadata_and_status(self, tmp_path):
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(
                type="missing_page",
                title="Queue placeholder",
                description="Needs a placeholder page",
                source_path="summaries/source.md",
                action_type="create_placeholder",
                payload={"path": "concepts/queue-placeholder.md", "title": "Queue Placeholder"},
                status="pending",
            )
        ])

        loaded = ReviewQueue(openkb_dir).list()
        assert len(loaded) == 1
        assert loaded[0].action_type == "create_placeholder"
        assert loaded[0].payload == {
            "path": "concepts/queue-placeholder.md",
            "title": "Queue Placeholder",
        }
        assert loaded[0].status == "pending"

    @pytest.mark.parametrize(
        ("action_type", "payload", "target_path", "expected_fragment", "seed_content"),
        [
            (
                "create_placeholder",
                {"path": "concepts/new-topic.md", "title": "New Topic"},
                "concepts/new-topic.md",
                "# New Topic",
                None,
            ),
            (
                "alias_concept",
                {
                    "path": "concepts/graph-alias.md",
                    "alias": "Graph Alias",
                    "target": "Graph Theory",
                },
                "concepts/graph-alias.md",
                "[[Graph Theory]]",
                None,
            ),
            (
                "mark_stale",
                {"path": "concepts/stale-topic.md", "reason": "Needs refresh"},
                "concepts/stale-topic.md",
                "Needs refresh",
                "# Existing Topic\n",
            ),
        ],
    )
    def test_apply_review_action_supports_stage2_action_types(
        self,
        tmp_path,
        action_type,
        payload,
        target_path,
        expected_fragment,
        seed_content,
    ):
        from openkb.review import apply_review_action

        kb_dir = tmp_path / "kb"
        target = kb_dir / "wiki" / target_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if seed_content is not None:
            target.write_text(seed_content, encoding="utf-8")

        item = ReviewItem(
            type="suggestion",
            title=f"{action_type} item",
            description="Apply the suggested wiki change",
            source_path="summaries/source.md",
            action_type=action_type,
            payload=payload,
        )

        changed_path = apply_review_action(kb_dir, item)

        assert changed_path == target
        assert target.exists()
        assert expected_fragment in target.read_text(encoding="utf-8")

    def test_apply_executes_action_and_removes_item_on_success(self, tmp_path):
        from openkb.review import apply_review_action

        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(
                type="missing_page",
                title="Applied placeholder",
                description="Create the missing page",
                source_path="summaries/source.md",
                action_type="create_placeholder",
                payload={"path": "concepts/applied.md", "title": "Applied Topic"},
            )
        ])

        applied = q.apply(0, lambda queued_item: apply_review_action(kb_dir, queued_item))

        assert applied.status == "applied"
        assert len(q.list()) == 0
        assert (kb_dir / "wiki" / "concepts" / "applied.md").exists()

    def test_apply_failure_keeps_pending_item(self, tmp_path):
        from openkb.review import apply_review_action

        kb_dir = tmp_path / "kb"
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir(parents=True)
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(
                type="suggestion",
                title="Missing stale page",
                description="Mark a missing page as stale",
                source_path="summaries/source.md",
                action_type="mark_stale",
                payload={"path": "concepts/missing.md", "reason": "Needs refresh"},
            )
        ])

        with pytest.raises(FileNotFoundError):
            q.apply(0, lambda queued_item: apply_review_action(kb_dir, queued_item))

        reloaded = ReviewQueue(openkb_dir).list()
        assert len(reloaded) == 1
        assert reloaded[0].title == "Missing stale page"
        assert reloaded[0].status == "pending"


# ---------------------------------------------------------------------------
# CLI review command
# ---------------------------------------------------------------------------


class TestCLIReview:
    """Tests for the 'openkb review' CLI subcommand."""

    def _make_kb(self, tmp_path):
        """Set up a minimal KB directory structure."""
        kb_dir = tmp_path / "kb"
        kb_dir.mkdir()
        openkb_dir = kb_dir / ".openkb"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
        (kb_dir / "wiki").mkdir()
        (kb_dir / "raw").mkdir()
        return kb_dir

    def test_review_no_items(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli

        kb_dir = self._make_kb(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review"])
        assert result.exit_code == 0
        assert "No pending review items" in result.output

    def test_review_lists_items(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli
        from openkb.review import ReviewQueue

        kb_dir = self._make_kb(tmp_path)
        openkb_dir = kb_dir / ".openkb"
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(type="contradiction", title="Conflict in X",
                        description="X is defined differently", source_path="summaries/a.md"),
        ])
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review"])
        assert result.exit_code == 0
        assert "Conflict in X" in result.output
        assert "contradiction" in result.output

    def test_review_accept(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli
        from openkb.review import ReviewQueue

        kb_dir = self._make_kb(tmp_path)
        openkb_dir = kb_dir / ".openkb"
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(type="confirm", title="Verify Y",
                        description="Please verify Y", source_path="summaries/b.md"),
        ])
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review", "--accept", "0"])
        assert result.exit_code == 0
        assert "Accepted" in result.output or "accepted" in result.output
        # Queue should be empty now — reload from disk
        q2 = ReviewQueue(openkb_dir)
        assert len(q2.list()) == 0

    def test_review_skip(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli
        from openkb.review import ReviewQueue

        kb_dir = self._make_kb(tmp_path)
        openkb_dir = kb_dir / ".openkb"
        q = ReviewQueue(openkb_dir)
        q.add([
            ReviewItem(type="suggestion", title="Improve Z",
                        description="Consider improving Z", source_path="summaries/c.md"),
        ])
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review", "--skip", "0"])
        assert result.exit_code == 0
        assert "Skipped" in result.output or "skipped" in result.output
        # Queue should be empty now — reload from disk
        q2 = ReviewQueue(openkb_dir)
        assert len(q2.list()) == 0

    def test_review_accept_out_of_range(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli

        kb_dir = self._make_kb(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review", "--accept", "5"])
        assert result.exit_code == 0
        assert "No pending review items" in result.output or "Invalid" in result.output

    def test_review_no_queue_file(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli

        kb_dir = self._make_kb(tmp_path)
        # Don't create any queue file — should show "No pending review items"
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review"])
        assert result.exit_code == 0
        assert "No pending review items" in result.output
