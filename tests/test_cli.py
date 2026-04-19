import json
import inspect
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from openkb.cli import _normalize_background_insights_cooldown, _run_async_entrypoint, cli
from openkb.config import DEFAULT_CONFIG
from openkb.frontmatter import parse_fm
from openkb.review import ReviewItem, ReviewQueue
from openkb.schema import AGENTS_MD


def test_init_creates_structure(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        from pathlib import Path
        cwd = Path(".")

        # Directories
        assert (cwd / "raw").is_dir()
        assert (cwd / "wiki" / "sources" / "images").is_dir()
        assert (cwd / "wiki" / "summaries").is_dir()
        assert (cwd / "wiki" / "concepts").is_dir()
        assert (cwd / ".openkb").is_dir()

        # Files
        assert (cwd / "wiki" / "AGENTS.md").is_file()
        assert (cwd / "wiki" / "log.md").is_file()
        assert (cwd / "wiki" / "index.md").is_file()
        assert (cwd / ".openkb" / "config.yaml").is_file()
        assert (cwd / ".openkb" / "hashes.json").is_file()

        # hashes.json is empty object
        hashes = json.loads((cwd / ".openkb" / "hashes.json").read_text())
        assert hashes == {}

        # index.md header
        index_content = (cwd / "wiki" / "index.md").read_text()
        assert index_content == "# Knowledge Base Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n"


def test_init_schema_content(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        from pathlib import Path
        agents_content = Path("wiki/AGENTS.md").read_text()
        assert agents_content == AGENTS_MD


def test_init_already_exists(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        # First run should succeed
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        # Second run should print already initialized message
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "already initialized" in result.output


def test_init_writes_background_insights_cooldown_default(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        from pathlib import Path
        import yaml

        config = yaml.safe_load(Path(".openkb/config.yaml").read_text())
        assert (
            config["background_insights_cooldown_seconds"]
            == DEFAULT_CONFIG["background_insights_cooldown_seconds"]
        )


def test_init_writes_insights_cooldown_default(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        from pathlib import Path
        import yaml

        config = yaml.safe_load(Path(".openkb/config.yaml").read_text())
        assert config["insights_cooldown"] == DEFAULT_CONFIG["insights_cooldown"]


def test_init_writes_provider_and_effort_defaults_without_env_file(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0

        from pathlib import Path
        import yaml

        config = yaml.safe_load(Path(".openkb/config.yaml").read_text())
        assert config["provider"] == DEFAULT_CONFIG["provider"]
        assert config["effort"] == DEFAULT_CONFIG["effort"]
        assert not Path(".env").exists()
        assert "LLM API Key" not in result.output


def test_add_triggers_background_insights_after_success(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text(
        "model: sonnet\nbackground_insights_cooldown_seconds: 45\n",
        encoding="utf-8",
    )
    doc = tmp_path / "paper.md"
    doc.write_text("# Paper", encoding="utf-8")

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file") as mock_add_single_file, \
         patch("openkb.cli.maybe_trigger_insights") as mock_trigger:
        result = runner.invoke(cli, ["add", str(doc)])

    assert result.exit_code == 0
    mock_add_single_file.assert_called_once_with(doc, kb_dir)
    mock_trigger.assert_called_once_with(kb_dir)


def test_add_prefers_insights_cooldown_alias_when_present(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text(
        "model: sonnet\ninsights_cooldown: 60\nbackground_insights_cooldown_seconds: 45\n",
        encoding="utf-8",
    )
    doc = tmp_path / "paper.md"
    doc.write_text("# Paper", encoding="utf-8")

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file", return_value=True), \
         patch("openkb.cli.maybe_trigger_insights") as mock_trigger:
        result = runner.invoke(cli, ["add", str(doc)])

    assert result.exit_code == 0
    mock_trigger.assert_called_once_with(kb_dir)


def test_add_normalizes_background_insights_cooldown_from_string_config(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text(
        "model: sonnet\nbackground_insights_cooldown_seconds: '45'\n",
        encoding="utf-8",
    )
    doc = tmp_path / "paper.md"
    doc.write_text("# Paper", encoding="utf-8")

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file", return_value=True), \
         patch("openkb.cli.maybe_trigger_insights") as mock_trigger:
        result = runner.invoke(cli, ["add", str(doc)])

    assert result.exit_code == 0
    mock_trigger.assert_called_once_with(kb_dir)


def test_add_falls_back_to_default_background_insights_cooldown_when_invalid(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text(
        "model: sonnet\nbackground_insights_cooldown_seconds: invalid\n",
        encoding="utf-8",
    )
    doc = tmp_path / "paper.md"
    doc.write_text("# Paper", encoding="utf-8")

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file", return_value=True), \
         patch("openkb.cli.maybe_trigger_insights") as mock_trigger:
        result = runner.invoke(cli, ["add", str(doc)])

    assert result.exit_code == 0
    mock_trigger.assert_called_once_with(kb_dir)


def test_add_does_not_trigger_background_insights_when_add_returns_false(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text(
        "model: sonnet\nbackground_insights_cooldown_seconds: 45\n",
        encoding="utf-8",
    )
    doc = tmp_path / "paper.md"
    doc.write_text("# Paper", encoding="utf-8")

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file", return_value=False), \
         patch("openkb.cli.maybe_trigger_insights") as mock_trigger:
        result = runner.invoke(cli, ["add", str(doc)])

    assert result.exit_code == 0
    mock_trigger.assert_not_called()


def test_add_uses_default_background_insights_cooldown_when_config_missing(tmp_path):
    kb_dir = tmp_path
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text(
        "model: sonnet\n",
        encoding="utf-8",
    )
    doc = tmp_path / "paper.md"
    doc.write_text("# Paper", encoding="utf-8")

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.cli.add_single_file"), \
         patch("openkb.cli.maybe_trigger_insights") as mock_trigger:
        result = runner.invoke(cli, ["add", str(doc)])

    assert result.exit_code == 0
    mock_trigger.assert_called_once_with(kb_dir)


def test_config_set_insights_cooldown(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        init_result = runner.invoke(cli, ["init"])
        assert init_result.exit_code == 0

        result = runner.invoke(cli, ["config", "set", "insights_cooldown", "60"])
        assert result.exit_code == 0

        from pathlib import Path
        import yaml

        config = yaml.safe_load(Path(".openkb/config.yaml").read_text())
        assert config["insights_cooldown"] == 60


def test_config_get_insights_cooldown(tmp_path):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path), \
         patch("openkb.cli.register_kb"):
        init_result = runner.invoke(cli, ["init"])
        assert init_result.exit_code == 0

        result = runner.invoke(cli, ["config", "get", "insights_cooldown"])
        assert result.exit_code == 0
        assert str(DEFAULT_CONFIG["insights_cooldown"]) in result.output


def test_add_sources_uses_parallel_workers(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / ".openkb").mkdir()

    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "one.md").write_text(
        "---\nurl: https://example.com/one\nsource_type: article\n---\n",
        encoding="utf-8",
    )
    (source_dir / "two.md").write_text(
        "---\nurl: https://example.com/two\nsource_type: article\n---\n",
        encoding="utf-8",
    )

    def _fake_fetch(url: str):
        slug = url.rsplit("/", 1)[-1]
        return f"# {slug}\n", slug

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.url_fetch.is_url", return_value=True), \
         patch("openkb.url_fetch.fetch_url", side_effect=_fake_fetch), \
         patch("openkb.cli.add_single_file", return_value=True) as mock_add_single_file:
        result = runner.invoke(
            cli,
            ["add-sources", str(source_dir), "--concurrency", "2"],
        )

    assert result.exit_code == 0
    assert "Found 2 URLs to add." in result.output
    assert "Using 2 parallel workers." in result.output
    assert mock_add_single_file.call_count == 2
    assert (kb_dir / "raw" / "one.md").exists()
    assert (kb_dir / "raw" / "two.md").exists()


def test_add_sources_defaults_to_kb_local_sources_dir(tmp_path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / ".openkb").mkdir()
    source_dir = kb_dir / "sources"
    source_dir.mkdir()
    (source_dir / "one.md").write_text(
        "---\nurl: https://example.com/one\nsource_type: article\n---\n",
        encoding="utf-8",
    )

    def _fake_fetch(url: str):
        return "# one\n", "one"

    runner = CliRunner()
    with patch("openkb.cli._find_kb_dir", return_value=kb_dir), \
         patch("openkb.url_fetch.is_url", return_value=True), \
         patch("openkb.url_fetch.fetch_url", side_effect=_fake_fetch), \
         patch("openkb.cli.add_single_file", return_value=True) as mock_add_single_file:
        result = runner.invoke(cli, ["add-sources"])

    assert result.exit_code == 0
    assert "Found 1 URLs to add." in result.output
    mock_add_single_file.assert_called_once()


def test_run_async_entrypoint_closes_coroutine_when_asyncio_run_raises():
    async def _sample() -> None:
        return None

    coro = _sample()

    with patch("openkb.cli.asyncio.run", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError):
            _run_async_entrypoint(coro)

    assert inspect.getcoroutinestate(coro) == inspect.CORO_CLOSED


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (45.9, 45),
        ("45.9", 45),
        (True, DEFAULT_CONFIG["background_insights_cooldown_seconds"]),
        (-1, DEFAULT_CONFIG["background_insights_cooldown_seconds"]),
        (-0.5, DEFAULT_CONFIG["background_insights_cooldown_seconds"]),
        (float("inf"), DEFAULT_CONFIG["background_insights_cooldown_seconds"]),
    ],
)
def test_normalize_background_insights_cooldown_policy(raw_value, expected):
    assert _normalize_background_insights_cooldown(raw_value) == expected


def _make_review_kb(tmp_path):
    from pathlib import Path

    kb_dir = tmp_path / "review-kb"
    kb_dir.mkdir()
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text("model: gpt-5.4-mini\n", encoding="utf-8")
    (kb_dir / "wiki" / "concepts").mkdir(parents=True)
    (kb_dir / "raw").mkdir()
    return Path(kb_dir)


def _make_promotion_kb(tmp_path):
    kb_dir = tmp_path / "promotion-kb"
    kb_dir.mkdir()
    (kb_dir / ".openkb").mkdir()
    (kb_dir / ".openkb" / "config.yaml").write_text("model: gpt-5.4-mini\n", encoding="utf-8")
    (kb_dir / "wiki" / "explorations").mkdir(parents=True)
    (kb_dir / "wiki" / "queries").mkdir(parents=True)
    (kb_dir / "wiki" / "concepts").mkdir(parents=True)
    return kb_dir


def test_review_apply_mutates_wiki_and_consumes_item(tmp_path):
    from openkb.review import ReviewItem, ReviewQueue

    kb_dir = _make_review_kb(tmp_path)
    queue = ReviewQueue(kb_dir / ".openkb")
    queue.add([
        ReviewItem(
            type="missing_page",
            title="Create Applied Topic",
            description="Create a placeholder page from review",
            source_path="summaries/applied.md",
            action_type="create_placeholder",
            payload={"path": "concepts/applied-topic.md", "title": "Applied Topic"},
        )
    ])

    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review", "--apply", "0"])

    assert result.exit_code == 0
    assert "Applied:" in result.output
    target = kb_dir / "wiki" / "concepts" / "applied-topic.md"
    assert target.exists()
    assert "# Applied Topic" in target.read_text(encoding="utf-8")
    assert ReviewQueue(kb_dir / ".openkb").list() == []


def test_review_accept_does_not_mutate_wiki(tmp_path):
    from openkb.review import ReviewItem, ReviewQueue

    kb_dir = _make_review_kb(tmp_path)
    queue = ReviewQueue(kb_dir / ".openkb")
    queue.add([
        ReviewItem(
            type="missing_page",
            title="Accept Only Topic",
            description="Accept without changing wiki",
            source_path="summaries/accepted.md",
            action_type="create_placeholder",
            payload={"path": "concepts/accepted-topic.md", "title": "Accepted Topic"},
        )
    ])

    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review", "--accept", "0"])

    assert result.exit_code == 0
    assert "Accepted" in result.output
    assert not (kb_dir / "wiki" / "concepts" / "accepted-topic.md").exists()
    assert ReviewQueue(kb_dir / ".openkb").list() == []


def test_review_apply_failure_surfaces_error_and_keeps_pending_item(tmp_path):
    from openkb.review import ReviewItem, ReviewQueue

    kb_dir = _make_review_kb(tmp_path)
    queue = ReviewQueue(kb_dir / ".openkb")
    queue.add([
        ReviewItem(
            type="suggestion",
            title="Missing stale target",
            description="Try to mark a missing file as stale",
            source_path="summaries/stale.md",
            action_type="mark_stale",
            payload={"path": "concepts/missing-topic.md", "reason": "Needs refresh"},
        )
    ])

    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "review", "--apply", "0"])

    assert result.exit_code == 0
    assert "Failed to apply review item" in result.output
    remaining = ReviewQueue(kb_dir / ".openkb").list()
    assert len(remaining) == 1
    assert remaining[0].status == "pending"


def test_promote_cli_creates_query_page(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
    source = kb_dir / "wiki" / "explorations" / "attention.md"
    source.write_text(
        (
            "---\n"
            'query: "What is attention?"\n'
            "---\n\n"
            "# Saved Answer\n\n"
            "Attention is a mechanism.\n"
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--kb-dir", str(kb_dir), "promote", "wiki/explorations/attention.md", "--mode", "query_page"],
    )

    assert result.exit_code == 0
    assert "Promoted exploration to query page" in result.output
    target = kb_dir / "wiki" / "queries" / "attention.md"
    meta, body = parse_fm(target.read_text(encoding="utf-8"))
    assert meta["type"] == "wiki-query"
    assert meta["promoted_from"] == "wiki/explorations/attention.md"
    assert "Attention is a mechanism." in body


def test_promote_cli_enqueues_concept_seed_review_item(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
    source = kb_dir / "wiki" / "explorations" / "context-window.md"
    source.write_text("# Context Window\n\nExploration body.\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--kb-dir", str(kb_dir), "promote", "explorations/context-window.md", "--mode", "concept_seed"],
    )

    assert result.exit_code == 0
    assert "Queued concept seed review item" in result.output
    items = ReviewQueue(kb_dir / ".openkb").list()
    assert len(items) == 1
    assert items[0].payload["path"] == "concepts/context-window.md"


def test_promote_cli_rejects_missing_exploration(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
    runner = CliRunner()

    result = runner.invoke(
        cli,
        ["--kb-dir", str(kb_dir), "promote", "explorations/missing.md", "--mode", "query_page"],
    )

    assert result.exit_code == 0
    assert "[ERROR] Promotion failed:" in result.output


def test_quality_cli_prints_convergence_summary(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
    runner = CliRunner()

    with patch(
        "openkb.quality_loop.run_quality_convergence",
        return_value={
            "structural_issue_count": 3,
            "structural_report": "wiki/reports/structural_latest.md",
            "semantic_report": "wiki/reports/semantic_latest.md",
            "quality_report": "wiki/reports/quality_latest.md",
            "pending_review_count": 2,
            "insights": {
                "status": "ready",
                "summary": "Last insights run: 2026-04-19T10:00:00Z",
                "last_run": "2026-04-19T10:00:00Z",
                "report_path": ".openkb/insights.md",
            },
        },
    ):
        result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "quality"])

    assert result.exit_code == 0
    assert "Structural issues: 3" in result.output
    assert "Semantic report: wiki/reports/semantic_latest.md" in result.output
    assert "Pending review items: 2" in result.output
    assert "Last insights run: 2026-04-19T10:00:00Z" in result.output
    assert "Quality report: wiki/reports/quality_latest.md" in result.output


def test_refresh_plan_cli_lists_stale_pages(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
    (kb_dir / "sources").mkdir(parents=True, exist_ok=True)
    source = kb_dir / "sources" / "paper.md"
    source.write_text("# Paper\n", encoding="utf-8")
    concept = kb_dir / "wiki" / "concepts" / "attention.md"
    concept.write_text(
        (
            "---\n"
            "updated_at: 2020-01-01T00:00:00Z\n"
            "supporting_sources:\n"
            "  - sources/paper.md\n"
            "---\n\n"
            "# Attention\n"
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "refresh", "--plan"])

    assert result.exit_code == 0
    assert "1 stale page(s) found" in result.output
    assert "concepts/attention.md" in result.output
    assert "supporting source newer than page: sources/paper.md" in result.output


def test_refresh_plan_cli_lists_missing_supporting_source(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
    concept = kb_dir / "wiki" / "concepts" / "attention.md"
    concept.write_text(
        (
            "---\n"
            "updated_at: 2020-01-01T00:00:00Z\n"
            "supporting_sources:\n"
            "  - sources/missing.md\n"
            "---\n\n"
            "# Attention\n"
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "refresh", "--plan"])

    assert result.exit_code == 0
    assert "supporting source missing: sources/missing.md" in result.output


def test_refresh_plan_cli_lists_pending_mark_stale_item(tmp_path):
    kb_dir = _make_promotion_kb(tmp_path)
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

    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(kb_dir), "refresh", "--plan"])

    assert result.exit_code == 0
    assert "concepts/attention.md" in result.output
    assert "pending mark_stale review item: Needs refresh" in result.output
