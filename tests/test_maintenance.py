"""Tests for internal KB maintenance helpers."""
from __future__ import annotations

from pathlib import Path

import yaml

from openkb.frontmatter import parse_fm
from openkb.maintenance import collect_structural_issues, run_internal_maintenance


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_kb(tmp_path: Path) -> Path:
    kb_dir = tmp_path / "kb"
    (kb_dir / "raw").mkdir(parents=True)
    (kb_dir / "sources").mkdir(parents=True)
    (kb_dir / "wiki" / "summaries").mkdir(parents=True)
    (kb_dir / "wiki" / "concepts").mkdir(parents=True)
    (kb_dir / "wiki" / "reports").mkdir(parents=True)
    (kb_dir / ".openkb").mkdir(parents=True)
    _write(kb_dir / ".openkb" / "config.yaml", "language: ko\nmodel: gpt-5.4-mini\n")
    _write(kb_dir / "wiki" / "index.md", "---\ntype: wiki-index\nupdated_at: 2026-04-19\ntopic_count: 0\n---\n\n# 위키 토픽 카탈로그\n")
    return kb_dir


class TestSourceRepair:
    def test_repairs_stale_source_links_and_frontmatter(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        actual_rel = "sources/2026/03/28/노정석-ai와-노동의-미래-100x-엔지니어와-1인-체제-실험.md"
        _write(
            kb_dir / actual_rel,
            "---\ntitle: Test Source\nurl: https://example.com/test\n---\n\n# Test Source\n",
        )
        _write(
            kb_dir / "wiki" / "100x-engineer.md",
            (
                "---\n"
                "type: wiki\n"
                "topic: 100x-engineer\n"
                "source_count: 1\n"
                "sources:\n"
                "  - sources/2026/03/28/노정석-AI-노동-미래-100x-엔지니어-1인-체제.md\n"
                "---\n\n"
                "# 100x Engineer\n\n"
                "- [[sources/2026/03/28/노정석-AI-노동-미래-100x-엔지니어-1인-체제.md]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, body = parse_fm((kb_dir / "wiki" / "100x-engineer.md").read_text(encoding="utf-8"))
        assert meta["sources"] == [actual_rel]
        assert meta["source_count"] == 1
        assert "[[sources/2026/03/28/노정석-ai와-노동의-미래-100x-엔지니어와-1인-체제-실험]]" in body

        issues = collect_structural_issues(kb_dir)
        assert issues["broken_source_links"] == []

    def test_canonical_source_links_are_idempotent(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        actual_rel = "sources/2026/03/28/canonical-source.md"
        _write(
            kb_dir / actual_rel,
            "---\ntitle: Canonical Source\nurl: https://example.com/canonical\n---\n\n# Canonical Source\n",
        )
        page = kb_dir / "wiki" / "canonical-wiki.md"
        _write(
            page,
            (
                "---\n"
                "type: wiki\n"
                "topic: canonical-wiki\n"
                "source_count: 1\n"
                "sources:\n"
                f"  - {actual_rel}\n"
                "---\n\n"
                "# Canonical Wiki\n\n"
                f"- [[{actual_rel[:-3]}]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)
        first = page.read_text(encoding="utf-8")
        run_internal_maintenance(kb_dir)
        second = page.read_text(encoding="utf-8")

        assert first == second

    def test_normalizes_absolute_source_links_inside_source_docs(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "sources" / "2026" / "03" / "28" / "child-source.md",
            "---\ntitle: Child\nurl: https://example.com/child\n---\n\n# Child\n",
        )
        source_doc = kb_dir / "sources" / "2026" / "03" / "29" / "parent-source.md"
        _write(
            source_doc,
            (
                "---\ntitle: Parent\nurl: https://example.com/parent\nrelated: []\n---\n\n"
                f"[Child]({kb_dir}/sources/2026/03/28/child-source.md)\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = source_doc.read_text(encoding="utf-8")
        assert "[[sources/2026/03/28/child-source|Child]]" in text

    def test_root_wiki_sources_can_use_query_hints(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / ".openkb" / "wiki_support_hints.yaml",
            (
                "wiki_support_hints:\n"
                "  macro-economics:\n"
                "    - wiki/queries/2026-04-17-gdp.md\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "queries" / "2026-04-17-gdp.md",
            "---\ntype: wiki-query\nquestion: GDP\n---\n\n# GDP\n",
        )
        page = kb_dir / "wiki" / "macro-economics.md"
        _write(
            page,
            (
                "---\n"
                "type: wiki\n"
                "topic: macro-economics\n"
                "source_count: 0\n"
                "sources: []\n"
                "---\n\n"
                "# Macro Economics\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, _ = parse_fm(page.read_text(encoding="utf-8"))
        assert meta["sources"] == ["wiki/queries/2026-04-17-gdp.md"]
        assert meta["source_count"] == 1

    def test_hints_override_bloated_existing_root_wiki_sources(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / ".openkb" / "wiki_support_hints.yaml",
            (
                "wiki_support_hints:\n"
                "  macro-economics:\n"
                "    - wiki/queries/2026-04-17-gdp.md\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "queries" / "2026-04-17-gdp.md",
            "---\ntype: wiki-query\nquestion: GDP\n---\n\n# GDP\n",
        )
        page = kb_dir / "wiki" / "macro-economics.md"
        _write(
            page,
            (
                "---\n"
                "type: wiki\n"
                "topic: macro-economics\n"
                "source_count: 2\n"
                "sources:\n"
                "  - sources/2026/01/01/noisy-a.md\n"
                "  - sources/2026/01/01/noisy-b.md\n"
                "---\n\n"
                "# Macro Economics\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, _ = parse_fm(page.read_text(encoding="utf-8"))
        assert meta["sources"] == ["wiki/queries/2026-04-17-gdp.md"]
        assert meta["source_count"] == 1

    def test_hints_merge_with_explicit_query_links(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / ".openkb" / "wiki_support_hints.yaml",
            (
                "wiki_support_hints:\n"
                "  stock-screening:\n"
                "    - sources/2026/04/01/source-a.md\n"
            ),
        )
        _write(
            kb_dir / "sources" / "2026" / "04" / "01" / "source-a.md",
            "---\nurl: https://example.com/a\n---\n\n# Source A\n",
        )
        page = kb_dir / "wiki" / "stock-screening.md"
        _write(
            page,
            (
                "---\n"
                "type: wiki\n"
                "topic: stock-screening\n"
                "source_count: 0\n"
                "sources: []\n"
                "---\n\n"
                "# Stock Screening\n\n"
                "[[wiki/queries/2026-04-17-per-pbr-roe]]\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "queries" / "2026-04-17-per-pbr-roe.md",
            "---\ntype: wiki-query\nquestion: per pbr roe\n---\n\n# Query\n",
        )

        run_internal_maintenance(kb_dir)

        meta, _ = parse_fm(page.read_text(encoding="utf-8"))
        assert meta["sources"] == [
            "wiki/queries/2026-04-17-per-pbr-roe.md",
            "sources/2026/04/01/source-a.md",
        ]
        assert meta["source_count"] == 2

    def test_root_wiki_repair_preserves_stage1_provenance_fields(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        actual_rel = "sources/2026/04/01/source-a.md"
        _write(
            kb_dir / actual_rel,
            "---\nurl: https://example.com/a\n---\n\n# Source A\n",
        )
        page = kb_dir / "wiki" / "product-strategy.md"
        _write(
            page,
            (
                "---\n"
                "type: wiki\n"
                "topic: product-strategy\n"
                "updated_at: 2026-04-19T04:16:18Z\n"
                "generation_mode: concept_update\n"
                "source_count: 1\n"
                "sources:\n"
                "  - sources/2026/04/01/source-a-stale.md\n"
                "supporting_sources:\n"
                "  - sources/2026/04/01/source-a-stale.md\n"
                "supporting_pages:\n"
                "  - summaries/source-a.md\n"
                "---\n\n"
                "# Product Strategy\n\n"
                "- [[sources/2026/04/01/source-a-stale]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, _ = parse_fm(page.read_text(encoding="utf-8"))
        assert meta["sources"] == [actual_rel]
        assert meta["supporting_sources"] == [actual_rel]
        assert meta["supporting_pages"] == ["summaries/source-a.md"]
        assert meta["source_count"] == 1
        assert meta["generation_mode"] == "concept_update"
        assert meta["updated_at"] == "2026-04-19T04:16:18Z"


class TestConceptRepair:
    def test_mixed_policy_canonicalizes_creates_and_plainifies(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "MCP.md",
            "---\nbrief: Existing MCP page.\nentity_type: concept\nsources: []\n---\n\n# MCP\n",
        )
        _write(
            kb_dir / "wiki" / "summaries" / "alpha.md",
            "# Alpha\n\nUses [[concepts/mcp]] and [[concepts/tool-calling]].\n",
        )
        _write(
            kb_dir / "wiki" / "summaries" / "beta.md",
            "# Beta\n\nAgain mentions [[concepts/tool-calling]] with [[concepts/mcp]].\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "context-management.md",
            "# Context Management\n\nOne-off [[concepts/ghost-only-once]] mention.\n",
        )

        run_internal_maintenance(kb_dir)

        alpha = (kb_dir / "wiki" / "summaries" / "alpha.md").read_text(encoding="utf-8")
        beta = (kb_dir / "wiki" / "summaries" / "beta.md").read_text(encoding="utf-8")
        context = (kb_dir / "wiki" / "concepts" / "context-management.md").read_text(encoding="utf-8")
        tool_calling = kb_dir / "wiki" / "concepts" / "tool-calling.md"

        assert "[[concepts/MCP]]" in alpha
        assert "[[concepts/MCP]]" in beta
        assert tool_calling.exists()
        assert "[[concepts/tool-calling]]" in alpha
        assert "[[concepts/tool-calling]]" in beta
        assert "[[concepts/ghost-only-once]]" not in context
        assert "ghost only once" in context.lower()

        issues = collect_structural_issues(kb_dir)
        assert issues["broken_concept_links"] == []

    def test_backfills_related_concepts_idempotently(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )
        _write(
            kb_dir / "wiki" / "summaries" / "paper.md",
            "# Paper\n\nTalks about [[concepts/attention]].\n",
        )

        run_internal_maintenance(kb_dir)
        run_internal_maintenance(kb_dir)

        text = (kb_dir / "wiki" / "summaries" / "paper.md").read_text(encoding="utf-8")
        assert text.count("## Related Concepts") == 1
        assert text.count("[[concepts/attention]]") >= 1

    def test_repeated_concept_links_across_sections_are_preserved(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "focus.md",
            (
                "---\nbrief: Focus.\nentity_type: concept\nsources: []\n---\n\n"
                "# Focus\n\n"
                "[[concepts/attention]]는 본문 설명에 등장한다.\n\n"
                "## 관련 개념\n\n"
                "- [[concepts/attention]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)
        run_internal_maintenance(kb_dir)

        text = (kb_dir / "wiki" / "concepts" / "focus.md").read_text(encoding="utf-8")
        assert text.count("[[concepts/attention]]") == 2

    def test_backfills_summary_related_concepts_from_source_topics(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )
        _write(
            kb_dir / "sources" / "2026" / "04" / "01" / "source-a.md",
            (
                "---\n"
                "url: https://example.com/a\n"
                "topics:\n"
                "  - name: attention\n"
                "    type: concept\n"
                "---\n\n# Source A\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "sources" / "summary-a.md",
            "---\nsource_url: https://example.com/a\n---\n\nsource\n",
        )
        _write(
            kb_dir / "wiki" / "summaries" / "summary-a.md",
            "---\ndoc_type: short\nfull_text: sources/summary-a.md\n---\n\n# Summary A\n\nNo explicit wikilinks.\n",
        )

        run_internal_maintenance(kb_dir)

        text = (kb_dir / "wiki" / "summaries" / "summary-a.md").read_text(encoding="utf-8")
        assert "## Related Concepts" in text
        assert "[[concepts/attention]]" in text

    def test_creates_concepts_from_repeated_summary_source_topics(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "product-strategy.md",
            (
                "---\n"
                "type: wiki\n"
                "topic: product-strategy\n"
                "source_count: 1\n"
                "sources:\n"
                "  - sources/2026/04/01/source-0.md\n"
                "---\n\n"
                "# Product Strategy\n\n"
                "Product strategy paragraph with enough detail for a concept brief.\n"
            ),
        )

        for idx in range(5):
            url = f"https://example.com/{idx}"
            _write(
                kb_dir / "sources" / "2026" / "04" / "01" / f"source-{idx}.md",
                (
                    "---\n"
                    f"url: {url}\n"
                    "topics:\n"
                    "  - name: product-strategy\n"
                    "    type: concept\n"
                    "---\n\n"
                    f"# Source {idx}\n"
                ),
            )
            _write(
                kb_dir / "wiki" / "sources" / f"summary-{idx}.md",
                f"---\nsource_url: {url}\n---\n\nsource\n",
            )
            _write(
                kb_dir / "wiki" / "summaries" / f"summary-{idx}.md",
                f"---\ndoc_type: short\nfull_text: sources/summary-{idx}.md\n---\n\n# Summary {idx}\n\nNo related concepts yet.\n",
            )

        run_internal_maintenance(kb_dir)

        concept = kb_dir / "wiki" / "concepts" / "product-strategy.md"
        assert concept.exists()
        summary_text = (kb_dir / "wiki" / "summaries" / "summary-0.md").read_text(encoding="utf-8")
        assert "[[concepts/product-strategy]]" in summary_text

    def test_backfills_related_concepts_from_connection_sections_and_inline_lists(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "context-harness.md",
            "---\nbrief: Context harness.\nentity_type: concept\nsources: []\n---\n\n# Context Harness\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "agentic-operations.md",
            "---\nbrief: Agentic operations.\nentity_type: concept\nsources: []\n---\n\n# Agentic Operations\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "deep-work.md",
            "---\nbrief: Deep work.\nentity_type: concept\nsources: []\n---\n\n# Deep Work\n",
        )
        _write(
            kb_dir / "wiki" / "summaries" / "paper.md",
            (
                "# Paper\n\n"
                "이 자료는 집중 설계를 다룬다.\n\n"
                "## 연결 개념\n"
                "- Context Harness\n"
                "- Agentic Operations\n\n"
                "관련 주제는 attention, deep work로 묶어 볼 수 있다.\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = (kb_dir / "wiki" / "summaries" / "paper.md").read_text(encoding="utf-8")
        assert "## Related Concepts" in text
        assert "[[concepts/context-harness]]" in text
        assert "[[concepts/agentic-operations]]" in text
        assert "[[concepts/attention]]" in text
        assert "[[concepts/deep-work]]" in text

    def test_creates_root_backed_concepts_from_single_summary_source_topic(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "technical-analysis.md",
            (
                "---\n"
                "type: wiki\n"
                "topic: technical-analysis\n"
                "source_count: 2\n"
                "sources:\n"
                "  - sources/2026/04/01/source-0.md\n"
                "  - sources/2026/04/01/source-1.md\n"
                "---\n\n"
                "# Technical Analysis\n\n"
                "Technical analysis paragraph with enough detail for a concept brief that should be shortened when promoted.\n"
            ),
        )
        _write(
            kb_dir / "sources" / "2026" / "04" / "01" / "source-0.md",
            (
                "---\n"
                "url: https://example.com/a\n"
                "topics:\n"
                "  - name: technical-analysis\n"
                "    type: concept\n"
                "---\n\n# Source 0\n"
            ),
        )
        _write(
            kb_dir / "sources" / "2026" / "04" / "01" / "source-1.md",
            "---\nurl: https://example.com/b\n---\n\n# Source 1\n",
        )
        _write(
            kb_dir / "wiki" / "sources" / "summary-a.md",
            "---\nsource_url: https://example.com/a\n---\n\nsource\n",
        )
        _write(
            kb_dir / "wiki" / "summaries" / "summary-a.md",
            "---\ndoc_type: short\nfull_text: sources/summary-a.md\n---\n\n# Summary A\n\nNo explicit wikilinks.\n",
        )

        run_internal_maintenance(kb_dir)

        concept = kb_dir / "wiki" / "concepts" / "technical-analysis.md"
        assert concept.exists()
        summary_text = (kb_dir / "wiki" / "summaries" / "summary-a.md").read_text(encoding="utf-8")
        assert "[[concepts/technical-analysis]]" in summary_text

    def test_normalizes_existing_root_backed_concept_metadata(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        concept = kb_dir / "wiki" / "concepts" / "ai.md"
        _write(
            concept,
            (
                "---\n"
                "brief: AI는 모델 자체보다 작업 분해와 운영 경계를 어떻게 설계하느냐를 길게 설명하는 매우 긴 개요 문장으로 적혀 있어서 frontmatter brief로는 지나치게 길다.\n"
                "entity_type: concept\n"
                "sources:\n"
                + "".join(f"  - sources/2026/04/01/source-{idx}.md\n" for idx in range(12))
                + "---\n\n"
                "# AI\n\n"
                "## 개요\n"
                "AI는 작업 분해, 상태 관리, 검증 루프를 어떻게 운영에 연결하느냐에 따라 성패가 갈린다. 이 문단은 brief 후보로 충분하다.\n\n"
                "## 관련 위키\n\n"
                "- [[wiki/ai]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, _ = parse_fm(concept.read_text(encoding="utf-8"))
        assert len(meta["brief"]) <= 120
        assert len(meta["sources"]) == 8

    def test_creates_summary_backed_concepts_from_explicit_related_phrase_lists(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        summary = kb_dir / "wiki" / "summaries" / "shop.md"
        _write(
            summary,
            (
                "# Shop\n\n"
                "## 관련 개념\n"
                "- AI Software Factory: 기업 내부에서 AI 기반 개발 흐름을 운영하는 방식\n"
                "- Tool Use\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        ai_factory = kb_dir / "wiki" / "concepts" / "ai-software-factory.md"
        tool_use = kb_dir / "wiki" / "concepts" / "tool-use.md"
        assert ai_factory.exists()
        assert tool_use.exists()

        text = summary.read_text(encoding="utf-8")
        assert "[[concepts/ai-software-factory]]" in text
        assert "[[concepts/tool-use]]" in text

    def test_extracts_additional_concept_heading_and_inline_patterns(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        summary = kb_dir / "wiki" / "summaries" / "work.md"
        _write(
            summary,
            (
                "# Work\n\n"
                "관련 주제로는 Terminal Productivity, Task Automation, Productivity Optimization이 연결될 수 있다.\n\n"
                "## 다룬 개념\n"
                "- 로컬 퍼스트: 서버 의존 없이 바로 쓰는 방식\n"
                "- 프라이버시 우선 도구: 데이터 수집 부담이 적은 도구\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = summary.read_text(encoding="utf-8")
        for slug in [
            "terminal-productivity",
            "task-automation",
            "productivity-optimization",
            "로컬-퍼스트",
            "프라이버시-우선-도구",
        ]:
            assert (kb_dir / "wiki" / "concepts" / f"{slug}.md").exists()
            assert f"[[concepts/{slug}]]" in text

    def test_extracts_last_resort_definition_sentences(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        summary = kb_dir / "wiki" / "summaries" / "duo.md"
        _write(
            summary,
            (
                "# Duo\n\n"
                "글의 핵심은 AI 제품 개발과 소규모 팀 생산성이다.\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = summary.read_text(encoding="utf-8")
        for slug in ["AI-제품-개발", "소규모-팀-생산성"]:
            assert (kb_dir / "wiki" / "concepts" / f"{slug}.md").exists()
            assert f"[[concepts/{slug}]]" in text

    def test_extracts_last_resort_bullet_labels_inline_english_and_line_end_tags(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "MCP.md",
            "---\nbrief: MCP.\nentity_type: concept\nsources: []\n---\n\n# MCP\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "의사결정.md",
            "---\nbrief: 의사결정.\nentity_type: concept\nsources: []\n---\n\n# 의사결정\n",
        )
        summary = kb_dir / "wiki" / "summaries" / "lead.md"
        _write(
            summary,
            (
                "# Lead\n\n"
                "- **MCP 서버 제공**: Claude 같은 외부 AI 도구와의 연결을 암시한다.\n"
                "- 시니어 엔지니어의 다음 단계는 System Design과 연결된 역할이다.\n"
                "- 리드 역할은 기술을 많이 쓰는 것보다 상황을 정리하고 선택하는 Decision Making 능력에 가깝다.\n"
                "- 이 관점은 Engineering Leadership의 역할 정의와 맞닿아 있다.\n"
                "- 어떤 워크플로를 에이전트가 실제로 사용할 수 있는지 쉽게 확인하게 해 준다. Project Isolation\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = summary.read_text(encoding="utf-8")
        assert "[[concepts/MCP]]" in text
        assert "[[concepts/의사결정]]" in text
        for slug in ["system-design", "engineering-leadership", "project-isolation"]:
            assert (kb_dir / "wiki" / "concepts" / f"{slug}.md").exists()
            assert f"[[concepts/{slug}]]" in text

    def test_last_resort_extraction_ignores_author_metadata_names(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        summary = kb_dir / "wiki" / "summaries" / "author.md"
        _write(
            summary,
            (
                "# Author\n\n"
                "- 작성자: @harumak_11 (Haruki Yano / Haruma-K)\n"
                "- 작성일: 2026-04-17\n"
                "- 이 관점은 Engineering Leadership의 역할 정의와 맞닿아 있다.\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = summary.read_text(encoding="utf-8")
        assert "[[concepts/engineering-leadership]]" in text
        assert not (kb_dir / "wiki" / "concepts" / "haruki-yano.md").exists()

    def test_concept_curation_aliases_prevent_and_merge_noisy_summary_backed_concepts(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / ".openkb" / "concept_curation.yaml",
            (
                "concept_curation:\n"
                "  aliases:\n"
                "    noisy-metadata-label: clean-concept\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "concepts" / "clean-concept.md",
            (
                "---\n"
                "brief: Clean concept.\n"
                "entity_type: concept\n"
                "sources: []\n"
                "---\n\n"
                "# Clean Concept\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "concepts" / "noisy-metadata-label.md",
            (
                "---\n"
                "brief: Noisy concept.\n"
                "entity_type: concept\n"
                "sources:\n"
                "  - summaries/sample.md\n"
                "---\n\n"
                "# Noisy Metadata Label\n\n"
                "## Related Documents\n\n"
                "- [[summaries/sample]]\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "summaries" / "sample.md",
            (
                "# Sample\n\n"
                "## Related Concepts\n\n"
                "- [[concepts/noisy-metadata-label]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        summary_text = (kb_dir / "wiki" / "summaries" / "sample.md").read_text(encoding="utf-8")
        assert "[[concepts/clean-concept]]" in summary_text
        assert "[[concepts/noisy-metadata-label]]" not in summary_text
        assert not (kb_dir / "wiki" / "concepts" / "noisy-metadata-label.md").exists()

        merged_meta, merged_body = parse_fm((kb_dir / "wiki" / "concepts" / "clean-concept.md").read_text(encoding="utf-8"))
        assert "summaries/sample.md" in merged_meta["sources"]
        assert "- [[summaries/sample]]" in merged_body
        assert "- - [[summaries/sample]]" not in merged_body
        assert merged_body.count("## Related Documents") == 1

    def test_creates_repeated_related_section_concepts_and_rewrites_links(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "summaries" / "alpha.md",
            (
                "# Alpha\n\n"
                "Alpha explains why prompt engineering matters.\n\n"
                "## Related Concepts\n\n"
                "- 프롬프트 엔지니어링\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "summaries" / "beta.md",
            (
                "# Beta\n\n"
                "Beta also recommends better prompt engineering habits.\n\n"
                "## Related Concepts\n\n"
                "- 프롬프트 엔지니어링\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        concept = kb_dir / "wiki" / "concepts" / "프롬프트-엔지니어링.md"
        assert concept.exists()
        alpha_text = (kb_dir / "wiki" / "summaries" / "alpha.md").read_text(encoding="utf-8")
        beta_text = (kb_dir / "wiki" / "summaries" / "beta.md").read_text(encoding="utf-8")
        assert "[[concepts/프롬프트-엔지니어링]]" in alpha_text
        assert "[[concepts/프롬프트-엔지니어링]]" in beta_text

    def test_moves_summary_links_out_of_related_concepts(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        concept = kb_dir / "wiki" / "concepts" / "source-metadata.md"
        _write(
            concept,
            (
                "---\nbrief: Meta concept.\nentity_type: concept\nsources: []\n---\n\n"
                "# Source Metadata\n\n"
                "## 관련 개념\n\n"
                "- [[summaries/sample]]\n"
                "- [[concepts/attention]]\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )

        run_internal_maintenance(kb_dir)

        text = concept.read_text(encoding="utf-8")
        related_match = "## 관련 개념\n\n- [[concepts/attention]]"
        assert related_match in text
        assert "[[summaries/sample]]" not in text.split("## 관련 개념", 1)[1].split("##", 1)[0]
        assert "## Related Documents" in text
        assert "- [[summaries/sample]]" in text

    def test_applies_curated_brief_overrides(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / ".openkb" / "concept_curation.yaml",
            (
                "concept_curation:\n"
                "  brief_overrides:\n"
                "    source-metadata: 링크형 자료의 메타데이터 상태를 기록하는 개념이다.\n"
            ),
        )
        concept = kb_dir / "wiki" / "concepts" / "source-metadata.md"
        _write(
            concept,
            (
                "---\nbrief: 매우 긴 기존 설명.\nentity_type: concept\nsources: []\n---\n\n"
                "# Source Metadata\n\n"
                "## 개요\n"
                "매우 긴 기존 설명.\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, body = parse_fm(concept.read_text(encoding="utf-8"))
        assert meta["brief"] == "링크형 자료의 메타데이터 상태를 기록하는 개념이다."
        assert "## 개요\n링크형 자료의 메타데이터 상태를 기록하는 개념이다." in body

    def test_collapses_duplicate_related_concepts_sections(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )
        summary = kb_dir / "wiki" / "summaries" / "dup.md"
        _write(
            summary,
            (
                "# Dup\n\n"
                "## Related Concepts\n\n"
                "- [[concepts/attention]]\n\n"
                "## Related Concepts\n\n"
                "- [[concepts/attention]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        text = summary.read_text(encoding="utf-8")
        assert text.count("## Related Concepts") == 1

    def test_refreshes_summary_backed_concepts_with_contextual_brief(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "summaries" / "duo.md",
            (
                "# Duo\n\n"
                "글의 핵심은 AI 제품 개발과 소규모 팀 생산성이다.\n\n"
                "## 핵심 내용\n"
                "- AI를 활용하면 적은 인원과 짧은 기간으로도 영향력 있는 제품을 만들 수 있다.\n"
            ),
        )
        concept = kb_dir / "wiki" / "concepts" / "AI-제품-개발.md"
        _write(
            concept,
            (
                "---\n"
                "brief: AI 제품 개발는 내부 summary가 관련 개념으로 직접 묶어 둔 주제다.\n"
                "entity_type: concept\n"
                "sources:\n"
                "  - summaries/duo.md\n"
                "---\n\n"
                "# AI 제품 개발\n\n"
                "## 개요\n"
                "AI 제품 개발는 내부 summary가 관련 개념으로 직접 묶어 둔 주제다. 이 페이지는 source 원문이 아니라 현재 KB 요약 문맥만 기준으로 정리했다.\n\n"
                "## Related Documents\n\n"
                "- [[summaries/duo]]\n"
            ),
        )

        run_internal_maintenance(kb_dir)

        meta, body = parse_fm(concept.read_text(encoding="utf-8"))
        assert "내부 summary가 관련 개념으로 직접 묶어 둔 주제" not in meta["brief"]
        assert not meta["brief"].startswith("원문:")
        assert "## 관찰된 문맥" in body
        assert "적은 인원과 짧은 기간" in body

    def test_refresh_replaces_metadata_context_and_sanitizes_existing_bullets(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "summaries" / "lead.md",
            (
                "# Lead\n\n"
                "- 성장의 기준은 코드 생산량이나 처리 속도가 아니다.\n"
                "- 시니어 엔지니어의 다음 단계는 System Design과 연결된 역할이다.\n"
                "- 리드 역할은 기술을 많이 쓰는 것보다 상황을 정리하고 선택하는 Decision Making 능력에 가깝다.\n"
            ),
        )
        concept = kb_dir / "wiki" / "concepts" / "engineering-leadership.md"
        _write(
            concept,
            (
                "---\n"
                "brief: 성장의 기준은 코드 생산량이나 처리 속도가 아니다.\n"
                "entity_type: concept\n"
                "sources:\n"
                "  - summaries/lead.md\n"
                "---\n\n"
                "# Engineering Leadership\n\n"
                "## 개요\n"
                "성장의 기준은 코드 생산량이나 처리 속도가 아니다.\n\n"
                "## 관찰된 문맥\n\n"
                "- 원문: https://example.com/post\n"
                "- 작성자: Example Writer\n\n"
                "## Related Documents\n\n"
                "- [[summaries/lead]]\n"
                "- [[summaries/lead]]\n\n"
                "## 관련 개념\n\n"
                "- [[concepts/engineering-leadership]]\n"
                "- [[concepts/system-design]]\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "concepts" / "system-design.md",
            "---\nbrief: System Design.\nentity_type: concept\nsources: []\n---\n\n# System Design\n",
        )
        _write(
            kb_dir / "wiki" / "concepts" / "의사결정.md",
            "---\nbrief: 의사결정.\nentity_type: concept\nsources: []\n---\n\n# 의사결정\n",
        )

        run_internal_maintenance(kb_dir)

        meta, body = parse_fm(concept.read_text(encoding="utf-8"))
        assert meta["brief"] == "성장의 기준은 코드 생산량이나 처리 속도가 아니다."
        assert "- 원문:" not in body
        assert "- 작성자:" not in body
        assert body.count("[[summaries/lead]]") == 1
        assert "[[concepts/engineering-leadership]]" not in body
        assert "[[concepts/system-design]]" in body
        assert "Decision Making" in body or "[[concepts/의사결정]]" in body

    def test_refresh_picks_label_relevant_contexts_over_first_repo_path_bullets(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "summaries" / "starter-pack.md",
            (
                "# Starter Pack\n\n"
                "## 포함된 구성\n"
                "- `projects/oakwoods/`: Phaser 기반 2D 플랫폼러 스타터다. 이동, 전투, 패럴랙스 같은 전형적인 플랫포머 요소를 다루며, 외부 Oak Woods 아트팩이 필요하다. Phaser Game Dev\n"
                "- `projects/tinyswords/`: 타일맵, 계층형 UI, 턴제 아이디어에 잘 맞는 2D 전술 프로토타입이다. Tilemap Layering\n\n"
                "## 사용 흐름\n"
                "권장 흐름은 저장소를 내려받은 뒤 루트 전체를 열고 `START-HERE.md`를 읽는 것이다. 필요하면 각 스타터를 새 작업용 저장소로 복사해 깔끔한 잼 프로젝트로 시작할 수 있다. Project Isolation\n"
            ),
        )
        for slug, title in [
            ("project-isolation", "Project Isolation"),
            ("tiny-swords", "Tiny Swords"),
        ]:
            _write(
                kb_dir / "wiki" / "concepts" / f"{slug}.md",
                (
                    "---\n"
                    "brief: 'projects/oakwoods/: Phaser 기반 2D 플랫폼러 스타터다.'\n"
                    "entity_type: concept\n"
                    "sources:\n"
                    "  - summaries/starter-pack.md\n"
                    "---\n\n"
                    f"# {title}\n\n"
                    "## 개요\n"
                    "projects/oakwoods/: Phaser 기반 2D 플랫폼러 스타터다.\n\n"
                    "## 관찰된 문맥\n\n"
                    "- projects/oakwoods/: Phaser 기반 2D 플랫폼러 스타터다. 이동, 전투, 패럴랙스 같은 전형적인 플랫포머 요소를 다룬다. Phaser Game Dev\n\n"
                    "## Related Documents\n\n"
                    "- [[summaries/starter-pack]]\n"
                ),
            )

        run_internal_maintenance(kb_dir)

        project_meta, project_body = parse_fm((kb_dir / "wiki" / "concepts" / "project-isolation.md").read_text(encoding="utf-8"))
        tiny_meta, tiny_body = parse_fm((kb_dir / "wiki" / "concepts" / "tiny-swords.md").read_text(encoding="utf-8"))

        assert not project_meta["brief"].startswith("projects/")
        assert "START-HERE" in project_body
        assert "Project Isolation." not in project_body
        assert not tiny_meta["brief"].startswith("projects/")
        assert "타일맵, 계층형 UI" in tiny_body

    def test_refresh_normalizes_runtime_paths_and_urlish_briefs(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "summaries" / "codeburn.md",
            (
                "# Codeburn\n\n"
                "codeburn은 Claude Code가 이미 저장하는 세션 트랜스크립트를 ~/.claude/projects/에서 읽어 분석한다.\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "summaries" / "missing-link.md",
            (
                "# Missing Link\n\n"
                "링크 대상: 공개적으로 확인 가능한 렌더링에서는 x.com/i/article/2044775003556646912가 현재 페이지 없음으로 표시된다.\n"
            ),
        )
        for slug, title, source in [
            ("세션-트랜스크립트-분석", "세션 트랜스크립트 분석", "summaries/codeburn.md"),
            ("source-metadata", "Source Metadata", "summaries/missing-link.md"),
        ]:
            _write(
                kb_dir / "wiki" / "concepts" / f"{slug}.md",
                (
                    "---\n"
                    f"brief: {title}는 내부 summary가 관련 개념으로 직접 묶어 둔 주제다.\n"
                    "entity_type: concept\n"
                    "sources:\n"
                    f"  - {source}\n"
                    "---\n\n"
                    f"# {title}\n\n"
                    "## 개요\n"
                    f"{title}는 내부 summary가 관련 개념으로 직접 묶어 둔 주제다. 이 페이지는 source 원문이 아니라 현재 KB 요약 문맥만 기준으로 정리했다.\n\n"
                    "## Related Documents\n\n"
                    f"- [[{source[:-3]}]]\n"
                ),
            )

        run_internal_maintenance(kb_dir)

        codeburn_meta, codeburn_body = parse_fm((kb_dir / "wiki" / "concepts" / "세션-트랜스크립트-분석.md").read_text(encoding="utf-8"))
        missing_meta, missing_body = parse_fm((kb_dir / "wiki" / "concepts" / "source-metadata.md").read_text(encoding="utf-8"))

        assert "~/.claude/" not in codeburn_meta["brief"]
        assert "~/.claude/" not in codeburn_body
        assert "x.com/" not in missing_meta["brief"]
        assert "원문 링크" in missing_body

    def test_cleans_duplicate_and_double_dash_bullet_sections_globally(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        summary = kb_dir / "wiki" / "summaries" / "messy.md"
        _write(
            summary,
            (
                "# Messy\n\n"
                "## Related Concepts\n\n"
                "- [[concepts/attention]]\n\n"
                "## Related Concepts\n\n"
                "- [[concepts/attention]]\n"
            ),
        )
        concept = kb_dir / "wiki" / "concepts" / "messy-concept.md"
        _write(
            concept,
            (
                "---\nbrief: messy.\nentity_type: concept\nsources: [summaries/messy.md]\n---\n\n"
                "# Messy Concept\n\n"
                "## Related Documents\n\n"
                "- - [[summaries/messy]]\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Attention.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )

        run_internal_maintenance(kb_dir)

        summary_text = summary.read_text(encoding="utf-8")
        concept_text = concept.read_text(encoding="utf-8")
        assert summary_text.count("## Related Concepts") == 1
        assert "- - [[summaries/messy]]" not in concept_text
        assert "- [[summaries/messy]]" in concept_text


class TestCatalogMaintenance:
    def test_rebuilds_wiki_index_catalog(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "topic-a.md",
            (
                "---\n"
                "type: wiki\n"
                "topic: topic-a\n"
                "category: AI/ML\n"
                "entity_type: project\n"
                "source_count: 2\n"
                "---\n\n"
                "# Topic A\n\n"
                "Topic A summary paragraph with enough detail.\n"
            ),
        )
        _write(
            kb_dir / "wiki" / "concepts" / "attention.md",
            "---\nbrief: Selective focus mechanism.\nentity_type: concept\nsources: []\n---\n\n# Attention\n",
        )

        run_internal_maintenance(kb_dir)

        index_text = (kb_dir / "wiki" / "index.md").read_text(encoding="utf-8")
        meta = yaml.safe_load(index_text.split("---", 2)[1])
        assert meta["topic_count"] == 1
        assert "[[concepts/attention]] — Selective focus mechanism." in index_text
        assert "[[wiki/topic-a]]" in index_text


class TestStructuralIssues:
    def test_collects_internal_only_issue_buckets(self, tmp_path):
        kb_dir = _make_kb(tmp_path)
        _write(
            kb_dir / "wiki" / "mental-health.md",
            (
                "---\n"
                "type: wiki\n"
                "topic: mental-health\n"
                "source_count: 0\n"
                "---\n\n"
                "# Mental Health\n\n"
                "[[wiki/not-a-real-page]] and [[concepts/missing-concept]] and "
                "[[sources/2026/01/01/missing-source]].\n"
            ),
        )
        _write(kb_dir / "wiki" / "summaries" / "lonely.md", "# Lonely\n\nNo related section.\n")

        issues = collect_structural_issues(kb_dir)

        assert issues["broken_source_links"]
        assert issues["broken_wiki_links"]
        assert issues["broken_concept_links"]
        assert "mental-health" in issues["wiki_zero_sources"]
        assert "lonely" in issues["summaries_missing_related"]
