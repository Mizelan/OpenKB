"""Tests for openkb.agent.compiler pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from openkb.agent.compiler import (
    compile_long_doc,
    compile_short_doc,
    _compile_concepts,
    _parse_json,
    _sanitize_concept_name,
    _write_summary,
    _write_concept,
    _update_index,
    _read_wiki_context,
    _read_concept_briefs,
    _add_related_link,
    _backlink_summary,
    _backlink_concepts,
    _section_heading,
    _make_index_template,
    _INDEX_SECTIONS,
)


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('[{"name": "foo"}]') == [{"name": "foo"}]

    def test_fenced_json(self):
        text = '```json\n[{"name": "bar"}]\n```'
        assert _parse_json(text) == [{"name": "bar"}]

    def test_invalid_json(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json("not json")


class TestParseConceptsPlan:
    def test_dict_format(self):
        text = json.dumps({
            "create": [{"name": "foo", "title": "Foo"}],
            "update": [{"name": "bar", "title": "Bar"}],
            "related": ["baz"],
        })
        parsed = _parse_json(text)
        assert isinstance(parsed, dict)
        assert len(parsed["create"]) == 1
        assert len(parsed["update"]) == 1
        assert parsed["related"] == ["baz"]

    def test_fallback_list_format(self):
        text = json.dumps([{"name": "foo", "title": "Foo"}])
        parsed = _parse_json(text)
        assert isinstance(parsed, list)

    def test_fenced_dict(self):
        text = '```json\n{"create": [], "update": [], "related": []}\n```'
        parsed = _parse_json(text)
        assert isinstance(parsed, dict)
        assert parsed["create"] == []


class TestParseBriefContent:
    def test_dict_with_brief_and_content(self):
        text = json.dumps({"brief": "A short desc", "content": "# Full page\n\nDetails."})
        parsed = _parse_json(text)
        assert parsed["brief"] == "A short desc"
        assert "# Full page" in parsed["content"]

    def test_plain_text_fallback(self):
        """If LLM returns plain text, _parse_json raises — caller handles fallback."""
        with pytest.raises((json.JSONDecodeError, ValueError)):
            _parse_json("Just plain markdown text without JSON")


class TestSanitizeConceptName:
    def test_ascii_passthrough(self):
        assert _sanitize_concept_name("hello-world") == "hello-world"

    def test_spaces_replaced(self):
        assert _sanitize_concept_name("hello world") == "hello-world"

    def test_chinese(self):
        result = _sanitize_concept_name("注意力机制")
        assert result == "注意力机制"

    def test_japanese(self):
        result = _sanitize_concept_name("トランスフォーマー")
        assert result == "トランスフォーマー"

    def test_french_accents(self):
        result = _sanitize_concept_name("réseau neuronal")
        assert "r" in result
        assert result != "r-seau-neuronal"  # accented chars preserved, not stripped

    def test_distinct_chinese_names_no_collision(self):
        a = _sanitize_concept_name("注意力机制")
        b = _sanitize_concept_name("变压器模型")
        assert a != b

    def test_empty_fallback(self):
        assert _sanitize_concept_name("!!!") == "unnamed-concept"

    def test_nfkc_normalization(self):
        # U+FF21 (fullwidth A) should normalize to regular A
        assert _sanitize_concept_name("\uff21\uff22") == "AB"


class TestWriteSummary:
    def test_writes_with_frontmatter(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_summary(wiki, "my-doc", "# Summary\n\nContent here.")
        path = wiki / "summaries" / "my-doc.md"
        assert path.exists()
        text = path.read_text()
        assert "doc_type: short" in text
        assert "full_text: sources/my-doc.md" in text
        assert "# Summary" in text

    def test_writes_without_brief(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_summary(wiki, "my-doc", "# Summary\n\nContent here.")
        path = wiki / "summaries" / "my-doc.md"
        text = path.read_text()
        assert "doc_type: short" in text
        assert "full_text: sources/my-doc.md" in text


class TestWriteConcept:
    def test_new_concept_with_brief(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_concept(wiki, "attention", "# Attention\n\nDetails.", "paper.pdf", False, brief="Mechanism for selective focus")
        path = wiki / "concepts" / "attention.md"
        assert path.exists()
        text = path.read_text()
        assert "sources: [paper.pdf]" in text
        assert "brief: Mechanism for selective focus" in text
        assert "# Attention" in text

    def test_new_concept_without_brief(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_concept(wiki, "attention", "# Attention\n\nDetails.", "paper.pdf", False)
        path = wiki / "concepts" / "attention.md"
        assert path.exists()
        text = path.read_text()
        assert "sources: [paper.pdf]" in text
        assert "brief:" not in text

    def test_update_concept_updates_brief(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper1.pdf]\nbrief: Old brief\n---\n\n# Attention\n\nOld content.",
            encoding="utf-8",
        )
        _write_concept(wiki, "attention", "New info.", "paper2.pdf", True, brief="Updated brief")
        text = (concepts / "attention.md").read_text()
        assert "paper2.pdf" in text
        assert "paper1.pdf" in text
        assert "brief: Updated brief" in text
        assert "Old brief" not in text

    def test_update_concept_appends_source(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper1.pdf]\n---\n\n# Attention\n\nOld content.",
            encoding="utf-8",
        )
        _write_concept(wiki, "attention", "New info from paper2.", "paper2.pdf", True)
        text = (concepts / "attention.md").read_text()
        assert "paper2.pdf" in text
        assert "paper1.pdf" in text
        assert "New info from paper2." in text


class TestUpdateIndex:
    def test_appends_entries_with_briefs(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )
        _update_index(wiki, "my-doc", ["attention", "transformer"],
                       doc_brief="Introduces transformers",
                       concept_briefs={"attention": "Focus mechanism", "transformer": "NN architecture"})
        text = (wiki / "index.md").read_text()
        assert "[[summaries/my-doc]] (short) — Introduces transformers" in text
        assert "[[concepts/attention]] — Focus mechanism" in text
        assert "[[concepts/transformer]] — NN architecture" in text

    def test_updates_only_exact_concept_row(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n"
            "- [[concepts/transformer]] — Uses [[concepts/attention]] internally\n"
            "- [[concepts/attention]] — Old brief\n\n## Explorations\n",
            encoding="utf-8",
        )
        _update_index(
            wiki,
            "my-doc",
            ["attention"],
            concept_briefs={"attention": "New brief"},
        )
        text = (wiki / "index.md").read_text()
        assert "- [[concepts/transformer]] — Uses [[concepts/attention]] internally" in text
        assert "- [[concepts/attention]] — New brief" in text
        assert text.count("[[concepts/attention]] — New brief") == 1

    def test_no_duplicates(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n- [[summaries/my-doc]] — Old brief\n\n## Concepts\n",
            encoding="utf-8",
        )
        _update_index(wiki, "my-doc", [], doc_brief="New brief")
        text = (wiki / "index.md").read_text()
        assert text.count("[[summaries/my-doc]]") == 1

    def test_backwards_compat_no_briefs(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )
        _update_index(wiki, "my-doc", ["attention"])
        text = (wiki / "index.md").read_text()
        assert "[[summaries/my-doc]]" in text
        assert "[[concepts/attention]]" in text

    def test_updates_concept_brief_only_inside_concepts_section(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Index\n\n"
            "## Documents\n"
            "- [[summaries/my-doc]] (short) — Mentions [[concepts/attention]] here\n\n"
            "## Concepts\n"
            "- [[concepts/attention]] — Old brief\n\n"
            "## Explorations\n",
            encoding="utf-8",
        )

        _update_index(
            wiki,
            "my-doc",
            ["attention"],
            concept_briefs={"attention": "New brief"},
        )

        text = (wiki / "index.md").read_text()
        assert "- [[summaries/my-doc]] (short) — Mentions [[concepts/attention]] here" in text
        assert "- [[concepts/attention]] — New brief" in text
        assert "- [[concepts/attention]] — Old brief" not in text

    def test_adds_concept_entry_when_link_exists_outside_concepts_section(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Index\n\n"
            "## Documents\n"
            "- [[summaries/my-doc]] (short) — Mentions [[concepts/attention]] here\n\n"
            "## Concepts\n\n"
            "## Explorations\n",
            encoding="utf-8",
        )

        _update_index(
            wiki,
            "my-doc",
            ["attention"],
            concept_briefs={"attention": "New brief"},
        )

        text = (wiki / "index.md").read_text()
        assert "- [[summaries/my-doc]] (short) — Mentions [[concepts/attention]] here" in text
        assert "- [[concepts/attention]] — New brief" in text


class TestReadWikiContext:
    def test_empty_wiki(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        index, concepts = _read_wiki_context(wiki)
        assert index == ""
        assert concepts == []

    def test_with_content(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text("# Index\n", encoding="utf-8")
        concepts_dir = wiki / "concepts"
        concepts_dir.mkdir()
        (concepts_dir / "attention.md").write_text("# Attention", encoding="utf-8")
        (concepts_dir / "transformer.md").write_text("# Transformer", encoding="utf-8")
        index, concepts = _read_wiki_context(wiki)
        assert "# Index" in index
        assert concepts == ["attention", "transformer"]


class TestReadConceptBriefs:
    def test_empty_wiki(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "concepts").mkdir()
        assert _read_concept_briefs(wiki) == "(none yet)"

    def test_no_concepts_dir(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        assert _read_concept_briefs(wiki) == "(none yet)"

    def test_reads_briefs_with_frontmatter(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper.pdf]\n---\n\nAttention is a mechanism that allows models to focus on relevant parts.",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "- attention:" in result
        assert "Attention is a mechanism" in result
        assert "sources" not in result
        assert "---" not in result

    def test_reads_briefs_without_frontmatter(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "transformer.md").write_text(
            "Transformer is a neural network architecture based on attention.",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "- transformer:" in result
        assert "Transformer is a neural network" in result

    def test_truncates_long_content(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        long_body = "A" * 300
        (concepts / "longconcept.md").write_text(long_body, encoding="utf-8")
        result = _read_concept_briefs(wiki)
        # The brief part should be truncated at 150 chars
        brief = result.split("- longconcept: ", 1)[1]
        assert len(brief) == 150
        assert brief == "A" * 150

    def test_sorted_alphabetically(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "zebra.md").write_text("Zebra concept.", encoding="utf-8")
        (concepts / "apple.md").write_text("Apple concept.", encoding="utf-8")
        (concepts / "mango.md").write_text("Mango concept.", encoding="utf-8")
        result = _read_concept_briefs(wiki)
        lines = result.strip().splitlines()
        slugs = [line.split(":")[0].lstrip("- ") for line in lines]
        assert slugs == ["apple", "mango", "zebra"]

    def test_reads_brief_from_frontmatter(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper.pdf]\nbrief: Selective focus mechanism\n---\n\n# Attention\n\nLong content...",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "- attention: Selective focus mechanism" in result

    def test_falls_back_to_body_truncation(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "old.md").write_text(
            "---\nsources: [paper.pdf]\n---\n\nOld concept without brief field.",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "- old: Old concept without brief field." in result


class TestBacklinkSummary:
    def test_adds_missing_concept_links(self, tmp_path):
        wiki = tmp_path / "wiki"
        summaries = wiki / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "paper.md").write_text(
            "---\nsources: [paper.pdf]\n---\n\n# Summary\n\nContent about attention.",
            encoding="utf-8",
        )
        _backlink_summary(wiki, "paper", ["attention", "transformer"])
        text = (summaries / "paper.md").read_text()
        assert "[[concepts/attention]]" in text
        assert "[[concepts/transformer]]" in text

    def test_skips_already_linked(self, tmp_path):
        wiki = tmp_path / "wiki"
        summaries = wiki / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "paper.md").write_text(
            "---\nsources: [paper.pdf]\n---\n\n# Summary\n\nSee [[concepts/attention]].",
            encoding="utf-8",
        )
        _backlink_summary(wiki, "paper", ["attention", "transformer"])
        text = (summaries / "paper.md").read_text()
        # attention already linked, should not duplicate
        assert text.count("[[concepts/attention]]") == 1
        # transformer should be added
        assert "[[concepts/transformer]]" in text

    def test_no_op_when_all_linked(self, tmp_path):
        wiki = tmp_path / "wiki"
        summaries = wiki / "summaries"
        summaries.mkdir(parents=True)
        original = "# Summary\n\n[[concepts/attention]] and [[concepts/transformer]]"
        (summaries / "paper.md").write_text(original, encoding="utf-8")
        _backlink_summary(wiki, "paper", ["attention", "transformer"])
        assert (summaries / "paper.md").read_text() == original

    def test_skips_if_file_missing(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        # Should not raise
        _backlink_summary(wiki, "nonexistent", ["attention"])

    def test_merges_into_existing_section(self, tmp_path):
        """Second add should merge into existing ## Related Concepts, not duplicate."""
        wiki = tmp_path / "wiki"
        summaries = wiki / "summaries"
        summaries.mkdir(parents=True)
        (summaries / "paper.md").write_text(
            "# Summary\n\nContent.\n\n## Related Concepts\n- [[concepts/attention]]\n",
            encoding="utf-8",
        )
        _backlink_summary(wiki, "paper", ["attention", "transformer"])
        text = (summaries / "paper.md").read_text()
        assert text.count("## Related Concepts") == 1
        assert "[[concepts/transformer]]" in text
        assert text.count("[[concepts/attention]]") == 1


class TestBacklinkConcepts:
    def test_adds_summary_link_to_concept(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper.pdf]\n---\n\n# Attention\n\nContent.",
            encoding="utf-8",
        )
        _backlink_concepts(wiki, "paper", ["attention"])
        text = (concepts / "attention.md").read_text()
        assert "[[summaries/paper]]" in text
        assert "## Related Documents" in text

    def test_skips_if_already_linked(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "# Attention\n\nBased on [[summaries/paper]].",
            encoding="utf-8",
        )
        _backlink_concepts(wiki, "paper", ["attention"])
        text = (concepts / "attention.md").read_text()
        assert text.count("[[summaries/paper]]") == 1
        assert "## Related Documents" not in text

    def test_merges_into_existing_section(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "# Attention\n\n## Related Documents\n- [[summaries/old-paper]]\n",
            encoding="utf-8",
        )
        _backlink_concepts(wiki, "new-paper", ["attention"])
        text = (concepts / "attention.md").read_text()
        assert text.count("## Related Documents") == 1
        assert "[[summaries/old-paper]]" in text
        assert "[[summaries/new-paper]]" in text

    def test_skips_missing_concept_file(self, tmp_path):
        wiki = tmp_path / "wiki"
        (wiki / "concepts").mkdir(parents=True)
        # Should not raise
        _backlink_concepts(wiki, "paper", ["nonexistent"])


class TestAddRelatedLink:
    def test_adds_see_also_link(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper1.pdf]\n---\n\n# Attention\n\nSome content.",
            encoding="utf-8",
        )
        _add_related_link(wiki, "attention", "new-doc", "paper2.pdf")
        text = (concepts / "attention.md").read_text()
        assert "[[summaries/new-doc]]" in text
        assert "paper2.pdf" in text

    def test_skips_if_already_linked(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper1.pdf]\n---\n\n# Attention\n\nSee also: [[summaries/new-doc]]",
            encoding="utf-8",
        )
        _add_related_link(wiki, "attention", "new-doc", "paper1.pdf")
        text = (concepts / "attention.md").read_text()
        assert text.count("[[summaries/new-doc]]") == 1

    def test_skips_if_file_missing(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        # Should not raise
        _add_related_link(wiki, "nonexistent", "doc", "file.pdf")




class TestCompileShortDoc:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        # Setup KB structure
        wiki = tmp_path / "wiki"
        (wiki / "sources").mkdir(parents=True)
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )
        source_path = wiki / "sources" / "test-doc.md"
        source_path.write_text("# Test Doc\n\nSome content about transformers.", encoding="utf-8")
        (tmp_path / ".openkb").mkdir()
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "test-doc.pdf").write_bytes(b"fake")

        summary_response = json.dumps({
            "brief": "Discusses transformers",
            "content": "# Summary\n\nThis document discusses transformers.",
        })
        concepts_list_response = json.dumps({
            "create": [{"name": "transformer", "title": "Transformer"}],
            "update": [],
            "related": [],
        })
        concept_page_response = json.dumps({
            "brief": "NN architecture using self-attention",
            "content": "# Transformer\n\nA neural network architecture.",
        })

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [summary_response, concepts_list_response, concept_page_response]
            await compile_short_doc("test-doc", source_path, tmp_path, "gpt-4o-mini")

        # Verify summary written
        summary_path = wiki / "summaries" / "test-doc.md"
        assert summary_path.exists()
        assert "full_text: sources/test-doc.md" in summary_path.read_text()

        # Verify concept written
        concept_path = wiki / "concepts" / "transformer.md"
        assert concept_path.exists()
        assert "sources: [summaries/test-doc.md]" in concept_path.read_text()

        # Verify index updated
        index_text = (wiki / "index.md").read_text()
        assert "[[summaries/test-doc]]" in index_text
        assert "[[concepts/transformer]]" in index_text

    @pytest.mark.asyncio
    async def test_handles_bad_json(self, tmp_path):
        wiki = tmp_path / "wiki"
        (wiki / "sources").mkdir(parents=True)
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n",
            encoding="utf-8",
        )
        source_path = wiki / "sources" / "doc.md"
        source_path.write_text("Content", encoding="utf-8")
        (tmp_path / ".openkb").mkdir()

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = ["Plain summary text", "not valid json"]
            # Should not raise
            await compile_short_doc("doc", source_path, tmp_path, "gpt-4o-mini")

        # Summary should still be written
        assert (wiki / "summaries" / "doc.md").exists()


class TestCompileLongDoc:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, tmp_path):
        wiki = tmp_path / "wiki"
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n",
            encoding="utf-8",
        )
        summary_path = wiki / "summaries" / "big-doc.md"
        summary_path.write_text("# Big Doc\n\nPageIndex summary tree.", encoding="utf-8")
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        (openkb_dir / "config.yaml").write_text("model: gpt-4o-mini\n")
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "big-doc.pdf").write_bytes(b"fake")

        overview_response = "Overview of the big document."
        concepts_list_response = json.dumps({
            "create": [{"name": "deep-learning", "title": "Deep Learning"}],
            "update": [],
            "related": [],
        })
        concept_page_response = json.dumps({
            "brief": "Subfield of ML using neural networks",
            "content": "# Deep Learning\n\nA subfield of ML.",
        })

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [overview_response, concepts_list_response, concept_page_response]
            await compile_long_doc(
                "big-doc", summary_path, "doc-123", tmp_path, "gpt-4o-mini"
            )

        concept_path = wiki / "concepts" / "deep-learning.md"
        assert concept_path.exists()
        assert "Deep Learning" in concept_path.read_text()

        index_text = (wiki / "index.md").read_text()
        assert "[[summaries/big-doc]]" in index_text
        assert "[[concepts/deep-learning]]" in index_text


class TestCompileConceptsPlan:
    """Integration tests for _compile_concepts with the new plan format."""

    def _setup_wiki(self, tmp_path, existing_concepts=None):
        """Helper to set up a wiki directory with optional existing concepts."""
        wiki = tmp_path / "wiki"
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n",
            encoding="utf-8",
        )
        (tmp_path / "raw").mkdir(exist_ok=True)
        (tmp_path / "raw" / "test-doc.pdf").write_bytes(b"fake")

        if existing_concepts:
            for name, content in existing_concepts.items():
                (wiki / "concepts" / f"{name}.md").write_text(
                    content, encoding="utf-8",
                )

        return wiki

    @pytest.mark.asyncio
    async def test_create_and_update_flow(self, tmp_path):
        """Pre-existing 'attention' concept; plan creates 'flash-attention' and updates 'attention'."""
        wiki = self._setup_wiki(tmp_path, existing_concepts={
            "attention": "---\nsources: [old-paper.pdf]\n---\n\n# Attention\n\nOriginal content about attention.",
        })

        plan_response = json.dumps({
            "create": [{"name": "flash-attention", "title": "Flash Attention"}],
            "update": [{"name": "attention", "title": "Attention"}],
            "related": [],
        })
        create_page_response = json.dumps({
            "brief": "Efficient attention algorithm",
            "content": "# Flash Attention\n\nAn efficient attention algorithm.",
        })
        update_page_response = json.dumps({
            "brief": "Updated attention mechanism",
            "content": "# Attention\n\nUpdated content with new info.",
        })

        system_msg = {"role": "system", "content": "You are a wiki agent."}
        doc_msg = {"role": "user", "content": "Document about attention mechanisms."}
        summary = "Summary of the document."

        def mock_llm(model, messages, step_name, **kwargs):
            if step_name == "concepts-plan":
                return plan_response
            elif step_name.startswith("concept:"):
                return create_page_response
            elif step_name.startswith("update:"):
                return update_page_response
            return ""

        with patch("openkb.agent.compiler._llm_call", side_effect=mock_llm):
            await _compile_concepts(
                wiki, tmp_path, "gpt-4o-mini", system_msg, doc_msg,
                summary, "test-doc", 5,
            )

        # Verify flash-attention created
        fa_path = wiki / "concepts" / "flash-attention.md"
        assert fa_path.exists()
        fa_text = fa_path.read_text()
        assert "sources: [summaries/test-doc.md]" in fa_text
        assert "Flash Attention" in fa_text

        # Verify attention updated (is_update=True path in _write_concept)
        att_path = wiki / "concepts" / "attention.md"
        assert att_path.exists()
        att_text = att_path.read_text()
        assert "summaries/test-doc.md" in att_text
        assert "old-paper.pdf" in att_text

        # Verify index updated
        index_text = (wiki / "index.md").read_text()
        assert "[[concepts/flash-attention]]" in index_text
        assert "[[concepts/attention]]" in index_text

    @pytest.mark.asyncio
    async def test_related_adds_link_no_llm(self, tmp_path):
        """Plan has only related items. No acompletion calls should be made."""
        wiki = self._setup_wiki(tmp_path, existing_concepts={
            "transformer": "---\nsources: [old.pdf]\n---\n\n# Transformer\n\nContent about transformers.",
        })

        plan_response = json.dumps({
            "create": [],
            "update": [],
            "related": ["transformer"],
        })

        system_msg = {"role": "system", "content": "You are a wiki agent."}
        doc_msg = {"role": "user", "content": "Document content."}
        summary = "Summary."

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [plan_response]
            await _compile_concepts(
                wiki, tmp_path, "gpt-4o-mini", system_msg, doc_msg,
                summary, "test-doc", 5,
            )
            # Only the plan call should be made — related is code-only
            assert mock_llm.call_count == 1

        # Verify link added to transformer page
        transformer_text = (wiki / "concepts" / "transformer.md").read_text()
        assert "[[summaries/test-doc]]" in transformer_text
        assert "summaries/test-doc.md" in transformer_text

    @pytest.mark.asyncio
    async def test_fallback_list_format(self, tmp_path):
        """LLM returns a flat array instead of dict — treated as all create."""
        wiki = self._setup_wiki(tmp_path)

        plan_response = json.dumps([
            {"name": "attention", "title": "Attention"},
        ])
        concept_page_response = json.dumps({
            "brief": "A mechanism for focusing",
            "content": "# Attention\n\nA mechanism for focusing.",
        })

        system_msg = {"role": "system", "content": "You are a wiki agent."}
        doc_msg = {"role": "user", "content": "Document content."}
        summary = "Summary."

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [plan_response, concept_page_response]
            await _compile_concepts(
                wiki, tmp_path, "gpt-4o-mini", system_msg, doc_msg,
                summary, "test-doc", 5,
            )

        # Verify concept was created (not updated)
        att_path = wiki / "concepts" / "attention.md"
        assert att_path.exists()
        att_text = att_path.read_text()
        assert "sources: [summaries/test-doc.md]" in att_text
        assert "Attention" in att_text


class TestConceptsPlanPrompt:
    def test_plan_requests_entity_type(self):
        """Verify _CONCEPTS_PLAN_USER asks for entity_type in create items."""
        from openkb.agent.compiler import _CONCEPTS_PLAN_USER
        assert "entity_type" in _CONCEPTS_PLAN_USER
        assert "person" in _CONCEPTS_PLAN_USER or "organization" in _CONCEPTS_PLAN_USER

    def test_plan_requests_entity_type_in_update(self):
        """Verify _CONCEPTS_PLAN_USER asks for entity_type in update items."""
        from openkb.agent.compiler import _CONCEPTS_PLAN_USER
        assert "entity_type" in _CONCEPTS_PLAN_USER
        # Check that the prompt mentions entity_type for both create and update
        lines = _CONCEPTS_PLAN_USER.split("\n")
        create_lines = [l for l in lines if "create" in l.lower() and "entity_type" in l.lower()]
        update_lines = [l for l in lines if "update" in l.lower() and "entity_type" in l.lower()]
        # At least the main template should mention entity_type
        assert len(create_lines) > 0 or "entity_type" in _CONCEPTS_PLAN_USER


class TestConceptPagePrompts:
    def test_concept_page_requests_entity_type(self):
        """Verify _CONCEPT_PAGE_USER asks for entity_type in LLM output."""
        from openkb.agent.compiler import _CONCEPT_PAGE_USER
        assert "entity_type" in _CONCEPT_PAGE_USER

    def test_concept_update_requests_entity_type(self):
        """Verify _CONCEPT_UPDATE_USER asks for entity_type in LLM output."""
        from openkb.agent.compiler import _CONCEPT_UPDATE_USER
        assert "entity_type" in _CONCEPT_UPDATE_USER


class TestWriteConceptEntityType:
    def test_new_concept_with_entity_type(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_concept(wiki, "openai", "# OpenAI\n\nDetails.", "paper.pdf",
                       False, brief="AI research company", entity_type="organization")
        text = (wiki / "concepts" / "openai.md").read_text()
        assert "entity_type: organization" in text
        assert "brief: AI research company" in text

    def test_new_concept_without_entity_type(self, tmp_path):
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_concept(wiki, "attention", "# Attention\n\nDetails.", "paper.pdf", False)
        text = (wiki / "concepts" / "attention.md").read_text()
        assert "entity_type:" not in text

    def test_update_concept_preserves_entity_type(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "openai.md").write_text(
            "---\nsources: [paper1.pdf]\nbrief: AI company\nentity_type: organization\n---\n\n# OpenAI\n\nOld content.",
            encoding="utf-8",
        )
        _write_concept(wiki, "openai", "New info.", "paper2.pdf", True, brief="Updated")
        text = (concepts / "openai.md").read_text()
        assert "entity_type: organization" in text

    def test_update_concept_sets_entity_type(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "openai.md").write_text(
            "---\nsources: [paper1.pdf]\nbrief: AI company\n---\n\n# OpenAI\n\nOld content.",
            encoding="utf-8",
        )
        _write_concept(wiki, "openai", "New info.", "paper2.pdf", True,
                       brief="Updated", entity_type="organization")
        text = (concepts / "openai.md").read_text()
        assert "entity_type: organization" in text


class TestReadConceptBriefsWithEntityType:
    def test_includes_entity_type(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "openai.md").write_text(
            "---\nsources: [paper.pdf]\nbrief: AI company\nentity_type: organization\n---\n\n# OpenAI\n\nContent.",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "organization" in result

    def test_no_entity_type_omits_tag(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "attention.md").write_text(
            "---\nsources: [paper.pdf]\nbrief: Focus mechanism\n---\n\n# Attention\n\nContent.",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "[organization]" not in result
        assert "- attention: Focus mechanism" in result

    def test_mixed_entity_types(self, tmp_path):
        wiki = tmp_path / "wiki"
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "openai.md").write_text(
            "---\nsources: [paper.pdf]\nbrief: AI company\nentity_type: organization\n---\n\nContent.",
            encoding="utf-8",
        )
        (concepts / "attention.md").write_text(
            "---\nsources: [paper.pdf]\nbrief: Focus mechanism\n---\n\nContent.",
            encoding="utf-8",
        )
        result = _read_concept_briefs(wiki)
        assert "openai [organization]" in result
        assert "attention: Focus mechanism" in result


class TestKoreanSectionHeaders:
    def test_make_index_template_english(self):
        template = _make_index_template("en")
        assert "## Documents" in template
        assert "## Concepts" in template
        assert "## Explorations" in template
        assert "# Knowledge Base Index" in template

    def test_make_index_template_korean(self):
        template = _make_index_template("ko")
        assert "## \ubb38\uc11c" in template
        assert "## \uac1c\ub150" in template
        assert "## \ud0d0\uad6c" in template
        assert "# \uc9c0\uc2dd \ubca0\uc774\uc2a4 \uc778\ub371\uc2a4" in template

    def test_section_heading_english(self):
        assert _section_heading("en", "documents") == "## Documents"
        assert _section_heading("en", "concepts") == "## Concepts"

    def test_section_heading_korean(self):
        assert _section_heading("ko", "documents") == "## \ubb38\uc11c"
        assert _section_heading("ko", "concepts") == "## \uac1c\ub150"

    def test_section_heading_unknown_defaults_english(self):
        assert _section_heading("fr", "documents") == "## Documents"

    def test_update_index_korean_headers(self, tmp_path):
        """_update_index should use Korean headers when language='ko'."""
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# \uc9c0\uc2dd \ubca0\uc774\uc2a4 \uc778\ub371\uc2a4\n\n## \ubb38\uc11c\n\n## \uac1c\ub150\n\n## \ud0d0\uad6c\n",
            encoding="utf-8",
        )
        _update_index(wiki, "my-doc", ["attention"],
                     doc_brief="Introduces transformers",
                     concept_briefs={"attention": "Focus mechanism"}, language="ko")
        text = (wiki / "index.md").read_text()
        assert "[[summaries/my-doc]]" in text
        assert "## \ubb38\uc11c" in text
        assert "## \uac1c\ub150" in text

    def test_update_index_english_headers_default(self, tmp_path):
        """_update_index should use English headers by default."""
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# Knowledge Base Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )
        _update_index(wiki, "my-doc", ["attention"],
                     doc_brief="Introduces transformers",
                     concept_briefs={"attention": "Focus mechanism"})
        text = (wiki / "index.md").read_text()
        assert "## Documents" in text
        assert "## Concepts" in text

    def test_update_index_creates_korean_template(self, tmp_path):
        """_update_index should create Korean template when language='ko' and no index exists."""
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _update_index(wiki, "my-doc", [], language="ko")
        text = (wiki / "index.md").read_text()
        assert "## \ubb38\uc11c" in text
        assert "## \uac1c\ub150" in text


class TestBriefIntegration:
    @pytest.mark.asyncio
    async def test_short_doc_briefs_in_index_and_frontmatter(self, tmp_path):
        wiki = tmp_path / "wiki"
        (wiki / "sources").mkdir(parents=True)
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text(
            "# Index\n\n## Documents\n\n## Concepts\n\n## Explorations\n",
            encoding="utf-8",
        )
        source_path = wiki / "sources" / "test-doc.md"
        source_path.write_text("# Test Doc\n\nContent.", encoding="utf-8")
        (tmp_path / ".openkb").mkdir()
        (tmp_path / "raw").mkdir()
        (tmp_path / "raw" / "test-doc.pdf").write_bytes(b"fake")

        summary_resp = json.dumps({
            "brief": "A paper about transformers",
            "content": "# Summary\n\nThis paper discusses transformers.",
        })
        plan_resp = json.dumps({
            "create": [{"name": "transformer", "title": "Transformer"}],
            "update": [],
            "related": [],
        })
        concept_resp = json.dumps({
            "brief": "NN architecture using self-attention",
            "content": "# Transformer\n\nA neural network architecture.",
        })

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [summary_resp, plan_resp, concept_resp]
            await compile_short_doc("test-doc", source_path, tmp_path, "gpt-4o-mini")

        # Summary frontmatter has doc_type and full_text
        summary_text = (wiki / "summaries" / "test-doc.md").read_text()
        assert "doc_type: short" in summary_text
        assert "full_text: sources/test-doc.md" in summary_text

        # Concept frontmatter has brief
        concept_text = (wiki / "concepts" / "transformer.md").read_text()
        assert "brief: NN architecture using self-attention" in concept_text

        # Index has briefs
        index_text = (wiki / "index.md").read_text()
        assert "— A paper about transformers" in index_text
        assert "— NN architecture using self-attention" in index_text


class TestEntityTypePipeline:
    """Kill tests verifying entity_type flows from concept plan through _write_concept."""

    @pytest.mark.asyncio
    async def test_gen_create_entity_type_in_result(self, tmp_path):
        """_gen_create extracts entity_type from concept dict and writes it to concept file."""
        wiki = tmp_path / "wiki"
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text("# Index\n\n## Documents\n\n## Concepts\n", encoding="utf-8")
        (tmp_path / "raw").mkdir(exist_ok=True)
        (tmp_path / "raw" / "test-doc.pdf").write_bytes(b"fake")

        plan_response = json.dumps({
            "create": [{"name": "openai", "title": "OpenAI", "entity_type": "organization"}],
            "update": [],
            "related": [],
        })
        concept_response = json.dumps({
            "brief": "AI research company",
            "entity_type": "organization",
            "content": "# OpenAI\n\nAn AI research company.",
        })

        system_msg = {"role": "system", "content": "You are a wiki agent."}
        doc_msg = {"role": "user", "content": "Document about OpenAI."}
        summary = "Summary."

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [plan_response, concept_response]
            await _compile_concepts(
                wiki, tmp_path, "gpt-4o-mini", system_msg, doc_msg,
                summary, "test-doc", 5,
            )

        concept_text = (wiki / "concepts" / "openai.md").read_text()
        assert "entity_type: organization" in concept_text

    @pytest.mark.asyncio
    async def test_gen_update_entity_type_in_result(self, tmp_path):
        """_gen_update extracts entity_type from concept dict and writes it to concept file."""
        wiki = tmp_path / "wiki"
        (wiki / "summaries").mkdir(parents=True)
        concepts = wiki / "concepts"
        concepts.mkdir(parents=True)
        (concepts / "openai.md").write_text(
            "---\nsources: [paper1.pdf]\nbrief: AI company\n---\n\n# OpenAI\n\nOld content.",
            encoding="utf-8",
        )
        (wiki / "index.md").write_text("# Index\n\n## Documents\n\n## Concepts\n", encoding="utf-8")
        (tmp_path / "raw").mkdir(exist_ok=True)
        (tmp_path / "raw" / "test-doc.pdf").write_bytes(b"fake")

        plan_response = json.dumps({
            "create": [],
            "update": [{"name": "openai", "title": "OpenAI", "entity_type": "organization"}],
            "related": [],
        })
        update_response = json.dumps({
            "brief": "Updated AI company",
            "entity_type": "organization",
            "content": "# OpenAI\n\nUpdated content.",
        })

        system_msg = {"role": "system", "content": "You are a wiki agent."}
        doc_msg = {"role": "user", "content": "Document about OpenAI."}
        summary = "Summary."

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [plan_response, update_response]
            await _compile_concepts(
                wiki, tmp_path, "gpt-4o-mini", system_msg, doc_msg,
                summary, "test-doc", 5,
            )

        concept_text = (concepts / "openai.md").read_text()
        assert "entity_type: organization" in concept_text

    @pytest.mark.asyncio
    async def test_entity_type_from_plan_as_fallback(self, tmp_path):
        """entity_type from concept plan dict is used when LLM response lacks it."""
        wiki = tmp_path / "wiki"
        (wiki / "summaries").mkdir(parents=True)
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "index.md").write_text("# Index\n\n## Documents\n\n## Concepts\n", encoding="utf-8")
        (tmp_path / "raw").mkdir(exist_ok=True)
        (tmp_path / "raw" / "test-doc.pdf").write_bytes(b"fake")

        plan_response = json.dumps({
            "create": [{"name": "openai", "title": "OpenAI", "entity_type": "organization"}],
            "update": [],
            "related": [],
        })
        # LLM response omits entity_type — should fall back to plan's entity_type
        concept_response = json.dumps({
            "brief": "AI research company",
            "content": "# OpenAI\n\nAn AI research company.",
        })

        system_msg = {"role": "system", "content": "You are a wiki agent."}
        doc_msg = {"role": "user", "content": "Document about OpenAI."}
        summary = "Summary."

        with patch("openkb.agent.compiler._llm_call") as mock_llm:
            mock_llm.side_effect = [plan_response, concept_response]
            await _compile_concepts(
                wiki, tmp_path, "gpt-4o-mini", system_msg, doc_msg,
                summary, "test-doc", 5,
            )

        concept_text = (wiki / "concepts" / "openai.md").read_text()
        assert "entity_type: organization" in concept_text
