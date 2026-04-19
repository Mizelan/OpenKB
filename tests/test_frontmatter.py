"""Tests for openkb.frontmatter — pyyaml-based frontmatter utilities."""
from __future__ import annotations

import pytest


class TestParseFm:
    def test_parse_simple_frontmatter(self):
        from openkb.frontmatter import parse_fm
        text = "---\nsources: [a.md]\nbrief: test\n---\n\nBody content."
        meta, body = parse_fm(text)
        assert meta["sources"] == ["a.md"]
        assert meta["brief"] == "test"
        assert body.strip() == "Body content."

    def test_parse_no_frontmatter(self):
        from openkb.frontmatter import parse_fm
        text = "# Title\n\nNo frontmatter here."
        meta, body = parse_fm(text)
        assert meta == {}
        assert body == text

    def test_parse_empty_frontmatter(self):
        from openkb.frontmatter import parse_fm
        text = "---\n---\n\nBody."
        meta, body = parse_fm(text)
        assert meta == {}
        assert body.strip() == "Body."

    def test_parse_entity_type(self):
        from openkb.frontmatter import parse_fm
        text = "---\nentity_type: organization\n---\n\nContent."
        meta, body = parse_fm(text)
        assert meta["entity_type"] == "organization"

    def test_parse_multiline_sources(self):
        from openkb.frontmatter import parse_fm
        text = "---\nsources:\n- a.md\n- b.md\n---\n\nContent."
        meta, body = parse_fm(text)
        assert meta["sources"] == ["a.md", "b.md"]

    def test_parse_entities_list(self):
        from openkb.frontmatter import parse_fm
        text = "---\nentities:\n- name: OpenAI\n  type: organization\n---\n\nContent."
        meta, body = parse_fm(text)
        assert len(meta["entities"]) == 1
        assert meta["entities"][0]["name"] == "OpenAI"

    def test_parse_doc_type(self):
        from openkb.frontmatter import parse_fm
        text = "---\ndoc_type: short\nfull_text: sources/test.md\n---\n\nSummary."
        meta, body = parse_fm(text)
        assert meta["doc_type"] == "short"
        assert meta["full_text"] == "sources/test.md"

    def test_parse_values_with_triple_hyphen_without_truncating(self):
        from openkb.frontmatter import parse_fm
        text = (
            "---\n"
            "sources:\n"
            "- summaries/github---chongdashu-vibejam-starter-pack--free-vibe-jam-starter-pack---battle-te.md\n"
            "brief: test\n"
            "---\n\n"
            "# Body\n"
        )
        meta, body = parse_fm(text)
        assert meta["sources"] == ["summaries/github---chongdashu-vibejam-starter-pack--free-vibe-jam-starter-pack---battle-te.md"]
        assert meta["brief"] == "test"
        assert body.strip() == "# Body"


class TestSerializeFm:
    def test_serialize_simple(self):
        from openkb.frontmatter import serialize_fm
        meta = {"sources": ["a.md"], "brief": "test"}
        body = "Body content."
        text = serialize_fm(meta, body)
        assert text.startswith("---\n")
        assert "---\n\n" in text
        assert "Body content." in text

    def test_roundtrip(self):
        from openkb.frontmatter import parse_fm, serialize_fm
        original = "---\nsources: [a.md]\nbrief: test\n---\n\nBody."
        meta, body = parse_fm(original)
        result = serialize_fm(meta, body)
        meta2, body2 = parse_fm(result)
        assert meta2 == meta
        assert body2.strip() == body.strip()

    def test_roundtrip_preserves_provenance_contract_fields(self):
        from openkb.frontmatter import parse_fm, serialize_fm

        original = (
            "---\n"
            "updated_at: 2026-04-19T04:00:12Z\n"
            "source_count: 1\n"
            "supporting_sources:\n"
            "- sources/test.md\n"
            "supporting_pages:\n"
            "- summaries/test.md\n"
            "generation_mode: summary_write\n"
            "---\n\n"
            "Body."
        )

        meta, body = parse_fm(original)

        assert meta["updated_at"] == "2026-04-19T04:00:12Z"
        assert meta["source_count"] == 1
        assert meta["supporting_sources"] == ["sources/test.md"]
        assert meta["supporting_pages"] == ["summaries/test.md"]
        assert meta["generation_mode"] == "summary_write"

        result = serialize_fm(meta, body)
        meta2, body2 = parse_fm(result)

        assert meta2 == meta
        assert body2.strip() == body.strip()

    def test_empty_meta(self):
        from openkb.frontmatter import serialize_fm
        text = serialize_fm({}, "Just body.")
        assert "Just body." in text


class TestUpdateFm:
    def test_add_new_field(self):
        from openkb.frontmatter import update_fm
        text = "---\nsources: [a.md]\n---\n\nBody."
        result = update_fm(text, brief="New brief")
        assert "brief: New brief" in result
        assert "sources:" in result

    def test_replace_existing_field(self):
        from openkb.frontmatter import update_fm
        text = "---\nbrief: Old\n---\n\nBody."
        result = update_fm(text, brief="New")
        assert "brief: New" in result
        assert "Old" not in result

    def test_add_field_to_no_fm(self):
        from openkb.frontmatter import update_fm
        text = "# No FM\n\nBody."
        result = update_fm(text, sources="[a.md]")
        assert "sources:" in result
        assert "# No FM" in result

    def test_multiple_fields(self):
        from openkb.frontmatter import update_fm
        text = "---\n---\n\nBody."
        result = update_fm(text, brief="B", entity_type="organization")
        assert "brief: B" in result
        assert "entity_type: organization" in result
