# Codebase Quality Fix Implementation Plan

> **For Claude Code:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 5개 검증 이슈 근원 해결 — frontmatter 파이프라인 재설계, entities → graph 활용, GraphLoadError, tuple return

**Architecture:** pyyaml 기반 frontmatter 유틸리티 도입으로 문자열 조작 제거. entities를 summary frontmatter에 저장 → build_graph에서 entity_mention 엣지 생성 → relevance에 5번째 시그널 추가. GraphLoadError 예외로 손상 감지.

**Tech Stack:** Python, pyyaml (기존 의존성), networkx, pytest

---

### Task 1: Create `openkb/frontmatter.py` with parse_fm/serialize_fm/update_fm

**Files:**
- Create: `openkb/frontmatter.py`
- Test: `tests/test_frontmatter.py`

**Step 1: Write the failing tests**

```python
# tests/test_frontmatter.py
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
        assert "sources: [a.md]" in result

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
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_frontmatter.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'openkb.frontmatter'`

**Step 3: Write implementation**

```python
# openkb/frontmatter.py
"""PyYAML-based frontmatter utilities for OpenKB.

Provides parse → dict modify → serialize flow, replacing manual string
manipulation and regex-based field injection.
"""
from __future__ import annotations

import yaml


def parse_fm(text: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown text.

    Returns (metadata_dict, body_string).
    If no valid frontmatter, returns ({}, original_text).
    """
    if not text.startswith("---"):
        return {}, text

    end = text.find("---", 3)
    if end == -1:
        return {}, text

    fm_block = text[3:end].strip()
    body = text[end + 3:].lstrip("\n")

    try:
        meta = yaml.safe_load(fm_block)
    except yaml.YAMLError:
        return {}, text

    if not isinstance(meta, dict):
        return {}, text

    return meta, body


def serialize_fm(meta: dict, body: str) -> str:
    """Serialize metadata dict + body back to markdown with YAML frontmatter.

    Empty dict produces body-only output (no frontmatter block).
    """
    if not meta:
        return body

    fm = yaml.dump(meta, allow_unicode=True, default_flow_style=False).strip()
    return f"---\n{fm}\n---\n\n{body}"


def update_fm(text: str, **fields) -> str:
    """Parse frontmatter, update specified fields, re-serialize.

    Single-pass field injection — no regex, no re-parsing.
    """
    meta, body = parse_fm(text)
    for key, value in fields.items():
        meta[key] = value
    return serialize_fm(meta, body)
```

**Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_frontmatter.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/frontmatter.py tests/test_frontmatter.py
git commit -m "feat: add pyyaml-based frontmatter utilities (parse_fm/serialize_fm/update_fm)"
```

---

### Task 2: Migrate compiler.py to use frontmatter.py

**Files:**
- Modify: `openkb/agent/compiler.py` (lines 434-564: _split_frontmatter, _inject_fm_field, _strip_frontmatter, _write_concept, _write_summary)
- Modify: `tests/test_compiler.py`

**Step 1: Write failing tests for _write_summary with entities**

```python
# Add to tests/test_compiler.py

class TestWriteSummaryWithEntities:
    def test_writes_entities_to_frontmatter(self, tmp_path):
        from openkb.agent.compiler import _write_summary
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        entities = [{"name": "OpenAI", "type": "organization"}]
        _write_summary(wiki, "my-doc", "# Summary\n\nContent.", entities=entities)
        text = (wiki / "summaries" / "my-doc.md").read_text()
        assert "OpenAI" in text
        assert "organization" in text

    def test_write_summary_without_entities(self, tmp_path):
        from openkb.agent.compiler import _write_summary
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        _write_summary(wiki, "my-doc", "# Summary\n\nContent.")
        text = (wiki / "summaries" / "my-doc.md").read_text()
        assert "entities:" not in text
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_compiler.py::TestWriteSummaryWithEntities -v
```
Expected: FAIL — `_write_summary() got an unexpected keyword argument 'entities'`

**Step 3: Migrate compiler.py**

Changes:
1. Add `from openkb.frontmatter import parse_fm, serialize_fm, update_fm` at top imports
2. Delete `_split_frontmatter`, `_inject_fm_field`, `_strip_frontmatter` functions (lines 434-457)
3. Rewrite `_write_concept` (lines 520-564):
```python
def _write_concept(wiki_dir: Path, name: str, content: str, source_file: str, is_update: bool, brief: str = "", entity_type: str = "") -> None:
    """Write or update a concept page, managing the sources frontmatter."""
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_concept_name(name)
    path = (concepts_dir / f"{safe_name}.md").resolve()
    if not path.is_relative_to(concepts_dir.resolve()):
        logger.warning("Concept name escapes concepts dir: %s", name)
        return

    clean_body = parse_fm(content)[1] if content.startswith("---") else content

    if is_update and path.exists():
        existing = path.read_text(encoding="utf-8")
        meta, _ = parse_fm(existing)
        sources = meta.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        if source_file not in sources:
            sources.append(source_file)
        meta["sources"] = sources
        if brief:
            meta["brief"] = brief
        if entity_type:
            meta["entity_type"] = entity_type
        path.write_text(serialize_fm(meta, clean_body), encoding="utf-8")
    else:
        meta = {"sources": [source_file]}
        if brief:
            meta["brief"] = brief
        if entity_type:
            meta["entity_type"] = entity_type
        path.write_text(serialize_fm(meta, clean_body), encoding="utf-8")
```

4. Rewrite `_write_summary` (lines 413-428):
```python
def _write_summary(wiki_dir: Path, doc_name: str, summary: str,
                   doc_type: str = "short", entities: list | None = None) -> None:
    """Write summary page with frontmatter."""
    clean_summary = parse_fm(summary)[1] if summary.startswith("---") else summary
    summaries_dir = wiki_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    ext = "md" if doc_type == "short" else "json"
    meta = {
        "doc_type": doc_type,
        "full_text": f"sources/{doc_name}.{ext}",
    }
    if entities:
        meta["entities"] = entities
    path = summaries_dir / f"{doc_name}.md"
    path.write_text(serialize_fm(meta, clean_summary), encoding="utf-8")
```

5. In `compile_short_doc` (line ~962): pass entities to `_write_summary`:
```python
_write_summary(wiki_dir, doc_name, summary, entities=analysis.get("entities", []))
```

6. In `compile_long_doc` (line ~1024): same:
```python
_write_summary(wiki_dir, doc_name, overview, entities=analysis.get("entities", []))
```

Note: For `compile_long_doc`, overview is used instead of summary. Adjust accordingly — the `_write_summary` call may not exist yet; check if long doc needs it added.

7. Remove old imports of `_split_frontmatter`, `_inject_fm_field`, `_strip_frontmatter` from test file imports if present.

**Step 4: Run all compiler tests**

```bash
python -m pytest tests/test_compiler.py -v
```
Expected: PASS (existing tests may need minor adjustments for YAML format changes — e.g., `sources: [a.md]` becomes `sources:\n- a.md` in pyyaml output)

**Step 5: Commit**

```bash
git add openkb/agent/compiler.py tests/test_compiler.py
git commit -m "refactor: migrate compiler.py to pyyaml frontmatter utilities"
```

---

### Task 3: Migrate build.py to use frontmatter.py + add GraphLoadError + entity_mention edges + tuple return

**Files:**
- Modify: `openkb/graph/build.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing tests**

```python
# Add to tests/test_graph.py

class TestGraphLoadError:
    def test_load_graph_raises_on_corrupted_json(self, tmp_path):
        from openkb.graph.build import load_graph, GraphLoadError
        bad_path = tmp_path / "graph.json"
        bad_path.write_text("{invalid json", encoding="utf-8")
        with pytest.raises(GraphLoadError):
            load_graph(bad_path)

    def test_load_graph_raises_on_missing_file(self, tmp_path):
        from openkb.graph.build import load_graph, GraphLoadError
        with pytest.raises(GraphLoadError):
            load_graph(tmp_path / "nonexistent.json")

class TestBuildAndSaveReturnsTuple:
    def test_returns_graph_and_path(self, tmp_path):
        from openkb.graph.build import build_and_save_graph
        wiki = _make_wiki(tmp_path)
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        result = build_and_save_graph(wiki, openkb_dir)
        assert isinstance(result, tuple)
        assert len(result) == 2
        graph, path = result
        assert graph.number_of_nodes() > 0
        assert path.exists()

class TestEntityMentionEdges:
    def test_entity_mention_edges_created(self, tmp_path):
        """Pages sharing an entity should get entity_mention edges."""
        from openkb.graph.build import build_graph
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "summaries").mkdir()
        (wiki / "concepts").mkdir()
        (wiki / "explorations").mkdir()
        (wiki / "summaries" / "doc1.md").write_text(
            "---\ndoc_type: short\nentities:\n- name: OpenAI\n  type: organization\n---\n\nDoc1 content.",
            encoding="utf-8",
        )
        (wiki / "summaries" / "doc2.md").write_text(
            "---\ndoc_type: short\nentities:\n- name: OpenAI\n  type: organization\n---\n\nDoc2 content.",
            encoding="utf-8",
        )
        g = build_graph(wiki)
        assert g.has_edge("summaries/doc1", "summaries/doc2")
        edge_data = g.get_edge_data("summaries/doc1", "summaries/doc2")
        assert edge_data["edge_type"] == "entity_mention"

    def test_no_entity_mention_without_shared_entities(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "summaries").mkdir()
        (wiki / "concepts").mkdir()
        (wiki / "explorations").mkdir()
        (wiki / "summaries" / "doc1.md").write_text(
            "---\ndoc_type: short\nentities:\n- name: OpenAI\n  type: organization\n---\n\nContent.",
            encoding="utf-8",
        )
        (wiki / "summaries" / "doc2.md").write_text(
            "---\ndoc_type: short\nentities:\n- name: Google\n  type: organization\n---\n\nContent.",
            encoding="utf-8",
        )
        g = build_graph(wiki)
        if g.has_edge("summaries/doc1", "summaries/doc2"):
            edge_data = g.get_edge_data("summaries/doc1", "summaries/doc2")
            assert edge_data["edge_type"] != "entity_mention"
```

**Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_graph.py::TestGraphLoadError tests/test_graph.py::TestBuildAndSaveReturnsTuple tests/test_graph.py::TestEntityMentionEdges -v
```
Expected: FAIL

**Step 3: Implement changes in build.py**

1. Replace `_parse_frontmatter` with `parse_fm` import:
```python
from openkb.frontmatter import parse_fm
```

2. Add `GraphLoadError`:
```python
class GraphLoadError(Exception):
    """Raised when graph.json cannot be loaded (corrupted or missing)."""
```

3. Modify `build_graph` to use `parse_fm` instead of `_parse_frontmatter`:
```python
# In the loop over md_file:
meta, body = parse_fm(text)
```
Remove `_parse_frontmatter` function entirely.

4. Add entity_mention edge logic after source_overlap edges in `build_graph`:
```python
# Add entity_mention edges
entity_to_nodes: dict[str, list[str]] = {}
for nid, data in page_data.items():
    for ent in data["meta"].get("entities", []):
        if isinstance(ent, dict) and "name" in ent and "type" in ent:
            key = f"{ent['name']}|{ent['type']}"
            entity_to_nodes.setdefault(key, []).append(nid)
    # Also store mentioned_entities as node attribute
    ent_names = []
    for ent in data["meta"].get("entities", []):
        if isinstance(ent, dict) and "name" in ent:
            ent_names.append(ent["name"])
    g.nodes[nid]["mentioned_entities"] = ent_names

for key, nodes in entity_to_nodes.items():
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i], nodes[j]
            if g.has_edge(a, b):
                edge_data = g.get_edge_data(a, b)
                current_weight = edge_data.get("weight", 1)
                g[a][b]["weight"] = current_weight + 1
                existing_type = edge_data.get("edge_type", "wikilink")
                g[a][b]["edge_type"] = f"{existing_type}+entity_mention"
            else:
                g.add_edge(a, b, edge_type="entity_mention", weight=1)
```

5. Modify `load_graph` to raise `GraphLoadError`:
```python
def load_graph(path: Path) -> nx.Graph:
    """Deserialise graph from JSON file. Raises GraphLoadError on failure."""
    if not path.exists():
        raise GraphLoadError(str(path), "File not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise GraphLoadError(str(path), str(exc)) from exc
    # ... rest unchanged
```

6. Modify `build_and_save_graph` return type:
```python
def build_and_save_graph(wiki_dir: Path, openkb_dir: Path | None = None) -> tuple[nx.Graph, Path]:
    """Build graph and save to .openkb/graph.json. Returns (graph, path)."""
    # ... existing logic ...
    return g, graph_path
```

7. Update `save_graph` to include `mentioned_entities` in node serialization (it should already be included via `dict(graph.nodes[nid])`).

**Step 4: Update existing tests that call `build_and_save_graph`**

In `tests/test_graph.py` and `tests/test_compiler.py`, update calls from:
```python
graph_path = build_and_save_graph(wiki, openkb_dir)
```
to:
```python
graph, graph_path = build_and_save_graph(wiki, openkb_dir)
```

Update `TestInsightsCLI.test_insights_command_runs` and `TestSearchRelatedPages` tests accordingly.

Also update `test_build_and_save_convenience` to unpack tuple.

**Step 5: Run all graph tests**

```bash
python -m pytest tests/test_graph.py -v
```
Expected: PASS

**Step 6: Commit**

```bash
git add openkb/graph/build.py tests/test_graph.py
git commit -m "feat: build.py — pyyaml FM, GraphLoadError, entity_mention edges, tuple return"
```

---

### Task 4: Add entity_mention signal to relevance.py

**Files:**
- Modify: `openkb/graph/relevance.py`
- Modify: `tests/test_graph.py`

**Step 1: Write failing test**

```python
# Add to TestRelevance class in tests/test_graph.py

def test_entity_mention_signal(self):
    from openkb.graph.relevance import relevance_score
    import networkx as nx
    g = nx.Graph()
    g.add_node("p", entity_type="", sources=[], mentioned_entities=["OpenAI", "GPT"])
    g.add_node("q", entity_type="", sources=[], mentioned_entities=["OpenAI", "LLM"])
    # 1 shared entity → 1 * 2.0 = 2.0
    score = relevance_score(g, "p", "q")
    assert score >= 2.0

def test_no_entity_mention_signal_without_overlap(self):
    from openkb.graph.relevance import relevance_score
    import networkx as nx
    g = nx.Graph()
    g.add_node("p", entity_type="", sources=[], mentioned_entities=["OpenAI"])
    g.add_node("q", entity_type="", sources=[], mentioned_entities=["Google"])
    score = relevance_score(g, "p", "q")
    assert score < 2.0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_graph.py::TestRelevance::test_entity_mention_signal -v
```
Expected: FAIL — score < 2.0 because entity_mention signal not implemented

**Step 3: Add entity_mention signal to relevance_score**

In `openkb/graph/relevance.py`, add after type_affinity:

```python
# 5. Entity mention overlap
entities_a = set(graph.nodes[node_a].get("mentioned_entities", []) or [])
entities_b = set(graph.nodes[node_b].get("mentioned_entities", []) or [])
shared_entities = len(entities_a & entities_b)
score += shared_entities * 2.0
```

Update module docstring to include 5th signal.

**Step 4: Run tests**

```bash
python -m pytest tests/test_graph.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/graph/relevance.py tests/test_graph.py
git commit -m "feat: add entity_mention signal (5th) to relevance scoring"
```

---

### Task 5: Update CLI for GraphLoadError handling + tuple return

**Files:**
- Modify: `openkb/cli.py` (lines 901-949: insights command)
- Modify: `tests/test_graph.py` (TestInsightsCLI)

**Step 1: Write failing test**

```python
# Add to TestInsightsCLI in tests/test_graph.py

def test_insights_reports_corruption(self, tmp_path):
    from click.testing import CliRunner
    from openkb.cli import cli
    openkb_dir = tmp_path / ".openkb"
    openkb_dir.mkdir()
    # Write corrupted graph.json
    (openkb_dir / "graph.json").write_text("{bad", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["--kb-dir", str(tmp_path), "insights"])
    assert "손상" in result.output or "corrupt" in result.output.lower()
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_graph.py::TestInsightsCLI::test_insights_reports_corruption -v
```
Expected: FAIL — current code silently returns empty graph

**Step 3: Update cli.py insights command**

Replace the graph loading block (lines ~917-922) with:

```python
from openkb.graph.build import build_and_save_graph, GraphLoadError, load_graph
from openkb.graph.insights import generate_insights

wiki_dir = kb_dir / "wiki"
graph_path = kb_dir / ".openkb" / "graph.json"

try:
    if graph_path.exists():
        graph = load_graph(graph_path)
    else:
        openkb_dir = kb_dir / ".openkb"
        graph, _ = build_and_save_graph(wiki_dir, openkb_dir)
except GraphLoadError:
    click.echo("graph.json이 손상되었습니다. `openkb add`로 재생성하세요.")
    return
```

**Step 4: Run all tests**

```bash
python -m pytest tests/test_graph.py tests/test_cli.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/cli.py tests/test_graph.py
git commit -m "fix: insights CLI handles GraphLoadError + uses tuple return"
```

---

### Task 6: Update tools.py for GraphLoadError handling + update compiler.py tuple return

**Files:**
- Modify: `openkb/agent/tools.py` (lines 197-225)
- Modify: `openkb/agent/compiler.py` (line 904: _compile_concepts graph rebuild)

**Step 1: Update search_related_pages in tools.py**

```python
def search_related_pages(page_name: str, top_k: int, kb_dir: str) -> str:
    """Return top-k related pages for *page_name* using the knowledge graph."""
    from openkb.graph.build import load_graph, GraphLoadError
    from openkb.graph.relevance import top_related

    graph_path = Path(kb_dir) / ".openkb" / "graph.json"
    if not graph_path.exists():
        return "Graph not available. Run 'openkb add' first."

    try:
        graph = load_graph(graph_path)
    except GraphLoadError:
        return "graph.json이 손상되었습니다. `openkb add`로 재생성하세요."

    if page_name not in graph.nodes:
        return f"Page '{page_name}' not found in graph."

    results = top_related(graph, page_name, k=top_k)
    if not results:
        return f"No related pages found for '{page_name}'."

    return "\n".join(f"{slug} (relevance: {score:.2f})" for slug, score in results)
```

**Step 2: Update _compile_concepts in compiler.py**

Change line ~904 from:
```python
build_and_save_graph(wiki_dir, kb_dir / ".openkb")
```
to:
```python
build_and_save_graph(wiki_dir, kb_dir / ".openkb")  # returns (graph, path) — unused here
```
No behavioral change needed since the return value is discarded in the try/except block.

**Step 3: Run all tests**

```bash
python -m pytest tests/ -v
```
Expected: PASS

**Step 4: Commit**

```bash
git add openkb/agent/tools.py openkb/agent/compiler.py
git commit -m "fix: tools.py handles GraphLoadError; compiler.py adapts to tuple return"
```

---

### Task 7: Fix _add_related_link and _backlink_* functions to use parse_fm/serialize_fm

**Files:**
- Modify: `openkb/agent/compiler.py` (lines 567-600: _add_related_link, _backlink_summary, _backlink_concepts)

**Step 1: Identify remaining manual frontmatter manipulations**

The `_add_related_link` and `_backlink_concepts` functions also do manual frontmatter string manipulation. Migrate them to use `parse_fm`/`serialize_fm`.

**Step 2: Rewrite _add_related_link**

Replace manual `text[:end + 3]` manipulation with:
```python
def _add_related_link(wiki_dir: Path, concept_slug: str, doc_name: str, source_file: str) -> None:
    concepts_dir = wiki_dir / "concepts"
    path = concepts_dir / f"{concept_slug}.md"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    link = f"[[summaries/{doc_name}]]"
    if link in text:
        return

    meta, body = parse_fm(text)
    sources = meta.get("sources", [])
    if isinstance(sources, str):
        sources = [sources]
    if source_file not in sources:
        sources.append(source_file)
    meta["sources"] = sources

    if "## Related Documents" in body:
        updated = body.replace("## Related Documents", f"## Related Documents\n- {link}")
    else:
        updated = body.rstrip() + f"\n\n## Related Documents\n- {link}\n"

    path.write_text(serialize_fm(meta, updated), encoding="utf-8")
```

**Step 3: Rewrite _backlink_concepts similarly**

```python
def _backlink_concepts(wiki_dir: Path, doc_name: str, concept_slugs: list[str]) -> None:
    # Uses parse_fm/serialize_fm for source update
```

**Step 4: Run compiler tests**

```bash
python -m pytest tests/test_compiler.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/agent/compiler.py
git commit -m "refactor: migrate _add_related_link and _backlink_* to pyyaml FM"
```

---

### Task 8: Final regression check

**Step 1: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: ALL PASS

**Step 2: Verify entities flow end-to-end**

Run the existing `TestCompileShortDoc` and `TestCompileLongDoc` integration tests to verify the entities passthrough doesn't break the pipeline.

**Step 3: Verify test coverage for new features**

- `test_frontmatter.py` — parse_fm, serialize_fm, update_fm
- `test_graph.py` — GraphLoadError, entity_mention edges, tuple return
- `test_compiler.py` — _write_summary with entities, _write_concept with pyyaml FM

**Step 4: Commit any remaining fixes**

```bash
git add -A
git commit -m "test: add coverage for frontmatter, GraphLoadError, entity_mention"
```