# LLM Wiki Feature Port — Design Document

## Goal

Port LLM Wiki's functional capabilities (2-step ingest, review system, graph engine, hybrid search, community detection) into OpenKB, solving concrete problems:

| Problem | Feature | Before | After |
|---------|---------|--------|-------|
| Concept duplication | 2-step ingest | "AI safety" / "AI 안전" / "인공지능 안전" as separate pages | Analysis step detects overlap, merges into 1 page |
| LLM errors auto-saved | Review system | Wrong info enters wiki unchecked | Discord alert → human confirmation |
| Isolated topics invisible | Graph insights | No way to know which topics are sparse | `openkb insights` surfaces gaps automatically |
| Shallow query results | 4-signal search | Only direct links found | Source overlap + Adamic-Adar expands results |
| No structural overview | Community detection | 200 pages in flat list | Auto-clustered communities with cohesion scores |

## Architecture Overview

```
pkm-bot (Go)                    OpenKB (Python fork)
────────────                     ─────────────────
intake → enrichment              2-step ingest (analyze → generate)
        │                              │
        └─→ raw/slug.md ──→ openkb add ──→ summaries/
                                             concepts/
                                             index.md
                                              │
                                    ┌─────────┼─────────┐
                                    │         │         │
                               relevance  community  insights
                               (4-signal) (Louvain)  (gaps)
                                    │         │         │
                                    └───── graph.json ──┘
                                              │
                              openkb query (hybrid search)
                              openkb lint  (structural + graph)
                              openkb review (judgment-deferred queue)
                              openkb insights (knowledge gap report)
```

## Key Design Decisions

1. **Feature branch strategy** — Each phase on a feature branch, PR to main. VectifyAI upstream sync remains easy.
2. **Existing executor** — All LLM calls use `_llm_call()` / `_llm_call_async()` in compiler.py. Graph analysis is pure CPU (networkx), no LLM needed.
3. **Auto graph rebuild** — `graph.json` rebuilt after every `openkb add`. `openkb insights` reads cached `graph.json`.
4. **pkm-bot integration** — Separate effort in the pkm-bot repo, not this PR. OpenKB provides CLI interfaces; pkm-bot wraps them.
5. **No LanceDB in initial phases** — Hybrid search starts with keyword + graph expansion. Vector search added later if needed.

## File Layout

```
openkb/
├── agent/
│   ├── compiler.py      # MODIFIED: add 2-step analysis step
│   ├── query.py          # MODIFIED: add graph expansion to search
│   └── ...
├── graph/
│   ├── __init__.py
│   ├── build.py          # NEW: build wikilink graph from wiki/
│   ├── relevance.py      # NEW: 4-signal relevance model
│   ├── community.py       # NEW: Louvain community detection
│   └── insights.py        # NEW: gap/surprising-connection detection
├── review/
│   ├── __init__.py
│   ├── parser.py          # NEW: parse ---REVIEW--- blocks from LLM output
│   ├── models.py           # NEW: ReviewItem dataclass
│   └── queue.py            # NEW: save/load review_queue.json
├── cli.py               # MODIFIED: add review, insights subcommands
├── config.py             # MODIFIED: add graph-related config defaults
└── ...
```

## Phase 0: Korean Setup + Entity Typing

**Goal**: Configure OpenKB for Korean output and add entity classification to concept planning.

### Changes

1. **`config.yaml` defaults** — `language: ko` (already supported)
2. **`compiler.py` prompt modifications**:
   - Section headers in Korean: `## 문서`, `## 개념`, `## 탐구`
   - `index.md` template uses Korean section headers
3. **Entity type in concept plan** — Extend `_CONCEPTS_PLAN_USER` prompt to include `entity_type` in each concept:
   ```json
   {"name": "concept-slug", "title": "Human-Readable Title", "entity_type": "person|org|technology|concept|event"}
   ```
4. **`_write_concept`** — Store `entity_type` in frontmatter

### Verification
- `openkb init` → config shows `language: ko`
- `openkb add` on a Korean document → Korean section headers, entity_type in frontmatter

## Phase 1: 2-Step Ingest + Review System

**Goal**: Add analysis step before generation; parse review blocks from LLM output; CLI review queue.

### 1A: Analysis Step

Add `_analyze_document()` prompt and integrate into `compile_short_doc`:

```python
# New prompt template
_ANALYSIS_USER = """\
Analyze this document BEFORE generating wiki content.

Existing wiki concepts:
{concept_briefs}

Analyze:
1. Key entities and their types (person, org, technology, concept, event)
2. Core arguments and claims
3. How this relates to existing wiki concepts (overlap, contradiction, extension)
4. Whether existing concepts need updating vs new ones needed

Return JSON:
- "entities": array of {{"name": "...", "type": "..."}}
- "concept_actions": array of {{"action": "create"|"update"|"skip", "name": "...", "reason": "..."}}
- "review_items": array of {{"type": "contradiction"|"duplicate"|"confirm", "title": "...", "description": "..."}}
"""
```

Pipeline change:
```
Before: doc → summary → concept_plan → concepts → index
After:  doc → analysis → summary → concept_plan → concepts → index
                         ↑                    ↑
                    analysis feeds      analysis feeds
                    review_items        concept_actions
```

The analysis `concept_actions` field feeds into `_CONCEPTS_PLAN_USER` as additional context, telling the LLM which concepts to create/update/skip with reasons. The existing concept plan prompt remains — analysis results are appended as a `Prior analysis:` section to guide better planning decisions.

### 1B: Review Block Parsing

LLM output may include `---REVIEW---` blocks. Parse and save to `.openkb/review_queue.json`:

```python
@dataclass
class ReviewItem:
    type: str          # "contradiction" | "duplicate" | "missing_page" | "confirm" | "suggestion"
    title: str
    description: str
    source_path: str
    affected_pages: list[str]
    search_queries: list[str]
    options: list[dict]  # [{"label": "Create Page"}, {"label": "Skip"}]
```

### 1C: CLI Review Command

```
openkb review              # List pending review items
openkb review --accept 1   # Accept item 1 (apply suggested action)
openkb review --skip 2     # Skip item 2 (dismiss)
```

### Verification
- `openkb add` produces `analysis` before `summary`
- Review items appear in `.openkb/review_queue.json`
- `openkb review` lists items; `--accept` / `--skip` modify queue
- Duplicate concepts are reduced (manual comparison before/after)

## Phase 2: Graph Engine + Insights

**Goal**: Build wikilink graph, compute relevance scores, detect communities, surface insights.

### 2A: Graph Construction (`openkb/graph/build.py`)

- Parse all `[[wikilinks]]` from `.md` files → directed edges
- Parse `sources:` frontmatter → source overlap edges
- Parse `entity_type:` frontmatter → node attributes
- Output: `networkx.Graph` with weighted edges

### 2B: 4-Signal Relevance (`openkb/graph/relevance.py`)

```
relevance(A, B) = (direct_link × 3.0)
               + (source_overlap × 4.0)
               + (adamic_adar × 1.5)
               + (type_affinity × 1.0)
```

- `direct_link`: A→B or B→A wikilink exists (0 or 3)
- `source_overlap`: shared `sources:` frontmatter count × 4
- `adamic_adar`: `1/log(degree)` sum for shared neighbors × 1.5
- `type_affinity`: same `entity_type` × 1

### 2C: Community Detection (`openkb/graph/community.py`)

- Use `python-louvain` for Louvain algorithm
- Compute cohesion per community (edge density / max possible)
- Flag communities with cohesion < 0.15 as "sparse"

### 2D: Insights (`openkb/graph/insights.py`)

- `find_orphans()`: nodes with degree ≤ 1
- `find_sparse_communities()`: communities with cohesion < 0.15
- `find_bridge_nodes()`: nodes connecting 3+ communities
- `find_surprising_connections()`: cross-community edges, type-variant edges

### 2E: Auto Graph Build

After every `openkb add`, rebuild `graph.json`:
```python
# In compile_short_doc / compile_long_doc, after index update:
from openkb.graph.build import build_and_save_graph
build_and_save_graph(wiki_dir)
```

### 2F: CLI Insights Command

```
openkb insights              # Show graph insights report
```

Output format:
```
Graph: 45 nodes, 78 edges, 5 communities

Surprising Connections:
  "regulation" ↔ "innovation" (cross-community, relevance: 8.2)
  "EU AI Act" ↔ "startup" (periphery-hub, relevance: 6.1)

Knowledge Gaps:
  Orphan nodes: quantum-computing (1 edge)
  Sparse communities: robotics (cohesion: 0.08)
  Bridge nodes: openai (3 communities)

Communities:
  AI/ML (28 pages, cohesion: 0.42)
  Regulation/Policy (15 pages, cohesion: 0.31)
  ...
```

### Verification
- `openkb add` creates/updates `.openkb/graph.json`
- `openkb insights` shows communities, gaps, surprising connections
- Graph rebuild is idempotent

## Phase 3: Hybrid Search (Keyword + Graph Expansion)

**Goal**: Enhance `openkb query` with graph-based result expansion.

### Changes

Modify `query.py`'s search strategy:

```
Current:
1. Read index.md → find relevant concepts → read pages → synthesize answer

Enhanced:
1. Read index.md → keyword match
2. Load graph.json → expand to related pages via relevance scores
3. Read top-k pages (keyword + expanded)
4. Synthesize answer with broader context
```

Implementation:
- Add graph expansion as a tool for the query agent
- The agent gets a new tool: `search_related(page_name, top_k)` that uses `graph.json`
- No LanceDB in this phase — graph expansion provides the "semantic" boost

### Verification
- `openkb query "AI regulation"` returns results including indirectly related pages
- Graph expansion correctly follows relevance scores

## Dependencies

```toml
# Added to pyproject.toml [project.dependencies]
dependencies = [
    # ... existing ...
    "networkx>=3.0",          # Graph construction and algorithms
    "python-louvain>=0.16",   # Louvain community detection
]

# Optional (Phase 3+)
[project.optional-dependencies]
vector = ["lancedb>=0.6"]    # Vector search (future)
```

## Estimated Scope

| Feature | New Python LOC | Modified Files |
|---------|---------------|-----------------|
| Korean setup + entity type | ~30 | compiler.py, schema.py |
| 2-step ingest | ~100 | compiler.py |
| Review system | ~120 | new: review/parser.py, review/models.py, review/queue.py |
| CLI review command | ~40 | cli.py |
| Graph build | ~60 | new: graph/build.py |
| 4-signal relevance | ~100 | new: graph/relevance.py |
| Community detection | ~40 | new: graph/community.py |
| Insights | ~80 | new: graph/insights.py |
| CLI insights command | ~30 | cli.py |
| Hybrid search expansion | ~80 | agent/query.py, agent/tools.py |
| **Total** | **~680** | 4 modified, 7 new |

## Phase Dependencies

```
Phase 0 (Korean + entity type)
    ↓
Phase 1 (2-step ingest + review)
    ↓  (entity_type needed for type_affinity signal)
Phase 2 (graph engine + insights)
    ↓  (graph.json needed for search expansion)
Phase 3 (hybrid search)
    ↓  (CLI interfaces stable)
Phase 4 (pkm-bot integration — separate repo)
```

## Not In Scope

- Chrome extension (pkm-bot has bookmark intake)
- Desktop UI / Tauri IPC
- KaTeX rendering
- Chat session persistence (Discord sessions replace this)
- LanceDB integration (deferred to post-Phase 3)
- OmegaWiki 9-edge typed system (optional future addition)
- Vector embedding search (deferred; graph expansion is sufficient for now)

## Testing Strategy

- **Unit tests** for each new module (graph/build, graph/relevance, graph/community, graph/insights, review/parser, review/queue)
- **Integration test**: `openkb add` → verify graph.json created → `openkb insights` shows output
- **Regression test**: existing `compile_short_doc` tests still pass after 2-step refactor
- **Manual verification**: Korean document ingestion produces Korean headers and entity_type frontmatter