# Codebase Quality Fix — Design

5개 검증 이슈 근원 해결 설계.

## Issue Summary

| # | Issue | Severity | Root Cause |
|---|-------|----------|------------|
| 1 | `entities` from analysis unused | Medium | Data produced but never consumed |
| 2 | `_write_concept` 4x frontmatter re-parse | Low | String-level FM manipulation, no in-memory model |
| 3 | `_inject_fm_field` regex injection risk | Low | `re.sub` replacement string unsanitized |
| 4 | Insights CLI: redundant build+load I/O | Low | `build_and_save_graph` returns path only |
| 5 | `load_graph` silent corruption | Low | Empty graph on error, no cause differentiation |

## Design

### 1. Frontmatter Pipeline Redesign (Issues #2, #3)

**New module**: `openkb/frontmatter.py`

Three functions:

- `parse_fm(text) -> (dict, body)` — YAML frontmatter to dict + body string. Returns `({}, text)` if no FM.
- `serialize_fm(meta, body) -> str` — dict to YAML frontmatter block + body.
- `update_fm(text, **fields) -> str` — Parse → modify dict → serialize. Single-pass field injection.

**Migration plan**:

| Old | New | Action |
|-----|-----|--------|
| `_split_frontmatter` | `parse_fm` | Replace in compiler.py |
| `_inject_fm_field` | `update_fm` / dict manipulation | Delete |
| `_strip_frontmatter` | `parse_fm` → use body only | Delete |
| `build.py _parse_frontmatter` | `parse_fm` | Replace |

**`_write_concept` rewrite** (compiler.py):

Current: 47 lines, 4x `_split_frontmatter` calls on incrementally modified string.

New flow (~30 lines):
```
if is_update and path.exists():
    meta, body = parse_fm(existing)
    # sources injection
    sources = meta.get("sources", [])
    if source_file not in sources:
        sources.append(source_file)
    meta["sources"] = sources
    # body replacement
    clean_body = parse_fm(content)[1] if content.startswith("---") else content
    # field updates
    if brief: meta["brief"] = brief
    if entity_type: meta["entity_type"] = entity_type
    path.write_text(serialize_fm(meta, clean_body))
else:
    meta = {"sources": [source_file]}
    if brief: meta["brief"] = brief
    if entity_type: meta["entity_type"] = entity_type
    clean_body = parse_fm(content)[1] if content.startswith("---") else content
    path.write_text(serialize_fm(meta, clean_body))
```

No regex. No re-parsing. Dict operations only.

### 2. Entities → Graph Enrichment (Issue #1)

**Data flow**:
```
_analyze_document → {entities: [{name, type}, ...]}
    ↓
compile_short_doc / compile_long_doc
    ↓
_write_summary(wiki_dir, doc_name, summary, entities=entities)
    ↓ frontmatter: entities: [{name: "OpenAI", type: "organization"}, ...]
build_graph
    ↓ read entities from summary node attributes
    ↓ create entity_mention edges between pages sharing same entity
relevance_score
    ↓ add entity_mention signal: shared_entities * 2.0
```

**Changes**:

- `compiler.py`:
  - `_write_summary`: accept `entities` param, write to frontmatter via `update_fm`
  - `compile_short_doc`: pass `analysis.get("entities", [])` to `_write_summary`
  - `compile_long_doc`: same

- `build.py`:
  - `build_graph`: read `entities` from node attributes (already parsed by `parse_fm`)
  - Add `entity_mention` edges: group nodes by shared `{name, type}` entity, connect pairs with `edge_type="entity_mention"`, `weight=shared_count`
  - Entity names also added as node `mentioned_entities` attribute for debugging

- `relevance.py`:
  - Add 5th signal: `entity_mention` — count shared entities between two nodes, `score += shared * 2.0`
  - Shared entities computed from `mentioned_entities` node attribute

**Updated relevance signals**:

| Signal | Weight | Source |
|--------|--------|--------|
| direct_link | 3.0 | wikilink edge |
| source_overlap | 4.0 * n | shared source files |
| adamic_adar | 1.5 * sum | shared neighbor rarity |
| type_affinity | 1.0 | same entity_type |
| entity_mention | 2.0 * n | shared named entities (NEW) |

### 3. Graph Loading/Building Improvements (Issues #4, #5)

**3a. `load_graph` error differentiation**:

- New exception: `GraphLoadError(Exception)` in `build.py`
- `load_graph`: raise `GraphLoadError(path, cause)` on JSON decode failure instead of returning empty graph
- Callers catch `GraphLoadError`:
  - `cli.py insights`: "graph.json 손상됨. `openkb add`로 재생성하세요"
  - `tools.py search_related_pages`: return error message with corruption hint
  - `compiler.py _compile_concepts`: existing try/except catches this (already catches Exception)

**3b. `build_and_save_graph` return value**:

- Change: `Path` → `(nx.Graph, Path)` tuple
- `compiler.py _compile_concepts`: use returned graph directly (no reload needed)
- `cli.py insights`: when building graph, use returned graph instead of `load_graph`
- Backward compat: callers that only need path can unpack `_, path = build_and_save_graph(...)`

## Changed Files Summary

| File | Change Type |
|------|-------------|
| `openkb/frontmatter.py` | NEW — pyyaml-based FM utilities |
| `openkb/agent/compiler.py` | MODIFY — use parse_fm/serialize_fm, entities passthrough |
| `openkb/graph/build.py` | MODIFY — use parse_fm, entity_mention edges, GraphLoadError, tuple return |
| `openkb/graph/relevance.py` | MODIFY — add entity_mention signal |
| `openkb/cli.py` | MODIFY — insights: use returned graph, GraphLoadError handling |
| `openkb/agent/tools.py` | MODIFY — search_related_pages: GraphLoadError handling |

## Validation

- `openkb add` → summary page frontmatter에 entities 필드 존재 확인
- `openkb insights` → graph.json에 entity_mention 엣지 존재 확인
- `openkb query` → search_related 결과에 entity_mention 시그널 반영 확인
- 손상된 graph.json → "손상됨" 메시지 출력 확인
- 기존 wiki 데이터로 regression 없음 확인