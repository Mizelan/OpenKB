# Phase 0: Korean Setup + Entity Typing — Implementation Plan

> **For Claude Code:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Configure OpenKB for Korean output and add entity classification to concept planning.

**Architecture:** Modify compiler prompts to output Korean section headers and `entity_type` fields. Extend `_write_concept` to store entity_type in YAML frontmatter. Extend `_update_index` and `_read_concept_briefs` to handle entity_type. Update schema defaults for Korean.

**Tech Stack:** Python 3.10+, pytest for testing

---

### Task 1: Add entity_type to concept plan prompt

**Files:**
- Modify: `openkb/agent/compiler.py:55-80` (prompt templates)
- Test: `tests/test_compiler.py`

**Step 1: Write the failing test**

In `tests/test_compiler.py`, add a test that verifies the concepts plan prompt requests `entity_type`:

```python
class TestConceptsPlanPrompt:
    def test_plan_requests_entity_type(self):
        """Verify _CONCEPTS_PLAN_USER asks for entity_type in create items."""
        from openkb.agent.compiler import _CONCEPTS_PLAN_USER
        assert "entity_type" in _CONCEPTS_PLAN_USER
        assert "person" in _CONCEPTS_PLAN_USER or "organization" in _CONCEPTS_PLAN_USER
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_compiler.py::TestConceptsPlanPrompt -v`
Expected: FAIL — "entity_type" not found in prompt

**Step 3: Update _CONCEPTS_PLAN_USER prompt**

In `openkb/agent/compiler.py`, modify `_CONCEPTS_PLAN_USER` to include `entity_type`:

```python
_CONCEPTS_PLAN_USER = """\
Based on the summary above, decide how to update the wiki's concept pages.

Existing concept pages:
{concept_briefs}

Return a JSON object with three keys:

1. "create" — new concepts not covered by any existing page. Array of objects:
   {{"name": "concept-slug", "title": "Human-Readable Title", "entity_type": "person|organization|technology|concept|event|location|work"}}

2. "update" — existing concepts that have significant new information from \
this document worth integrating. Array of objects:
   {{"name": "existing-slug", "title": "Existing Title", "entity_type": "person|organization|technology|concept|event|location|work"}}

3. "related" — existing concepts tangentially related to this document but \
not needing content changes, just a cross-reference link. Array of slug strings.

Rules:
- For the first few documents, create 2-3 foundational concepts at most.
- Do NOT create a concept that overlaps with an existing one — use "update".
- Do NOT create concepts that are just the document topic itself.
- "related" is for lightweight cross-linking only, no content rewrite needed.
- entity_type must be one of: person, organization, technology, concept, event, location, work

Return ONLY valid JSON, no fences, no explanation.
"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_compiler.py::TestConceptsPlanPrompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/agent/compiler.py tests/test_compiler.py
git commit -m "feat: add entity_type to concept plan prompt"
```

---

### Task 2: Store entity_type in concept frontmatter

**Files:**
- Modify: `openkb/agent/compiler.py:337-398` (`_write_concept` and `_gen_create`)
- Test: `tests/test_compiler.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_compiler.py::TestWriteConceptEntityType -v`
Expected: FAIL — `entity_type` parameter not yet accepted

**Step 3: Implement entity_type in _write_concept**

Modify `_write_concept` signature and logic in `openkb/agent/compiler.py`:

```python
def _write_concept(wiki_dir: Path, name: str, content: str, source_file: str,
                   is_update: bool, brief: str = "", entity_type: str = "") -> None:
```

In the new-concept branch (the `else` block around line 388), add entity_type to frontmatter:

```python
    else:
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].lstrip("\n")
        fm_lines = [f"sources: [{source_file}]"]
        if brief:
            fm_lines.append(f"brief: {brief}")
        if entity_type:
            fm_lines.append(f"entity_type: {entity_type}")
        frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n\n"
        path.write_text(frontmatter + content, encoding="utf-8")
```

In the update branch (around line 347-386), preserve existing entity_type:

After the existing brief-handling block (around line 381-386), add:

```python
        if entity_type and existing.startswith("---"):
            end = existing.find("---", 3)
            if end != -1:
                fm = existing[:end + 3]
                body = existing[end + 3:]
                if "entity_type:" in fm:
                    fm = re.sub(r"entity_type:.*", f"entity_type: {entity_type}", fm)
                else:
                    fm = fm.replace("---\n", f"---\nentity_type: {entity_type}\n", 1)
                existing = fm + body
```

**Step 4: Pass entity_type from _gen_create and _gen_update**

In `_compile_concepts`, modify `_gen_create` to extract entity_type from the plan:

```python
    async def _gen_create(concept: dict) -> tuple[str, str, bool, str, str]:
        name = concept["name"]
        title = concept.get("title", name)
        entity_type = concept.get("entity_type", "")
        async with semaphore:
            raw = await _llm_call_async(model, [
                system_msg,
                doc_msg,
                {"role": "assistant", "content": summary},
                {"role": "user", "content": _CONCEPT_PAGE_USER.format(
                    title=title, doc_name=doc_name,
                    update_instruction="",
                )},
            ], f"concept: {name}")
        try:
            parsed = _parse_json(raw)
            brief = parsed.get("brief", "")
            content = parsed.get("content", raw)
        except (json.JSONDecodeError, ValueError):
            brief, content = "", raw
        return name, content, False, brief, entity_type
```

Similarly for `_gen_update`:

```python
    async def _gen_update(concept: dict) -> tuple[str, str, bool, str, str]:
        name = concept["name"]
        title = concept.get("title", name)
        entity_type = concept.get("entity_type", "")
        # ... (rest same as existing, but return entity_type too)
        return name, content, True, brief, entity_type
```

Update the result processing to unpack 5-tuple and pass entity_type:

```python
        for r in results:
            if isinstance(r, Exception):
                logger.warning("Concept generation failed: %s", r)
                continue
            name, page_content, is_update, brief, entity_type = r
            _write_concept(wiki_dir, name, page_content, source_file, is_update,
                          brief=brief, entity_type=entity_type)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_compiler.py::TestWriteConceptEntityType -v`
Expected: PASS

**Step 6: Commit**

```bash
git add openkb/agent/compiler.py tests/test_compiler.py
git commit -m "feat: store entity_type in concept frontmatter"
```

---

### Task 3: Include entity_type in concept briefs

**Files:**
- Modify: `openkb/agent/compiler.py:218-254` (`_read_concept_briefs`)
- Test: `tests/test_compiler.py`

**Step 1: Write the failing test**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_compiler.py::TestReadConceptBriefsWithEntityType -v`
Expected: PASS (already works — briefs include the full frontmatter line scan)

Actually, let me verify this. The current `_read_concept_briefs` reads `brief:` from frontmatter but does NOT read `entity_type`. We need it to include entity_type in the one-line summary.

**Step 3: Modify _read_concept_briefs to include entity_type**

In `openkb/agent/compiler.py`, modify `_read_concept_briefs`:

```python
def _read_concept_briefs(wiki_dir: Path) -> str:
    """Read existing concept pages and return compact one-line summaries.

    For each concept, reads the ``brief:`` and ``entity_type:`` fields from
    YAML frontmatter if present; otherwise falls back to truncating the
    first 150 chars of the body. Formats each as
    ``- {slug} [{entity_type}]: {brief}``.

    Returns "(none yet)" if the concepts directory is missing or empty.
    """
    concepts_dir = wiki_dir / "concepts"
    if not concepts_dir.exists():
        return "(none yet)"

    md_files = sorted(concepts_dir.glob("*.md"))
    if not md_files:
        return "(none yet)"

    lines: list[str] = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        brief = ""
        entity_type = ""
        body = text
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                fm = text[:end + 3]
                body = text[end + 3:]
                for line in fm.split("\n"):
                    if line.startswith("brief:"):
                        brief = line[len("brief:"):].strip()
                    elif line.startswith("entity_type:"):
                        entity_type = line[len("entity_type:"):].strip()
        if not brief:
            brief = body.strip().replace("\n", " ")[:150]
        if brief:
            type_tag = f" [{entity_type}]" if entity_type else ""
            lines.append(f"- {path.stem}{type_tag}: {brief}")

    return "\n".join(lines) or "(none yet)"
```

**Step 4: Run tests**

Run: `pytest tests/test_compiler.py::TestReadConceptBriefsWithEntityType -v`
Expected: PASS

**Step 5: Commit**

```bash
git add openkb/agent/compiler.py tests/test_compiler.py
git commit -m "feat: include entity_type in concept briefs"
```

---

### Task 4: Add Korean section headers to index template

**Files:**
- Modify: `openkb/agent/compiler.py:503-506` (index template in `_update_index`)
- Modify: `openkb/cli.py:289-294` (index template in `init`)
- Modify: `openkb/schema.py:7-8` (AGENTS.md directory listing)
- Test: `tests/test_compiler.py`

**Step 1: Write the failing test**

```python
class TestKoreanIndexTemplate:
    def test_init_creates_korean_index_when_language_is_ko(self, tmp_path):
        """When language is ko, init should create index.md with Korean headers."""
        # This tests that the config-driven header system works
        from openkb.schema import AGENTS_MD
        # Verify AGENTS_MD contains the section structure
        assert "Documents" in AGENTS_MD or "문서" in AGENTS_MD
```

Wait — the design says Korean headers should be driven by config. But the current `_update_index` hardcodes English section headers `## Documents`, `## Concepts`, `## Explorations`. We need to make these language-dependent.

Actually, looking more carefully at the code, `_update_index` uses hardcoded `## Documents`, `## Concepts`, `## Explorations` section headers. The index template in `cli.py` init also hardcodes these. The design says to add Korean section headers.

The cleanest approach is to add a language parameter to `_update_index` and the init command, and map language codes to section headers. But this is a significant refactor. Let me think about a simpler approach.

**Simpler approach:** Make the section headers configurable in `_update_index` via a parameter, and derive them from language in the calling code.

**Step 1: Write the failing test**

```python
class TestKoreanSectionHeaders:
    def test_update_index_korean_headers(self, tmp_path):
        """_update_index should use Korean headers when language='ko'."""
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text(
            "# 지식 베이스 인덱스\n\n## 문서\n\n## 개념\n\n## 탐구\n",
            encoding="utf-8",
        )
        _update_index(wiki, "my-doc", ["attention"],
                     doc_brief="Introduces transformers",
                     concept_briefs={"attention": "Focus mechanism"})
        text = (wiki / "index.md").read_text()
        assert "[[summaries/my-doc]]" in text
        assert "## 문서" in text
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_compiler.py::TestKoreanSectionHeaders -v`
Expected: FAIL — `_update_index` hardcodes `## Documents`

**Step 3: Make section headers language-aware**

Add a helper function near the top of `compiler.py`:

```python
# Language-specific section headers for index.md
_INDEX_SECTIONS = {
    "en": {"documents": "Documents", "concepts": "Concepts", "explorations": "Explorations"},
    "ko": {"documents": "문서", "concepts": "개념", "explorations": "탐구"},
}


def _section_heading(language: str, section: str) -> str:
    """Return the localized heading for an index section."""
    lang = _INDEX_SECTIONS.get(language, _INDEX_SECTIONS["en"])
    return f"## {lang[section]}"
```

Modify `_update_index` to accept a `language` parameter:

```python
def _update_index(
    wiki_dir: Path, doc_name: str, concept_names: list[str],
    doc_brief: str = "", concept_briefs: dict[str, str] | None = None,
    doc_type: str = "short", language: str = "en",
) -> None:
```

Replace hardcoded `## Documents`, `## Concepts`, `## Explorations` with:

```python
doc_heading = _section_heading(language, "documents")
concept_heading = _section_heading(language, "concepts")
```

Update the `_section_contains_link`, `_replace_section_entry`, `_insert_section_entry` calls to use these headings instead of hardcoded strings.

Similarly update `compile_short_doc` and `compile_long_doc` to pass `language` to `_update_index`.

**Step 4: Update init command for Korean headers**

In `openkb/cli.py`, modify the `init` command to check config for language:

```python
# In init command, after saving config:
language = config.get("language", "en")
sections = _INDEX_SECTIONS.get(language, _INDEX_SECTIONS["en"])
Path("wiki/index.md").write_text(
    f"# {'지식 베이스 인덱스' if language == 'ko' else 'Knowledge Base Index'}\n\n"
    f"## {sections['documents']}\n\n"
    f"## {sections['concepts']}\n\n"
    f"## {sections['explorations']}\n",
    encoding="utf-8",
)
```

**Step 5: Run tests to verify they pass**

Run: `pytest tests/test_compiler.py -v`
Expected: ALL PASS

Also run existing tests to verify no regression:

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```bash
git add openkb/agent/compiler.py openkb/cli.py tests/test_compiler.py
git commit -m "feat: add Korean section headers for index.md"
```

---

### Task 5: Add entity_type to concept page prompt

**Files:**
- Modify: `openkb/agent/compiler.py:82-95` (`_CONCEPT_PAGE_USER`)

**Step 1: Update _CONCEPT_PAGE_USER to request entity_type in frontmatter**

The concept page prompt should instruct the LLM to include `entity_type` in its output, so it gets stored in frontmatter:

```python
_CONCEPT_PAGE_USER = """\
Write the concept page for: {title}

This concept relates to the document "{doc_name}" summarized above.
{update_instruction}

Return a JSON object with three keys:
- "brief": A single sentence (under 100 chars) defining this concept
- "entity_type": One of: person, organization, technology, concept, event, location, work
- "content": The full concept page in Markdown. Include clear explanation, \
key details from the source document, and [[wikilinks]] to related concepts \
and [[summaries/{doc_name}]]

Return ONLY valid JSON, no fences.
"""
```

Also update `_CONCEPT_UPDATE_USER` similarly:

```python
_CONCEPT_UPDATE_USER = """\
Update the concept page for: {title}

Current content of this page:
{existing_content}

New information from document "{doc_name}" (summarized above) should be \
integrated into this page. Rewrite the full page incorporating the new \
information naturally — do not just append. Maintain existing \
[[wikilinks]] and add new ones where appropriate.

Return a JSON object with three keys:
- "brief": A single sentence (under 100 chars) defining this concept (may differ from before)
- "entity_type": One of: person, organization, technology, concept, event, location, work
- "content": The rewritten full concept page in Markdown

Return ONLY valid JSON, no fences.
"""
```

**Step 2: Update _gen_create and _gen_update to extract entity_type from LLM response**

In the result parsing of both `_gen_create` and `_gen_update`:

```python
        try:
            parsed = _parse_json(raw)
            brief = parsed.get("brief", "")
            entity_type = parsed.get("entity_type", "")
            content = parsed.get("content", raw)
        except (json.JSONDecodeError, ValueError):
            brief, content, entity_type = "", raw, ""
        return name, content, False, brief, entity_type
```

**Step 3: Run existing tests**

Run: `pytest tests/test_compiler.py -v`
Expected: ALL PASS (entity_type defaults to "" when LLM doesn't provide it)

**Step 4: Commit**

```bash
git add openkb/agent/compiler.py
git commit -m "feat: request entity_type from LLM in concept page prompts"
```

---

### Task 6: Run full test suite and verify no regressions

**Files:** None (verification only)

**Step 1: Run complete test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

**Step 2: Manual integration test**

```bash
cd /path/to/test/kb
openkb init  # Select language "ko" when prompted
openkb add test-document.md  # Verify Korean headers and entity_type in output
```

Verify:
- `wiki/index.md` has Korean section headers (## 문서, ## 개념, ## 탐구)
- Concept pages have `entity_type:` in frontmatter
- `_read_concept_briefs` includes entity_type in output

**Step 3: Commit any fixes if needed**

```bash
git add -A
git commit -m "fix: phase 0 regression fixes"
```

---

### Summary

| Task | What | LOC | Files Modified |
|------|------|-----|----------------|
| 1 | entity_type in concept plan prompt | ~10 | compiler.py, test_compiler.py |
| 2 | Store entity_type in frontmatter | ~20 | compiler.py, test_compiler.py |
| 3 | Include entity_type in concept briefs | ~15 | compiler.py, test_compiler.py |
| 4 | Korean section headers | ~30 | compiler.py, cli.py, test_compiler.py |
| 5 | entity_type in concept page prompt | ~10 | compiler.py |
| 6 | Full test suite verification | 0 | None |
| **Total** | | **~85** | 3 modified, 1 test file |