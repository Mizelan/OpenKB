"""Wiki compilation pipeline for OpenKB.

Pipeline leveraging LLM prompt caching:
  Step 1: Build base context A (schema + document content).
  Step 2: A → generate summary.
  Step 3: A + summary → concepts plan (create/update/related).
  Step 4: Concurrent LLM calls (A cached) → generate new + rewrite updated concepts.
  Step 5: Code adds cross-ref links to related concepts, updates index.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import unicodedata
from pathlib import Path

from openkb.schema import get_agents_md

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are OpenKB's wiki compilation agent for a personal knowledge base.

{schema_md}

Write all content in {language} language.
Use [[wikilinks]] to connect related pages (e.g. [[concepts/attention]]).
"""

_SUMMARY_USER = """\
New document: {doc_name}

Full text:
{content}

Write a summary page for this document in Markdown.

Return a JSON object with two keys:
- "brief": A single sentence (under 100 chars) describing the document's main contribution
- "content": The full summary in Markdown. Include key concepts, findings, ideas, \
and [[wikilinks]] to concepts that could become cross-document concept pages

Return ONLY valid JSON, no fences.
"""


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

_LONG_DOC_SUMMARY_USER = """\
This is a PageIndex summary for long document "{doc_name}" (doc_id: {doc_id}):

{content}

Based on this structured summary, write a concise overview that captures \
the key themes and findings. This will be used to generate concept pages.

Return ONLY the Markdown content (no frontmatter, no code fences).
"""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _fmt_messages(messages: list[dict], max_content: int = 200) -> str:
    """Format messages for debug output, truncating long content."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if len(content) > max_content:
            preview = content[:max_content] + f"... ({len(content)} chars)"
        else:
            preview = content
        parts.append(f"      [{role}] {preview}")
    return "\n".join(parts)


def _llm_call(model: str, messages: list[dict], step_name: str, **kwargs) -> str:
    """Single LLM call via subprocess executor."""
    from openkb.executor import ExecutorConfig, run_llm_with_system
    logger.debug("LLM request [%s]:\n%s", step_name, _fmt_messages(messages))

    # Build system and user prompts from messages list
    system_parts = []
    user_parts = []
    for msg in messages:
        if msg.get("role") == "system":
            system_parts.append(msg["content"])
        else:
            user_parts.append(msg["content"])

    system_prompt = "\n".join(system_parts)
    user_prompt = "\n".join(user_parts)

    # Determine provider from model prefix (e.g. "anthropic/claude-sonnet-4-6" → "claude")
    cfg = kwargs.pop("executor_config", None) or ExecutorConfig()
    # If model starts with "anthropic/" or is a short alias like "sonnet", use claude provider
    if model and not model.startswith("anthropic/"):
        cfg.model = model

    result = run_llm_with_system(system_prompt, user_prompt, cfg)

    if result.error:
        logger.error("LLM error [%s]: %s", step_name, result.error)
        raise RuntimeError(f"LLM call failed ({step_name}): {result.error}")

    tokens_str = f"(in={result.input_tokens}, out={result.output_tokens})"
    sys.stdout.write(f"    {step_name}... {result.elapsed_seconds:.1f}s {tokens_str}\n")
    sys.stdout.flush()
    logger.debug("LLM response [%s]:\n%s", step_name, result.text[:500] + ("..." if len(result.text) > 500 else ""))
    return result.text.strip()


async def _llm_call_async(model: str, messages: list[dict], step_name: str, **kwargs) -> str:
    """Async LLM call — runs sync call in a thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: _llm_call(model, messages, step_name, **kwargs))


def _clean_concept_content(content: str) -> str:
    """Strip LLM meta-commentary from concept/summary page content.

    Removes patterns Claude CLI commonly injects when running with
    --dangerously-skip-permissions: insight blocks, conversational
    intros/outros, file listings, see-also lines, etc.
    """
    import re as _re

    # 1. Remove ★ Insight / ★ 인사이트 blocks (with or without backticks)
    #    Matches the opening line through the closing delimiter line.
    content = _re.sub(
        r'`?★\s*(?:Insight|인사이트)\s*─+`?\s*\n.*?(?:`?─+`?|──)\s*\n?',
        '', content, flags=_re.DOTALL
    )
    # Standalone ★ lines (no closing delimiter found above)
    content = _re.sub(r'^.*★\s*(?:Insight|인사이트).*$\n?', '', content, flags=_re.MULTILINE)

    # 2. Remove "생성된 파일:" section (file listing from tool use)
    content = _re.sub(
        r'생성된 파일:\n(?:- .*\n)*',
        '', content
    )

    # 3. Remove "Related Documents" section (index handles cross-references)
    content = _re.sub(
        r'\n## Related Documents\n(?:- \[\[.*\]\]\n)*',
        '', content
    )

    # 4. Remove "See also:" lines
    content = _re.sub(r'^See also:.*$\n?', '', content, flags=_re.MULTILINE)

    # 5. Strip leading meta-commentary before first markdown heading.
    #    Claude CLI often prepends conversational text before the actual content.
    #    Find the first line starting with # and discard everything before it.
    heading_match = _re.search(r'^#{1,3}\s', content, _re.MULTILINE)
    if heading_match:
        content = content[heading_match.start():]

    # 6. Remove any remaining isolated meta-commentary lines.
    #    These are lines that describe actions rather than content.
    meta_line_patterns = [
        # Korean action statements (하겠습니다, 했습니다, 합니다, etc.)
        r'^[^\n#|*\-]{0,80}(?:하겠습니다|했습니다|합니다|되었습니다|완료했습니다|추가하겠습니다|반영하겠습니다)\s*$',
        # English meta-commentary
        r'^(?:Let me|I will|I\'ll|I have|First|Now|Already|Creating|Generated|Writing|Updating|Updated)\b.*$',
    ]
    for pat in meta_line_patterns:
        content = _re.sub(pat, '', content, flags=_re.MULTILINE)

    # 7. Collapse multiple blank lines to max 2
    content = _re.sub(r'\n{3,}', '\n\n', content)

    return content.strip()


def _parse_json(text: str) -> list | dict:
    """Parse JSON from LLM response, handling fences, prose, and malformed JSON.

    Strips conversational wrapper text (greetings, commentary, insight blocks)
    that Claude CLI may emit before/after the actual JSON payload.
    """
    from json_repair import repair_json
    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # If the text doesn't start with { or [, find the first JSON structure.
    # This strips conversational wrapper text the LLM may emit.
    if cleaned and not cleaned.startswith(("{", "[")):
        brace_idx = cleaned.find("{")
        bracket_idx = cleaned.find("[")
        # Pick whichever comes first
        indices = [i for i in (brace_idx, bracket_idx) if i != -1]
        if indices:
            cleaned = cleaned[min(indices):]

    # Truncate trailing text after the closing bracket
    if cleaned:
        # Find the matching closing bracket for the opening one
        open_ch = cleaned[0]
        close_ch = "}" if open_ch == "{" else "]"
        depth = 0
        in_string = False
        escape = False
        end = len(cleaned)
        for i, c in enumerate(cleaned):
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        cleaned = cleaned[:end]

    result = json.loads(repair_json(cleaned.strip()))
    if not isinstance(result, (dict, list)):
        raise ValueError(f"Expected JSON object or array, got {type(result).__name__}")
    return result


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _read_wiki_context(wiki_dir: Path) -> tuple[str, list[str]]:
    """Read current index.md content and list of existing concept slugs."""
    index_path = wiki_dir / "index.md"
    index_content = index_path.read_text(encoding="utf-8") if index_path.exists() else ""

    concepts_dir = wiki_dir / "concepts"
    existing = sorted(p.stem for p in concepts_dir.glob("*.md")) if concepts_dir.exists() else []

    return index_content, existing


def _read_concept_briefs(wiki_dir: Path) -> str:
    """Read existing concept pages and return compact one-line summaries.

    For each concept, reads the ``brief:`` and ``entity_type:`` fields from
    YAML frontmatter if present; otherwise falls back to truncating the
    first 150 chars of the body (newlines collapsed to spaces). Formats
    each as ``- {slug} [{entity_type}]: {brief}``.

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


def _get_section_bounds(lines: list[str], heading: str) -> tuple[int, int] | None:
    """Return the [start, end) bounds for a Markdown H2 section."""
    for i, line in enumerate(lines):
        if line == heading:
            start = i + 1
            end = len(lines)
            for j in range(start, len(lines)):
                if lines[j].startswith("## "):
                    end = j
                    break
            return start, end
    return None


def _section_contains_link(lines: list[str], heading: str, link: str) -> bool:
    """Check whether an index entry already exists inside the named section."""
    bounds = _get_section_bounds(lines, heading)
    if bounds is None:
        return False

    start, end = bounds
    entry_prefix = f"- {link}"
    return any(line.startswith(entry_prefix) for line in lines[start:end])


def _replace_section_entry(lines: list[str], heading: str, link: str, entry: str) -> bool:
    """Replace the first matching entry within a specific section."""
    bounds = _get_section_bounds(lines, heading)
    if bounds is None:
        return False

    start, end = bounds
    entry_prefix = f"- {link}"
    for i in range(start, end):
        if lines[i].startswith(entry_prefix):
            lines[i] = entry
            return True
    return False


def _insert_section_entry(lines: list[str], heading: str, entry: str) -> bool:
    """Insert a new entry at the top of a specific section."""
    bounds = _get_section_bounds(lines, heading)
    if bounds is None:
        return False

    start, _ = bounds
    lines.insert(start, entry)
    return True



def _write_summary(wiki_dir: Path, doc_name: str, summary: str,
                    doc_type: str = "short") -> None:
    """Write summary page with frontmatter."""
    if summary.startswith("---"):
        end = summary.find("---", 3)
        if end != -1:
            summary = summary[end + 3:].lstrip("\n")
    summaries_dir = wiki_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    ext = "md" if doc_type == "short" else "json"
    fm_lines = [
        f"doc_type: {doc_type}",
        f"full_text: sources/{doc_name}.{ext}",
    ]
    frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n\n"
    (summaries_dir / f"{doc_name}.md").write_text(frontmatter + summary, encoding="utf-8")


_SAFE_NAME_RE = re.compile(r'[^\w\-]')


def _sanitize_concept_name(name: str) -> str:
    """Sanitize a concept name for safe use as a filename."""
    name = unicodedata.normalize("NFKC", name)
    sanitized = _SAFE_NAME_RE.sub("-", name).strip("-")
    return sanitized or "unnamed-concept"


def _write_concept(wiki_dir: Path, name: str, content: str, source_file: str, is_update: bool, brief: str = "", entity_type: str = "") -> None:
    """Write or update a concept page, managing the sources frontmatter."""
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_concept_name(name)
    path = (concepts_dir / f"{safe_name}.md").resolve()
    if not path.is_relative_to(concepts_dir.resolve()):
        logger.warning("Concept name escapes concepts dir: %s", name)
        return

    if is_update and path.exists():
        existing = path.read_text(encoding="utf-8")
        if source_file not in existing:
            if existing.startswith("---"):
                end = existing.find("---", 3)
                if end != -1:
                    fm = existing[:end + 3]
                    body = existing[end + 3:]
                    if "sources:" in fm:
                        fm = fm.replace("sources: [", f"sources: [{source_file}, ")
                    else:
                        fm = fm.replace("---\n", f"---\nsources: [{source_file}]\n", 1)
                    existing = fm + body
            else:
                existing = f"---\nsources: [{source_file}]\n---\n\n" + existing
        # Strip frontmatter from LLM content to avoid duplicate blocks
        clean = content
        if clean.startswith("---"):
            end = clean.find("---", 3)
            if end != -1:
                clean = clean[end + 3:].lstrip("\n")
        # Replace body with LLM rewrite (prompt asks for full rewrite, not delta)
        if existing.startswith("---"):
            end = existing.find("---", 3)
            if end != -1:
                existing = existing[:end + 3] + "\n\n" + clean
            else:
                existing = clean
        else:
            existing = clean
        if brief and existing.startswith("---"):
            end = existing.find("---", 3)
            if end != -1:
                fm = existing[:end + 3]
                body = existing[end + 3:]
                if "brief:" in fm:
                    fm = re.sub(r"brief:.*", f"brief: {brief}", fm)
                else:
                    fm = fm.replace("---\n", f"---\nbrief: {brief}\n", 1)
                existing = fm + body
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
        path.write_text(existing, encoding="utf-8")
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


def _add_related_link(wiki_dir: Path, concept_slug: str, doc_name: str, source_file: str) -> None:
    """Add a cross-reference link to an existing concept page (no LLM call)."""
    concepts_dir = wiki_dir / "concepts"
    path = concepts_dir / f"{concept_slug}.md"
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    link = f"[[summaries/{doc_name}]]"
    if link in text:
        return

    # Update sources in frontmatter
    if source_file not in text:
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                fm = text[:end + 3]
                body = text[end + 3:]
                if "sources:" in fm:
                    fm = fm.replace("sources: [", f"sources: [{source_file}, ")
                else:
                    fm = fm.replace("---\n", f"---\nsources: [{source_file}]\n", 1)
                text = fm + body
        else:
            text = f"---\nsources: [{source_file}]\n---\n\n" + text

    text += f"\n\nSee also: {link}"
    path.write_text(text, encoding="utf-8")


def _backlink_summary(wiki_dir: Path, doc_name: str, concept_slugs: list[str]) -> None:
    """Append missing concept wikilinks to the summary page (no LLM call).

    After all concepts are generated, this ensures the summary page links
    back to every related concept — closing the bidirectional link that
    concept pages already have toward the summary.

    If a ``## Related Concepts`` section already exists, new links are
    appended into it rather than creating a duplicate section.
    """
    summary_path = wiki_dir / "summaries" / f"{doc_name}.md"
    if not summary_path.exists():
        return

    text = summary_path.read_text(encoding="utf-8")
    missing = [slug for slug in concept_slugs if f"[[concepts/{slug}]]" not in text]
    if not missing:
        return

    new_links = "\n".join(f"- [[concepts/{s}]]" for s in missing)
    if "## Related Concepts" in text:
        # Append into existing section
        text = text.replace("## Related Concepts\n", f"## Related Concepts\n{new_links}\n", 1)
    else:
        text += f"\n\n## Related Concepts\n{new_links}\n"
    summary_path.write_text(text, encoding="utf-8")


def _backlink_concepts(wiki_dir: Path, doc_name: str, concept_slugs: list[str]) -> None:
    """Append missing summary wikilink to each concept page (no LLM call).

    Ensures every concept page links back to the source document's summary,
    regardless of whether the LLM included the link in its output.

    If a ``## Related Documents`` section already exists, the link is
    appended into it rather than creating a duplicate section.
    """
    link = f"[[summaries/{doc_name}]]"
    concepts_dir = wiki_dir / "concepts"

    for slug in concept_slugs:
        path = concepts_dir / f"{slug}.md"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if link in text:
            continue
        if "## Related Documents" in text:
            text = text.replace("## Related Documents\n", f"## Related Documents\n- {link}\n", 1)
        else:
            text += f"\n\n## Related Documents\n- {link}\n"
        path.write_text(text, encoding="utf-8")


# Language-specific section headers for index.md
_INDEX_SECTIONS = {
    "en": {"title": "Knowledge Base Index", "documents": "Documents", "concepts": "Concepts", "explorations": "Explorations"},
    "ko": {"title": "\uc9c0\uc2dd \ubca0\uc774\uc2a4 \uc778\ub371\uc2a4", "documents": "\ubb38\uc11c", "concepts": "\uac1c\ub150", "explorations": "\ud0d0\uad6c"},
}


def _section_heading(language: str, section: str) -> str:
    """Return the localized heading for an index section."""
    lang = _INDEX_SECTIONS.get(language, _INDEX_SECTIONS["en"])
    return f"## {lang[section]}"


def _make_index_template(language: str = "en") -> str:
    """Return the initial index.md content for the given language."""
    lang = _INDEX_SECTIONS.get(language, _INDEX_SECTIONS["en"])
    return (
        f"# {lang['title']}\n\n"
        f"## {lang['documents']}\n\n"
        f"## {lang['concepts']}\n\n"
        f"## {lang['explorations']}\n"
    )


def _update_index(
    wiki_dir: Path, doc_name: str, concept_names: list[str],
    doc_brief: str = "", concept_briefs: dict[str, str] | None = None,
    doc_type: str = "short", language: str = "en",
) -> None:
    """Append document and concept entries to index.md.

    When ``doc_brief`` or entries in ``concept_briefs`` are provided, entries
    are written as ``- [[link]] (type) — brief text``. Existing entries are
    detected within their own section by exact entry prefix and skipped to
    avoid duplicates.
    ``doc_type`` is ``"short"`` or ``"pageindex"`` — shown in the entry so the
    query agent knows how to access detailed content.
    ``language`` controls section headers (``"en"`` or ``"ko"``).
    """
    if concept_briefs is None:
        concept_briefs = {}

    doc_heading = _section_heading(language, "documents")
    concept_heading = _section_heading(language, "concepts")

    index_path = wiki_dir / "index.md"
    if not index_path.exists():
        index_path.write_text(_make_index_template(language), encoding="utf-8")

    lines = index_path.read_text(encoding="utf-8").split("\n")

    doc_link = f"[[summaries/{doc_name}]]"
    if not _section_contains_link(lines, doc_heading, doc_link):
        doc_entry = f"- {doc_link} ({doc_type})"
        if doc_brief:
            doc_entry += f" — {doc_brief}"
        _insert_section_entry(lines, doc_heading, doc_entry)

    for name in concept_names:
        concept_link = f"[[concepts/{name}]]"
        concept_entry = f"- {concept_link}"
        if name in concept_briefs:
            concept_entry += f" — {concept_briefs[name]}"
        if _section_contains_link(lines, concept_heading, concept_link):
            if name in concept_briefs:
                _replace_section_entry(lines, concept_heading, concept_link, concept_entry)
        else:
            _insert_section_entry(lines, concept_heading, concept_entry)

    index_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_COMPILE_CONCURRENCY = 5


async def _compile_concepts(
    wiki_dir: Path,
    kb_dir: Path,
    model: str,
    system_msg: dict,
    doc_msg: dict,
    summary: str,
    doc_name: str,
    max_concurrency: int,
    doc_brief: str = "",
    doc_type: str = "short",
    language: str = "en",
) -> None:
    """Shared Steps 2-4: concepts plan → generate/update → index.

    Uses ``_CONCEPTS_PLAN_USER`` to get a plan with create/update/related
    actions, then executes each action type accordingly.
    """
    source_file = f"summaries/{doc_name}.md"

    # --- Step 2: Get concepts plan (A cached) ---
    concept_briefs = _read_concept_briefs(wiki_dir)

    plan_raw = _llm_call(model, [
        system_msg,
        doc_msg,
        {"role": "assistant", "content": summary},
        {"role": "user", "content": _CONCEPTS_PLAN_USER.format(
            concept_briefs=concept_briefs,
        )},
    ], "concepts-plan", max_tokens=1024)

    try:
        parsed = _parse_json(plan_raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse concepts plan: %s", exc)
        logger.debug("Raw: %s", plan_raw)
        _update_index(wiki_dir, doc_name, [], doc_brief=doc_brief, doc_type=doc_type, language=language)
        return

    # Fallback: if LLM returns a flat list, treat all items as "create"
    if isinstance(parsed, list):
        plan = {"create": parsed, "update": [], "related": []}
    else:
        plan = {
            "create": parsed.get("create", []),
            "update": parsed.get("update", []),
            "related": parsed.get("related", []),
        }

    create_items = plan["create"]
    update_items = plan["update"]
    related_items = plan["related"]

    if not create_items and not update_items and not related_items:
        _update_index(wiki_dir, doc_name, [], doc_brief=doc_brief, doc_type=doc_type, language=language)
        return

    # --- Step 3: Generate/update concept pages concurrently (A cached) ---
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _gen_create(concept: dict) -> tuple[str, str, bool, str, str]:
        name = concept.get("name") or concept.get("title") or "unnamed-concept"
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
            entity_type = parsed.get("entity_type", "") or entity_type
            content = _clean_concept_content(parsed.get("content", raw))
        except (json.JSONDecodeError, ValueError):
            brief, content = "", _clean_concept_content(raw)
        return name, content, False, brief, entity_type

    async def _gen_update(concept: dict) -> tuple[str, str, bool, str, str]:
        name = concept.get("name") or concept.get("title") or "unnamed-concept"
        title = concept.get("title", name)
        entity_type = concept.get("entity_type", "")
        concept_path = wiki_dir / "concepts" / f"{_sanitize_concept_name(name)}.md"
        if concept_path.exists():
            raw_text = concept_path.read_text(encoding="utf-8")
            if raw_text.startswith("---"):
                parts = raw_text.split("---", 2)
                existing_content = parts[2].strip() if len(parts) >= 3 else raw_text
            else:
                existing_content = raw_text
        else:
            existing_content = "(page not found — create from scratch)"
        async with semaphore:
            raw = await _llm_call_async(model, [
                system_msg,
                doc_msg,
                {"role": "assistant", "content": summary},
                {"role": "user", "content": _CONCEPT_UPDATE_USER.format(
                    title=title, doc_name=doc_name,
                    existing_content=existing_content,
                )},
            ], f"update: {name}")
        try:
            parsed = _parse_json(raw)
            brief = parsed.get("brief", "")
            entity_type = parsed.get("entity_type", "") or entity_type
            content = _clean_concept_content(parsed.get("content", raw))
        except (json.JSONDecodeError, ValueError):
            brief, content = "", _clean_concept_content(raw)
        return name, content, True, brief, entity_type

    tasks = []
    tasks.extend(_gen_create(c) for c in create_items)
    tasks.extend(_gen_update(c) for c in update_items)

    concept_names: list[str] = []
    concept_briefs_map: dict[str, str] = {}

    if tasks:
        total = len(tasks)
        sys.stdout.write(f"    Generating {total} concept(s) (concurrency={max_concurrency})...\n")
        sys.stdout.flush()

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.warning("Concept generation failed: %s", r)
                continue
            name, page_content, is_update, brief, entity_type = r
            _write_concept(wiki_dir, name, page_content, source_file, is_update,
                          brief=brief, entity_type=entity_type)
            safe_name = _sanitize_concept_name(name)
            concept_names.append(safe_name)
            if brief:
                concept_briefs_map[safe_name] = brief

    # --- Step 3b: Process related items (code only, no LLM) ---
    sanitized_related = [_sanitize_concept_name(s) for s in related_items]
    for slug in sanitized_related:
        _add_related_link(wiki_dir, slug, doc_name, source_file)

    # --- Step 3c: Backlink — summary ↔ concepts (code only) ---
    all_concept_slugs = concept_names + sanitized_related
    if all_concept_slugs:
        _backlink_summary(wiki_dir, doc_name, all_concept_slugs)
        _backlink_concepts(wiki_dir, doc_name, all_concept_slugs)

    # --- Step 4: Update index (code only) ---
    _update_index(wiki_dir, doc_name, concept_names,
                  doc_brief=doc_brief, concept_briefs=concept_briefs_map,
                  doc_type=doc_type, language=language)


async def compile_short_doc(
    doc_name: str,
    source_path: Path,
    kb_dir: Path,
    model: str,
    max_concurrency: int = DEFAULT_COMPILE_CONCURRENCY,
) -> None:
    """Compile a short document using a multi-step LLM pipeline with caching.

    Step 1: Build base context A (schema + doc content), generate summary.
    Steps 2-4: Delegated to ``_compile_concepts``.
    """
    from openkb.config import load_config

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    language: str = config.get("language", "en")

    wiki_dir = kb_dir / "wiki"
    schema_md = get_agents_md(wiki_dir)
    content = source_path.read_text(encoding="utf-8")

    # Base context A: system + document
    system_msg = {"role": "system", "content": _SYSTEM_TEMPLATE.format(
        schema_md=schema_md, language=language,
    )}
    doc_msg = {"role": "user", "content": _SUMMARY_USER.format(
        doc_name=doc_name, content=content,
    )}

    # --- Step 1: Generate summary ---
    summary_raw = _llm_call(model, [system_msg, doc_msg], "summary")
    try:
        summary_parsed = _parse_json(summary_raw)
        doc_brief = summary_parsed.get("brief", "")
        summary = _clean_concept_content(summary_parsed.get("content", summary_raw))
    except (json.JSONDecodeError, ValueError):
        doc_brief = ""
        summary = _clean_concept_content(summary_raw)
    _write_summary(wiki_dir, doc_name, summary)

    # --- Steps 2-4: Concept plan → generate/update → index ---
    await _compile_concepts(
        wiki_dir, kb_dir, model, system_msg, doc_msg,
        summary, doc_name, max_concurrency, doc_brief=doc_brief,
        doc_type="short", language=language,
    )


async def compile_long_doc(
    doc_name: str,
    summary_path: Path,
    doc_id: str,
    kb_dir: Path,
    model: str,
    doc_description: str = "",
    max_concurrency: int = DEFAULT_COMPILE_CONCURRENCY,
) -> None:
    """Compile a long (PageIndex) document's concepts and index.

    The summary page is already written by the indexer. This function
    generates concept pages and updates the index.
    """
    from openkb.config import load_config

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    language: str = config.get("language", "en")

    wiki_dir = kb_dir / "wiki"
    schema_md = get_agents_md(wiki_dir)
    summary_content = summary_path.read_text(encoding="utf-8")

    # Base context A
    system_msg = {"role": "system", "content": _SYSTEM_TEMPLATE.format(
        schema_md=schema_md, language=language,
    )}
    doc_msg = {"role": "user", "content": _LONG_DOC_SUMMARY_USER.format(
        doc_name=doc_name, doc_id=doc_id, content=summary_content,
    )}

    # --- Step 1: Generate overview ---
    overview = _llm_call(model, [system_msg, doc_msg], "overview")

    # --- Steps 2-4: Concept plan → generate/update → index ---
    await _compile_concepts(
        wiki_dir, kb_dir, model, system_msg, doc_msg,
        overview, doc_name, max_concurrency, doc_brief=doc_description,
        doc_type="pageindex", language=language,
    )
