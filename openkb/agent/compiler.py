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
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from openkb.frontmatter import parse_fm, serialize_fm
from openkb.json_utils import extract_json
from openkb.schema import get_agents_md
from openkb.review import ReviewItem, parse_review_blocks, ReviewQueue

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

_ANALYSIS_USER = """\
Analyze this document against the existing wiki concepts.

Existing concept pages:
{concept_briefs}

Document content:
{content}

Return a JSON object with three keys:

1. "entities" — key entities mentioned in the document. Array of objects:
   {{"name": "entity-name", "type": "person|organization|technology|concept|event|location|work"}}

2. "concept_actions" — how this document relates to existing concepts and what new ones to create. Array of objects:
   {{"action": "create|update|skip", "name": "concept-slug", "reason": "brief justification"}}

3. "review_items" — issues or suggestions that need human review before proceeding. Array of objects:
   {{"type": "contradiction|duplicate|missing_page|confirm|suggestion", \
"title": "short title", "description": "what needs review", \
"source_path": "path in wiki", "affected_pages": ["list of pages"], \
"search_queries": ["queries to find related info"], "options": [{{"action": "create|merge|skip"}}]}}

Be conservative with review_items — only flag genuine conflicts, duplicates, or gaps.
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


_CONCEPTS_ONLY_DOC_CONTEXT_USER = """\
Regenerate concept pages and index links for document "{doc_name}".
Use only the summary provided in the assistant message as the source of truth.
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
    from openkb.executor import ExecutorConfig, build_executor_config, run_llm_with_system
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

    # Determine provider from model name when no explicit config given
    cfg = kwargs.pop("executor_config", None)
    if cfg is None:
        cfg = build_executor_config(model=model)

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

    Uses extract_json for bracket matching, then attempts json_repair
    on failure as a fallback for malformed LLM output.
    """
    from json_repair import repair_json
    result = extract_json(text)
    if result is not None:
        if not isinstance(result, (dict, list)):
            raise ValueError(f"Expected JSON object or array, got {type(result).__name__}")
        return result

    # Fallback: json_repair on the raw text for malformed JSON
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1:] if first_nl != -1 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
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
            brief = _embedded_json_brief(body)
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


def _utcnow_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _summary_source_path(doc_name: str, doc_type: str) -> str:
    ext = "md" if doc_type == "short" else "json"
    return f"sources/{doc_name}.{ext}"


def _summary_page_path(doc_name: str) -> str:
    return f"summaries/{doc_name}.md"


def _build_provenance_meta(
    *,
    generation_mode: str,
    supporting_sources: list[str],
    supporting_pages: list[str],
    previous_meta: dict | None = None,
) -> dict:
    previous_meta = previous_meta or {}
    legacy_sources = _as_string_list(previous_meta.get("sources"))
    prior_supporting_sources = _as_string_list(previous_meta.get("supporting_sources"))
    prior_supporting_pages = _as_string_list(previous_meta.get("supporting_pages"))

    merged_sources = _dedupe_strings(
        prior_supporting_sources
        + [source for source in legacy_sources if not source.startswith("summaries/")]
        + supporting_sources
    )
    merged_pages = _dedupe_strings(
        prior_supporting_pages
        + [source for source in legacy_sources if source.startswith("summaries/")]
        + supporting_pages
    )

    return {
        "updated_at": _utcnow_iso(),
        "source_count": len(merged_sources),
        "supporting_sources": merged_sources,
        "supporting_pages": merged_pages,
        "generation_mode": generation_mode,
    }


def _write_summary(wiki_dir: Path, doc_name: str, summary: str,
                    doc_type: str = "short", entities: list | None = None) -> None:
    """Write summary page with frontmatter."""
    clean_summary = parse_fm(summary)[1] if summary.startswith("---") else summary
    summaries_dir = wiki_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    full_text_path = _summary_source_path(doc_name, doc_type)
    summary_path = _summary_page_path(doc_name)
    meta = {
        "doc_type": doc_type,
        "full_text": full_text_path,
    }
    meta.update(
        _build_provenance_meta(
            generation_mode="summary_write",
            supporting_sources=[full_text_path],
            supporting_pages=[summary_path],
        )
    )
    if entities:
        meta["entities"] = entities
    path = summaries_dir / f"{doc_name}.md"
    path.write_text(serialize_fm(meta, clean_summary), encoding="utf-8")


def _summary_brief_from_body(body: str) -> str:
    """Extract a short one-line brief from an existing summary body."""
    text = body.strip()
    if not text:
        return ""

    m = re.search(r"^## (?:한줄 요약|Summary)\s*\n+(.+?)(?:\n## |\Z)", text, re.MULTILINE | re.DOTALL)
    if m:
        brief = m.group(1).strip()
    else:
        brief = ""
        for block in re.split(r"\n\s*\n", text):
            line = block.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("- "):
                line = line[2:].strip()
            brief = line
            break

    brief = re.sub(r"\[\[([^\]]+)\]\]", r"\1", brief)
    brief = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", brief)
    brief = re.sub(r"`([^`]+)`", r"\1", brief)
    brief = re.sub(r"\s+", " ", brief).strip()
    return brief[:100]


def _embedded_json_brief(body: str) -> str:
    """Extract a brief from legacy JSON-shaped concept content if present."""
    text = body.lstrip()
    if not text.startswith("{"):
        return ""

    for key in ("concept_page", "concept", "summary_page", "summary"):
        m = re.search(rf'"{key}"\s*:\s*\{{.*?"brief"\s*:\s*"([^"]+)"', text, re.DOTALL)
        if m:
            brief = re.sub(r"\s+", " ", m.group(1)).strip()
            return brief[:100]
    return ""


def _load_existing_summary(wiki_dir: Path, doc_name: str) -> tuple[dict, str, str]:
    """Load summary metadata/body/brief for concept-only regeneration."""
    path = wiki_dir / "summaries" / f"{doc_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Existing summary not found: {path}")
    meta, body = parse_fm(path.read_text(encoding="utf-8"))
    return meta, body, _summary_brief_from_body(body)


_SAFE_NAME_RE = re.compile(r'[^\w\-]')


def _sanitize_concept_name(name: str) -> str:
    """Sanitize a concept name for safe use as a filename."""
    name = unicodedata.normalize("NFKC", name)
    sanitized = _SAFE_NAME_RE.sub("-", name).strip("-")
    return sanitized or "unnamed-concept"


# ---------------------------------------------------------------------------
# Analysis step
# ---------------------------------------------------------------------------


def _analyze_document(
    model: str,
    system_msg: dict,
    doc_msg: dict,
    concept_briefs: str,
) -> dict:
    """Run the analysis step and return entities, concept_actions, and review_items.

    Returns a dict with keys: entities, concept_actions, review_items.
    On failure (bad JSON, LLM error), returns empty defaults.
    """
    analysis_raw = _llm_call(model, [
        system_msg,
        doc_msg,
        {"role": "user", "content": _ANALYSIS_USER.format(
            concept_briefs=concept_briefs,
            content=doc_msg["content"],
        )},
    ], "analysis")

    try:
        parsed = _parse_json(analysis_raw)
        if not isinstance(parsed, dict):
            logger.warning("Analysis response was not a dict, using defaults")
            return {"entities": [], "concept_actions": [], "review_items": []}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse analysis response: %s", exc)
        return {"entities": [], "concept_actions": [], "review_items": []}

    entities = parsed.get("entities", [])
    concept_actions = parsed.get("concept_actions", [])
    raw_review_items = parsed.get("review_items", [])

    # Parse review items through ReviewItem for validation
    review_items: list[ReviewItem] = []
    for raw in raw_review_items:
        try:
            review_items.append(ReviewItem.from_dict(raw))
        except (KeyError, ValueError) as exc:
            logger.warning("Skipping invalid review item from analysis: %s", exc)

    return {
        "entities": entities,
        "concept_actions": concept_actions,
        "review_items": review_items,
    }


def _write_concept(
    wiki_dir: Path,
    name: str,
    content: str,
    source_file: str,
    is_update: bool,
    brief: str = "",
    entity_type: str = "",
    supporting_source: str | None = None,
    supporting_page: str | None = None,
) -> None:
    """Write or update a concept page, managing the sources frontmatter."""
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_concept_name(name)
    path = (concepts_dir / f"{safe_name}.md").resolve()
    if not path.is_relative_to(concepts_dir.resolve()):
        logger.warning("Concept name escapes concepts dir: %s", name)
        return

    clean_body = parse_fm(content)[1] if content.startswith("---") else content
    generation_mode = "concept_update" if is_update else "concept_create"

    if is_update and path.exists():
        existing = path.read_text(encoding="utf-8")
        meta, _ = parse_fm(existing)
        meta["sources"] = _dedupe_strings(_as_string_list(meta.get("sources")) + [source_file])
        meta.update(
            _build_provenance_meta(
                generation_mode=generation_mode,
                supporting_sources=[supporting_source or source_file],
                supporting_pages=[supporting_page or source_file],
                previous_meta=meta,
            )
        )
        if brief:
            meta["brief"] = brief
        if entity_type:
            meta["entity_type"] = entity_type
        path.write_text(serialize_fm(meta, clean_body), encoding="utf-8")
    else:
        meta = {"sources": [source_file]}
        meta.update(
            _build_provenance_meta(
                generation_mode=generation_mode,
                supporting_sources=[supporting_source or source_file],
                supporting_pages=[supporting_page or source_file],
            )
        )
        if brief:
            meta["brief"] = brief
        if entity_type:
            meta["entity_type"] = entity_type
        path.write_text(serialize_fm(meta, clean_body), encoding="utf-8")


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
        meta, body = parse_fm(text)
        if "## Related Documents" in body:
            updated = body.replace("## Related Documents\n", f"## Related Documents\n- {link}\n", 1)
        else:
            updated = body.rstrip() + f"\n\n## Related Documents\n- {link}\n"
        path.write_text(serialize_fm(meta, updated), encoding="utf-8")


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
    analysis_context: dict | None = None,
) -> None:
    """Shared Steps 2-4: concepts plan → generate/update → index.

    Uses ``_CONCEPTS_PLAN_USER`` to get a plan with create/update/related
    actions, then executes each action type accordingly.

    If *analysis_context* is provided, concept_actions from the analysis
    step are injected as prior context into the concept plan prompt.
    """
    source_file = _summary_page_path(doc_name)
    supporting_source = _summary_source_path(doc_name, doc_type)

    # --- Step 2: Get concepts plan (A cached) ---
    concept_briefs = _read_concept_briefs(wiki_dir)

    # Build concept plan prompt, optionally with analysis context
    plan_prompt = _CONCEPTS_PLAN_USER.format(concept_briefs=concept_briefs)
    if analysis_context and analysis_context.get("concept_actions"):
        actions_text = "\n".join(
            f"- {a.get('action', '?')}: {a.get('name', '?')} — {a.get('reason', '')}"
            for a in analysis_context["concept_actions"]
        )
        plan_prompt = f"Prior analysis suggests the following concept actions:\n{actions_text}\n\n{plan_prompt}"

    plan_raw = _llm_call(model, [
        system_msg,
        doc_msg,
        {"role": "assistant", "content": summary},
        {"role": "user", "content": plan_prompt},
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

    # Filter out non-dict items from create/update lists (LLM may return malformed items)
    create_items = [c for c in plan["create"] if isinstance(c, dict)]
    update_items = [c for c in plan["update"] if isinstance(c, dict)]
    related_items = [r for r in plan["related"] if isinstance(r, str)]

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
            if not isinstance(parsed, dict):
                raise ValueError("concept response was not a JSON object")
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
            if not isinstance(parsed, dict):
                raise ValueError("concept response was not a JSON object")
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
                          brief=brief, entity_type=entity_type,
                          supporting_source=supporting_source,
                          supporting_page=source_file)
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

    # --- Step 5: Auto graph rebuild (non-blocking) ---
    try:
        from openkb.graph.build import build_and_save_graph
        build_and_save_graph(wiki_dir, kb_dir / ".openkb")
    except Exception as exc:
        logger.warning("Graph rebuild failed (non-fatal): %s", exc)

    # --- Step 6: Internal maintenance (non-blocking) ---
    try:
        from openkb.maintenance import run_internal_maintenance
        run_internal_maintenance(kb_dir)
    except Exception as exc:
        logger.warning("Internal maintenance failed (non-fatal): %s", exc)


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
    summary_only = os.environ.get("OPENKB_SUMMARY_ONLY") == "1"
    concepts_only = os.environ.get("OPENKB_CONCEPTS_ONLY") == "1"

    # Base context A: system + document
    system_msg = {"role": "system", "content": _SYSTEM_TEMPLATE.format(
        schema_md=schema_md, language=language,
    )}

    analysis: dict = {}
    if concepts_only:
        _, summary, doc_brief = _load_existing_summary(wiki_dir, doc_name)
        doc_msg = {"role": "user", "content": _CONCEPTS_ONLY_DOC_CONTEXT_USER.format(
            doc_name=doc_name,
        )}
    else:
        content = source_path.read_text(encoding="utf-8")
        doc_msg = {"role": "user", "content": _SUMMARY_USER.format(
            doc_name=doc_name, content=content,
        )}

        if not summary_only:
            # --- Step 0: Analysis step ---
            concept_briefs = _read_concept_briefs(wiki_dir)
            analysis = _analyze_document(
                model=model,
                system_msg=system_msg,
                doc_msg=doc_msg,
                concept_briefs=concept_briefs,
            )

            # Save review items to queue
            review_items = analysis.get("review_items", [])
            if review_items:
                openkb_dir = kb_dir / ".openkb"
                queue = ReviewQueue(openkb_dir)
                queue.add(review_items)

        # --- Step 1: Generate summary ---
        summary_raw = _llm_call(model, [system_msg, doc_msg], "summary")
        try:
            summary_parsed = _parse_json(summary_raw)
            if isinstance(summary_parsed, dict):
                doc_brief = summary_parsed.get("brief", "")
                summary = _clean_concept_content(summary_parsed.get("content", summary_raw))
            else:
                raise ValueError("summary response was not a JSON object")
        except (json.JSONDecodeError, ValueError):
            doc_brief = ""
            summary = _clean_concept_content(summary_raw)
        _write_summary(wiki_dir, doc_name, summary, entities=analysis.get("entities", []))

    if summary_only:
        return

    # --- Steps 2-4: Concept plan → generate/update → index ---
    await _compile_concepts(
        wiki_dir, kb_dir, model, system_msg, doc_msg,
        summary, doc_name, max_concurrency, doc_brief=doc_brief,
        doc_type="short", language=language,
        analysis_context=analysis if analysis.get("concept_actions") else None,
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
    summary_only = os.environ.get("OPENKB_SUMMARY_ONLY") == "1"
    concepts_only = os.environ.get("OPENKB_CONCEPTS_ONLY") == "1"

    # Base context A
    system_msg = {"role": "system", "content": _SYSTEM_TEMPLATE.format(
        schema_md=schema_md, language=language,
    )}

    analysis: dict = {}
    if concepts_only:
        _, overview = parse_fm(summary_content)
        doc_msg = {"role": "user", "content": _CONCEPTS_ONLY_DOC_CONTEXT_USER.format(
            doc_name=doc_name,
        )}
        if not doc_description:
            doc_description = _summary_brief_from_body(overview)
    else:
        doc_msg = {"role": "user", "content": _LONG_DOC_SUMMARY_USER.format(
            doc_name=doc_name, doc_id=doc_id, content=summary_content,
        )}

        if not summary_only:
            # --- Step 0: Analysis step ---
            concept_briefs = _read_concept_briefs(wiki_dir)
            analysis = _analyze_document(
                model=model,
                system_msg=system_msg,
                doc_msg=doc_msg,
                concept_briefs=concept_briefs,
            )

            # Save review items to queue
            review_items = analysis.get("review_items", [])
            if review_items:
                openkb_dir = kb_dir / ".openkb"
                queue = ReviewQueue(openkb_dir)
                queue.add(review_items)

        # --- Step 1: Generate overview ---
        overview = _llm_call(model, [system_msg, doc_msg], "overview")

    if summary_only:
        return

    # --- Steps 2-4: Concept plan → generate/update → index ---
    await _compile_concepts(
        wiki_dir, kb_dir, model, system_msg, doc_msg,
        overview, doc_name, max_concurrency, doc_brief=doc_description,
        doc_type="pageindex", language=language,
        analysis_context=analysis if analysis.get("concept_actions") else None,
    )
