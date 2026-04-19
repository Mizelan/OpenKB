"""Internal knowledge-base maintenance helpers.

Repairs stale internal links, backfills related-concept sections, rebuilds
catalog-style indexes, and emits a structural report that only depends on
the current KB contents.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import re
import unicodedata

import yaml

from openkb.frontmatter import parse_fm, serialize_fm

_GENERIC_EXCLUDED = {"AGENTS.md", "SCHEMA.md", "log.md"}
_SOURCE_LINK_RE = re.compile(r"\[\[sources/([^\]|]+)(?:\|([^\]]+))?\]\]")
_CONCEPT_LINK_RE = re.compile(r"\[\[concepts/([^\]|]+)(?:\|([^\]]+))?\]\]")
_WIKI_LINK_RE = re.compile(r"\[\[wiki/([^\]|]+)(?:\|([^\]]+))?\]\]")
_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_WIKI_INDEX_TYPE = "wiki-index"
_WIKI_QUERY_PREFIX = "wiki/queries/"
_ACRONYMS = {
    "ai": "AI",
    "llm": "LLM",
    "mcp": "MCP",
    "rag": "RAG",
    "ui": "UI",
    "ux": "UX",
    "kb": "KB",
    "hud": "HUD",
    "api": "API",
    "cli": "CLI",
    "gui": "GUI",
}
_BROKEN_WIKI_FALLBACKS = {
    "decision-making": "[[concepts/의사결정]]",
    "learning": "학습",
}
_KB_LOCAL_SOURCE_PATH_RE = re.compile(r"\[([^\]]+)\]\((/[^)]+?/sources/[^)]+?\.md)\)")
_SUMMARY_CONCEPT_HEADINGS = {
    "## 관련 개념",
    "## 핵심 개념",
    "## 연결 개념",
    "## 연결 가능한 개념",
    "## 다룬 개념",
    "## 확장 가능한 개념",
    "## 읽을 거리",
    "## 강조된 가치",
}
_INLINE_SUMMARY_CONCEPT_PATTERNS = (
    re.compile(r"관련 개념(?:으로는|은)?\s+(.+?)(?:\s*가 있다|\s*를 들 수 있다|\s*가 적절하다|\s*이다|\s*로 정리할 수 있다|\s*로 묶어 볼 수 있다|\s*로 묶을 수 있다|$)"),
    re.compile(r"관련 주제는\s+(.+?)(?:\s*이다|\s*로 묶어 볼 수 있다|\s*로 묶을 수 있다|$)"),
    re.compile(r"관련 주제로는\s+(.+?)(?:(?:이|가)\s*연결될 수 있다|\s*이다|$)"),
    re.compile(r"(.+?)\s+같은\s+(?:메타\s+)?개념(?:\s+페이지)?(?:과 연결하는 것이 적절하다|으로 묶어두면|로 연결해 묶을 수 있다)"),
    re.compile(r"(.+?)\s+같은\s+관점과 연결해 볼 수 있다"),
    re.compile(r"(.+?)\s+같은\s+주제로 확장될 수 있다"),
    re.compile(r"이 부분은\s+(.+?)(?:로 확장 가능하다|로 확장해 볼 수 있다|로 볼 수 있다)(?:\.|$)"),
)
_LAST_RESORT_DEFINITION_PATTERNS = (
    re.compile(r"^(?:글의\s+)?핵심(?:은|는)?\s+(.+?)(?:이다|다)(?:\.|$)"),
    re.compile(r"^(?:중심\s+메시지|핵심\s+메시지|메시지)(?:은|는)?\s+(.+?)(?:이다|다)(?:\.|$)"),
)
_DISCOURSE_PREFIXES = (
    "따라서",
    "이런 유형의 문서는",
    "이 항목은",
    "이 문서는",
    "현재 정보만으로는",
)
_ENGLISH_TITLE_PHRASE_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9]+|[A-Z]{2,})){1,5}(?=$|[^A-Za-z0-9])"
)
_CONCEPT_ALIAS_TARGETS = {
    "decision making": "의사결정",
    "decision-making": "의사결정",
    "mcp server": "MCP",
    "mcp server 제공": "MCP",
    "model context protocol": "MCP",
}
_METADATA_BULLET_RE = re.compile(r"^(?:- +)?(?:원문|작성자|작성일|출처|유형)\s*:")
_REPO_PATH_PREFIX_RE = re.compile(r"^(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+/?:\s*")
_TRAILING_TITLE_TAG_RE = re.compile(r"^(.*?\.)\s*([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){1,5})\.$")
_RUNTIME_PATH_RE = re.compile(r"(?:~|/Users/[^/]+)/(?:\.claude|\.codex)/[A-Za-z0-9_./-]+/?")
_INLINE_URLISH_RE = re.compile(r"(?:https?://|x\.com/|github\.com/)[^\s)]+")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _scan_wiki_docs(wiki_dir: Path) -> list[Path]:
    return [
        path for path in wiki_dir.rglob("*.md")
        if path.name not in _GENERIC_EXCLUDED
        and "reports" not in path.relative_to(wiki_dir).parts
    ]


def _normalize_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = text.removesuffix(".md")
    return re.sub(r"[-_\s./]+", "", text)


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold().strip()
    text = text.removesuffix(".md")
    return [token for token in re.split(r"[-_\s./]+", text) if token]


def _slug_to_label(slug: str) -> str:
    tokens = re.split(r"[-_\s]+", slug)
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        lower = token.lower()
        if lower in _ACRONYMS:
            out.append(_ACRONYMS[lower])
        elif token.isascii() and token.lower() == token:
            out.append(token.capitalize())
        else:
            out.append(token)
    return " ".join(out) if out else slug


def _strip_md_suffix(text: str) -> str:
    return text[:-3] if text.endswith(".md") else text


def _resolve_internal_doc_target(target: str, kb_dir: Path, source_inventory: dict | None = None) -> str | None:
    clean = target.strip().strip("/")
    if clean.startswith("sources/"):
        inventory = source_inventory or _collect_source_inventory(kb_dir)
        return _resolve_source_target(clean, inventory)

    candidate = kb_dir / clean
    if candidate.exists():
        return str(Path(clean)).replace("\\", "/")

    if not clean.endswith(".md"):
        md_candidate = kb_dir / f"{clean}.md"
        if md_candidate.exists():
            return f"{clean}.md".replace("\\", "/")
    return None


def _load_wiki_support_hints(kb_dir: Path) -> dict[str, list[str]]:
    hints_path = kb_dir / ".openkb" / "wiki_support_hints.yaml"
    if not hints_path.exists():
        return {}
    raw = yaml.safe_load(_read_text(hints_path)) or {}
    hints = raw.get("wiki_support_hints", raw)
    if not isinstance(hints, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for topic, refs in hints.items():
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list):
            continue
        normalized[str(topic)] = [str(ref) for ref in refs if str(ref).strip()]
    return normalized


def _load_concept_curation_hints(kb_dir: Path) -> dict:
    hints_path = kb_dir / ".openkb" / "concept_curation.yaml"
    if not hints_path.exists():
        return {"aliases": {}, "phrase_aliases": {}, "brief_overrides": {}}
    raw = yaml.safe_load(_read_text(hints_path)) or {}
    hints = raw.get("concept_curation", raw)
    if not isinstance(hints, dict):
        return {"aliases": {}, "phrase_aliases": {}, "brief_overrides": {}}
    aliases = hints.get("aliases", {})
    if not isinstance(aliases, dict):
        aliases = {}
    phrase_aliases = hints.get("phrase_aliases", {})
    if not isinstance(phrase_aliases, dict):
        phrase_aliases = {}
    brief_overrides = hints.get("brief_overrides", {})
    if not isinstance(brief_overrides, dict):
        brief_overrides = {}
    normalized: dict[str, str] = {}
    for src, dst in aliases.items():
        src_slug = _topic_name_to_slug(str(src))
        dst_slug = _topic_name_to_slug(str(dst))
        if src_slug and dst_slug and src_slug != dst_slug:
            normalized[src_slug] = dst_slug
    normalized_phrase_aliases: dict[str, str] = {}
    for src, dst in phrase_aliases.items():
        src_key = _normalize_key(_summary_phrase_label(str(src)))
        dst_slug = _topic_name_to_slug(str(dst))
        if src_key and dst_slug:
            normalized_phrase_aliases[src_key] = dst_slug
    normalized_brief_overrides: dict[str, str] = {}
    for slug, brief in brief_overrides.items():
        slug_key = _topic_name_to_slug(str(slug))
        brief_text = str(brief).strip()
        if slug_key and brief_text:
            normalized_brief_overrides[slug_key] = brief_text
    return {
        "aliases": normalized,
        "phrase_aliases": normalized_phrase_aliases,
        "brief_overrides": normalized_brief_overrides,
    }


def _collect_source_inventory(kb_dir: Path) -> dict:
    source_root = kb_dir / "sources"
    by_key: dict[str, str] = {}
    by_path: set[str] = set()
    by_dir: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    source_meta: dict[str, dict] = {}

    if not source_root.exists():
        return {
            "paths": by_path,
            "keys": by_key,
            "by_dir": by_dir,
            "meta": source_meta,
        }

    for path in sorted(source_root.rglob("*.md")):
        rel = str(path.relative_to(kb_dir)).replace("\\", "/")
        rel_no_ext = _strip_md_suffix(rel)
        by_path.add(rel)
        by_path.add(rel_no_ext)
        by_key[_normalize_key(rel)] = rel
        by_key[_normalize_key(rel_no_ext)] = rel

        rel_path = Path(rel)
        rel_no_ext_path = Path(rel_no_ext)
        dir_key = str(rel_path.parent).replace("\\", "/")
        basename = rel_no_ext_path.name
        by_dir[dir_key].append((basename, rel))

        meta, _ = parse_fm(_read_text(path))
        source_meta[rel] = meta

    return {
        "paths": by_path,
        "keys": by_key,
        "by_dir": by_dir,
        "meta": source_meta,
    }


def _collect_original_source_meta(kb_dir: Path) -> dict[str, dict]:
    source_root = kb_dir / "sources"
    by_url: dict[str, dict] = {}
    if not source_root.exists():
        return by_url

    for path in sorted(source_root.rglob("*.md")):
        meta, _ = parse_fm(_read_text(path))
        url = str(meta.get("url", "")).strip()
        if not url:
            continue
        by_url[url] = {
            "path": path,
            "meta": meta,
        }
    return by_url


def _collect_wiki_source_urls(kb_dir: Path) -> dict[str, str]:
    wiki_sources = kb_dir / "wiki" / "sources"
    by_stem: dict[str, str] = {}
    if not wiki_sources.exists():
        return by_stem

    for path in sorted(wiki_sources.glob("*.md")):
        meta, _ = parse_fm(_read_text(path))
        url = str(meta.get("source_url") or meta.get("url") or "").strip()
        if url:
            by_stem[path.stem] = url
    return by_stem


def _resolve_source_target(target: str, inventory: dict) -> str | None:
    clean = target.strip().strip("/")
    if clean.startswith("sources/"):
        clean = clean[len("sources/") :]

    candidates = [
        f"sources/{clean}",
        f"sources/{_strip_md_suffix(clean)}",
    ]
    for candidate in candidates:
        if candidate in inventory["paths"]:
            return candidate if candidate.endswith(".md") else f"{candidate}.md"

    norm_matches = {
        inventory["keys"].get(_normalize_key(candidate))
        for candidate in candidates
        if inventory["keys"].get(_normalize_key(candidate))
    }
    norm_matches.discard(None)
    if len(norm_matches) == 1:
        return next(iter(norm_matches))

    clean_path = Path(clean)
    dir_key = f"sources/{clean_path.parent}".replace("\\", "/")
    target_tokens = set(_tokenize(clean_path.name))
    if not target_tokens:
        return None

    scored: list[tuple[float, str]] = []
    for candidate_name, candidate_rel in inventory["by_dir"].get(dir_key, []):
        candidate_tokens = set(_tokenize(candidate_name))
        if not candidate_tokens:
            continue
        overlap = len(target_tokens & candidate_tokens)
        if not overlap:
            continue
        score = overlap / max(len(target_tokens), len(candidate_tokens))
        scored.append((score, candidate_rel))

    if not scored:
        return None

    scored.sort(reverse=True)
    top_score, top_rel = scored[0]
    if top_score < 0.55:
        return None
    if len(scored) > 1 and abs(top_score - scored[1][0]) < 0.10:
        return None
    return top_rel


def _replace_source_links(text: str, inventory: dict) -> tuple[str, int]:
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        target, alias = match.group(1), match.group(2)
        resolved = _resolve_source_target(target, inventory)
        if resolved:
            link_target = _strip_md_suffix(resolved)
            if alias:
                replacement = f"[[{link_target}|{alias}]]"
            else:
                replacement = f"[[{link_target}]]"
            if replacement != match.group(0):
                changed += 1
            return replacement
        if alias:
            replacement = alias
        else:
            replacement = _slug_to_label(Path(target).stem)
        if replacement != match.group(0):
            changed += 1
        return replacement

    return _SOURCE_LINK_RE.sub(repl, text), changed


def _repair_sources_frontmatter(meta: dict, inventory: dict) -> bool:
    sources = meta.get("sources")
    if not isinstance(sources, list):
        return _repair_provenance_frontmatter(meta, inventory)

    repaired: list[str] = []
    seen: set[str] = set()
    changed = False
    for source in sources:
        resolved = _resolve_source_target(str(source), inventory)
        if resolved is None:
            changed = True
            continue
        if resolved in seen:
            changed = True
            continue
        seen.add(resolved)
        repaired.append(resolved)
        if resolved != source:
            changed = True

    meta["sources"] = repaired
    changed = _repair_provenance_frontmatter(meta, inventory) or changed
    count_basis = meta.get("supporting_sources") if isinstance(meta.get("supporting_sources"), list) else repaired
    if "source_count" in meta and meta.get("source_count") != len(count_basis):
        meta["source_count"] = len(count_basis)
        changed = True
    return changed


def _repair_provenance_frontmatter(meta: dict, inventory: dict) -> bool:
    changed = False

    supporting_sources = meta.get("supporting_sources")
    if isinstance(supporting_sources, str):
        supporting_sources = [supporting_sources]
        changed = True
    if isinstance(supporting_sources, list):
        repaired_supporting: list[str] = []
        seen_sources: set[str] = set()
        for source in supporting_sources:
            resolved = _resolve_source_target(str(source), inventory)
            if resolved is None:
                changed = True
                continue
            if resolved in seen_sources:
                changed = True
                continue
            seen_sources.add(resolved)
            repaired_supporting.append(resolved)
            if resolved != source:
                changed = True
        meta["supporting_sources"] = repaired_supporting

    supporting_pages = meta.get("supporting_pages")
    if isinstance(supporting_pages, str):
        supporting_pages = [supporting_pages]
        changed = True
    if isinstance(supporting_pages, list):
        deduped_pages: list[str] = []
        seen_pages: set[str] = set()
        for page in supporting_pages:
            page_str = str(page)
            if page_str in seen_pages:
                changed = True
                continue
            seen_pages.add(page_str)
            deduped_pages.append(page_str)
        meta["supporting_pages"] = deduped_pages

    return changed


def _gather_root_wiki_support_sources(
    kb_dir: Path,
    topic: str,
    body: str,
    current_sources: list[str],
    source_inventory: dict,
    hints: dict[str, list[str]],
) -> list[str]:
    support: list[str] = []
    seen: set[str] = set()
    hinted = topic in hints and bool(hints.get(topic))

    def add(ref: str) -> None:
        resolved = _resolve_internal_doc_target(ref, kb_dir, source_inventory)
        if not resolved or resolved in seen:
            return
        seen.add(resolved)
        support.append(resolved)

    if not hinted:
        for source in current_sources:
            add(str(source))

    for match in _SOURCE_LINK_RE.finditer(body):
        add(f"sources/{match.group(1)}")

    for match in _WIKI_LINK_RE.finditer(body):
        target = match.group(1)
        if target.startswith("queries/"):
            add(f"wiki/{target}")

    for hint in hints.get(topic, []):
        add(hint)

    return support


def _repair_root_wiki_sources(
    kb_dir: Path,
    meta: dict,
    body: str,
    source_inventory: dict,
    hints: dict[str, list[str]],
) -> bool:
    if meta.get("type") != "wiki":
        return _repair_sources_frontmatter(meta, source_inventory)

    topic = str(meta.get("topic", "")).strip()
    current_sources = meta.get("sources", [])
    if isinstance(current_sources, str):
        current_sources = [current_sources]
    if not isinstance(current_sources, list):
        current_sources = []

    repaired = _gather_root_wiki_support_sources(
        kb_dir=kb_dir,
        topic=topic,
        body=body,
        current_sources=[str(source) for source in current_sources],
        source_inventory=source_inventory,
        hints=hints,
    )

    changed = repaired != current_sources
    meta["sources"] = repaired
    changed = _repair_provenance_frontmatter(meta, source_inventory) or changed
    count_basis = meta.get("supporting_sources") if isinstance(meta.get("supporting_sources"), list) else repaired
    if meta.get("source_count") != len(count_basis):
        meta["source_count"] = len(count_basis)
        changed = True
    return changed


def _normalize_source_doc_links(body: str, kb_dir: Path, source_inventory: dict) -> tuple[str, int]:
    changed = 0
    kb_name = kb_dir.name

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        label, abs_path = match.group(1), match.group(2)
        marker = f"/{kb_name}/sources/"
        if marker not in abs_path:
            return match.group(0)
        rel = abs_path.split(marker, 1)[1]
        resolved = _resolve_source_target(rel, source_inventory)
        if not resolved:
            return match.group(0)
        replacement = f"[[{_strip_md_suffix(resolved)}|{label}]]"
        if replacement != match.group(0):
            changed += 1
        return replacement

    return _KB_LOCAL_SOURCE_PATH_RE.sub(repl, body), changed


def _repair_source_documents(kb_dir: Path) -> int:
    source_root = kb_dir / "sources"
    if not source_root.exists():
        return 0

    inventory = _collect_source_inventory(kb_dir)
    updated = 0
    for path in sorted(source_root.rglob("*.md")):
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        new_body, changed_links = _normalize_source_doc_links(body, kb_dir, inventory)

        current_related = meta.get("related")
        changed_meta = False
        if isinstance(current_related, list):
            deduped: list[str] = []
            seen: set[str] = set()
            for item in current_related:
                item_str = str(item)
                if item_str in seen:
                    changed_meta = True
                    continue
                seen.add(item_str)
                deduped.append(item_str)
            if deduped != current_related:
                meta["related"] = deduped
                changed_meta = True

        if changed_links or changed_meta:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _dedupe_lines(body: str) -> str:
    lines = body.splitlines()
    if not lines:
        return body

    out: list[str] = []
    previous: str | None = None
    for line in lines:
        if (
            line == previous
            and ("[[sources/" in line or "[[concepts/" in line)
        ):
            continue
        out.append(line)
        previous = line
    result = "\n".join(out)
    if body.endswith("\n"):
        result += "\n"
    return result


def _repair_source_links(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    inventory = _collect_source_inventory(kb_dir)
    hints = _load_wiki_support_hints(kb_dir)
    updated = 0
    for path in wiki_dir.glob("*.md"):
        if path.name in _GENERIC_EXCLUDED or path.name == "index.md":
            continue
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        body, changed_links = _replace_source_links(body, inventory)
        changed_meta = _repair_root_wiki_sources(kb_dir, meta, body, inventory, hints)
        deduped = _dedupe_lines(body)
        if changed_links or changed_meta or deduped != body:
            path.write_text(serialize_fm(meta, deduped), encoding="utf-8")
            updated += 1
    return updated


def _normalize_concept_map(existing: set[str]) -> dict[str, str | None]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for slug in existing:
        grouped[_normalize_key(slug)].append(slug)
    mapping: dict[str, str | None] = {}
    for key, slugs in grouped.items():
        mapping[key] = slugs[0] if len(slugs) == 1 else None
    return mapping


def _topic_name_to_slug(name: str) -> str:
    text = unicodedata.normalize("NFKC", name).strip()
    text = re.sub(r"\[\[concepts/([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"[^0-9A-Za-z가-힣\s\-_]+", " ", text)
    text = re.sub(r"[\s_]+", "-", text).strip("-")
    return text.casefold() if text.isascii() else text


def _resolve_concept_phrase(phrase: str, mapping: dict[str, str | None], existing: set[str]) -> str | None:
    candidate = phrase.strip()
    if not candidate:
        return None

    norm_candidate = _normalize_key(candidate)
    for alias, target in _CONCEPT_ALIAS_TARGETS.items():
        if norm_candidate == _normalize_key(alias) and target in existing:
            return target

    if candidate.startswith("[[concepts/") and candidate.endswith("]]"):
        inner = candidate[11:-2].split("|", 1)[0]
        return _canonical_concept_slug(inner, mapping, existing)

    labeled = _summary_phrase_label(candidate)
    probes = [
        candidate,
        labeled,
        _topic_name_to_slug(candidate),
        _topic_name_to_slug(labeled),
        candidate.replace(" ", ""),
        labeled.replace(" ", ""),
    ]
    for probe in probes:
        resolved = _canonical_concept_slug(probe, mapping, existing)
        if resolved:
            return resolved

    for slug in existing:
        if norm_candidate and norm_candidate in _normalize_key(slug):
            return slug
    return None


def _resolve_concept_phrase_curated(
    phrase: str,
    mapping: dict[str, str | None],
    existing: set[str],
    curation: dict,
) -> str | None:
    phrase_aliases = curation.get("phrase_aliases", {})
    aliases = curation.get("aliases", {})
    label = _summary_phrase_label(phrase)
    target = phrase_aliases.get(_normalize_key(label))
    if target:
        resolved = _canonical_concept_slug(target, mapping, existing, aliases)
        if resolved:
            return resolved
    return _resolve_concept_phrase(label, mapping, existing)


def _split_inline_concept_phrase_list(text: str) -> list[str]:
    cleaned = re.sub(r"[`*_]+", "", text).strip().strip(".")
    cleaned = re.sub(r"\s+(?:와|과|및)\s+", ", ", cleaned)
    cleaned = re.sub(r"(?<=[0-9A-Za-z가-힣])(?:와|과)\s+", ", ", cleaned)
    parts = re.split(r"\s*(?:,|/|·|;)\s*", cleaned)
    return [part.strip().strip(".") for part in parts if part.strip().strip(".")]


def _summary_phrase_label(phrase: str) -> str:
    label = re.sub(r"[`*_]+", "", phrase).strip().strip(".")
    label = re.sub(r"\s+", " ", label)
    if ":" in label:
        head, _ = label.split(":", 1)
        if head.strip():
            label = head.strip()
    label = re.sub(r"^\d+(?:[.,]\d+)?%?\s+", "", label)
    if label.endswith(" 서버 제공"):
        label = label[: -len(" 서버 제공")].strip()
    if label.endswith(" 기능"):
        label = label[: -len(" 기능")].strip()
    if label.endswith(" 개발자 대상"):
        label = label[: -len("자 대상")].strip()
    return label


def _extract_summary_concept_phrases(body: str) -> list[str]:
    phrases: list[str] = []
    capture = False
    for line in body.splitlines():
        if line.startswith("## "):
            capture = line.strip() in _SUMMARY_CONCEPT_HEADINGS
            continue
        stripped = line.strip()
        if capture:
            match = re.match(r"^- (.+)$", stripped)
            if match:
                phrase = match.group(1).strip().strip(".")
                if phrase:
                    phrases.append(phrase)
                continue
        if not stripped:
            continue
        for pattern in _INLINE_SUMMARY_CONCEPT_PATTERNS:
            match = pattern.search(stripped)
            if not match:
                continue
            phrases.extend(_split_inline_concept_phrase_list(match.group(1)))
    return phrases


def _looks_like_concept_phrase(phrase: str) -> bool:
    candidate = _summary_phrase_label(phrase)
    if not candidate or len(candidate) > 60:
        return False
    if any(candidate.startswith(prefix) for prefix in _DISCOURSE_PREFIXES):
        return False
    if candidate in _ACRONYMS.values() or re.fullmatch(r"[A-Z]{2,10}", candidate):
        return True
    if "http" in candidate or "/" in candidate or candidate.startswith("@"):
        return False
    if re.fullmatch(r"[0-9% ]+", candidate):
        return False
    if re.search(r"(이다|다|있다|없다|한다|보인다|알린다|소개한다|강조한다|시사한다|보여준다|가깝다|필요하다|적절하다)$", candidate):
        return False
    if _ENGLISH_TITLE_PHRASE_RE.fullmatch(candidate):
        return True
    if re.search(r"[가-힣]", candidate):
        if len(candidate.split()) > 5:
            return False
        if any(candidate.endswith(suffix) for suffix in (
            "도구",
            "워크플로",
            "개발",
            "생산성",
            "설계",
            "리더십",
            "프로토콜",
            "편집",
            "시스템",
            "플랫폼",
            "대안",
            "참조",
            "튜터링",
            "학습",
            "교육",
        )):
            return True
        return 2 <= len(candidate.split()) <= 5
    return False


def _extract_last_resort_summary_concept_phrases(body: str) -> list[str]:
    phrases: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        label = _summary_phrase_label(candidate)
        if not _looks_like_concept_phrase(label):
            return
        key = label.casefold()
        if key in seen:
            return
        seen.add(key)
        phrases.append(label)

    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        plain = re.sub(r"[`*_]+", "", stripped)
        is_bullet = plain.startswith("- ")
        if re.match(r"^- +(원문|작성자|작성일|출처|유형)\s*:", plain):
            continue
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", plain) if segment.strip()]
        for sentence in sentences:
            for pattern in _LAST_RESORT_DEFINITION_PATTERNS:
                match = pattern.search(sentence)
                if match:
                    for item in _split_inline_concept_phrase_list(match.group(1)):
                        add(item)

        label_match = re.match(r"^- +([^:]+):", plain)
        if label_match:
            raw_label = label_match.group(1).strip()
            normalized_label = _summary_phrase_label(raw_label)
            if (
                not re.search(r"\d", raw_label)
                and (
                    normalized_label in _ACRONYMS.values()
                    or raw_label.endswith((" 기능", " 서버 제공", " 개발자 대상"))
                )
            ):
                add(raw_label)

        if is_bullet:
            for match in _ENGLISH_TITLE_PHRASE_RE.finditer(plain):
                add(match.group(0))

        if ". " in plain:
            tail = plain.rsplit(". ", 1)[-1].strip()
            add(tail)

    return phrases


def _canonicalize_summary_phrase_slug(slug: str, curation: dict[str, dict[str, str]]) -> str:
    aliases = curation.get("aliases", {})
    return aliases.get(slug, slug)


def _concept_alias_target_slug(target: str, aliases: dict[str, str]) -> str:
    norm_target = _topic_name_to_slug(target)
    return aliases.get(norm_target, target)


def _summary_source_record(
    kb_dir: Path,
    summary_stem: str,
    wiki_source_urls: dict[str, str],
    original_sources: dict[str, dict],
) -> dict | None:
    url = wiki_source_urls.get(summary_stem)
    if not url:
        return None
    return original_sources.get(url)


def _topic_concept_names(meta: dict) -> list[str]:
    names: list[str] = []
    for topic in meta.get("topics") or []:
        if not isinstance(topic, dict):
            continue
        if topic.get("type") != "concept":
            continue
        name = str(topic.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _canonical_concept_slug(
    target: str,
    mapping: dict[str, str | None],
    existing: set[str],
    aliases: dict[str, str] | None = None,
) -> str | None:
    if aliases:
        target = _concept_alias_target_slug(target, aliases)
    if target in existing:
        return target
    key = _normalize_key(target)
    resolved = mapping.get(key)
    return resolved if resolved in existing else None


def _scan_missing_concept_refs(
    kb_dir: Path,
    existing: set[str],
    mapping: dict[str, str | None],
    aliases: dict[str, str] | None = None,
) -> dict[str, dict]:
    wiki_dir = kb_dir / "wiki"
    refs: dict[str, dict] = {}
    for path in _scan_wiki_docs(wiki_dir):
        if path.name == "index.md":
            continue
        rel = path.relative_to(wiki_dir)
        text = _read_text(path)
        for match in _CONCEPT_LINK_RE.finditer(text):
            target = match.group(1)
            if _canonical_concept_slug(target, mapping, existing, aliases):
                continue
            item = refs.setdefault(target, {"total": set(), "summaries": set(), "root_wiki": set()})
            item["total"].add(str(rel))
            if rel.parts[0] == "summaries":
                item["summaries"].add(str(rel))
            elif len(rel.parts) == 1:
                item["root_wiki"].add(str(rel))
    return refs


def _qualifies_for_creation(refs: dict) -> bool:
    return len(refs["summaries"]) >= 2 or (len(refs["root_wiki"]) >= 1 and len(refs["total"]) >= 2)


def _clean_context_line(line: str, missing_slug: str) -> str:
    text = line.strip()
    text = _SOURCE_LINK_RE.sub(lambda m: m.group(2) or _slug_to_label(Path(m.group(1)).stem), text)
    text = _WIKI_LINK_RE.sub(lambda m: m.group(2) or _slug_to_label(m.group(1)), text)
    text = _CONCEPT_LINK_RE.sub(lambda m: m.group(2) or _slug_to_label(m.group(1)), text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    if text.startswith("## "):
        return ""
    if len(text) < 12:
        return ""
    if _normalize_key(text) == _normalize_key(missing_slug):
        return ""
    return text.rstrip(".") + "."


def _select_context_lines(kb_dir: Path, slug: str, refs: dict) -> list[str]:
    wiki_dir = kb_dir / "wiki"
    lines: list[str] = []
    seen: set[str] = set()
    needle = f"[[concepts/{slug}"
    for rel in sorted(refs["total"]):
        path = wiki_dir / rel
        for raw_line in _read_text(path).splitlines():
            if needle not in raw_line:
                continue
            cleaned = _clean_context_line(raw_line, slug)
            key = cleaned.casefold()
            if cleaned and key not in seen:
                seen.add(key)
                lines.append(cleaned)
    return lines[:5]


def _cooccurring_existing_concepts(kb_dir: Path, slug: str, refs: dict, existing: set[str], mapping: dict[str, str | None]) -> list[str]:
    wiki_dir = kb_dir / "wiki"
    counts: Counter[str] = Counter()
    for rel in refs["total"]:
        text = _read_text(wiki_dir / rel)
        for target in _CONCEPT_LINK_RE.findall(text):
            other = _canonical_concept_slug(target[0], mapping, existing)
            if other and other != slug:
                counts[other] += 1
    return [name for name, _ in counts.most_common(4)]


def _build_concept_stub(kb_dir: Path, slug: str, refs: dict, existing: set[str], mapping: dict[str, str | None]) -> str:
    title = _slug_to_label(slug)
    context_lines = _select_context_lines(kb_dir, slug, refs)
    overview = (
        f"{title}는 내부 summary와 wiki 문맥에서 반복 등장하는 개념이다. "
        "이 페이지는 source 원문이 아니라 현재 KB 안의 문맥만 기준으로 정리했다."
    )
    related_concepts = _cooccurring_existing_concepts(kb_dir, slug, refs, existing, mapping)
    related_docs = [Path(rel).stem for rel in sorted(refs["summaries"])]

    parts = [f"# {title}", "", "## 개요", overview]
    if context_lines:
        parts.extend(["", "## 관찰된 문맥", ""])
        parts.extend([f"- {line}" for line in context_lines])
    if related_concepts:
        parts.extend(["", "## 관련 개념", ""])
        parts.extend([f"- [[concepts/{name}]]" for name in related_concepts])
    if related_docs:
        parts.extend(["", "## Related Documents", ""])
        parts.extend([f"- [[summaries/{name}]]" for name in related_docs])
    return "\n".join(parts).strip() + "\n"


def _build_root_wiki_backed_concept(
    kb_dir: Path,
    slug: str,
    root_meta: dict,
    root_body: str,
    summary_stems: list[str],
) -> tuple[dict, str]:
    overview = _first_paragraph(root_body)
    title = _slug_to_label(slug)
    meta = {
        "brief": _compact_brief(overview),
        "entity_type": "concept",
        "sources": _cap_sources(root_meta.get("sources", []) or []),
    }
    lines = [f"# {title}", "", "## 개요", overview]
    if summary_stems:
        lines.extend(["", "## Related Documents", ""])
        lines.extend([f"- [[summaries/{stem}]]" for stem in summary_stems[:8]])
    lines.extend(["", "## 관련 위키", "", f"- [[wiki/{slug}]]"])
    return meta, "\n".join(lines).strip() + "\n"


def _build_summary_backed_concept(slug: str, label: str, summary_stems: list[str]) -> tuple[dict, str]:
    title = _slug_to_label(slug)
    overview = (
        f"{label}는 내부 summary가 관련 개념으로 직접 묶어 둔 주제다. "
        "이 페이지는 source 원문이 아니라 현재 KB 요약 문맥만 기준으로 정리했다."
    )
    meta = {
        "brief": _compact_brief(overview),
        "entity_type": "concept",
        "sources": [f"summaries/{stem}.md" for stem in summary_stems[:8]],
    }
    lines = [f"# {title}", "", "## 개요", overview]
    if summary_stems:
        lines.extend(["", "## Related Documents", ""])
        lines.extend([f"- [[summaries/{stem}]]" for stem in summary_stems[:8]])
    return meta, "\n".join(lines).strip() + "\n"


def _merge_bullet_block_lines(body: str, heading: str, items: list[str]) -> str:
    if not items:
        return body
    escaped = re.escape(heading)
    pattern = re.compile(rf"({escaped}\n(?:\n)?)(.*?)(?=\n## |\Z)", re.S)
    existing: list[str] = []
    match = pattern.search(body)
    if match:
        existing = _dedupe_bullet_items(re.findall(r"^- (.+)$", match.group(2), re.M))
    merged = existing[:]
    seen = {line.casefold() for line in existing}
    for item in items:
        normalized = item.removeprefix("- ").strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(normalized)
    replacement = heading + "\n\n" + "\n".join(f"- {line}" for line in merged)
    if match:
        return _collapse_duplicate_bullet_sections(body[:match.start()] + replacement + body[match.end():], heading)
    return _collapse_duplicate_bullet_sections(body.rstrip() + "\n\n" + replacement + "\n", heading)


def _create_missing_concepts_from_summary_phrases(kb_dir: Path) -> set[str]:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    existing = {path.stem for path in concepts_dir.glob("*.md")}
    missing_summaries = set(collect_structural_issues(kb_dir)["summaries_missing_related"])
    curation = _load_concept_curation_hints(kb_dir)
    candidates: defaultdict[str, dict[str, list[str] | str]] = defaultdict(lambda: {"label": "", "summaries": []})
    created: set[str] = set()

    for stem in sorted(missing_summaries):
        path = wiki_dir / "summaries" / f"{stem}.md"
        meta, body = parse_fm(_read_text(path))
        del meta
        phrases = _extract_summary_concept_phrases(body)
        if not phrases:
            phrases = _extract_last_resort_summary_concept_phrases(body)
        for phrase in phrases:
            label = _summary_phrase_label(phrase)
            if not label or len(label) > 60:
                continue
            slug = _canonicalize_summary_phrase_slug(_topic_name_to_slug(label), curation)
            if not slug or slug in existing:
                continue
            item = candidates[slug]
            if not item["label"]:
                item["label"] = label
            item["summaries"].append(stem)

    for slug, info in sorted(candidates.items()):
        concept_path = concepts_dir / f"{slug}.md"
        if concept_path.exists():
            continue
        label = str(info["label"]).strip() or _slug_to_label(slug)
        summary_stems = list(dict.fromkeys(str(stem) for stem in info["summaries"]))
        root_path = wiki_dir / f"{slug}.md"
        if root_path.exists():
            root_meta, root_body = parse_fm(_read_text(root_path))
            if root_meta.get("type") == "wiki":
                meta, body = _build_root_wiki_backed_concept(kb_dir, slug, root_meta, root_body, summary_stems)
            else:
                meta, body = _build_summary_backed_concept(slug, label, summary_stems)
        else:
            meta, body = _build_summary_backed_concept(slug, label, summary_stems)
        concept_path.write_text(serialize_fm(meta, body), encoding="utf-8")
        existing.add(slug)
        created.add(slug)
    return created


def _collect_related_concept_phrase_refs(kb_dir: Path) -> dict[str, dict]:
    wiki_dir = kb_dir / "wiki"
    refs: dict[str, dict] = {}
    for path in _scan_wiki_docs(wiki_dir):
        if path.name == "index.md":
            continue
        rel = str(path.relative_to(wiki_dir))
        body = _read_text(path)
        for heading in ("## Related Concepts", "## 관련 개념"):
            for item in _read_bullet_section_items(body, heading):
                if item.startswith("[["):
                    continue
                label = _summary_phrase_label(item)
                if not _looks_like_concept_phrase(label):
                    continue
                key = _normalize_key(label)
                info = refs.setdefault(
                    key,
                    {
                        "label": label,
                        "total": set(),
                        "summaries": set(),
                        "concepts": set(),
                        "root_wiki": set(),
                    },
                )
                info["total"].add(rel)
                rel_path = Path(rel)
                if rel_path.parts[0] == "summaries":
                    info["summaries"].add(rel)
                elif rel_path.parts[0] == "concepts":
                    info["concepts"].add(rel)
                elif len(rel_path.parts) == 1:
                    info["root_wiki"].add(rel)
    return refs


def _summary_stems_from_related_docs(kb_dir: Path, refs: dict) -> list[str]:
    wiki_dir = kb_dir / "wiki"
    stems: list[str] = []
    seen: set[str] = set()

    def add(stem: str) -> None:
        if not stem or stem in seen:
            return
        seen.add(stem)
        stems.append(stem)

    for rel in sorted(refs["total"]):
        rel_path = Path(rel)
        if rel_path.parts[0] == "summaries":
            add(rel_path.stem)
            continue
        path = wiki_dir / rel
        meta, _ = parse_fm(_read_text(path))
        sources = meta.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            source_str = str(source)
            if source_str.startswith("summaries/"):
                add(Path(source_str).stem)
    return stems


def _create_missing_related_concepts_from_sections(kb_dir: Path) -> set[str]:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    curation = _load_concept_curation_hints(kb_dir)
    aliases = curation.get("aliases", {})
    refs = _collect_related_concept_phrase_refs(kb_dir)
    existing = {path.stem for path in concepts_dir.glob("*.md")}
    mapping = _normalize_concept_map(existing)
    created: set[str] = set()

    for info in refs.values():
        label = str(info["label"]).strip()
        if _resolve_concept_phrase_curated(label, mapping, existing | created, curation):
            continue
        slug = _canonicalize_summary_phrase_slug(_topic_name_to_slug(label), curation)
        if not slug or slug in existing or slug in created:
            continue

        summary_stems = _summary_stems_from_related_docs(kb_dir, info)
        root_path = wiki_dir / f"{slug}.md"
        if root_path.exists():
            root_meta, root_body = parse_fm(_read_text(root_path))
            if root_meta.get("type") == "wiki":
                meta, body = _build_root_wiki_backed_concept(kb_dir, slug, root_meta, root_body, summary_stems)
            elif summary_stems:
                meta, body = _build_summary_backed_concept(slug, label, summary_stems)
            else:
                meta = {"brief": _compact_brief(f"{label}는 내부 문서에서 반복 등장하는 관련 개념이다."), "entity_type": "concept", "sources": []}
                body = _build_concept_stub(kb_dir, slug, info, existing | created, mapping)
        elif summary_stems:
            meta, body = _build_summary_backed_concept(slug, label, summary_stems)
        else:
            meta = {"brief": _compact_brief(f"{label}는 내부 문서에서 반복 등장하는 관련 개념이다."), "entity_type": "concept", "sources": []}
            body = _build_concept_stub(kb_dir, slug, info, existing | created, mapping)

        concept_path = concepts_dir / f"{slug}.md"
        concept_path.write_text(serialize_fm(meta, body), encoding="utf-8")
        created.add(slug)
    return created


def _rewrite_specific_concept_slug(text: str, source_slug: str, target_slug: str) -> tuple[str, int]:
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        target, alias = match.group(1), match.group(2)
        if target != source_slug:
            return match.group(0)
        changed += 1
        if alias:
            return f"[[concepts/{target_slug}|{alias}]]"
        return f"[[concepts/{target_slug}]]"

    return _CONCEPT_LINK_RE.sub(repl, text), changed


def _merge_aliased_concepts(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    curation = _load_concept_curation_hints(kb_dir)
    aliases = curation.get("aliases", {})
    if not aliases:
        return 0

    updated = 0
    for source_slug, target_slug in aliases.items():
        source_path = concepts_dir / f"{source_slug}.md"
        target_path = concepts_dir / f"{target_slug}.md"
        if not source_path.exists() and not target_path.exists():
            continue

        if source_path.exists():
            source_meta, source_body = parse_fm(_read_text(source_path))
        else:
            source_meta, source_body = {}, ""

        if target_path.exists():
            target_meta, target_body = parse_fm(_read_text(target_path))
        else:
            title = _slug_to_label(target_slug)
            target_meta = {
                "brief": f"{title} 관련 개념 정리.",
                "entity_type": "concept",
                "sources": [],
            }
            target_body = f"# {title}\n"

        source_sources = source_meta.get("sources", [])
        target_sources = target_meta.get("sources", [])
        if isinstance(source_sources, str):
            source_sources = [source_sources]
        if isinstance(target_sources, str):
            target_sources = [target_sources]
        merged_sources = _cap_sources(list(target_sources) + list(source_sources), limit=32)
        target_meta["sources"] = merged_sources

        related_docs = re.findall(r"^- \[\[summaries/[^\]]+\]\]$", source_body, re.M)
        target_body = _merge_bullet_block_lines(target_body, "## Related Documents", related_docs)

        if not target_path.exists() or serialize_fm(target_meta, target_body) != _read_text(target_path):
            target_path.write_text(serialize_fm(target_meta, target_body), encoding="utf-8")
            updated += 1

        if source_path.exists():
            source_path.unlink()
            updated += 1

        for path in _scan_wiki_docs(wiki_dir):
            raw = _read_text(path)
            meta, body = parse_fm(raw)
            new_body, changed = _rewrite_specific_concept_slug(body, source_slug, target_slug)
            if changed:
                path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
                updated += 1

    return updated


def _create_missing_concepts_from_summary_topics(kb_dir: Path, *, min_count: int = 1) -> set[str]:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    existing = {path.stem for path in concepts_dir.glob("*.md")}
    wiki_source_urls = _collect_wiki_source_urls(kb_dir)
    original_sources = _collect_original_source_meta(kb_dir)
    missing_summaries = set(collect_structural_issues(kb_dir)["summaries_missing_related"])

    candidate_docs: defaultdict[str, list[str]] = defaultdict(list)
    for stem in sorted(missing_summaries):
        record = _summary_source_record(kb_dir, stem, wiki_source_urls, original_sources)
        if not record:
            continue
        for topic_name in _topic_concept_names(record["meta"]):
            slug = _topic_name_to_slug(topic_name)
            if not slug or slug in existing:
                continue
            candidate_docs[slug].append(stem)

    created: set[str] = set()
    for slug, stems in sorted(candidate_docs.items()):
        unique_stems = list(dict.fromkeys(stems))
        root_path = wiki_dir / f"{slug}.md"
        if not root_path.exists():
            continue
        root_meta, root_body = parse_fm(_read_text(root_path))
        if root_meta.get("type") != "wiki":
            continue
        if len(unique_stems) < min_count:
            continue

        concept_path = concepts_dir / f"{slug}.md"
        if concept_path.exists():
            continue

        meta, body = _build_root_wiki_backed_concept(kb_dir, slug, root_meta, root_body, unique_stems)
        concept_path.write_text(serialize_fm(meta, body), encoding="utf-8")
        existing.add(slug)
        created.add(slug)
    return created


def _create_missing_concepts(kb_dir: Path) -> set[str]:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.stem for path in concepts_dir.glob("*.md")}
    mapping = _normalize_concept_map(existing)
    aliases = _load_concept_curation_hints(kb_dir).get("aliases", {})
    refs = _scan_missing_concept_refs(kb_dir, existing, mapping, aliases)
    created: set[str] = set()

    for slug, ref_info in refs.items():
        if not _qualifies_for_creation(ref_info):
            continue
        safe_slug = slug
        path = concepts_dir / f"{safe_slug}.md"
        if path.exists():
            continue
        body = _build_concept_stub(kb_dir, safe_slug, ref_info, existing, mapping)
        meta = {
            "brief": f"{_slug_to_label(safe_slug)}는 내부 문서 문맥에서 반복 등장하는 개념이다.",
            "entity_type": "concept",
            "sources": [f"summaries/{Path(rel).name}" for rel in sorted(ref_info["summaries"])],
        }
        path.write_text(serialize_fm(meta, body), encoding="utf-8")
        created.add(safe_slug)
        existing.add(safe_slug)
        mapping = _normalize_concept_map(existing)
    return created


def _rewrite_concept_links(text: str, existing: set[str], created: set[str], aliases: dict[str, str] | None = None) -> tuple[str, int]:
    mapping = _normalize_concept_map(existing | created)
    changed = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal changed
        target, alias = match.group(1), match.group(2)
        canonical = _canonical_concept_slug(target, mapping, existing | created, aliases)
        if canonical:
            if canonical != target:
                changed += 1
            if alias:
                return f"[[concepts/{canonical}|{alias}]]"
            return f"[[concepts/{canonical}]]"
        changed += 1
        return alias or _slug_to_label(target)

    return _CONCEPT_LINK_RE.sub(repl, text), changed


def _repair_concept_links(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    existing = {path.stem for path in concepts_dir.glob("*.md")}
    curation = _load_concept_curation_hints(kb_dir)
    aliases = curation.get("aliases", {})

    # First pass creates qualifying concepts from current internal references.
    created = _create_missing_concepts(kb_dir)
    created |= _create_missing_concepts_from_summary_phrases(kb_dir)
    created |= _create_missing_concepts_from_summary_topics(kb_dir)
    existing |= created

    updated = 0
    for path in _scan_wiki_docs(wiki_dir):
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        new_body, changed = _rewrite_concept_links(body, existing, created, aliases)
        new_body = _dedupe_lines(new_body)
        if changed or new_body != body:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _repair_wiki_links(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    updated = 0
    for path in wiki_dir.glob("*.md"):
        if path.name in _GENERIC_EXCLUDED or path.name == "index.md":
            continue
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        changed = 0

        def repl(match: re.Match[str]) -> str:
            nonlocal changed
            target, alias = match.group(1), match.group(2)
            if target not in _BROKEN_WIKI_FALLBACKS:
                return match.group(0)
            changed += 1
            replacement = _BROKEN_WIKI_FALLBACKS[target]
            if replacement.startswith("[[") and alias:
                if replacement.endswith("]]"):
                    inner = replacement[2:-2].split("|", 1)[0]
                    return f"[[{inner}|{alias}]]"
            return alias or replacement

        new_body = _WIKI_LINK_RE.sub(repl, body)
        if changed:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _merge_bullet_section(body: str, heading: str, items: list[str]) -> str:
    if not items:
        return body

    escaped = re.escape(heading)
    pattern = re.compile(rf"({escaped}\n(?:\n)?)(.*?)(?=\n## |\Z)", re.S)
    existing: list[str] = []
    match = pattern.search(body)
    if match:
        existing = _dedupe_bullet_items(re.findall(r"^- (.+)$", match.group(2), re.M))
    merged = existing[:]
    seen = {line.casefold() for line in existing}
    for item in items:
        normalized = item.removeprefix("- ").strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key not in seen:
            seen.add(key)
            merged.append(normalized)
    replacement = heading + "\n\n" + "\n".join(f"- {line}" for line in merged)
    if match:
        result = body[:match.start()] + replacement + body[match.end():]
        if body.endswith("\n") and not result.endswith("\n"):
            result += "\n"
        return _collapse_duplicate_bullet_sections(result, heading)
    return _collapse_duplicate_bullet_sections(body.rstrip() + "\n\n" + replacement + "\n", heading)


def _collapse_duplicate_bullet_sections(body: str, heading: str) -> str:
    escaped = re.escape(heading)
    pattern = re.compile(rf"({escaped}\n(?:\n)?)(.*?)(?=\n## |\Z)", re.S)
    matches = list(pattern.finditer(body))
    if len(matches) <= 1:
        return body

    merged: list[str] = []
    seen: set[str] = set()
    for match in matches:
        for item in re.findall(r"^- (.+)$", match.group(2), re.M):
            normalized = item.removeprefix("- ").strip()
            if not normalized:
                continue
            key = normalized.casefold()
            if key not in seen:
                seen.add(key)
                merged.append(normalized)

    replacement = heading + "\n\n" + "\n".join(f"- {line}" for line in merged)
    parts = [body[: matches[0].start()], replacement]
    cursor = matches[0].end()
    for match in matches[1:]:
        parts.append(body[cursor: match.start()])
        cursor = match.end()
    parts.append(body[cursor:])
    result = "".join(parts)
    if body.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result


def _dedupe_bullet_items(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.removeprefix("- ").strip()
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    return cleaned


def _read_bullet_section_items(body: str, heading: str) -> list[str]:
    escaped = re.escape(heading)
    pattern = re.compile(rf"{escaped}\n(?:\n)?(.*?)(?=\n## |\Z)", re.S)
    match = pattern.search(body)
    if not match:
        return []
    return _dedupe_bullet_items(re.findall(r"^- (.+)$", match.group(1), re.M))


def _rewrite_bullet_section(body: str, heading: str, items: list[str]) -> str:
    normalized = _dedupe_bullet_items(items)
    escaped = re.escape(heading)
    pattern = re.compile(rf"({escaped}\n(?:\n)?)(.*?)(?=\n## |\Z)", re.S)
    match = pattern.search(body)
    if not normalized:
        if not match:
            return body
        result = body[:match.start()] + body[match.end():]
        return result.lstrip("\n") if not body.startswith("\n") else result
    replacement = heading + "\n\n" + "\n".join(f"- {line}" for line in normalized)
    if match:
        result = body[:match.start()] + replacement + body[match.end():]
    else:
        result = body.rstrip() + "\n\n" + replacement + "\n"
    if body.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return _collapse_duplicate_bullet_sections(result, heading)


def _replace_markdown_section(body: str, heading: str, content: str) -> str:
    escaped = re.escape(heading)
    pattern = re.compile(rf"({escaped}\n)(.*?)(?=\n## |\Z)", re.S)
    replacement = heading + "\n" + content.strip() + "\n"
    match = pattern.search(body)
    if match:
        result = body[:match.start()] + replacement + body[match.end():]
        if body.endswith("\n") and not result.endswith("\n"):
            result += "\n"
        return result
    return body.rstrip() + "\n\n" + replacement


def _insert_section_before(body: str, before_heading: str, heading: str, items: list[str]) -> str:
    if not items:
        return body
    section = heading + "\n\n" + "\n".join(f"- {item}" for item in items) + "\n"
    marker = f"\n{before_heading}\n"
    idx = body.find(marker)
    if idx != -1:
        return body[:idx] + "\n\n" + section + body[idx:]
    return body.rstrip() + "\n\n" + section


def _clean_summary_context_line(line: str) -> str:
    text = line.strip()
    text = re.sub(r"^- +", "", text)
    text = re.sub(r"[`*_]+", "", text)
    text = _SOURCE_LINK_RE.sub(lambda m: m.group(2) or _slug_to_label(Path(m.group(1)).stem), text)
    text = _WIKI_LINK_RE.sub(lambda m: m.group(2) or _slug_to_label(m.group(1)), text)
    text = _CONCEPT_LINK_RE.sub(lambda m: m.group(2) or _slug_to_label(m.group(1)), text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    if not text or text.startswith("## "):
        return ""
    return text.rstrip(".") + "."


def _is_metadata_context_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _METADATA_BULLET_RE.match(stripped):
        return True
    return "http://" in stripped or "https://" in stripped


def _context_relevance_score(label: str, line: str) -> int:
    norm_label = _normalize_key(label)
    norm_line = _normalize_key(line)
    score = 0
    if norm_label and norm_label in norm_line:
        score += 100
    for token in _tokenize(label):
        norm_token = _normalize_key(token)
        if len(norm_token) >= 3 and norm_token in norm_line:
            score += 10
    return score


def _normalize_context_for_label(label: str, line: str) -> str:
    normalized = _REPO_PATH_PREFIX_RE.sub("", line).strip()
    normalized = re.sub(rf"\s+{_RUNTIME_PATH_RE.pattern}에서", " ", normalized)
    normalized = _RUNTIME_PATH_RE.sub("내부 저장 경로", normalized)
    normalized = _INLINE_URLISH_RE.sub("원문 링크", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    match = _TRAILING_TITLE_TAG_RE.match(normalized)
    if match and _normalize_key(match.group(2)) == _normalize_key(label):
        normalized = match.group(1).strip()
    return normalized or line


def _summary_backed_context_lines(kb_dir: Path, summary_stems: list[str], label: str) -> list[str]:
    wiki_dir = kb_dir / "wiki"
    lines: list[str] = []
    seen: set[str] = set()
    norm_label = _normalize_key(label)
    candidates: list[tuple[int, int, str]] = []
    fallback = ""

    def add(candidate: str) -> None:
        cleaned = _clean_summary_context_line(candidate)
        if len(cleaned) < 24:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        lines.append(cleaned)

    for stem in summary_stems[:8]:
        path = wiki_dir / "summaries" / f"{stem}.md"
        if not path.exists():
            continue
        _, body = parse_fm(_read_text(path))
        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if (
                not stripped
                or stripped.startswith("#")
                or "[[concepts/" in stripped
                or _is_metadata_context_line(stripped)
            ):
                continue
            cleaned = _clean_summary_context_line(stripped)
            if not cleaned:
                continue
            score = _context_relevance_score(label, cleaned)
            if norm_label and norm_label in _normalize_key(cleaned):
                candidates.append((score, len(candidates), cleaned))
            elif stripped.startswith("- "):
                candidates.append((score, len(candidates), cleaned))
        if not candidates and not fallback:
            fallback = _first_paragraph(body)

    if candidates:
        ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
        if any(score > 0 for score, _, _ in ranked):
            ranked = [item for item in ranked if item[0] > 0] + [item for item in ranked if item[0] <= 0]
        for _, _, cleaned in ranked:
            add(_normalize_context_for_label(label, cleaned))
    elif fallback:
        add(_normalize_context_for_label(label, fallback))

    return lines[:3]


def _refresh_summary_backed_concepts(kb_dir: Path) -> int:
    concepts_dir = kb_dir / "wiki" / "concepts"
    updated = 0
    for path in sorted(concepts_dir.glob("*.md")):
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        brief = str(meta.get("brief", "")).strip()
        sources = meta.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        summary_stems = [
            Path(str(source)).stem
            for source in sources
            if str(source).startswith("summaries/")
        ]
        if not summary_stems:
            continue
        is_generic = "내부 summary가 관련 개념으로 직접 묶어 둔 주제" in brief or "내부 summary가 관련 개념으로 직접 묶어 둔 주제" in body
        is_metadata_brief = brief.startswith(("원문:", "작성자:", "출처:")) or "http://" in brief or "https://" in brief
        existing_contexts = _read_bullet_section_items(body, "## 관찰된 문맥")
        has_metadata_context = any(_is_metadata_context_line(item) for item in existing_contexts)
        is_repo_path_brief = bool(_REPO_PATH_PREFIX_RE.match(brief)) or "/:" in brief
        is_runtime_path_brief = bool(_RUNTIME_PATH_RE.search(brief)) or "x.com/" in brief or "github.com/" in brief
        has_repo_path_context = any(_REPO_PATH_PREFIX_RE.match(item) for item in existing_contexts)
        has_runtime_path_context = any(_RUNTIME_PATH_RE.search(item) or "x.com/" in item or "github.com/" in item for item in existing_contexts)
        if not is_generic and not is_metadata_brief and not is_repo_path_brief and not is_runtime_path_brief and "## 관찰된 문맥" in body and not has_metadata_context and not has_repo_path_context and not has_runtime_path_context:
            continue

        label = _slug_to_label(path.stem)
        contexts = _summary_backed_context_lines(kb_dir, summary_stems, label)
        if not contexts:
            continue
        new_brief = _compact_brief(contexts[0])
        if not new_brief or new_brief == "_아직 작성되지 않음._":
            continue

        changed = False
        if new_brief != brief:
            meta["brief"] = new_brief
            changed = True

        if is_generic or is_metadata_brief or is_repo_path_brief or is_runtime_path_brief:
            new_body = _replace_markdown_section(body, "## 개요", new_brief)
        else:
            new_body = body
        if "## 관찰된 문맥" in new_body:
            new_body = _rewrite_bullet_section(new_body, "## 관찰된 문맥", contexts)
        else:
            new_body = _insert_section_before(new_body, "## Related Documents", "## 관찰된 문맥", contexts)
            new_body = _merge_bullet_section(new_body, "## 관찰된 문맥", contexts)
        if new_body != body:
            body = new_body
            changed = True

        if changed:
            path.write_text(serialize_fm(meta, body), encoding="utf-8")
            updated += 1
    return updated


def _sanitize_existing_bullet_sections(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    updated = 0
    for path in _scan_wiki_docs(wiki_dir):
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        new_body = body.replace("- - [[summaries/", "- [[summaries/")
        for heading in ("## Related Concepts", "## Related Documents", "## 관련 개념", "## 관찰된 문맥"):
            items = _read_bullet_section_items(new_body, heading)
            if heading == "## 관련 개념" and path.parent.name == "concepts":
                self_ref = f"[[concepts/{path.stem}]]".casefold()
                items = [item for item in items if item.casefold() != self_ref]
            if items:
                new_body = _rewrite_bullet_section(new_body, heading, items)
            else:
                new_body = _collapse_duplicate_bullet_sections(new_body, heading)
        if new_body != body:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _normalize_related_concept_sections(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    curation = _load_concept_curation_hints(kb_dir)
    aliases = curation.get("aliases", {})
    concepts_dir = wiki_dir / "concepts"
    existing = {path.stem for path in concepts_dir.glob("*.md")}
    mapping = _normalize_concept_map(existing)
    updated = 0

    for path in _scan_wiki_docs(wiki_dir):
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        new_body = body
        collected_docs: list[str] = []

        for heading in ("## Related Concepts", "## 관련 개념"):
            items = _read_bullet_section_items(new_body, heading)
            if not items and heading not in new_body:
                continue
            normalized: list[str] = []
            for item in items:
                concept_match = _CONCEPT_LINK_RE.fullmatch(item)
                if concept_match:
                    target = concept_match.group(1)
                    alias = concept_match.group(2)
                    resolved = _canonical_concept_slug(target, mapping, existing, aliases)
                    if resolved:
                        normalized.append(f"[[concepts/{resolved}|{alias}]]" if alias else f"[[concepts/{resolved}]]")
                    continue

                if item.startswith("[[summaries/") or item.startswith("[[sources/") or item.startswith("[[wiki/"):
                    collected_docs.append(item)
                    continue

                resolved = _resolve_concept_phrase_curated(item, mapping, existing, curation)
                if not resolved:
                    slug = _canonicalize_summary_phrase_slug(_topic_name_to_slug(_summary_phrase_label(item)), curation)
                    resolved = _canonical_concept_slug(slug, mapping, existing, aliases) if slug else None
                if resolved:
                    normalized.append(f"[[concepts/{resolved}]]")
                else:
                    normalized.append(item)

            if path.parent.name == "concepts":
                self_link = f"[[concepts/{path.stem}]]".casefold()
                normalized = [item for item in normalized if item.casefold() != self_link]

            if normalized:
                rewritten = _rewrite_bullet_section(new_body, heading, normalized)
                new_body = rewritten
            elif heading in new_body:
                new_body = _rewrite_bullet_section(new_body, heading, [])

        if collected_docs:
            new_body = _merge_bullet_section(new_body, "## Related Documents", collected_docs)

        if new_body != body:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _backfill_summary_related_concepts(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    existing = {path.stem for path in concepts_dir.glob("*.md")}
    mapping = _normalize_concept_map(existing)
    wiki_source_urls = _collect_wiki_source_urls(kb_dir)
    original_sources = _collect_original_source_meta(kb_dir)
    updated = 0
    for path in sorted((wiki_dir / "summaries").glob("*.md")):
        raw = _read_text(path)
        meta, body = parse_fm(raw)
        ordered: list[str] = []
        seen: set[str] = set()
        for match in _CONCEPT_LINK_RE.finditer(body):
            slug = match.group(1)
            resolved = _canonical_concept_slug(slug, mapping, existing)
            if resolved and resolved not in seen:
                seen.add(resolved)
                ordered.append(f"[[concepts/{resolved}]]")

        record = _summary_source_record(kb_dir, path.stem, wiki_source_urls, original_sources)
        if record:
            source_meta = record["meta"]
            for topic_name in _topic_concept_names(source_meta):
                resolved = _resolve_concept_phrase(topic_name, mapping, existing)
                if resolved and resolved not in seen:
                    seen.add(resolved)
                    ordered.append(f"[[concepts/{resolved}]]")

        for phrase in _extract_summary_concept_phrases(body):
            resolved = _resolve_concept_phrase(phrase, mapping, existing)
            if resolved and resolved not in seen:
                seen.add(resolved)
                ordered.append(f"[[concepts/{resolved}]]")
        if not ordered:
            for phrase in _extract_last_resort_summary_concept_phrases(body):
                resolved = _resolve_concept_phrase(phrase, mapping, existing)
                if resolved and resolved not in seen:
                    seen.add(resolved)
                    ordered.append(f"[[concepts/{resolved}]]")
        if not ordered:
            continue
        new_body = _merge_bullet_section(body, "## Related Concepts", ordered)
        if new_body != body:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _backfill_concept_related_concepts(kb_dir: Path) -> int:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    existing = {path.stem for path in concepts_dir.glob("*.md")}
    updated = 0

    for path in sorted(concepts_dir.glob("*.md")):
        meta, body = parse_fm(_read_text(path))
        slug = path.stem
        counts: Counter[str] = Counter()
        sources = meta.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        for source in sources:
            source_path = wiki_dir / str(source)
            if not source_path.exists():
                continue
            text = _read_text(source_path)
            for match in _CONCEPT_LINK_RE.finditer(text):
                other = match.group(1)
                if other in existing and other != slug:
                    counts[other] += 1
        related = [f"[[concepts/{name}]]" for name, _ in counts.most_common(4)]
        if not related:
            continue
        new_body = _merge_bullet_section(body, "## 관련 개념", related)
        if new_body != body:
            path.write_text(serialize_fm(meta, new_body), encoding="utf-8")
            updated += 1
    return updated


def _compact_brief(text: str, *, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip().strip(".")
    if not cleaned:
        return "_아직 작성되지 않음._"
    first_sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip().strip(".")
    if len(first_sentence) <= limit:
        return first_sentence + "."
    if len(cleaned) <= limit:
        return cleaned + "."
    return cleaned[: limit - 1].rstrip() + "…"


def _cap_sources(sources: list | tuple, *, limit: int = 8) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for source in sources:
        item = str(source).strip()
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered[:limit]


def _normalize_concept_pages(kb_dir: Path) -> int:
    concepts_dir = kb_dir / "wiki" / "concepts"
    curation = _load_concept_curation_hints(kb_dir)
    brief_overrides = curation.get("brief_overrides", {})
    updated = 0
    for path in sorted(concepts_dir.glob("*.md")):
        meta, body = parse_fm(_read_text(path))
        changed = False

        brief = str(meta.get("brief", "")).strip()
        candidate = _first_paragraph(body)
        new_brief = _compact_brief(candidate if len(brief) > 120 and candidate != "_아직 작성되지 않음._" else brief or candidate)
        if path.stem in brief_overrides:
            new_brief = brief_overrides[path.stem]
        if new_brief != brief:
            meta["brief"] = new_brief
            changed = True
            if "## 개요" in body:
                body = _replace_markdown_section(body, "## 개요", new_brief)

        sources = meta.get("sources", [])
        if isinstance(sources, str):
            sources = [sources]
        if "## 관련 위키" in body:
            new_sources = _cap_sources(sources)
        else:
            new_sources = list(dict.fromkeys(str(source).strip() for source in sources if str(source).strip()))
        if new_sources != sources:
            meta["sources"] = new_sources
            changed = True

        if changed:
            path.write_text(serialize_fm(meta, body), encoding="utf-8")
            updated += 1
    return updated


def _first_paragraph(body: str) -> str:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue
        if stripped.startswith("#") or stripped.startswith("- ") or stripped.startswith("|"):
            continue
        current.append(stripped)
    if current:
        paragraphs.append(" ".join(current).strip())
    for paragraph in paragraphs:
        paragraph = re.sub(r"\s+", " ", paragraph).strip()
        if len(paragraph) >= 30:
            return paragraph.rstrip(".") + "."
    return "_아직 작성되지 않음._"


def _rebuild_catalog_index(kb_dir: Path) -> bool:
    wiki_dir = kb_dir / "wiki"
    index_path = wiki_dir / "index.md"
    if not index_path.exists():
        return False

    meta, _ = parse_fm(_read_text(index_path))
    if meta.get("type") != _WIKI_INDEX_TYPE:
        return False

    wiki_pages: list[dict] = []
    for path in sorted(wiki_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        page_meta, body = parse_fm(_read_text(path))
        if page_meta.get("type") != "wiki":
            continue
        wiki_pages.append(
            {
                "slug": page_meta.get("topic", path.stem),
                "entity_type": page_meta.get("entity_type", ""),
                "category": page_meta.get("category", "") or "미분류",
                "source_count": int(page_meta.get("source_count", 0) or 0),
                "summary": _first_paragraph(body),
            }
        )

    concepts: list[dict] = []
    for path in sorted((wiki_dir / "concepts").glob("*.md")):
        concept_meta, _ = parse_fm(_read_text(path))
        brief = str(concept_meta.get("brief", "")).strip()
        concepts.append({"slug": path.stem, "brief": brief.rstrip(".") + "." if brief else f"{path.stem} 관련 개념 정리."})

    lines = [
        "---",
        f"type: {_WIKI_INDEX_TYPE}",
        "updated_at: 2026-04-19",
        f"topic_count: {len(wiki_pages)}",
        "---",
        "",
        "# 위키 토픽 카탈로그",
        "",
    ]
    for entity_type, heading in (("person", "인물"), ("organization", "조직"), ("project", "프로젝트")):
        rows = [row for row in wiki_pages if row["entity_type"] == entity_type]
        if not rows:
            continue
        lines.extend([f"## {heading}", "", "| slug | 요약 | 소스 수 | wiki |", "|---|---|---|---|"])
        for row in sorted(rows, key=lambda item: item["slug"]):
            summary = row["summary"].replace("|", "\\|")
            lines.append(f"| {row['slug']} | {summary} | {row['source_count']} | [[wiki/{row['slug']}]] |")
        lines.append("")

    lines.extend(["## 개념", ""])
    for concept in concepts:
        lines.append(f"- [[concepts/{concept['slug']}]] — {concept['brief']}")
    lines.extend(["", "## 카테고리별", ""])

    by_category: defaultdict[str, list[dict]] = defaultdict(list)
    for row in wiki_pages:
        by_category[row["category"]].append(row)
    for category in sorted(by_category):
        lines.extend([f"### {category}", ""])
        for row in sorted(by_category[category], key=lambda item: item["slug"]):
            lines.append(f"- [[wiki/{row['slug']}]] — {row['summary']} ({row['source_count']})")
        lines.append("")

    index_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return True


def collect_structural_issues(kb_dir: Path) -> dict:
    wiki_dir = kb_dir / "wiki"
    concepts_dir = wiki_dir / "concepts"
    summaries_dir = wiki_dir / "summaries"
    existing_concepts = {path.stem for path in concepts_dir.glob("*.md")}
    aliases = _load_concept_curation_hints(kb_dir).get("aliases", {})
    source_inventory = _collect_source_inventory(kb_dir)

    broken_source_links: list[str] = []
    broken_wiki_links: list[str] = []
    broken_concept_links: list[str] = []
    incoming_concepts: Counter[str] = Counter()
    summary_missing_related: list[str] = []

    root_wiki_topics: set[str] = set()
    wiki_zero_sources: list[str] = []
    for path in wiki_dir.glob("*.md"):
        if path.name == "index.md":
            continue
        meta, _ = parse_fm(_read_text(path))
        if meta.get("type") == "wiki":
            topic = meta.get("topic", path.stem)
            root_wiki_topics.add(topic)
            if int(meta.get("source_count", 0) or 0) == 0:
                wiki_zero_sources.append(topic)

    for path in _scan_wiki_docs(wiki_dir):
        text = _read_text(path)
        rel = str(path.relative_to(wiki_dir))
        if path.parent == summaries_dir and "## Related Concepts" not in text:
            summary_missing_related.append(path.stem)
        for match in _SOURCE_LINK_RE.finditer(text):
            target = match.group(1)
            if _resolve_source_target(target, source_inventory) is None:
                broken_source_links.append(f"{rel} -> sources/{target}")
        for match in _WIKI_LINK_RE.finditer(text):
            target = match.group(1)
            if target not in root_wiki_topics and target not in {"queries"} and target not in _BROKEN_WIKI_FALLBACKS:
                target_path = wiki_dir / f"{target}.md"
                if not target_path.exists():
                    broken_wiki_links.append(f"{rel} -> wiki/{target}")
        concept_mapping = _normalize_concept_map(existing_concepts)
        for match in _CONCEPT_LINK_RE.finditer(text):
            target = match.group(1)
            canonical = _canonical_concept_slug(target, concept_mapping, existing_concepts, aliases)
            if canonical:
                incoming_concepts[canonical] += 1
            else:
                broken_concept_links.append(f"{rel} -> concepts/{target}")

    duplicate_groups: list[list[str]] = []
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for slug in existing_concepts:
        grouped[_normalize_key(slug)].append(slug)
    for slugs in grouped.values():
        if len(slugs) > 1:
            duplicate_groups.append(sorted(slugs))

    concept_orphans = sorted(slug for slug in existing_concepts if incoming_concepts[slug] == 0)
    weak_concepts: list[tuple[str, int, int]] = []
    for path in sorted(concepts_dir.glob("*.md")):
        text = _read_text(path)
        related_docs = len(re.findall(r"^- \[\[summaries/", text, re.MULTILINE))
        related_concepts = len(re.findall(r"^- \[\[concepts/", text, re.MULTILINE))
        if related_docs <= 2 or related_concepts == 0:
            weak_concepts.append((path.stem, related_docs, related_concepts))

    return {
        "broken_source_links": sorted(broken_source_links),
        "broken_wiki_links": sorted(broken_wiki_links),
        "broken_concept_links": sorted(broken_concept_links),
        "duplicate_slug_groups": sorted(duplicate_groups),
        "concept_orphans": concept_orphans,
        "wiki_zero_sources": sorted(wiki_zero_sources),
        "summaries_missing_related": sorted(summary_missing_related),
        "weak_concepts": weak_concepts,
    }


def render_structural_report(issues: dict) -> str:
    lines = ["## Internal Structural Report", ""]
    for key in (
        "broken_source_links",
        "broken_wiki_links",
        "broken_concept_links",
        "duplicate_slug_groups",
        "concept_orphans",
        "wiki_zero_sources",
        "summaries_missing_related",
    ):
        value = issues.get(key, [])
        lines.append(f"### {key} ({len(value)})")
        if value:
            for item in value[:40]:
                if isinstance(item, list):
                    lines.append(f"- {', '.join(item)}")
                else:
                    lines.append(f"- {item}")
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def run_internal_maintenance(kb_dir: Path) -> dict:
    """Repair internal links and write a KB-local structural report."""
    reports_dir = kb_dir / "wiki" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "updated_source_docs": _repair_source_documents(kb_dir),
        "updated_source_pages": _repair_source_links(kb_dir),
        "updated_wiki_pages": _repair_wiki_links(kb_dir),
        "updated_concept_pages": _repair_concept_links(kb_dir),
        "merged_aliased_concepts": _merge_aliased_concepts(kb_dir),
        "created_related_concepts": len(_create_missing_related_concepts_from_sections(kb_dir)),
        "summary_related_backfills": _backfill_summary_related_concepts(kb_dir),
        "concept_related_backfills": _backfill_concept_related_concepts(kb_dir),
        "refreshed_summary_backed_concepts": _refresh_summary_backed_concepts(kb_dir),
        "normalized_related_sections": _normalize_related_concept_sections(kb_dir),
        "sanitized_bullet_sections": _sanitize_existing_bullet_sections(kb_dir),
        "normalized_concept_pages": _normalize_concept_pages(kb_dir),
        "rebuild_index": _rebuild_catalog_index(kb_dir),
    }
    issues = collect_structural_issues(kb_dir)
    report = render_structural_report(issues)
    report_path = reports_dir / "structural_latest.md"
    report_path.write_text(report, encoding="utf-8")
    return {"stats": stats, "issues": issues, "report_path": report_path}
