"""OpenKB CLI — command-line interface for the knowledge base workflow."""
from __future__ import annotations

# Silence import-time warnings (e.g. pydub's missing-ffmpeg warning emitted
# when markitdown pulls it in). markitdown later clobbers the filters during
# its own import, so we re-apply after all imports below.
import warnings
warnings.filterwarnings("ignore")

import asyncio
import json
import logging
import math
import time
from pathlib import Path

import os

import click
from dotenv import load_dotenv

from openkb.config import DEFAULT_CONFIG, load_config, save_config, load_global_config, register_kb
from openkb.converter import convert_document
from openkb.graph.insights_bg import maybe_trigger_insights
from openkb.log import append_log
from openkb.schema import AGENTS_MD
from openkb.agent.compiler import _make_index_template
from openkb.review import ReviewQueue, apply_review_action

# Suppress warnings after all imports — markitdown overrides filters at import time
import warnings
warnings.filterwarnings("ignore")

load_dotenv()  # load from cwd (covers running inside the KB dir)

logger = logging.getLogger(__name__)


def _run_async_entrypoint(coro):
    """Run a coroutine from synchronous CLI code and close it on early failure."""
    try:
        return asyncio.run(coro)
    except Exception:
        coro.close()
        raise


def _compile_with_retry(fn, *, attempts: int = 2, delay: float = 2.0) -> None:
    """Run an async compile fn with one retry on failure."""
    for attempt in range(attempts):
        try:
            fn()
            return
        except Exception as exc:
            if attempt == 0:
                click.echo(f"  Retrying compilation in {delay:.0f}s...")
                time.sleep(delay)
            else:
                click.echo(f"  [ERROR] Compilation failed: {exc}")
                logger.debug("Compilation traceback:", exc_info=True)
                raise

# Supported document extensions for the `add` command
SUPPORTED_EXTENSIONS = {
    ".pdf", ".md", ".markdown", ".docx", ".pptx", ".xlsx",
    ".html", ".htm", ".txt", ".csv",
}

# Map raw doc types to display types
_TYPE_DISPLAY_MAP = {
    "long_pdf": "pageindex",
}

_SHORT_DOC_TYPES = {"pdf", "docx", "md", "markdown", "html", "htm", "txt", "csv", "pptx", "xlsx"}


def _display_type(raw_type: str) -> str:
    """Map a raw stored doc type to a display type string."""
    if raw_type in _TYPE_DISPLAY_MAP:
        return _TYPE_DISPLAY_MAP[raw_type]
    if raw_type in _SHORT_DOC_TYPES:
        return "short"
    return raw_type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_kb_dir(override: Path | None = None) -> Path | None:
    """Find the KB root: explicit override → walk up from cwd → global default_kb."""
    # 0. Explicit override (--kb-dir or OPENKB_DIR)
    if override is not None:
        if (override / ".openkb").is_dir():
            return override
        return None
    # 1. Walk up from cwd
    current = Path.cwd().resolve()
    while True:
        if (current / ".openkb").is_dir():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    # 2. Fall back to global config default_kb
    gc = load_global_config()
    default = gc.get("default_kb")
    if default:
        p = Path(default)
        if (p / ".openkb").is_dir():
            return p
    return None


def add_single_file(file_path: Path, kb_dir: Path) -> bool:
    """Convert, index, and compile a single document into the knowledge base.

    Steps:
    1. Load config to get the model name.
    2. Convert the document (hash-check; skip if already known).
    3. If long doc: run PageIndex then compile_long_doc.
    4. Else: compile_short_doc.
    """
    from openkb.agent.compiler import compile_long_doc, compile_short_doc
    from openkb.state import HashRegistry

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    model: str = config.get("model", DEFAULT_CONFIG["model"])
    registry = HashRegistry(openkb_dir / "hashes.json")

    # 2. Convert document
    click.echo(f"Adding: {file_path.name}")
    try:
        result = convert_document(file_path, kb_dir)
    except Exception as exc:
        click.echo(f"  [ERROR] Conversion failed: {exc}")
        logger.debug("Conversion traceback:", exc_info=True)
        return False

    if result.skipped:
        click.echo(f"  [SKIP] Already in knowledge base: {file_path.name}")
        return False

    doc_name = file_path.stem

    # 3/4. Index and compile
    if result.is_long_doc:
        click.echo(f"  Long document detected — indexing with PageIndex...")
        try:
            from openkb.indexer import index_long_document
            index_result = index_long_document(result.raw_path, kb_dir)
        except Exception as exc:
            click.echo(f"  [ERROR] Indexing failed: {exc}")
            logger.debug("Indexing traceback:", exc_info=True)
            return False

        summary_path = kb_dir / "wiki" / "summaries" / f"{doc_name}.md"
        click.echo(f"  Compiling long doc (doc_id={index_result.doc_id})...")
        _compile_with_retry(
            lambda: _run_async_entrypoint(
                compile_long_doc(doc_name, summary_path, index_result.doc_id, kb_dir, model,
                                 doc_description=index_result.description)
            )
        )
    else:
        click.echo(f"  Compiling short doc...")
        _compile_with_retry(
            lambda: _run_async_entrypoint(
                compile_short_doc(doc_name, result.source_path, kb_dir, model)
            )
        )

    # Register hash only after successful compilation
    if result.file_hash:
        doc_type = "long_pdf" if result.is_long_doc else file_path.suffix.lstrip(".")
        registry.add(result.file_hash, {"name": file_path.name, "type": doc_type})

    append_log(kb_dir / "wiki", "ingest", file_path.name)
    click.echo(f"  [OK] {file_path.name} added to knowledge base.")
    return True


def _normalize_background_insights_cooldown(raw_value: object) -> int:
    """Coerce YAML-loaded cooldown values to a safe integer boundary."""
    default_cooldown = DEFAULT_CONFIG["insights_cooldown"]
    if isinstance(raw_value, bool):
        return default_cooldown
    try:
        parsed_value = float(raw_value)
    except (TypeError, ValueError, OverflowError):
        return default_cooldown
    if not math.isfinite(parsed_value) or parsed_value < 0:
        return default_cooldown
    return int(parsed_value)


def _trigger_background_insights_after_add(kb_dir: Path) -> None:
    """Start cooldown-gated insights refresh after a successful add."""
    maybe_trigger_insights(kb_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option("-v", "--verbose", is_flag=True, default=False, help="Enable verbose logging.")
@click.option("--kb-dir", "kb_dir_override", default=None, type=click.Path(exists=True, file_okay=False, resolve_path=True), help="Path to a KB root directory (overrides auto-detection).")
@click.pass_context
def cli(ctx, verbose, kb_dir_override):
    """OpenKB — Karpathy's LLM Knowledge Base workflow, powered by PageIndex."""
    logging.basicConfig(
        format="%(name)s %(levelname)s: %(message)s",
        level=logging.WARNING,
    )
    if verbose:
        logging.getLogger("openkb").setLevel(logging.DEBUG)
    ctx.ensure_object(dict)
    if kb_dir_override:
        ctx.obj["kb_dir_override"] = Path(kb_dir_override)
    else:
        env_kb = os.environ.get("OPENKB_DIR")
        if env_kb:
            ctx.obj["kb_dir_override"] = Path(env_kb).resolve()
        else:
            ctx.obj["kb_dir_override"] = None


@cli.command()
@click.argument("path", default=".")
def use(path):
    """Set PATH as the default knowledge base."""
    target = Path(path).resolve()
    if not (target / ".openkb").is_dir():
        click.echo(f"Not a knowledge base: {target}")
        return
    register_kb(target)
    click.echo(f"Default KB set to: {target}")


@cli.command()
def init():
    """Initialise a new knowledge base in the current directory."""
    openkb_dir = Path(".openkb")
    if openkb_dir.exists():
        click.echo("Knowledge base already initialized.")
        return

    # Interactive prompts
    click.echo("Pick an executor provider and model:")
    click.echo("  claude     -> sonnet, opus, claude-sonnet-4-6")
    click.echo("  codex_app  -> gpt-5.4-mini, gpt-5.4")
    click.echo("  codex      -> gpt-5.4-mini, gpt-5.4")
    click.echo("  ollama     -> llama3, mistral, qwen2")
    click.echo()
    provider = click.prompt(
        f"Provider (enter for default {DEFAULT_CONFIG['provider']})",
        default=DEFAULT_CONFIG["provider"],
        show_default=False,
    ).strip()
    model = click.prompt(
        f"Model (enter for default {DEFAULT_CONFIG['model']})",
        default=DEFAULT_CONFIG["model"],
        show_default=False,
    ).strip()
    effort = click.prompt(
        f"Effort (enter for default {DEFAULT_CONFIG['effort']})",
        default=DEFAULT_CONFIG["effort"],
        show_default=False,
    ).strip()
    # Create directory structure
    Path("raw").mkdir(exist_ok=True)
    Path("wiki/sources/images").mkdir(parents=True, exist_ok=True)
    Path("wiki/summaries").mkdir(parents=True, exist_ok=True)
    Path("wiki/concepts").mkdir(parents=True, exist_ok=True)

    # Write wiki files
    Path("wiki/AGENTS.md").write_text(AGENTS_MD, encoding="utf-8")
    language = DEFAULT_CONFIG["language"]
    Path("wiki/index.md").write_text(_make_index_template(language), encoding="utf-8")
    Path("wiki/log.md").write_text("# Operations Log\n\n", encoding="utf-8")

    # Create .openkb/ state directory
    openkb_dir.mkdir()
    config = {
        "provider": provider or DEFAULT_CONFIG["provider"],
        "model": model,
        "effort": effort or DEFAULT_CONFIG["effort"],
        "language": language,
        "pageindex_threshold": DEFAULT_CONFIG["pageindex_threshold"],
        "insights_cooldown": DEFAULT_CONFIG["insights_cooldown"],
        "background_insights_cooldown_seconds": DEFAULT_CONFIG["background_insights_cooldown_seconds"],
    }
    save_config(openkb_dir / "config.yaml", config)
    (openkb_dir / "hashes.json").write_text(json.dumps({}), encoding="utf-8")

    # Register this KB in the global config
    register_kb(Path.cwd())

    click.echo("Knowledge base initialized.")


def _coerce_config_value(value: str) -> object:
    """Coerce CLI config values to int/float when possible."""
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


@cli.group()
@click.pass_context
def config(ctx):
    """Manage OpenKB configuration."""


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.pass_context
def config_set(ctx, key, value):
    """Set a config value: openkb config set KEY VALUE."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    config_path = kb_dir / ".openkb" / "config.yaml"
    config_data = load_config(config_path)
    coerced_value = _coerce_config_value(value)

    config_data[key] = coerced_value
    if key == "insights_cooldown":
        config_data["background_insights_cooldown_seconds"] = coerced_value
    elif key == "background_insights_cooldown_seconds":
        config_data["insights_cooldown"] = coerced_value

    save_config(config_path, config_data)
    click.echo(f"Set {key} = {coerced_value}")


@config.command("get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key):
    """Get a config value: openkb config get KEY."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    config_data = load_config(kb_dir / ".openkb" / "config.yaml")
    if key == "insights_cooldown":
        value = config_data.get(
            "insights_cooldown",
            config_data.get(
                "background_insights_cooldown_seconds",
                DEFAULT_CONFIG["insights_cooldown"],
            ),
        )
    else:
        value = config_data.get(key)

    if value is None:
        click.echo(f"{key}: (not set)")
    else:
        click.echo(f"{key}: {value}")


@cli.command()
@click.argument("path")
@click.pass_context
def add(ctx, path):
    """Add a document, URL, or directory at PATH to the knowledge base."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    # --- URL path ---
    from openkb.url_fetch import is_url, fetch_url, FetchError

    if is_url(path):
        click.echo(f"Fetching URL: {path}")
        try:
            markdown, slug = fetch_url(path)
        except FetchError as exc:
            click.echo(f"  [ERROR] {exc}")
            return
        raw_dir = kb_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_path = raw_dir / f"{slug}.md"
        raw_path.write_text(markdown, encoding="utf-8")
        click.echo(f"  Saved to {raw_path.relative_to(kb_dir)}")
        if add_single_file(raw_path, kb_dir):
            _trigger_background_insights_after_add(kb_dir)
        return

    # --- Local file path ---
    target = Path(path)
    if not target.exists():
        click.echo(f"Path does not exist: {path}")
        return

    if target.is_dir():
        files = [
            f for f in sorted(target.rglob("*"))
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not files:
            click.echo(f"No supported files found in {path}.")
            return
        total = len(files)
        click.echo(f"Found {total} supported file(s) in {path}.")
        for i, f in enumerate(files, 1):
            click.echo(f"\n[{i}/{total}] ", nl=False)
            if add_single_file(f, kb_dir):
                _trigger_background_insights_after_add(kb_dir)
    else:
        if target.suffix.lower() not in SUPPORTED_EXTENSIONS:
            click.echo(
                f"Unsupported file type: {target.suffix}. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )
            return
        if add_single_file(target, kb_dir):
            _trigger_background_insights_after_add(kb_dir)


def _parse_frontmatter(path: Path) -> dict | None:
    """Parse YAML frontmatter from a markdown file. Returns dict or None."""
    import yaml
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None


@cli.command("add-sources")
@click.argument("source_dir", required=False, type=click.Path(exists=True, file_okay=False))
@click.option("--type", "source_type_filter", default=None,
              help="Only add URLs with this source_type (e.g. twitter, youtube, article).")
@click.option("--limit", "max_count", type=int, default=None,
              help="Maximum number of URLs to process.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Show what would be added without fetching.")
@click.option("--concurrency", "max_workers", type=int, default=1,
              help="Number of parallel compilation workers (default: 1).")
@click.pass_context
def add_sources(ctx, source_dir, source_type_filter, max_count, dry_run, max_workers):
    """Scan a source directory for frontmatter URLs and add them to the KB."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.url_fetch import is_url, fetch_url, FetchError

    source_path = Path(source_dir) if source_dir else (kb_dir / "sources")
    if not source_path.exists():
        click.echo(f"Source directory not found: {source_path}")
        return
    md_files = sorted(source_path.rglob("*.md"))

    urls_to_add: list[tuple[str, str, Path]] = []  # (url, source_type, source_file)
    for md_file in md_files:
        fm = _parse_frontmatter(md_file)
        if not fm:
            continue
        url = str(fm.get("url", "")).strip('"').strip("'")
        stype = str(fm.get("source_type", ""))
        if not url or not is_url(url):
            continue
        if source_type_filter and stype != source_type_filter:
            continue
        urls_to_add.append((url, stype, md_file))

    if max_count:
        urls_to_add = urls_to_add[:max_count]

    click.echo(f"Found {len(urls_to_add)} URLs to add.")
    if dry_run:
        for i, (url, stype, src_file) in enumerate(urls_to_add, 1):
            click.echo(f"  [{i}] [{stype}] {url}")
        return

    raw_dir = kb_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    def _process_one(item: tuple[int, str, str, Path]) -> str:
        """Fetch + compile a single URL. Returns status string."""
        i, url, stype, src_file = item
        tag = f"[{i}/{len(urls_to_add)}]"
        try:
            markdown, slug = fetch_url(url)
        except FetchError as exc:
            return f"{tag} [ERROR] {exc}"
        raw_path = raw_dir / f"{slug}.md"
        raw_path.write_text(markdown, encoding="utf-8")
        # add_single_file uses click.echo which is not thread-safe,
        # so redirect output to a buffer and print from main thread.
        import io
        buf = io.StringIO()
        old_echo = click.echo
        click.echo = lambda msg, **kw: buf.write(str(msg) + "\n")
        try:
            add_single_file(raw_path, kb_dir)
        finally:
            click.echo = old_echo
        output = buf.getvalue()
        # Determine result from output
        if "[SKIP]" in output:
            return f"{tag} [SKIP] {slug}"
        if "[ERROR]" in output:
            return f"{tag} [ERROR] {slug}: {output.strip()[-200:]}"
        return f"{tag} [OK] {slug}"

    items = [(i, url, stype, src_file) for i, (url, stype, src_file) in enumerate(urls_to_add, 1)]

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        click.echo(f"Using {max_workers} parallel workers.")
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one, item): item for item in items}
            for future in as_completed(futures):
                result = future.result()
                click.echo(result)
    else:
        for item in items:
            result = _process_one(item)
            click.echo(result)


@cli.command()
@click.argument("question")
@click.option("--save", is_flag=True, default=False, help="Save the answer to wiki/explorations/.")
@click.option(
    "--raw", "raw",
    is_flag=True, default=False,
    help="Show raw markdown source instead of rendered output (keeps tool-call colors).",
)
@click.pass_context
def query(ctx, question, save, raw):
    """Query the knowledge base with QUESTION."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.agent.query import run_query

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    model: str = config.get("model", DEFAULT_CONFIG["model"])

    try:
        answer = asyncio.run(run_query(question, kb_dir, model, stream=True, raw=raw))
    except Exception as exc:
        click.echo(f"[ERROR] Query failed: {exc}")
        return

    append_log(kb_dir / "wiki", "query", question)

    if save and answer:
        import re
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")[:60]
        explore_dir = kb_dir / "wiki" / "explorations"
        explore_dir.mkdir(parents=True, exist_ok=True)
        explore_path = explore_dir / f"{slug}.md"
        explore_path.write_text(
            f"---\nquery: \"{question}\"\n---\n\n{answer}\n", encoding="utf-8"
        )
        click.echo(f"\nSaved to {explore_path}")


@cli.command()
@click.argument("path")
@click.option(
    "--mode",
    "mode",
    required=True,
    type=click.Choice(["query_page", "concept_seed"], case_sensitive=False),
    help="Promotion target mode.",
)
@click.pass_context
def promote(ctx, path, mode):
    """Promote a saved exploration into a durable query page or concept seed."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.promotion import promote_exploration

    try:
        result = promote_exploration(kb_dir, path, mode=mode)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"[ERROR] Promotion failed: {exc}")
        return

    if mode == "query_page":
        click.echo(f"Promoted exploration to query page: {result['target_path']}")
    else:
        click.echo(f"Queued concept seed review item: {result['target_path']}")


@cli.command()
@click.option(
    "--resume", "-r", "resume",
    is_flag=False, flag_value="__latest__", default=None, metavar="[ID]",
    help="Resume the latest chat session, or a specific one by id or prefix.",
)
@click.option(
    "--list", "list_sessions_flag",
    is_flag=True, default=False,
    help="List chat sessions.",
)
@click.option(
    "--delete", "delete_id",
    default=None, metavar="ID",
    help="Delete a chat session by id or prefix.",
)
@click.option(
    "--no-color", "no_color",
    is_flag=True, default=False,
    help="Disable colored output.",
)
@click.option(
    "--raw", "raw",
    is_flag=True, default=False,
    help="Show raw markdown source instead of rendered output (keeps prompt and tool-call colors).",
)
@click.pass_context
def chat(ctx, resume, list_sessions_flag, delete_id, no_color, raw):
    """Start an interactive chat with the knowledge base."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.agent.chat_session import (
        ChatSession,
        delete_session,
        list_sessions,
        load_session,
        relative_time,
        resolve_session_id,
    )

    if list_sessions_flag:
        sessions = list_sessions(kb_dir)
        if not sessions:
            click.echo("No chat sessions yet.")
            return
        click.echo(f"  {'ID':<22} {'TURNS':<6} {'UPDATED':<12} TITLE")
        click.echo(f"  {'-'*22} {'-'*6} {'-'*12} {'-'*30}")
        for s in sessions:
            rel = relative_time(s.get("updated_at", ""))
            title = s.get("title") or "(empty)"
            click.echo(
                f"  {s['id']:<22} {s['turn_count']:<6} {rel:<12} {title}"
            )
        click.echo(
            f"\n{len(sessions)} session(s) in {kb_dir / '.openkb' / 'chats'}"
        )
        return

    if delete_id is not None:
        try:
            resolved = resolve_session_id(kb_dir, delete_id)
        except ValueError as exc:
            click.echo(f"[ERROR] {exc}")
            return
        if not resolved:
            click.echo(f"No matching session: {delete_id}")
            return
        if delete_session(kb_dir, resolved):
            click.echo(f"Deleted session {resolved}")
        else:
            click.echo(f"Could not delete session: {resolved}")
        return

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")

    if resume is not None:
        try:
            resolved = resolve_session_id(kb_dir, resume)
        except ValueError as exc:
            click.echo(f"[ERROR] {exc}")
            return
        if not resolved:
            if resume == "__latest__":
                click.echo("No previous chat sessions to resume.")
            else:
                click.echo(f"No matching session: {resume}")
            return
        session = load_session(kb_dir, resolved)
    else:
        model: str = config.get("model", DEFAULT_CONFIG["model"])
        language: str = config.get("language", "en")
        session = ChatSession.new(kb_dir, model, language)

    from openkb.agent.chat import run_chat

    try:
        asyncio.run(run_chat(kb_dir, session, no_color=no_color, raw=raw))
    except Exception as exc:
        click.echo(f"[ERROR] Chat failed: {exc}")


@cli.command()
@click.pass_context
def watch(ctx):
    """Watch the raw/ directory for new documents and process them automatically."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.watcher import watch_directory

    raw_dir = kb_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    def on_new_files(paths):
        for p in paths:
            fp = Path(p)
            if fp.suffix.lower() not in SUPPORTED_EXTENSIONS:
                click.echo(
                    f"Skipping unsupported file type: {fp.suffix}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                )
                continue
            add_single_file(fp, kb_dir)

    click.echo(f"Watching {raw_dir} for new documents. Press Ctrl+C to stop.")
    watch_directory(raw_dir, on_new_files)


async def run_lint(kb_dir: Path) -> Path | None:
    """Run structural + knowledge lint, write report, return report path.

    Returns ``None`` if the KB has no indexed documents (nothing to lint).
    Async because knowledge lint uses an LLM agent. Usable from CLI
    (via ``asyncio.run``) and directly from the chat REPL.
    """
    from openkb.lint import run_structural_lint
    from openkb.agent.linter import run_knowledge_lint

    openkb_dir = kb_dir / ".openkb"

    # Skip lint entirely when the KB has no indexed documents
    hashes_file = openkb_dir / "hashes.json"
    if hashes_file.exists():
        hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
    else:
        hashes = {}
    if not hashes:
        click.echo("Nothing to lint — no documents indexed yet. Run `openkb add` first.")
        return

    config = load_config(openkb_dir / "config.yaml")
    model: str = config.get("model", DEFAULT_CONFIG["model"])

    click.echo("Running structural lint...")
    structural_report = run_structural_lint(kb_dir)
    click.echo(structural_report)

    click.echo("Running knowledge lint...")
    try:
        knowledge_report = await run_knowledge_lint(kb_dir, model)
    except Exception as exc:
        knowledge_report = f"Knowledge lint failed: {exc}"
    click.echo(knowledge_report)

    # Write combined report
    reports_dir = kb_dir / "wiki" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"lint_{timestamp}.md"
    report_content = f"# Lint Report — {timestamp}\n\n## Structural\n\n{structural_report}\n\n## Semantic\n\n{knowledge_report}\n"
    report_path.write_text(report_content, encoding="utf-8")
    append_log(kb_dir / "wiki", "lint", f"report → {report_path.name}")
    click.echo(f"\nReport written to {report_path}")
    return report_path


@cli.command()
@click.option("--fix", is_flag=True, default=False, help="Automatically fix lint issues (not yet implemented).")
@click.pass_context
def lint(ctx, fix):
    """Lint the knowledge base for structural and semantic inconsistencies."""
    if fix:
        click.echo("Warning: --fix is not yet implemented. Running lint in report-only mode.")
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return
    asyncio.run(run_lint(kb_dir))


@cli.command()
@click.pass_context
def quality(ctx):
    """Aggregate current KB quality signals into one latest report."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.quality_loop import run_quality_convergence

    config = load_config(kb_dir / ".openkb" / "config.yaml")
    model: str = config.get("model", DEFAULT_CONFIG["model"])
    result = run_quality_convergence(kb_dir, model)
    click.echo(f"Structural issues: {result['structural_issue_count']}")
    click.echo(f"Semantic report: {result['semantic_report']}")
    click.echo(f"Pending review items: {result['pending_review_count']}")
    click.echo(result["insights"]["summary"])
    click.echo(f"Quality report: {result['quality_report']}")


@cli.command()
@click.option("--plan", "plan_only", is_flag=True, default=False, help="Show stale refresh plan.")
@click.pass_context
def refresh(ctx, plan_only):
    """Show stale pages that should be refreshed."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return
    if not plan_only:
        click.echo("This stage only supports planning. Re-run with --plan.")
        return

    from openkb.refresh import collect_stale_pages, render_refresh_plan

    click.echo(render_refresh_plan(collect_stale_pages(kb_dir)))


def print_list(kb_dir: Path) -> None:
    """Print all documents in the knowledge base. Usable from CLI and chat REPL."""
    openkb_dir = kb_dir / ".openkb"
    hashes_file = openkb_dir / "hashes.json"
    if not hashes_file.exists():
        click.echo("No documents indexed yet.")
        return

    hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
    if not hashes:
        click.echo("No documents indexed yet.")
        return

    # Display documents table with count in header
    doc_count = len(hashes)
    click.echo(f"Documents ({doc_count}):")
    click.echo(f"  {'Name':<40} {'Type':<12} {'Pages':<8}")
    click.echo(f"  {'-'*40} {'-'*12} {'-'*8}")
    for file_hash, meta in hashes.items():
        name = meta.get("name", "unknown")
        raw_type = meta.get("type", "unknown")
        display = _display_type(raw_type)
        pages = meta.get("pages", "")
        pages_str = str(pages) if pages else ""
        click.echo(f"  {name:<40} {display:<12} {pages_str:<8}")

    # Display summaries
    summaries_dir = kb_dir / "wiki" / "summaries"
    if summaries_dir.exists():
        summaries = sorted(p.stem for p in summaries_dir.glob("*.md"))
        if summaries:
            click.echo(f"\nSummaries ({len(summaries)}):")
            for s in summaries:
                click.echo(f"  - {s}")

    # Display concepts
    concepts_dir = kb_dir / "wiki" / "concepts"
    if concepts_dir.exists():
        concepts = sorted(p.stem for p in concepts_dir.glob("*.md"))
        if concepts:
            click.echo(f"\nConcepts ({len(concepts)}):")
            for c in concepts:
                click.echo(f"  - {c}")

    # Display reports
    reports_dir = kb_dir / "wiki" / "reports"
    if reports_dir.exists():
        reports = sorted(p.name for p in reports_dir.glob("*.md"))
        if reports:
            click.echo(f"\nReports ({len(reports)}):")
            for r in reports:
                click.echo(f"  - {r}")


@cli.command(name="list")
@click.pass_context
def list_cmd(ctx):
    """List all documents in the knowledge base."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return
    print_list(kb_dir)


def print_status(kb_dir: Path) -> None:
    """Print knowledge base status. Usable from CLI and chat REPL."""
    wiki_dir = kb_dir / "wiki"
    subdirs = ["sources", "summaries", "concepts", "reports"]

    click.echo("Knowledge Base Status:")
    click.echo(f"  {'Directory':<20} {'Files':<10}")
    click.echo(f"  {'-'*20} {'-'*10}")

    for subdir in subdirs:
        path = wiki_dir / subdir
        if path.exists():
            count = len(list(path.glob("*.md")))
        else:
            count = 0
        click.echo(f"  {subdir:<20} {count:<10}")

    # Raw files
    raw_dir = kb_dir / "raw"
    if raw_dir.exists():
        raw_count = len([f for f in raw_dir.iterdir() if f.is_file()])
        click.echo(f"  {'raw':<20} {raw_count:<10}")

    # Hash registry summary
    openkb_dir = kb_dir / ".openkb"
    hashes_file = openkb_dir / "hashes.json"
    if hashes_file.exists():
        hashes = json.loads(hashes_file.read_text(encoding="utf-8"))
        click.echo(f"\n  Total indexed: {len(hashes)} document(s)")

    # Last compile time: newest file in wiki/summaries/
    summaries_dir = wiki_dir / "summaries"
    if summaries_dir.exists():
        summaries = list(summaries_dir.glob("*.md"))
        if summaries:
            newest_summary = max(summaries, key=lambda p: p.stat().st_mtime)
            import datetime
            mtime = datetime.datetime.fromtimestamp(newest_summary.stat().st_mtime)
            click.echo(f"  Last compile:  {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Last lint time: newest file in wiki/reports/
    reports_dir = wiki_dir / "reports"
    if reports_dir.exists():
        reports = list(reports_dir.glob("*.md"))
        if reports:
            newest_report = max(reports, key=lambda p: p.stat().st_mtime)
            import datetime
            mtime = datetime.datetime.fromtimestamp(newest_report.stat().st_mtime)
            click.echo(f"  Last lint:     {mtime.strftime('%Y-%m-%d %H:%M:%S')}")


@cli.command()
@click.option("--accept", "accept_idx", type=int, default=None, help="Accept and remove the item at the given index.")
@click.option("--apply", "apply_idx", type=int, default=None, help="Apply the item action and remove it on success.")
@click.option("--skip", "skip_idx", type=int, default=None, help="Skip and remove the item at the given index.")
@click.pass_context
def review(ctx, accept_idx, apply_idx, skip_idx):
    """Review pending items from the analysis step."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    openkb_dir = kb_dir / ".openkb"
    queue = ReviewQueue(openkb_dir)
    items = queue.list()
    operation_count = sum(idx is not None for idx in (accept_idx, apply_idx, skip_idx))
    if operation_count > 1:
        click.echo("Choose only one of --accept, --apply, or --skip.")
        return

    if accept_idx is not None:
        if not items or accept_idx < 0 or accept_idx >= len(items):
            click.echo("Invalid index. No item at that position.")
            return
        item = queue.accept(accept_idx)
        click.echo(
            f"Accepted without applying changes: [{item.type}] {item.title}\n"
            f"  {item.description}"
        )
        return

    if apply_idx is not None:
        if not items or apply_idx < 0 or apply_idx >= len(items):
            click.echo("Invalid index. No item at that position.")
            return
        try:
            item = queue.apply(apply_idx, lambda queued_item: apply_review_action(kb_dir, queued_item))
        except Exception as exc:
            click.echo(f"[ERROR] Failed to apply review item at index {apply_idx}: {exc}")
            return
        click.echo(
            f"Applied: [{item.type}] {item.title}\n"
            f"  Action: {item.action_type}\n"
            f"  {item.description}"
        )
        return

    if skip_idx is not None:
        if not items or skip_idx < 0 or skip_idx >= len(items):
            click.echo("Invalid index. No item at that position.")
            return
        queue.skip(skip_idx)
        click.echo(f"Skipped item at index {skip_idx}.")
        return

    if not items:
        click.echo("No pending review items.")
        return

    click.echo(f"Pending review items ({len(items)}):\n")
    for i, item in enumerate(items):
        click.echo(f"  [{i}] ({item.type}) {item.title}")
        click.echo(f"      {item.description}")
        if item.action_type:
            click.echo(f"      Action: {item.action_type} [{item.status}]")
        if item.affected_pages:
            click.echo(f"      Affected: {', '.join(item.affected_pages)}")
        click.echo()


@cli.command()
@click.pass_context
def status(ctx):
    """Show the current status of the knowledge base."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return
    print_status(kb_dir)


@cli.command(name="insights")
@click.pass_context
def insights(ctx):
    """Show graph insights: communities, knowledge gaps, and surprising connections."""
    kb_dir = _find_kb_dir(ctx.obj.get("kb_dir_override"))
    if kb_dir is None:
        click.echo("No knowledge base found. Run `openkb init` first.")
        return

    from openkb.graph.build import build_graph, load_graph, build_and_save_graph, GraphLoadError
    from openkb.graph.insights import generate_insights

    wiki_dir = kb_dir / "wiki"
    graph_path = kb_dir / ".openkb" / "graph.json"

    # Load or build graph
    try:
        if graph_path.exists():
            graph = load_graph(graph_path)
        else:
            openkb_dir = kb_dir / ".openkb"
            graph, _ = build_and_save_graph(wiki_dir, openkb_dir)
    except GraphLoadError:
        click.echo("graph.json이 손상되었습니다. `openkb add`로 재생성하세요.")
        return

    if graph.number_of_nodes() == 0:
        click.echo("No graph data. Add documents first with `openkb add`.")
        return

    result = generate_insights(graph)

    # Header
    n_comm = len(result["communities_summary"])
    click.echo(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, {n_comm} communities\n")

    # Surprising connections
    click.echo("== Surprising Connections ==")
    for src, tgt, reason, score in result["surprising_connections"][:10]:
        click.echo(f"  {src} <-> {tgt}  [{reason}]  score={score:.2f}")
    if not result["surprising_connections"]:
        click.echo("  (none)")

    # Knowledge gaps
    click.echo("\n== Knowledge Gaps ==")
    orphans = result["orphans"]
    if orphans:
        click.echo(f"  Orphan nodes ({len(orphans)}):")
        for name, deg in orphans[:10]:
            click.echo(f"    {name}  (degree={deg})")
    sparse = result["sparse_communities"]
    if sparse:
        click.echo(f"  Sparse communities ({len(sparse)}):")
        for cid, rep, coh in sparse[:10]:
            click.echo(f"    community {cid} (rep: {rep})  cohesion={coh:.3f}")
    bridges = result["bridge_nodes"]
    if bridges:
        click.echo(f"  Bridge nodes ({len(bridges)}):")
        for name, span in bridges[:10]:
            click.echo(f"    {name}  (spans {span} communities)")
    if not orphans and not sparse and not bridges:
        click.echo("  (none)")

    # Communities
    click.echo("\n== Communities ==")
    for cid, info in sorted(result["communities_summary"].items()):
        click.echo(f"  [{cid}] {info['representative']}  size={info['size']}  cohesion={info['cohesion']}")
