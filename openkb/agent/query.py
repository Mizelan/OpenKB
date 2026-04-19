"""Q&A agent for querying the OpenKB knowledge base."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from openkb.agent.executor_runtime import ExecutorAgent, ExecutorTool, run_executor_agent
from openkb.agent.tools import get_wiki_page_content, read_wiki_file, read_wiki_image, search_related_pages
from openkb.schema import get_agents_md

MAX_TURNS = 50

_QUERY_INSTRUCTIONS_TEMPLATE = """\
You are OpenKB, a knowledge-base Q&A agent. You answer questions by searching the wiki.

{schema_md}

## Search strategy
1. Read index.md to see all documents and concepts with brief summaries.
   Each document is marked (short) or (pageindex) to indicate its type.
2. Read relevant summary pages (summaries/) for document overviews.
   Summaries may omit details — if you need more, follow the summary's
   `full_text` frontmatter field to the source (see step 4).
3. Read concept pages (concepts/) for cross-document synthesis.
4. When you need detailed source document content, each summary page has a
   `full_text` frontmatter field with the path to the original document content:
   - Short documents (doc_type: short): read_file with that path.
   - PageIndex documents (doc_type: pageindex): use get_page_content(doc_name, pages)
     with tight page ranges. The summary shows document tree structure with page
     ranges to help you target. Never fetch the whole document.
4.5 After finding relevant concepts, use search_related to discover indirectly
    related pages via the knowledge graph. This expands your coverage beyond
    direct wikilinks and can surface pages that share sources or neighbours.
5. Source content may reference images (e.g. ![image](sources/images/doc/file.png)).
   Use the get_image tool when needed. Some executor providers may return a text
   fallback instead of inline image bytes, so answer conservatively when image
   evidence is incomplete.
6. Synthesize a clear, concise, well-cited answer grounded in wiki content.

Answer based only on wiki content. Be concise.
If you cannot find relevant information, say so clearly.
"""


def build_query_agent(
    wiki_root: str,
    model: str,
    language: str = "en",
    *,
    provider: str = "",
    effort: str = "medium",
) -> ExecutorAgent:
    """Build and return the Q&A agent."""
    schema_md = get_agents_md(Path(wiki_root))
    instructions = _QUERY_INSTRUCTIONS_TEMPLATE.format(schema_md=schema_md)
    instructions += f"\n\nIMPORTANT: Answer in {language} language."

    def read_file(path: str) -> str:
        return read_wiki_file(path, wiki_root)

    def get_page_content(doc_name: str, pages: str) -> str:
        return get_wiki_page_content(doc_name, pages, wiki_root)

    def get_image(image_path: str) -> str:
        result = read_wiki_image(image_path, wiki_root)
        if result["type"] == "image":
            return (
                f"Image available at {image_path}. "
                "Executor mode does not inline the raw image bytes in the transcript."
            )
        return result["text"]

    kb_dir = str(Path(wiki_root).parent)

    def search_related(page_name: str, top_k: int = 5) -> str:
        return search_related_pages(page_name, top_k, kb_dir)

    return ExecutorAgent(
        name="wiki-query",
        instructions=instructions,
        tools=[
            ExecutorTool(
                name="read_file",
                description="Read a Markdown file from the wiki by relative path.",
                handler=read_file,
            ),
            ExecutorTool(
                name="get_page_content",
                description="Read specific page ranges from a PageIndex document.",
                handler=get_page_content,
            ),
            ExecutorTool(
                name="get_image",
                description="Inspect an image reference from the wiki. May return a text fallback in executor mode.",
                handler=get_image,
            ),
            ExecutorTool(
                name="search_related",
                description="Find related pages using the knowledge graph.",
                handler=search_related,
            ),
        ],
        model=model,
        provider=provider,
        effort=effort,
        working_dir=kb_dir,
        max_turns=MAX_TURNS,
    )


async def run_query(
    question: str,
    kb_dir: Path,
    model: str,
    stream: bool = False,
    *,
    raw: bool = False,
) -> str:
    """Run a Q&A query against the knowledge base."""
    from openkb.config import load_config

    openkb_dir = kb_dir / ".openkb"
    config = load_config(openkb_dir / "config.yaml")
    language: str = config.get("language", "en")
    provider: str = config.get("provider", "")
    effort: str = config.get("effort", "medium")
    wiki_root = str(kb_dir / "wiki")

    agent = build_query_agent(
        wiki_root,
        model,
        language=language,
        provider=provider,
        effort=effort,
    )

    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR", "")
    streamed_text = False

    def _emit_tool_call(name: str, args: dict[str, object], reason: str) -> None:
        if not stream:
            return
        from openkb.agent.chat import _build_style, _fmt, _format_tool_line

        arg_text = json.dumps(args, ensure_ascii=False, sort_keys=True)
        style = _build_style(use_color)
        line = _format_tool_line(name, arg_text)
        if reason:
            line = f"{reason} {line}"
        _fmt(style, ("class:tool", line + "\n"))

    def _emit_text_delta(text: str) -> None:
        nonlocal streamed_text
        if not stream or not text:
            return
        streamed_text = True
        sys.stdout.write(text)
        sys.stdout.flush()

    result = await run_executor_agent(
        agent,
        question,
        on_tool_call=_emit_tool_call if stream else None,
        on_text_delta=_emit_text_delta if stream else None,
    )
    answer = result.final_output or ""

    if stream and streamed_text:
        sys.stdout.write("\n")
        sys.stdout.flush()
    elif stream:
        from openkb.agent.chat import _make_markdown, _make_rich_console

        if use_color and not raw:
            console = _make_rich_console()
            console.print(_make_markdown(answer))
        else:
            sys.stdout.write(answer + "\n")
            sys.stdout.flush()

    return answer
