"""URL fetch support for OpenKB.

Fetches content from URLs and converts to markdown for wiki compilation.
- X/Twitter URLs: uses ``bird`` CLI to fetch tweet JSON
- Other URLs: uses urllib + markitdown for HTML→markdown conversion
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import unicodedata
import urllib.error
import urllib.request
from io import BytesIO

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FetchError(Exception):
    """Raised when URL content cannot be fetched."""

    def __init__(self, url: str, detail: str):
        self.url = url
        self.detail = detail
        super().__init__(f"Failed to fetch {url}: {detail}")


# ---------------------------------------------------------------------------
# URL detection and classification
# ---------------------------------------------------------------------------

def is_url(s: str) -> bool:
    """Return True if *s* looks like an HTTP URL."""
    return s.startswith(("http://", "https://"))


def classify_url(url: str) -> str:
    """Classify a URL into ``'twitter'``, ``'youtube'``, or ``'article'``."""
    host = url.lower()
    if "x.com/" in host or "twitter.com/" in host:
        return "twitter"
    if "youtube.com/watch" in host or "youtu.be/" in host:
        return "youtube"
    return "article"


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    """NFKC-normalize and slugify *text* for safe use as a filename."""
    text = unicodedata.normalize("NFKC", text)
    slug = re.sub(r"[^\w\-]", "-", text.lower()).strip("-")
    return slug[:80] or "untitled"


def slug_from_url(url: str, title: str | None = None) -> str:
    """Generate a filesystem-safe slug from a URL or title.

    Strategy by URL type:
    - Twitter: ``username-tweetID``
    - YouTube: ``youtube-videoID``
    - Article: title if provided, else ``domain-path``
    """
    kind = classify_url(url)

    if kind == "twitter":
        # Extract tweet ID from URL
        m = re.search(r"/status/(\d+)", url)
        tweet_id = m.group(1) if m else "unknown"
        # Extract username
        m2 = re.search(r"(?:x\.com|twitter\.com)/([^/]+)/status", url)
        username = m2.group(1) if m2 else "unknown"
        return f"{username}-{tweet_id}"

    if kind == "youtube":
        # Extract video ID
        m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url)
        video_id = m.group(1) if m else "unknown"
        return f"youtube-{video_id}"

    # Article: use title if available, else derive from URL
    if title:
        return _slugify(title)

    # Fallback: domain + last path segment
    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.hostname or "unknown"
    path = (parsed.path or "/").rstrip("/").split("/")[-1]
    return _slugify(f"{domain}-{path}") if path else _slugify(domain)


# ---------------------------------------------------------------------------
# Tweet fetching (bird CLI)
# ---------------------------------------------------------------------------

def _find_bird() -> str:
    """Find the bird binary on PATH."""
    import shutil
    found = shutil.which("bird")
    return found or "bird"


def fetch_tweet(url: str) -> tuple[str, str]:
    """Fetch a tweet via ``bird read`` CLI and convert to markdown.

    Returns:
        ``(markdown_content, slug)``

    Raises:
        FetchError: if bird CLI fails or returns unparseable output
    """
    binary = _find_bird()
    args = [binary, url, "--json", "--plain"]

    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise FetchError(url, "bird CLI timed out after 60s")
    except FileNotFoundError:
        raise FetchError(url, f"bird CLI not found at '{binary}'. Install bird or ensure it is on PATH.")

    if proc.returncode != 0:
        stderr_preview = (proc.stderr or "")[:500]
        raise FetchError(url, f"bird CLI exit code {proc.returncode}: {stderr_preview}")

    # Parse JSON output
    try:
        tweet = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FetchError(url, f"Failed to parse bird JSON output: {exc}")

    # Extract fields
    text = tweet.get("text", "")
    author = tweet.get("author", {})
    username = author.get("username", "unknown")
    display_name = author.get("name", "")
    created_at = tweet.get("createdAt", "")
    tweet_id = tweet.get("id", "")
    media = tweet.get("media", [])

    # Build slug
    slug = f"{username}-{tweet_id}" if tweet_id else slug_from_url(url)

    # Build markdown
    # Parse date for frontmatter
    date_str = ""
    if created_at:
        # bird returns "Mon Apr 13 21:19:33 +0000 2026"
        m = re.search(r"(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{4})", created_at)
        if m:
            date_str = m.group(1)

    frontmatter = f"""---
source_url: "{url}"
source_type: twitter
author: "@{username}"
author_name: "{display_name}"
created_at: "{date_str}"
---"""

    # Tweet body
    body = f"@{username}\n\n{text}"

    # Media section
    media_lines = []
    for item in media:
        mtype = item.get("type", "photo")
        murl = item.get("url", "")
        if mtype == "photo" and murl:
            media_lines.append(f"- ![{mtype}]({murl})")
        elif mtype == "video":
            vurl = item.get("url", "")
            if vurl:
                media_lines.append(f"- [video]({vurl})")

    parts = [frontmatter, "", body]
    if media_lines:
        parts.append("")
        parts.append("## Media")
        parts.extend(media_lines)
    parts.append("")
    parts.append("---")
    parts.append(f"URL: {url}")

    return "\n".join(parts), slug


# ---------------------------------------------------------------------------
# Article fetching (urllib + markitdown)
# ---------------------------------------------------------------------------

def fetch_article(url: str) -> tuple[str, str]:
    """Fetch a web URL and convert HTML to markdown.

    Returns:
        ``(markdown_content, slug)``

    Raises:
        FetchError: on network errors or conversion failures
    """
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"},
    )

    # Use certifi CA bundle to avoid macOS Python SSL issues
    import ssl
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None

    try:
        kw = {"context": context} if context else {}
        with urllib.request.urlopen(req, timeout=30, **kw) as resp:
            html_bytes = resp.read()
    except urllib.error.URLError as exc:
        raise FetchError(url, f"Network error: {exc}")
    except Exception as exc:
        raise FetchError(url, f"HTTP error: {exc}")

    # Convert HTML to markdown via markitdown
    try:
        from markitdown import MarkItDown
        mid = MarkItDown()
        result = mid.convert_stream(BytesIO(html_bytes), extension=".html")
        markdown = result.text_content
    except Exception as exc:
        raise FetchError(url, f"markitdown conversion failed: {exc}")

    # Try to extract title from HTML
    title = ""
    html_text = html_bytes.decode("utf-8", errors="replace")
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()

    slug = slug_from_url(url, title=title if title else None)

    # Prepend frontmatter
    frontmatter = f"""---
source_url: "{url}"
source_type: article
title: "{title}"
---

"""
    return frontmatter + markdown, slug


# ---------------------------------------------------------------------------
# High-level dispatcher
# ---------------------------------------------------------------------------

def fetch_url(url: str) -> tuple[str, str]:
    """Fetch content from a URL and return ``(markdown, slug)``.

    Dispatches to :func:`fetch_tweet` for Twitter/X URLs and
    :func:`fetch_article` for everything else.

    Raises:
        FetchError: on any fetch failure
    """
    kind = classify_url(url)
    if kind == "twitter":
        return fetch_tweet(url)
    # youtube and article both use the same HTML fetch path for now
    return fetch_article(url)