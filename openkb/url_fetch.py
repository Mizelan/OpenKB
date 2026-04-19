"""URL fetch support for OpenKB.

Fetches content from URLs and converts to markdown for wiki compilation.
- X/Twitter URLs: uses ``bird`` CLI to fetch tweet JSON
- Other URLs: uses urllib + markitdown for HTML→markdown conversion
"""

from __future__ import annotations

import json
import logging
import re
import ssl
import subprocess
import unicodedata
import urllib.error
import urllib.request
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import parse_qsl, quote, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0 Safari/537.36"
)
_DROP_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "aem",
    "si",
    "s",
    "t",
    "tab",
}
_JINA_PREFIX = "https://r.jina.ai/http://"


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
    host = canonicalize_url(url).lower()
    if "/i/grok/share/" in host:
        return "article"
    if "x.com/" in host or "twitter.com/" in host:
        return "twitter"
    if "youtube.com/watch" in host or "youtu.be/" in host:
        return "youtube"
    return "article"


def canonicalize_url(url: str) -> str:
    """Return a stable fetch URL with tracking query params removed."""
    raw = url.strip()
    parsed = urlparse(raw)
    if not parsed.scheme:
        parsed = urlparse(f"https://{raw}")

    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"

    if netloc == "youtu.be":
        video_id = path.strip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"

    if netloc in {"twitter.com", "www.twitter.com"}:
        netloc = "x.com"

    if netloc.startswith("www.") and netloc not in {"www.youtube.com"}:
        netloc = netloc[4:]

    # X status/share URLs are stable without query params.
    if netloc == "x.com" and ("/status/" in path or "/i/grok/share/" in path):
        return urlunparse((scheme, netloc, path.rstrip("/") or path, "", "", ""))

    filtered_query = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        lk = key.lower()
        if lk.startswith("utm_") or lk in _DROP_QUERY_KEYS:
            continue
        filtered_query.append((key, value))

    query = urlencode(filtered_query, doseq=True)
    return urlunparse((scheme, netloc, path.rstrip("/") or path, "", query, ""))


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
    url = canonicalize_url(url)
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


def _ssl_context() -> ssl.SSLContext | None:
    """Return an SSL context backed by certifi when available."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _http_open(url: str, *, timeout: int = 30):
    """Open a URL with a browser-like user agent and certifi-backed TLS."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    context = _ssl_context()
    kw = {"context": context} if context else {}
    return urllib.request.urlopen(req, timeout=timeout, **kw)


def _looks_like_block_page(text: str) -> bool:
    """Detect challenge/placeholder pages that should use fallback fetching."""
    lowered = text.lower()
    markers = [
        "enable javascript and cookies to continue",
        "cf_chl_opt",
        "cf-mitigated",
        "checking your browser",
        "just a moment...",
        "javascript is not available",
        "something went wrong, but don’t fret",
    ]
    return any(marker in lowered for marker in markers)


def _split_jina_markdown(text: str) -> tuple[str, str]:
    """Extract title + markdown body from a Jina Reader response."""
    title = ""
    title_match = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()

    if "Markdown Content:" in text:
        markdown = text.split("Markdown Content:", 1)[1].lstrip()
    else:
        markdown = text.lstrip()
    return title, markdown


def _fetch_article_via_jina(url: str) -> tuple[str, str]:
    """Fetch article-style content via Jina Reader fallback."""
    canonical_url = canonicalize_url(url)
    proxy_url = _JINA_PREFIX + canonical_url.removeprefix("https://").removeprefix("http://")
    try:
        with _http_open(proxy_url, timeout=45) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise FetchError(canonical_url, f"Jina fallback failed: {exc}") from exc

    title, markdown = _split_jina_markdown(text)
    if not markdown.strip():
        raise FetchError(canonical_url, "Jina fallback returned empty markdown")

    slug = slug_from_url(canonical_url, title=title if title else None)
    frontmatter = f"""---
source_url: "{canonical_url}"
source_type: article
title: "{title}"
---

"""
    return frontmatter + markdown, slug


class _TweetOEmbedParser(HTMLParser):
    """Extract the main tweet text and published date from oEmbed HTML."""

    def __init__(self) -> None:
        super().__init__()
        self._in_p = False
        self._anchor_depth = 0
        self._current_anchor: list[str] = []
        self._text_parts: list[str] = []
        self._date_text = ""

    @property
    def text(self) -> str:
        text = "".join(self._text_parts)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @property
    def created_at(self) -> str:
        return self._date_text.strip()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "p":
            self._in_p = True
        elif self._in_p and tag == "br":
            self._text_parts.append("\n")
        elif tag == "a":
            self._anchor_depth += 1
            self._current_anchor = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "p":
            self._in_p = False
        elif tag == "a":
            anchor_text = "".join(self._current_anchor).strip()
            if self._in_p and anchor_text:
                self._text_parts.append(anchor_text)
            elif not self._in_p and anchor_text:
                self._date_text = anchor_text
            self._anchor_depth = max(0, self._anchor_depth - 1)
            self._current_anchor = []

    def handle_data(self, data: str) -> None:
        if self._anchor_depth > 0:
            self._current_anchor.append(data)
        elif self._in_p:
            self._text_parts.append(data)


def _fetch_tweet_oembed(url: str) -> tuple[str, str]:
    """Fetch tweet content from publish.twitter.com oEmbed."""
    url = canonicalize_url(url)
    tweet_id_match = re.search(r"/status/(\d+)", url)
    tweet_id = tweet_id_match.group(1) if tweet_id_match else ""
    oembed_url = (
        "https://publish.twitter.com/oembed?omit_script=1&url="
        + quote(url, safe="")
    )
    req = urllib.request.Request(
        oembed_url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    context = _ssl_context()
    kw = {"context": context} if context else {}
    with urllib.request.urlopen(req, timeout=30, **kw) as resp:
        data = json.load(resp)

    html = str(data.get("html", "")).strip()
    if not html:
        raise FetchError(url, "oEmbed returned empty HTML")

    parser = _TweetOEmbedParser()
    parser.feed(html)
    body_text = parser.text

    author_name = str(data.get("author_name", "")).strip()
    author_url = str(data.get("author_url", "")).strip()
    username = (urlparse(author_url).path or "").strip("/").split("/")[-1] if author_url else ""
    if not username:
        username_match = re.search(r"(?:x\.com|twitter\.com)/([^/]+)/status", url)
        username = username_match.group(1) if username_match else "unknown"

    slug = f"{username}-{tweet_id}" if tweet_id else slug_from_url(url)
    created_at = parser.created_at

    frontmatter = f"""---
source_url: "{url}"
source_type: twitter
author: "@{username}"
author_name: "{author_name}"
created_at: "{created_at}"
---"""

    body = f"@{username}\n\n{body_text or url}"
    parts = [frontmatter, "", body, "", "---", f"URL: {url}"]
    return "\n".join(parts), slug


def fetch_tweet(url: str) -> tuple[str, str]:
    """Fetch a tweet via ``bird read`` CLI and convert to markdown.

    Returns:
        ``(markdown_content, slug)``

    Raises:
        FetchError: if bird CLI fails or returns unparseable output
    """
    url = canonicalize_url(url)
    try:
        return _fetch_tweet_oembed(url)
    except Exception as exc:
        logger.debug("oEmbed fallback failed for %s: %s", url, exc)

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
    canonical_url = canonicalize_url(url)
    if "/i/grok/share/" in canonical_url:
        return _fetch_article_via_jina(canonical_url)

    try:
        with _http_open(canonical_url, timeout=30) as resp:
            html_bytes = resp.read()
            final_url = canonicalize_url(resp.geturl())
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 408, 409, 429} or 500 <= exc.code < 600:
            return _fetch_article_via_jina(canonical_url)
        raise FetchError(canonical_url, f"Network error: HTTP Error {exc.code}: {exc.reason}")
    except urllib.error.URLError as exc:
        return _fetch_article_via_jina(canonical_url)
    except Exception as exc:
        return _fetch_article_via_jina(canonical_url)

    html_text = html_bytes.decode("utf-8", errors="replace")
    if _looks_like_block_page(html_text):
        return _fetch_article_via_jina(canonical_url)

    # Convert HTML to markdown via markitdown
    try:
        from markitdown import MarkItDown
        mid = MarkItDown()
        result = mid.convert_stream(BytesIO(html_bytes), extension=".html")
        markdown = result.text_content
    except Exception as exc:
        return _fetch_article_via_jina(canonical_url)

    # Try to extract title from HTML
    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if m:
        title = m.group(1).strip()

    slug = slug_from_url(final_url, title=title if title else None)

    # Prepend frontmatter
    frontmatter = f"""---
source_url: "{final_url}"
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
    url = canonicalize_url(url)
    kind = classify_url(url)
    if kind == "twitter":
        return fetch_tweet(url)
    # youtube and article both use the same HTML fetch path for now
    return fetch_article(url)
