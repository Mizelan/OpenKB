"""Tests for openkb.url_fetch fallback and canonicalization."""
from __future__ import annotations

import urllib.error

from openkb.url_fetch import canonicalize_url, fetch_article


class _DummyResponse:
    def __init__(self, body: str, url: str):
        self._body = body.encode("utf-8")
        self._url = url

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body

    def geturl(self):
        return self._url


class TestCanonicalizeUrl:
    def test_strips_tracking_params(self):
        url = (
            "https://tech.kakao.com/posts/696"
            "?fbclid=abc&utm_source=test&utm_medium=email"
        )
        assert canonicalize_url(url) == "https://tech.kakao.com/posts/696"

    def test_normalizes_youtube_short_url(self):
        url = "https://youtu.be/AuF7V7bqsrQ?si=N5mkf2XciZ9IuK3a"
        assert canonicalize_url(url) == "https://www.youtube.com/watch?v=AuF7V7bqsrQ"

    def test_drops_x_status_query(self):
        url = "https://x.com/user/status/12345?s=20&t=abc"
        assert canonicalize_url(url) == "https://x.com/user/status/12345"


class TestFetchArticleFallback:
    def test_falls_back_to_jina_on_403(self, monkeypatch):
        url = "https://openai.com/index/introducing-gpt-5/"
        calls = []

        def fake_open(target, *, timeout=30):
            calls.append(target)
            if target.startswith("https://r.jina.ai/"):
                return _DummyResponse(
                    "Title: Introducing GPT-5\n\nURL Source: http://openai.com/index/introducing-gpt-5/\n\nMarkdown Content:\n# Introducing GPT-5\n\nReal body.\n",
                    target,
                )
            raise urllib.error.HTTPError(target, 403, "Forbidden", hdrs=None, fp=None)

        monkeypatch.setattr("openkb.url_fetch._http_open", fake_open)

        markdown, slug = fetch_article(url)
        assert calls[0] == "https://openai.com/index/introducing-gpt-5"
        assert calls[1].startswith("https://r.jina.ai/http://openai.com/index/introducing-gpt-5")
        assert 'source_url: "https://openai.com/index/introducing-gpt-5"' in markdown
        assert "# Introducing GPT-5" in markdown
        assert slug == "introducing-gpt-5"

    def test_falls_back_to_jina_on_block_page(self, monkeypatch):
        url = "https://x.com/i/grok/share/abcdef"
        calls = []

        def fake_open(target, *, timeout=30):
            calls.append(target)
            return _DummyResponse(
                "Title: Grok Share\n\nURL Source: http://x.com/i/grok/share/abcdef\n\nMarkdown Content:\nUseful content.\n",
                target,
            )

        monkeypatch.setattr("openkb.url_fetch._http_open", fake_open)

        markdown, slug = fetch_article(url)
        assert calls[0].startswith("https://r.jina.ai/http://x.com/i/grok/share/abcdef")
        assert "Useful content." in markdown
        assert slug == "grok-share"
