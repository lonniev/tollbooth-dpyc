"""Search the web, read a page, keep the part that matters: the steps before an LLM edits.

An operator that lets an LLM search for itself pays for every page the model reads, and
the model reads whole pages. Doing the searching here first — one search call, a few
fetched pages trimmed to the paragraphs that answer the question — hands the model a few
thousand tokens of sources instead of a few hundred thousand, and leaves it only the
editing.

The dpyc-oracle ranked wallets with its own Brave call; Good Earth looks up life cycles
with the same one. This module is that call, a polite page reader, and a trimmer.

**The key is the caller's.** ``BraveSearch`` is a value object holding the key passed in,
never one read from the environment, as ``llm_route.LlmRoute`` is. Two operators in one
process may search on two accounts.

Not this module's business: what to search for, which sites to trust, which words make a
paragraph relevant, and what to do when search fails. Those are the operator's.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urlsplit

import httpx

BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

#: How much of a page is read before it is cut off. A life-history account is well
#: inside this; a page that is not is mostly navigation and scripts.
DEFAULT_MAX_BYTES = 300_000


class WebSearchError(Exception):
    """The search did not answer. Carries the status alone, never the key or the body."""


@dataclass(frozen=True)
class Hit:
    """One web result."""

    title: str
    url: str
    description: str = ""
    snippets: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """Everything the result says, as one string."""
        return " ".join(p for p in (self.title, self.description, *self.snippets) if p)


def _clean(s: Any) -> str:
    """A result field as plain text: Brave marks matches with <strong>."""
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip() if isinstance(s, str) else ""


@dataclass(frozen=True)
class BraveSearch:
    """One Brave Search account. Hold one per client; the key is passed in."""

    api_key: str
    endpoint: str = BRAVE_URL
    timeout: float = 10.0

    async def search(self, query: str, *, count: int = 10, extra_snippets: bool = True,
                     client: httpx.AsyncClient | None = None) -> list[Hit]:
        """The web results for ``query``. Raises ``WebSearchError`` when it does not answer.

        The key goes only in the request header; the error names the status alone, so
        neither the key nor a response echoing it reaches a log.
        """
        params: dict[str, str | int] = {"q": query, "count": max(1, min(int(count), 20))}
        if extra_snippets:
            params["extra_snippets"] = "true"
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        try:
            if client is None:
                async with httpx.AsyncClient(timeout=self.timeout) as own:
                    resp = await own.get(self.endpoint, params=params, headers=headers)
            else:
                resp = await client.get(self.endpoint, params=params, headers=headers,
                                        timeout=self.timeout)
        except httpx.HTTPError as exc:
            raise WebSearchError(f"search unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise WebSearchError(f"search answered {resp.status_code}")
        try:
            results = resp.json().get("web", {}).get("results", [])
        except (ValueError, AttributeError):
            raise WebSearchError("search answered unreadable JSON") from None
        hits = []
        for r in results if isinstance(results, list) else []:
            if not isinstance(r, dict) or not isinstance(r.get("url"), str):
                continue
            snippets = r.get("extra_snippets")
            hits.append(Hit(
                title=_clean(r.get("title")),
                url=r["url"],
                description=_clean(r.get("description")),
                snippets=tuple(_clean(s) for s in snippets if isinstance(s, str))
                if isinstance(snippets, list) else (),
            ))
        return hits

    def __repr__(self) -> str:
        """Redact the key: a client is logged and caught in tracebacks."""
        return f"BraveSearch(endpoint={self.endpoint!r}, api_key='***')"

    __str__ = __repr__


def allowed(url: str, domains: tuple[str, ...] | list[str]) -> bool:
    """Whether ``url``'s host is one of ``domains`` or under one.

    A domain written with a leading dot (``.edu``) matches any host ending with it;
    otherwise the host must equal it or be a subdomain of it.
    """
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").lower().rstrip(".")
    if parts.scheme not in ("http", "https") or not host:
        return False
    for d in domains:
        d = d.lower()
        if d.startswith("."):
            if host.endswith(d):
                return True
        elif host == d or host.endswith("." + d):
            return True
    return False


# ── Reading a page ───────────────────────────────────────────────────────

#: Elements whose text is never the page's content.
_SKIP = frozenset({"script", "style", "noscript", "template", "svg", "nav", "header",
                   "footer", "aside", "form", "button", "select"})
#: Elements that end a line of text.
_BLOCK = frozenset({"p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "dd", "dt",
                    "tr", "div", "section", "article", "blockquote", "br", "caption",
                    "figcaption", "pre", "table", "ul", "ol", "dl"})
#: Void elements: no end tag, so they must never be counted as opened.
_VOID = frozenset({"br", "img", "hr", "meta", "link", "input", "wbr", "source", "area",
                   "base", "col", "embed", "param", "track"})


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self._line: list[str] = []
        self._skipping = 0

    def _end_line(self) -> None:
        line = " ".join(" ".join(self._line).split())
        if line:
            self.lines.append(line)
        self._line = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP and tag not in _VOID:
            self._skipping += 1
        elif tag in _BLOCK:
            self._end_line()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP:
            self._skipping = max(0, self._skipping - 1)
        elif tag in _BLOCK:
            self._end_line()

    def handle_data(self, data: str) -> None:
        if not self._skipping:
            self._line.append(data)

    def close(self) -> None:
        super().close()
        self._end_line()


def page_text(markup: str) -> str:
    """A page's readable text, one block per line; navigation, scripts and forms dropped."""
    parser = _Text()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # noqa: BLE001 — a malformed page yields what was read
        parser._end_line()
    return "\n".join(parser.lines)


async def fetch_text(url: str, *, user_agent: str, max_bytes: int = DEFAULT_MAX_BYTES,
                     timeout: float = 8.0, client: httpx.AsyncClient | None = None) -> str:
    """The readable text of the page at ``url``, or ``""`` when it cannot be read.

    Reads at most ``max_bytes``, so a huge page costs no more than a long one. A page that
    is not HTML or plain text, or does not answer, is ``""``: the caller has other pages.
    ``user_agent`` names the operator, as a polite reader does.
    """
    headers = {"User-Agent": user_agent, "Accept": "text/html,text/plain;q=0.9"}

    async def read(c: httpx.AsyncClient) -> str:
        async with c.stream("GET", url, headers=headers, timeout=timeout,
                            follow_redirects=True) as resp:
            if resp.status_code != 200:
                return ""
            kind = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if kind not in ("text/html", "application/xhtml+xml", "text/plain", ""):
                return ""
            body = bytearray()
            async for chunk in resp.aiter_bytes():
                body.extend(chunk)
                if len(body) >= max_bytes:
                    break
            decoded = bytes(body[:max_bytes]).decode(resp.encoding or "utf-8", errors="replace")
            return decoded.strip() if kind == "text/plain" else page_text(decoded)

    try:
        if client is None:
            async with httpx.AsyncClient() as own:
                return await read(own)
        return await read(client)
    except (httpx.HTTPError, LookupError):
        return ""


def passages(text: str, keywords: tuple[str, ...] | list[str], *, around: int = 1,
             max_chars: int = 2500) -> list[str]:
    """The lines of ``text`` that mention a keyword, with ``around`` neighbours each side.

    Neighbouring windows merge, so a paragraph is never quoted twice; the passages come
    in page order and stop at ``max_chars`` in all. Keywords match case-insensitively,
    as substrings, so ``"fledg"`` finds fledge, fledgling and fledging.
    """
    lines = text.splitlines()
    words = [k.lower() for k in keywords if k]
    if not lines or not words:
        return []
    hits = [i for i, line in enumerate(lines) if any(w in line.lower() for w in words)]
    spans: list[list[int]] = []
    for i in hits:
        lo, hi = max(0, i - around), min(len(lines) - 1, i + around)
        if spans and lo <= spans[-1][1] + 1:
            spans[-1][1] = max(spans[-1][1], hi)
        else:
            spans.append([lo, hi])
    out: list[str] = []
    used = 0
    for lo, hi in spans:
        chunk = "\n".join(lines[lo:hi + 1])
        if used + len(chunk) > max_chars:
            room = max_chars - used
            if room > 200:
                out.append(chunk[:room].rsplit(" ", 1)[0] + " …")
            break
        out.append(chunk)
        used += len(chunk)
    return out
