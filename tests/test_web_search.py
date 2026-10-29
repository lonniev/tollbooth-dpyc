"""Search, read, trim: key hygiene, the HTML reader, and the passage trimmer."""

from __future__ import annotations

import httpx
import pytest

from tollbooth.web_search import (
    BRAVE_URL,
    BraveSearch,
    Hit,
    WebSearchError,
    allowed,
    fetch_text,
    page_text,
    passages,
)

KEY = "brave-secret-key-123"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ── The search ───────────────────────────────────────────────────────────

def test_repr_redacts_the_key():
    s = BraveSearch(KEY)
    assert KEY not in repr(s) and KEY not in str(s)


async def test_search_sends_the_key_in_the_header_and_reads_hits():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"] = req.url, req.headers
        return httpx.Response(200, json={"web": {"results": [
            {"title": "Red-tailed <strong>Hawk</strong>", "url": "https://allaboutbirds.org/rth",
             "description": "Nests &amp; eggs", "extra_snippets": ["Incubation 28–35 days", 7]},
            {"title": "no url"},
            "junk",
        ]}})

    async with _client(handler) as c:
        hits = await BraveSearch(KEY).search("red-tailed hawk", count=50, client=c)

    assert seen["headers"]["X-Subscription-Token"] == KEY
    assert str(seen["url"]).startswith(BRAVE_URL)
    assert KEY not in str(seen["url"])
    assert seen["url"].params["count"] == "20"  # clamped to Brave's ceiling
    assert seen["url"].params["extra_snippets"] == "true"
    assert hits == [Hit("Red-tailed Hawk", "https://allaboutbirds.org/rth", "Nests & eggs",
                        ("Incubation 28–35 days",))]
    assert "Incubation" in hits[0].text


@pytest.mark.parametrize("response", [
    httpx.Response(429, text=f"rate limited for {KEY}"),
    httpx.Response(200, text="not json"),
])
async def test_a_failed_search_raises_without_the_key(response):
    async with _client(lambda req: response) as c:
        with pytest.raises(WebSearchError) as exc:
            await BraveSearch(KEY).search("q", client=c)
    assert KEY not in str(exc.value)


async def test_an_unreachable_search_raises():
    def handler(req):
        raise httpx.ConnectError(f"no route {KEY}")

    async with _client(handler) as c:
        with pytest.raises(WebSearchError) as exc:
            await BraveSearch(KEY).search("q", client=c)
    assert KEY not in str(exc.value)


# ── Which sites ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,ok", [
    ("https://extension.umn.edu/x", True),
    ("https://www.allaboutbirds.org/guide", True),
    ("https://allaboutbirds.org", True),
    ("https://notallaboutbirds.org", False),
    ("https://allaboutbirds.org.evil.com", False),
    ("https://example.com/.edu", False),
    ("ftp://extension.umn.edu", False),
    ("not a url", False),
])
def test_allowed(url, ok):
    assert allowed(url, (".edu", "allaboutbirds.org")) is ok


# ── Reading a page ───────────────────────────────────────────────────────

PAGE = """<html><head><title>T</title><style>p{color:red}</style>
<script>var nest = 1;</script></head><body>
<nav><a>Home</a> <a>Nesting menu</a></nav>
<header>Site header</header>
<h1>Red-tailed Hawk</h1>
<p>Clutch of 1&ndash;5 eggs.   Incubation
   lasts 28&ndash;35 days.</p>
<ul><li>Fledging at 42 days</li><li>Breeds March to May</li></ul>
<table><tr><td>Nest height</td><td>35 ft</td></tr></table>
<img src="x.png"><br>
<form><label>Search nests</label></form>
<footer>Copyright</footer></body></html>"""


def test_page_text_keeps_content_and_drops_chrome():
    text = page_text(PAGE)
    lines = text.splitlines()
    assert "Red-tailed Hawk" in lines
    assert "Clutch of 1–5 eggs. Incubation lasts 28–35 days." in lines
    assert "Fledging at 42 days" in lines and "Breeds March to May" in lines
    for gone in ("var nest", "color:red", "Nesting menu", "Site header", "Search nests",
                 "Copyright"):
        assert gone not in text


def test_page_text_survives_broken_markup():
    assert "kept" in page_text("<p>kept<div><span>unclosed")


async def test_fetch_text_reads_html_with_the_user_agent():
    seen = {}

    def handler(req):
        seen["ua"] = req.headers["User-Agent"]
        return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                              content=PAGE.encode())

    async with _client(handler) as c:
        text = await fetch_text("https://x.edu/p", user_agent="goodearth-mcp (+u)", client=c)
    assert seen["ua"] == "goodearth-mcp (+u)"
    assert "Fledging at 42 days" in text


async def test_fetch_text_stops_at_max_bytes():
    body = ("<p>" + "a" * 50 + "</p>") * 1000

    async with _client(lambda req: httpx.Response(
            200, headers={"content-type": "text/html"}, content=body.encode())) as c:
        text = await fetch_text("https://x.edu", user_agent="t", max_bytes=500, client=c)
    assert 0 < len(text) <= 500


@pytest.mark.parametrize("response", [
    httpx.Response(404, headers={"content-type": "text/html"}, content=b"<p>gone</p>"),
    httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"),
])
async def test_fetch_text_is_empty_for_what_it_cannot_read(response):
    async with _client(lambda req: response) as c:
        assert await fetch_text("https://x.edu", user_agent="t", client=c) == ""


async def test_fetch_text_is_empty_when_unreachable():
    def handler(req):
        raise httpx.ConnectTimeout("slow")

    async with _client(handler) as c:
        assert await fetch_text("https://x.edu", user_agent="t", client=c) == ""


# ── Trimming ─────────────────────────────────────────────────────────────

TEXT = "\n".join(f"line {i}" for i in range(20))


def test_passages_windows_merge_and_keep_page_order():
    text = TEXT.replace("line 3", "line 3 nest").replace("line 4", "line 4 NEST") \
               .replace("line 15", "line 15 fledging")
    assert passages(text, ["nest", "fledg"], around=1) == [
        "line 2\nline 3 nest\nline 4 NEST\nline 5",
        "line 14\nline 15 fledging\nline 16",
    ]


def test_passages_stop_at_max_chars():
    text = "\n".join(f"egg {'x' * 300} {i}" for i in range(20))
    out = passages(text, ["egg"], around=0, max_chars=1000)
    assert sum(len(p) for p in out) <= 1000 + 2
    assert out[-1].endswith("…")


def test_passages_need_a_keyword_and_text():
    assert passages(TEXT, []) == []
    assert passages("", ["nest"]) == []
    assert passages(TEXT, ["absent"]) == []
