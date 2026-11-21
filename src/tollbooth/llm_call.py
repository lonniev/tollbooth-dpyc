"""One LLM generation, sent and read back safely.

:mod:`tollbooth.llm_route` decides where a generation goes and builds its envelope.
This module sends it and reads the answer, the part goodearth-mcp grew for its
reference lookups and Task Booth needed for drafting a job: three operators make
a third copy of the same code, so it lives here.

The answer is adversarial input like any tool argument. A caller asks for ONE JSON
object inside a named tag, reads it with :func:`tagged`, and checks every field with
the small readers below before anything downstream sees it: a field out of range,
mistyped or empty comes back absent, never invented.

A refusal is :class:`LlmAnswerError`, a ``ValueError``, so a paid tool that lets it
raise refunds the fare. Its message is patron-safe; the raw status and provider body
stay in the operator's log. The key is never logged.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

import httpx

from tollbooth.llm_route import (
    TIER_WRITER,
    build_messages_request,
    error_message,
    llm_failure_situation,
    model_for,
    resolve_route,
    web_search_tool,
)

logger = logging.getLogger(__name__)

#: Reasoning effort a caller may ask for; ``default`` sends nothing, so the model
#: reasons as it always has. Sent as the Messages API's ``output_config.effort``,
#: which OpenRouter maps for every model.
EFFORTS = ("default", "none", "minimal", "low", "medium", "high")


class LlmAnswerError(ValueError):
    """The model could not answer. A ``ValueError``, so a paid call refunds."""


async def ask(
    api_key: str,
    *,
    system: str,
    user: str,
    max_tokens: int,
    model: str = TIER_WRITER,
    effort: str = "default",
    searches: int = 0,
    cache_system: bool = False,
    timeout_s: float = 90.0,
    purpose: str = "LLM",
) -> dict[str, Any]:
    """One generation, and the raw JSON it answered with.

    ``model`` is a tier or a ``vendor/model`` slug: the operator's property for this
    use. ``searches`` above 0 lets the model search the web that many rounds.
    ``cache_system`` marks the system prompt for the provider's prompt cache, so a
    fixed prompt sent again is billed as a cache read. ``purpose`` names the use in
    the log lines, which carry token counts and never the prompt or the key.
    """
    route = resolve_route(api_key=api_key, model=model_for(model))
    req = build_messages_request(
        route, system=system, user=user, max_tokens=max_tokens,
        tools=[web_search_tool(searches)] if searches > 0 else None,
        timeout_seconds=timeout_s, timeout_max_seconds=timeout_s,
    )
    if effort in EFFORTS and effort != "default":
        req["json"]["output_config"] = {"effort": effort}
    if cache_system:
        req["json"]["system"] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ]
    async with httpx.AsyncClient(timeout=req["timeout"]) as client:
        resp = await client.post(req["url"], headers=req["headers"], json=req["json"])
    if resp.status_code >= 400:
        try:
            body: object = resp.json()
        except ValueError:
            body = resp.text
        situation = llm_failure_situation(status=resp.status_code, body=body)
        logger.warning("%s refused: HTTP %s %s (%s)", purpose, resp.status_code,
                       situation.error_code, error_message(body)[:200])
        raise LlmAnswerError(situation.message)
    data = resp.json()
    if not isinstance(data, dict):
        raise LlmAnswerError("The AI provider answered with something unreadable. No fare was charged.")
    logger.info("%s: %s", purpose, usage_of(data))
    return data


def usage_of(data: dict[str, Any]) -> dict[str, Any]:
    """What one generation spent: model, web searches, tokens, how it stopped.

    The provider counts prompt-cache reads and writes apart from the rest of the
    input; billed input is all three. 2026-10-29: a lookup reported 113 input tokens
    that OpenRouter's log showed as 3,825.
    """
    usage = _dict(data.get("usage"))
    tool_use = _dict(usage.get("server_tool_use"))
    searches = tool_use.get("web_search_requests")
    if searches is None:
        searches = sum(1 for b in data.get("content", []) if isinstance(b, dict)
                       and b.get("type") == "server_tool_use" and b.get("name") == "web_search")
    cached = int(usage.get("cache_read_input_tokens") or 0)
    written = int(usage.get("cache_creation_input_tokens") or 0)
    blocks = [b for b in data.get("content", []) if isinstance(b, dict)]
    details = _dict(usage.get("output_tokens_details"))
    return {
        "model": str(data.get("model") or ""),
        "searches": int(searches or 0),
        "input_tokens": int(usage.get("input_tokens") or 0) + cached + written,
        "cached_input_tokens": cached,
        "output_tokens": int(usage.get("output_tokens") or 0),
        "reasoning_tokens": details.get("reasoning_tokens"),
        "thinking_chars": sum(len(str(b.get("thinking") or "")) for b in blocks if b.get("type") == "thinking"),
        "answer_chars": sum(len(str(b.get("text") or "")) for b in blocks if b.get("type") == "text"),
        "stop_reason": str(data.get("stop_reason") or ""),
    }


def _dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


def tagged(data: dict[str, Any], tag: str) -> str:
    """The text inside the last ``<tag>…</tag>``, across every text block."""
    full = "".join(b.get("text", "") for b in data.get("content", []) if isinstance(b, dict)
                   and b.get("type") == "text")
    found = re.findall(rf"<{tag}>(.*?)</{tag}>", full, re.DOTALL)
    if not found:
        if data.get("stop_reason") == "max_tokens":
            raise LlmAnswerError("The AI ran out of room before it answered. No fare was charged.")
        raise LlmAnswerError("The AI answered without the part this asks for. No fare was charged.")
    return found[-1].strip()


# ── Readers for one field of an answer ────────────────────────────────────


def number(v: Any, lo: float, hi: float) -> float | None:
    """A finite number in range, or nothing. A bool is not a number."""
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return None
    return float(v) if lo <= v <= hi else None


def whole(v: Any, lo: int, hi: int) -> int | None:
    """A whole number in range, rounded, or nothing."""
    n = number(v, lo, hi)
    return round(n) if n is not None else None


def text(v: Any, limit: int = 300) -> str:
    """A string, whitespace collapsed and cut to ``limit``; anything else is ''."""
    if not isinstance(v, str):
        return ""
    return re.sub(r"\s+", " ", v).strip()[:limit]


def url(v: Any) -> str:
    """An http(s) URL with nothing a page could misuse, or ''."""
    u = text(v, 500)
    return u if re.match(r"^https?://[^\s<>\"']+$", u) else ""


def one_of(v: Any, allowed: tuple[str, ...]) -> str | None:
    """``v`` lowercased when it is one of ``allowed``, or nothing."""
    t = text(v, 40).lower()
    return t if t in allowed else None
