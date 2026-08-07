"""Where an operator's LLM calls go, and what a refusal from that provider means.

Three operators independently grew the same code: excalibur-mcp (``resolve.py``,
``refine.py``), optionality-mcp (``claude.py`` — whose docstring records that it was
"Modeled after ``taxsort-mcp/tools/advisors.py``"), and taxsort-mcp itself. Each pinned
``https://api.anthropic.com/v1/messages``, each declared its own copy of the web-search
tool, and two of them carry a byte-identical ``clamp_timeout``. They had already drifted:
taxsort sits a model generation behind the other two. This module is the one place that
decides, so a model change is a restart rather than three releases.

**The account is per model client, never per fleet.** The Factory and the operators happen
to draw on one OpenRouter key today; that is resource allocation, not design. So a route
is a value object a caller constructs and holds — with its key passed *in* — and never a
module-level singleton read from the environment. Two clients in one process can hold two
different provider accounts with no change here. The environment supplies defaults; it is
not the authority.

The default route is a model router (OpenRouter) rather than a single lab, so the model
behind a tier is a configuration choice instead of a migration. Its Anthropic-compatible
endpoint reads ``x-api-key`` — the header this module already emits — and passes
Anthropic's server-side ``web_search`` / ``web_fetch`` declarations through even to
non-Anthropic models, so a caller's tool plumbing survives a model change untouched.

ROLLBACK to a direct Anthropic account: set ``TOLLBOOTH_LLM_ENDPOINT`` to
``https://api.anthropic.com/v1/messages`` and both model variables to ``claude-sonnet-4-6``,
then redeliver an Anthropic key to the operator's vault. Nothing else changes.

Not this module's business: prompts, response parsing, retry policy. Those are the
caller's domain and stay in the operator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from tollbooth.async_situation import AsyncJobSituation
from tollbooth.constants import ErrorCode

# The model router the fleet routes through by default. Overridable per deployment
# (env) and per client (argument) — see the module docstring on why neither is global.
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/messages"

# Tier aliases, matching the vocabulary of the Factory's CI-side resolver
# (dpyc-community/factory/actions/resolve-model). Same names so the estate reads as one
# system; resolved independently, because neither side may depend on the other's config.
#
#   writer — composes text a human publishes or acts on. Gets the stronger tier.
#   reader — judges, summarises, suggests. Cheaper, and measured to be sufficient.
TIER_WRITER = "writer"
TIER_READER = "reader"

DEFAULT_MODEL_WRITER = "x-ai/grok-4.5"
DEFAULT_MODEL_READER = "x-ai/grok-4.3"

ENV_ENDPOINT = "TOLLBOOTH_LLM_ENDPOINT"
ENV_MODEL_WRITER = "TOLLBOOTH_LLM_MODEL_WRITER"
ENV_MODEL_READER = "TOLLBOOTH_LLM_MODEL_READER"

# The Anthropic messages API version every provider on this route understands.
ANTHROPIC_VERSION = "2023-06-01"

# Anthropic's server-side web tools, in their dynamic-filtering variants. The provider
# runs both and folds the findings into the answer, so one request still returns finished
# text. ``max_uses`` bounds the ROUNDS, which are the dominant latency cost. Callers set
# their own budget; these are the type/name declarations that were duplicated per repo.
WEB_SEARCH_TYPE = "web_search_20260209"
WEB_FETCH_TYPE = "web_fetch_20260209"

# Request-timeout FALLBACKS, not policy. A ceiling on how long one generation may take is
# a judgement only the caller can make — a one-line rewrite and a catalogue sweep with a
# dozen web fetches are not the same request — so callers pass their own via ``maximum``
# and these apply only when nobody did. What they defend against is narrow: a stalled
# provider riding a client library's multi-minute default.
#
# Read as policy, ``_TIMEOUT_MAX`` silently capped an operator whose own configured budget
# was higher, and the shortfall surfaced as an unexplained truncation rather than as a
# refusal. Anything that wants a real ceiling should own it.
_TIMEOUT_MIN = 30
_TIMEOUT_MAX = 900
_TIMEOUT_DEFAULT = 210.0


@dataclass(frozen=True)
class LlmRoute:
    """One model client's provider, model, and account.

    Hold one per client. Two clients in the same process may hold routes with different
    ``api_key`` values — that is the point of this being a value object rather than
    module state.
    """

    endpoint: str
    model: str
    api_key: str

    def headers(self) -> dict[str, str]:
        """The Anthropic-shaped auth headers. OpenRouter reads ``x-api-key`` too."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def __repr__(self) -> str:
        """Redact the key.

        A route is passed around, logged in job specs, and captured in tracebacks. The
        dataclass default would print the operator's provider key into every one of
        those, so this override is load-bearing, not cosmetic.
        """
        return f"LlmRoute(endpoint={self.endpoint!r}, model={self.model!r}, api_key='***')"

    __str__ = __repr__


def model_for(tier: str) -> str:
    """Resolve a tier alias to a provider-scoped model slug.

    A value that already looks like a slug (``vendor/model``) passes through untouched,
    so an operator can pin a specific model without waiting on an SDK release. Raises
    ``ValueError`` on an unknown alias rather than guessing — a silent fallback to some
    default model is a bill nobody chose.
    """
    if "/" in tier:
        return tier
    normalized = tier.strip().lower()
    if normalized == TIER_WRITER:
        return os.environ.get(ENV_MODEL_WRITER) or DEFAULT_MODEL_WRITER
    if normalized == TIER_READER:
        return os.environ.get(ENV_MODEL_READER) or DEFAULT_MODEL_READER
    raise ValueError(
        f"unknown LLM tier {tier!r} — use {TIER_WRITER!r}, {TIER_READER!r}, "
        "or an explicit provider slug like 'x-ai/grok-4.5'"
    )


def resolve_route(
    *,
    api_key: str,
    tier: str = TIER_READER,
    endpoint: str | None = None,
    model: str | None = None,
) -> LlmRoute:
    """Build the route for one model client.

    ``api_key`` is required and explicit: it names the provider ACCOUNT this client bills
    to, which the SDK never infers. Pass a different key to give a second client a second
    account. ``endpoint`` and ``model`` override the deployment's environment, which in
    turn overrides the fleet defaults.
    """
    if not api_key:
        raise ValueError("LLM route requires an api_key — the caller names the account")
    return LlmRoute(
        endpoint=endpoint or os.environ.get(ENV_ENDPOINT) or DEFAULT_ENDPOINT,
        model=model or model_for(tier),
        api_key=api_key,
    )


def web_search_tool(max_uses: int) -> dict[str, Any]:
    """The server-side web-search declaration, bounded to ``max_uses`` rounds."""
    return {"type": WEB_SEARCH_TYPE, "name": "web_search", "max_uses": max_uses}


def web_fetch_tool(max_uses: int, allowed_domains: list[str] | None = None) -> dict[str, Any]:
    """The server-side web-fetch declaration.

    With ``allowed_domains`` the provider may only retrieve from those hosts; without it,
    fetch is still gated to URLs already present in the conversation.
    """
    tool: dict[str, Any] = {"type": WEB_FETCH_TYPE, "name": "web_fetch", "max_uses": max_uses}
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains
    return tool


def clamp_timeout(seconds: float | None, *, maximum: float | None = None) -> float:
    """Bound an LLM request timeout to a sane range.

    ``maximum`` is the CALLER's ceiling, and it wins. An operator that has configured a
    thirty-minute budget for one block means it; clamping that to this module's fallback
    would enforce a policy no operator chose. Omit it and :data:`_TIMEOUT_MAX` applies as a
    stalled-provider guard.
    """
    if not seconds or seconds <= 0:
        return _TIMEOUT_DEFAULT
    ceiling = maximum if maximum and maximum > 0 else _TIMEOUT_MAX
    return float(max(_TIMEOUT_MIN, min(int(seconds), int(ceiling))))


def build_messages_request(
    route: LlmRoute,
    *,
    system: str,
    user: str,
    max_tokens: int,
    tools: list[dict[str, Any]] | None = None,
    timeout_seconds: float | None = None,
    timeout_max_seconds: float | None = None,
) -> dict[str, Any]:
    """A declarative, JSON-serializable request envelope for one generation.

    Returns ``{method, url, headers, json, timeout}``. Building the envelope separately
    from sending it keeps one description of a generation usable by whoever runs it —
    in this process or on detached compute.

    ``timeout_max_seconds`` is the caller's ceiling, forwarded to :func:`clamp_timeout`.
    Without it an operator's configured budget would be silently cut back to this
    module's stalled-provider fallback.
    """
    body: dict[str, Any] = {
        "model": route.model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if tools:
        body["tools"] = tools
    return {
        "method": "POST",
        "url": route.endpoint,
        "headers": route.headers(),
        "json": body,
        "timeout": clamp_timeout(timeout_seconds, maximum=timeout_max_seconds),
    }


def error_message(body: object) -> str:
    """Pull ``error.message`` from a provider response body, if present.

    Anthropic and OpenRouter share this shape, so one reader serves both.
    """
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or "")
    return ""


# What each provider says when the operator's account is out of money. Anthropic reports
# it as a 400 whose message mentions the credit balance; OpenRouter as a 402 reading
# "Insufficient credits". Matching only Anthropic's wording — as all three operators did
# before this module — turns an exhausted router account into a generic transient error,
# so the operator is never told to feed it and patrons are told to retry forever.
_UNFUNDED_NEEDLES = (
    "credit balance",       # Anthropic
    "purchase credits",     # Anthropic
    "plans & billing",      # Anthropic
    "billing",              # Anthropic, broad
    "insufficient credit",  # OpenRouter ("Insufficient credits")
    "more credits",         # OpenRouter (request exceeds remaining balance)
    "quota",                # generic metered-API exhaustion
)

_AUTH_NEEDLES = ("authentication", "invalid api key", "user not found", "no auth credentials")

# A model slug the provider does not recognise — the signature of a marketplace renaming
# or retiring a model under a running deployment. Permanent until someone edits the
# config, so it must never be reported as retryable.
_MODEL_NEEDLES = ("not a valid model", "unknown model", "no endpoints found")


def classify_llm_failure(*, status: int | None = None, message: str = "") -> str:
    """Name what went wrong upstream, or ``""`` when nothing definite matched.

    Accepts a status, a message, or both, so the same rules serve a caller holding an
    HTTP response and a caller holding only an exception string from a client library.
    Returns an :class:`~tollbooth.constants.ErrorCode` value.
    """
    low = (message or "").lower()

    if any(n in low for n in _MODEL_NEEDLES):
        return ErrorCode.LLM_MODEL_UNKNOWN
    if status in (400, 402) and any(n in low for n in _UNFUNDED_NEEDLES):
        return ErrorCode.LLM_PROVIDER_UNFUNDED
    # A bare 402 carries no useful message but means exactly one thing from a metered
    # LLM provider: the account is empty. (Distinct from tollbooth.upstream_payment,
    # which reads a bare 402 from a BUSINESS API as a lapsed human subscription — that
    # renews, this tops up.)
    if status == 402:
        return ErrorCode.LLM_PROVIDER_UNFUNDED
    if status in (401, 403) or any(n in low for n in _AUTH_NEEDLES):
        return ErrorCode.LLM_PROVIDER_AUTH
    if status == 429 or "rate limit" in low or "too many requests" in low:
        return ErrorCode.UPSTREAM_RATE_LIMITED
    # No status to go on — fall back to the wording a client library leaves in an
    # exception, which is all publisher-style audit paths ever have.
    if status is None and any(n in low for n in _UNFUNDED_NEEDLES):
        return ErrorCode.LLM_PROVIDER_UNFUNDED
    return ""


def llm_failure_situation(
    *,
    status: int | None = None,
    message: str = "",
    body: object = None,
    fallback_code: str = "llm_call_failed",
    fallback_message: str = "The AI provider couldn't complete this request. No fare was charged.",
) -> AsyncJobSituation:
    """Curate a provider refusal into a patron-facing situation.

    The raw status and body stay operator-side; the patron gets a stable ``error_code``,
    safe copy, and an honest ``transient`` flag. Anything the operator's account must fix
    — no funds, bad key, retired model — is marked non-transient, so schedulers stop
    hammering an endpoint that cannot recover on its own and the operator gets told.
    """
    text = message or error_message(body)
    code = classify_llm_failure(status=status, message=text)

    if code == ErrorCode.LLM_PROVIDER_UNFUNDED:
        return AsyncJobSituation(
            error_code=code,
            message="This service's AI provider is temporarily unavailable, so the "
                    "request couldn't be completed. No fare was charged.",
            next_steps="Please try again later.",
            transient=False,
        )
    if code == ErrorCode.LLM_PROVIDER_AUTH:
        return AsyncJobSituation(
            error_code=code,
            message="This service's AI access is misconfigured, so the request couldn't "
                    "be completed. No fare was charged.",
            next_steps="Please try again later.",
            transient=False,
        )
    if code == ErrorCode.LLM_MODEL_UNKNOWN:
        return AsyncJobSituation(
            error_code=code,
            message="This service is configured for an AI model its provider no longer "
                    "offers, so the request couldn't be completed. No fare was charged.",
            next_steps="Please try again later.",
            transient=False,
        )
    if code == ErrorCode.UPSTREAM_RATE_LIMITED:
        return AsyncJobSituation(
            error_code=code,
            message="The AI provider is busy right now, so the request couldn't be "
                    "completed. No fare was charged.",
            next_steps="Try again in a minute.",
            transient=True,
        )
    return AsyncJobSituation(
        error_code=fallback_code,
        message=fallback_message,
        next_steps="Please try again.",
        transient=True,
    )
