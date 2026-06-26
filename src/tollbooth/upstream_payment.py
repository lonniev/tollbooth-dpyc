"""Generic upstream HTTP 402 (Payment Required) handling.

Some upstream APIs answer with a bare HTTP 402 to signal that the *paid
subscription or access tier* tied to the credentials this service uses has
lapsed or does not cover the request — e.g. an X (Twitter) developer plan
whose monthly billing failed, or a metered API past its quota. No payment this
server can make settles it: a human must renew or upgrade the plan at the
upstream provider.

This is distinct from the x402 *protocol* (Coinbase agentic micropayments),
which :class:`tollbooth.x402_client.X402Client` settles transparently as
Operator COGS. A machine-payable x402 challenge advertises its on-chain payment
terms in a ``payment-required`` header; a bare subscription 402 does not.
:func:`is_x402_payment_challenge` tells the two apart so callers route
machine-payable 402s to ``X402Client`` and human-actionable 402s to
:func:`upstream_payment_situation`.

Following the SDK's "situations, not failures" convention, a subscription 402
becomes a structured, human-facing situation (``error_code``
``upstream_subscription_required``) carrying clear renewal advice — not an
opaque error string, and explicitly marked ``transient: False`` so schedulers
and retry loops stop hammering an endpoint that cannot recover on its own.

This module has no heavy dependencies (it does not require the optional
``[x402]`` extra), so it is always importable.

Usage::

    from tollbooth.upstream_payment import classify_upstream_payment

    resp = await client.get(upstream_url)
    situation = classify_upstream_payment(
        resp, service="X (Twitter) API",
        renew_url="https://developer.x.com/en/portal/dashboard",
        audience="patron",
    )
    if situation is not None:
        return situation  # structured renewal advice for the calling app
"""

from __future__ import annotations

from typing import Any, Literal

from tollbooth.constants import ErrorCode

# Per the x402 HTTP transport spec, a machine-payable 402 advertises its
# payment terms in this header. Its ABSENCE on a 402 means the upstream is
# using the bare HTTP semantic ("renew your subscription"), not the protocol.
_X402_CHALLENGE_HEADER = "payment-required"

Audience = Literal["operator", "patron"]


def is_x402_payment_challenge(headers: Any) -> bool:
    """Return ``True`` if a 402's headers carry an x402 protocol challenge.

    Such a 402 is machine-payable via ``X402Client`` and is therefore NOT a
    human-subscription situation. The lookup is case-insensitive; ``headers``
    may be any mapping (``httpx.Headers``, ``dict``) or an iterable of
    ``(key, value)`` pairs.
    """
    if headers is None:
        return False
    target = _X402_CHALLENGE_HEADER.lower()
    try:
        items = headers.items()
    except AttributeError:
        try:
            items = list(headers)
        except TypeError:
            return False
    for pair in items:
        try:
            key = pair[0]
        except (TypeError, IndexError):
            continue
        if str(key).lower() == target:
            return True
    return False


def upstream_payment_situation(
    *,
    service: str,
    renew_url: str | None = None,
    audience: Audience = "operator",
    detail: str | None = None,
    status_code: int = 402,
) -> dict[str, Any]:
    """Build a structured, human-facing situation for a subscription 402.

    ``service`` names the upstream in human terms (e.g. ``"X (Twitter) API"``).
    ``renew_url`` is the provider's billing/portal page, woven into the advice
    when given. ``audience`` tunes whose subscription it is — ``"operator"``
    for the operator's own upstream account, ``"patron"`` for the patron's
    personally linked account. ``detail`` carries any upstream-supplied
    explanation verbatim.

    The returned dict follows the SDK situation shape (``success``,
    ``error_code``, ``error``, ``next_steps``) and adds ``transient: False`` so
    schedulers know not to re-fire, plus ``service`` (and ``renew_url`` /
    ``detail`` when present) for the calling app to render.
    """
    who = "your" if audience == "patron" else "the operator's"
    where = f" at {renew_url}" if renew_url else " in the provider's developer portal"
    message = (
        f"{service} returned 402 Payment Required: {who} subscription or "
        f"access tier for {service} has lapsed or does not cover this request. "
        f"A human needs to renew or upgrade the plan{where}. This is a billing "
        f"matter at {service} — not a re-authorization — and no DPYC credits "
        f"were charged."
    )
    next_steps = [
        f"Renew or upgrade the {service} subscription{where}",
        "Retry once the plan is active — no re-authorization needed",
    ]
    out: dict[str, Any] = {
        "success": False,
        "error_code": ErrorCode.UPSTREAM_SUBSCRIPTION_REQUIRED,
        "error": message,
        "next_steps": next_steps,
        "status_code": status_code,
        "service": service,
        "transient": False,
    }
    if renew_url:
        out["renew_url"] = renew_url
    if detail:
        out["detail"] = detail
    return out


def classify_upstream_payment(
    response: Any,
    *,
    service: str,
    renew_url: str | None = None,
    audience: Audience = "operator",
) -> dict[str, Any] | None:
    """Classify an upstream HTTP response for a subscription 402.

    Returns a structured situation (see :func:`upstream_payment_situation`)
    when ``response`` is a bare HTTP 402 carrying no x402 challenge header.
    Returns ``None`` for any other status, and for a 402 that IS a
    machine-payable x402 challenge — route those to ``X402Client`` instead.

    ``response`` is duck-typed: it needs a ``status_code`` int and a
    ``headers`` mapping, and optionally a ``json()`` method whose body may
    carry a short upstream explanation under ``detail`` / ``title`` /
    ``error``.
    """
    if getattr(response, "status_code", None) != 402:
        return None
    if is_x402_payment_challenge(getattr(response, "headers", None)):
        return None
    detail: str | None = None
    body_json = getattr(response, "json", None)
    if callable(body_json):
        try:
            body = body_json()
            if isinstance(body, dict):
                value = body.get("detail") or body.get("title") or body.get("error")
                detail = str(value) if value is not None else None
        except Exception:
            detail = None
    return upstream_payment_situation(
        service=service,
        renew_url=renew_url,
        audience=audience,
        detail=detail,
    )
