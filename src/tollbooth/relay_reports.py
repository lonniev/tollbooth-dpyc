"""Tell the Oracle when a relay could not be reached.

The Oracle serves the fleet's relay order, and ``dpyc-community/relays.json``
is only a curated guess about which relays are worth using — it cannot know
which one is down right now. Operators are the ones who find out, one failed
rendezvous at a time. This module carries that discovery back.

Three rules shape it:

**Only failures are reported.** Success is far too frequent to be worth
carrying, and a relay that works needs no announcement.

**Reporting never delays the operator.** ``note_relay_failure`` is synchronous,
non-blocking bookkeeping — it appends to a small buffer and returns. The
network call happens later, from an async seam the operator was going to pass
through anyway (``OperatorRuntime.courier``). A patron waiting on a tool call
never waits on telemetry.

**A report is a hint, not a verdict.** The Oracle probes the relay itself
before changing anything, so a report that turns out to be wrong — a local
network fault, a firewall — costs one probe and moves nothing. That is what
lets this be fire-and-forget: nobody has to be careful about false positives.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Enough to name every relay in the set several times over. A bound matters
# because this buffer is fed from failure paths, and failure paths are exactly
# where a runaway loop would otherwise accumulate unboundedly.
_MAX_BUFFERED = 16

# Don't re-report the same relay more often than this. A dead relay fails on
# every attempt; the Oracle only needs to hear about it once per demotion
# window, and re-probing on each failure would be the survey we are avoiding.
_REPORT_COOLDOWN_SECONDS = 300

# url -> (mode, first_noted_monotonic)
_pending: dict[str, tuple[str, float]] = {}
# url -> monotonic time of the last report actually sent
_last_sent: dict[str, float] = {}


def note_relay_failure(relay_url: str, mode: str = "unknown") -> None:
    """Record that a relay failed. Synchronous, non-blocking, never raises.

    Args:
        relay_url: The relay that could not be reached.
        mode: ``"send"``, ``"read"``, or ``"unknown"`` — what was being tried.
    """
    if not relay_url:
        return
    try:
        now = time.monotonic()
        last = _last_sent.get(relay_url)
        if last is not None and (now - last) < _REPORT_COOLDOWN_SECONDS:
            return
        if relay_url not in _pending and len(_pending) >= _MAX_BUFFERED:
            return
        _pending.setdefault(relay_url, (mode, now))
    except Exception as exc:  # noqa: BLE001 — telemetry must never break a caller
        logger.debug("Could not buffer relay failure for %s: %s", relay_url, exc)


def pending_relay_failures() -> dict[str, str]:
    """The relays currently awaiting a report, as ``{url: mode}``."""
    return {url: mode for url, (mode, _) in _pending.items()}


def _signed_report(relay_url: str, nsec_hex: str) -> str | None:
    """Sign a report naming the relay, so the Oracle can attribute it."""
    try:
        from pynostr.event import Event
    except ImportError:
        return None
    try:
        event = Event(content=f"DPYC relay unreachable: {relay_url}")
        event.sign(nsec_hex)
        # to_dict(), not to_message(): the latter wraps the event in the
        # ["EVENT", {...}] relay envelope, and the Oracle parses a bare event.
        return json.dumps(event.to_dict())
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not sign relay report for %s: %s", relay_url, exc)
        return None


async def flush_relay_failures(npub: str, nsec_hex: str) -> list[dict[str, Any]]:
    """Send buffered reports to the Oracle. Best-effort; never raises.

    Returns the Oracle's responses, for callers that want to know whether the
    fleet-wide order changed. An empty list means there was nothing to say or
    the Oracle could not be reached — neither is a fault worth surfacing.
    """
    if not _pending or not npub or not nsec_hex:
        return []

    batch = dict(_pending)
    _pending.clear()

    from tollbooth.oracle_client import OracleClientError, default_oracle_client

    responses: list[dict[str, Any]] = []
    for relay_url, (mode, _) in batch.items():
        signed = _signed_report(relay_url, nsec_hex)
        if signed is None:
            continue
        try:
            response = await default_oracle_client().call_tool(
                "report_relay_failure",
                {
                    "relay": relay_url,
                    "reporter_npub": npub,
                    "signed_event": signed,
                    "mode": mode,
                },
            )
        except OracleClientError as exc:
            # The Oracle being unreachable is not the operator's problem to
            # solve, and the relay report is not worth a retry queue.
            logger.debug("Relay report for %s not delivered: %s", relay_url, exc)
            continue
        except Exception as exc:  # noqa: BLE001 — telemetry never breaks a caller
            logger.debug("Relay report for %s failed: %s", relay_url, exc)
            continue

        _last_sent[relay_url] = time.monotonic()
        responses.append(response)
        logger.info(
            "Reported %s unreachable (%s); Oracle probe says %s",
            relay_url, mode, response.get("probed", "?"),
        )

    # A changed fleet order is worth having now rather than after the cache
    # TTL, so drop the local copy and let the next read pick up the new one.
    if any(r.get("order_changed") for r in responses):
        try:
            from tollbooth.relay_registry import invalidate_relays_cache
            invalidate_relays_cache()
            logger.info("Fleet relay order changed; local relay cache invalidated")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not invalidate relay cache: %s", exc)

    return responses


def reset_relay_reports() -> None:
    """Clear all buffered and cooldown state. For tests."""
    _pending.clear()
    _last_sent.clear()
