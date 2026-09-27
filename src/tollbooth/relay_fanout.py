"""One concurrent fan-out for every "ask each relay" read in the wheel.

A relay set is a list of independent peers, so asking them in series makes the
caller pay the SUM of their latencies, and the sum is dominated by the relays
that are down. Measured on 2026-09-26: eleven relays polled one after another
cost 16.6 s, of which 10 s was a single relay timing out, while every healthy
relay had answered inside 0.6 s. Asked together, the same set costs about the
slowest relay the caller still chooses to wait for.

``fan_out`` asks every relay at once and returns one :class:`Outcome` per relay
**in the order given** (registry order, primary first), never arrival order, so
callers that rank relays keep their ranking. It stops waiting when:

- every relay has answered, or
- ``deadline`` seconds have passed, or
- ``accept(value)`` was true for some answer and ``settle`` more seconds have
  passed since that first acceptable answer. The settle window is how a
  newest-wins reader still hears a slightly slower relay that holds a newer
  revision, without waiting on the relays that will never answer.

A relay that has not answered when waiting stops is ``abandoned``: slow, not
unreachable. Its thread runs on until its own socket timeout, so pass the same
budget to the per-relay function's timeouts. On a normal interpreter exit the
executor's threads are joined, so a CLI run can linger for up to that budget;
a long-lived server is ended by signal and never notices.

The primitive is synchronous by design (the wheel's relay I/O is synchronous
websocket-client); an async caller hops once with ``asyncio.to_thread``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one relay did: ``ok`` with a value, ``error`` with a message, or
    ``abandoned`` because the caller stopped waiting."""

    relay: str
    value: Any
    error: str | None
    elapsed: float
    state: str  # "ok" | "error" | "abandoned"

    @property
    def ok(self) -> bool:
        return self.state == "ok"


def fan_out(
    relays: Sequence[str],
    fn: Callable[[str], Any],
    *,
    deadline: float,
    settle: float | None = None,
    accept: Callable[[Any], bool] | None = None,
) -> list[Outcome]:
    """Run ``fn(relay)`` for every relay concurrently; outcomes in input order.

    ``deadline`` bounds the whole wait. ``accept`` names an answer worth
    stopping for; once one arrives the wait continues only ``settle`` more
    seconds (``None`` means stop at once). Without ``accept`` the wait runs to
    the deadline or until every relay has answered.
    """
    if not relays:
        return []

    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=len(relays), thread_name_prefix="relay-fanout")
    futures = [pool.submit(_timed, fn, relay) for relay in relays]
    try:
        pending = set(futures)
        settle_at: float | None = None
        while pending:
            now = time.monotonic()
            budget = deadline - (now - started)
            if settle_at is not None:
                budget = min(budget, settle_at - now)
            if budget <= 0:
                break
            done, pending = wait(pending, timeout=budget, return_when=FIRST_COMPLETED)
            if accept is not None and settle_at is None and any(_accepted(f, accept) for f in done):
                settle_at = time.monotonic() + (settle or 0.0)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    waited = time.monotonic() - started
    return [_outcome(relay, f, waited) for relay, f in zip(relays, futures, strict=True)]


def _timed(fn: Callable[[str], Any], relay: str) -> tuple[Any, str | None, float]:
    """Run ``fn`` on the worker thread; never lets an exception escape the future."""
    t0 = time.monotonic()
    try:
        return fn(relay), None, time.monotonic() - t0
    except Exception as exc:  # noqa: BLE001 — a relay's failure is data, not a fault
        return None, str(exc) or type(exc).__name__, time.monotonic() - t0


def _accepted(future: Future, accept: Callable[[Any], bool]) -> bool:
    value, error, _ = future.result()
    if error is not None:
        return False
    try:
        return bool(accept(value))
    except Exception:  # noqa: BLE001 — a predicate that raises is a "no"
        return False


def _outcome(relay: str, future: Future, waited: float) -> Outcome:
    if future.cancelled() or not future.done():
        logger.debug("Relay %s had not answered after %.1fs; abandoned", relay, waited)
        return Outcome(relay, None, f"no answer within {waited:.1f}s", waited, "abandoned")
    value, error, elapsed = future.result()
    if error is not None:
        return Outcome(relay, None, error, elapsed, "error")
    return Outcome(relay, value, None, elapsed, "ok")
