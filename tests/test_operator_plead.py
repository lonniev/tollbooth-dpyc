"""An operator's plea to its own human when an outside account runs dry.

``OperatorRuntime.plead`` sends a self-DM that stays on the relays for days and
repeats at most every twelve hours per cause; ``plea_resolved`` says once that
the cause cleared. Both ride ``send_dm``'s ``expires_in``, which every other DM
leaves at ten minutes.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from pynostr.key import PrivateKey

from tollbooth import runtime as rt_mod
from tollbooth.nostr_credentials import (
    DM_EXPIRES_IN,
    DM_EXPIRES_IN_MAX,
    CourierValidationError,
    NostrCredentialExchange,
)
from tollbooth.runtime import PLEA_LASTS_S, PLEA_REPEAT_S, OperatorRuntime

SELF = "npub1operatorself"


def _exchange() -> NostrCredentialExchange:
    return NostrCredentialExchange(nsec=PrivateKey().nsec, relays=["wss://relay.test.com"], templates={})


def _expiry(message: str) -> int:
    _, event = json.loads(message)
    return int(next(t[1] for t in event["tags"] if t[0] == "expiration"))


# ── expires_in ───────────────────────────────────────────────────────


def test_both_legs_keep_ten_minutes_unless_told_otherwise():
    ex = _exchange()
    to = PrivateKey().public_key.hex()
    now = int(time.time())
    assert abs(_expiry(ex._build_nip04_dm(to, "hi")) - (now + DM_EXPIRES_IN)) <= 2
    gift = ex._build_gift_wrap_with(ex._privkey_hex, ex._pubkey_hex, to, "hi")
    assert abs(_expiry(gift) - (now + DM_EXPIRES_IN)) <= 2


def test_both_legs_carry_the_expiry_they_are_given():
    ex = _exchange()
    to = PrivateKey().public_key.hex()
    now = int(time.time())
    week = 7 * 86400
    nip04 = ex._build_nip04_dm_with(ex._privkey_hex, ex._pubkey_hex, to, "hi", expires_in=week)
    gift = ex._build_gift_wrap_with(ex._privkey_hex, ex._pubkey_hex, to, "hi", expires_in=week)
    assert abs(_expiry(nip04) - (now + week)) <= 2
    assert abs(_expiry(gift) - (now + week)) <= 2


@pytest.mark.parametrize("bad", [0, 59, DM_EXPIRES_IN_MAX + 1, 600.5, True, "600"])
def test_send_dm_refuses_an_expiry_out_of_range(bad):
    with pytest.raises(CourierValidationError, match="expires_in"):
        _exchange().send_dm(SELF, "hi", expires_in=bad)


# ── plead / plea_resolved ────────────────────────────────────────────


class _FakeExchange:
    def __init__(self, fresh: bool = False) -> None:
        self.fresh = fresh
        self.sent: list[dict] = []
        self.asked: list[tuple] = []

    def has_recent_tagged_dm(self, npub, tag, *, within_seconds):
        self.asked.append((npub, tag, within_seconds))
        return self.fresh

    def send_dm(self, npub, text, **kw):
        self.sent.append({"npub": npub, "text": text, **kw})
        return True


class _Inline:
    """A Thread that runs its target at once, so a test sees what it sent."""

    def __init__(self, target, daemon=None):
        self._target = target

    def start(self):
        self._target()


def _runtime(exchange) -> SimpleNamespace:
    async def sender():
        return (exchange, SELF) if exchange is not None else None

    rt = SimpleNamespace(_open_pleas=set(), _self_dm_sender=sender)
    rt.plead = lambda cause, message: OperatorRuntime.plead(rt, cause, message)
    rt.plea_resolved = lambda cause, message: OperatorRuntime.plea_resolved(rt, cause, message)
    return rt


@pytest.fixture(autouse=True)
def _inline_threads(monkeypatch):
    monkeypatch.setattr(rt_mod.threading, "Thread", _Inline)


@pytest.mark.asyncio
async def test_a_plea_goes_to_the_operators_own_npub_tagged_and_kept_a_week():
    ex = _FakeExchange()
    rt = _runtime(ex)
    await rt.plead("x-credits", "Top up X.")
    assert ex.asked == [(SELF, "dpyc-plea-x-credits", PLEA_REPEAT_S)]
    assert ex.sent == [{"npub": SELF, "text": "Top up X.", "extra_tags": [["t", "dpyc-plea-x-credits"]],
                        "expires_in": PLEA_LASTS_S}]
    assert rt._open_pleas == {"x-credits"}


@pytest.mark.asyncio
async def test_a_plea_still_fresh_on_the_relays_is_not_repeated():
    ex = _FakeExchange(fresh=True)
    rt = _runtime(ex)
    await rt.plead("llm-credits", "Top up the LLM.")
    assert ex.sent == []
    # Still open here: this instance knows the outage, so it can say when it clears.
    assert rt._open_pleas == {"llm-credits"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["", "X-Credits", "x credits", "-x", "x" * 41, None])
async def test_a_cause_that_is_not_a_short_slug_is_refused(cause):
    ex = _FakeExchange()
    rt = _runtime(ex)
    await rt.plead(cause, "hi")
    assert ex.sent == [] and ex.asked == [] and rt._open_pleas == set()


@pytest.mark.asyncio
async def test_without_a_courier_a_plea_is_quietly_skipped():
    rt = _runtime(None)
    await rt.plead("x-credits", "hi")  # never raises


@pytest.mark.asyncio
async def test_the_message_is_capped():
    ex = _FakeExchange()
    await _runtime(ex).plead("x-credits", "a" * 10_000)
    assert len(ex.sent[0]["text"]) == 2000


@pytest.mark.asyncio
async def test_resolved_is_said_once_and_only_for_an_open_plea():
    ex = _FakeExchange()
    rt = _runtime(ex)
    await rt.plea_resolved("x-credits", "X is posting again.")
    assert ex.sent == []  # nothing was pleaded here: no relay query, no DM
    await rt.plead("x-credits", "Top up X.")
    await rt.plea_resolved("x-credits", "X is posting again.")
    await rt.plea_resolved("x-credits", "X is posting again.")
    assert [s["text"] for s in ex.sent] == ["Top up X.", "X is posting again."]
    assert ex.sent[1]["expires_in"] == PLEA_LASTS_S and "extra_tags" not in ex.sent[1]
    assert rt._open_pleas == set()
