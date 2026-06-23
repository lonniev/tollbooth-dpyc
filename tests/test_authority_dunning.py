"""Tests for the low-cert-balance self-notice path.

When an actor's ``purchase_credits`` is refused because *its own* balance at
its certifying Authority is exhausted (the ``certify_credits`` fee debits the
purchasing actor's ledger, not the Authority's), the SDK:

1. Returns a kind ``authority_insufficient_balance`` situation to the patron
   instead of leaking the raw "Insufficient balance: 0 sats ..." string, and
2. Sends a single marker-tagged reminder DM to the actor's **own** npub (a
   self-notice the operator/sub-Authority admin sees in Pricing Studio — the
   party that must refill by calling its Authority's ``purchase_credits``),
   deduped against the relay's own event store (relay-as-cache) so repeated
   patron attempts in the ~10-minute NIP-40 window do not spam.

These tests cover the two units that implement that: the NIP-04 marker tag
+ relay query helper (``nostr_credentials``) and the self-notice dispatcher
(``runtime._dun_self_low_cert_balance``).
"""

from __future__ import annotations

import json

import pytest
from pynostr.key import PrivateKey

from tollbooth import runtime as rt_mod
from tollbooth.nostr_credentials import NostrCredentialExchange


def _make_exchange(**kwargs) -> NostrCredentialExchange:
    return NostrCredentialExchange(
        nsec=kwargs.pop("nsec", PrivateKey().nsec),
        relays=kwargs.pop("relays", ["wss://relay.test.com"]),
        templates={},
        **kwargs,
    )


# ── NIP-04 marker tag ────────────────────────────────────────────────


def test_nip04_dm_carries_extra_tags():
    """extra_tags are appended to the kind-4 event alongside p + expiration."""
    ex = _make_exchange()
    recipient_hex = PrivateKey().public_key.hex()

    msg = ex._build_nip04_dm(
        recipient_hex, "hello", extra_tags=[["t", "dpyc-dunning"]],
    )
    _, event = json.loads(msg)

    assert event["kind"] == 4
    assert ["t", "dpyc-dunning"] in event["tags"]
    assert ["p", recipient_hex] in event["tags"]
    # expiration tag is still present (NIP-40 self-expiry powers the dedup)
    assert any(t[0] == "expiration" for t in event["tags"])


def test_nip04_dm_without_extra_tags_unchanged():
    """Omitting extra_tags leaves only the canonical p + expiration tags."""
    ex = _make_exchange()
    recipient_hex = PrivateKey().public_key.hex()

    _, event = json.loads(ex._build_nip04_dm(recipient_hex, "hi"))

    tag_names = [t[0] for t in event["tags"]]
    assert tag_names == ["p", "expiration"]


# ── relay-as-cache query ─────────────────────────────────────────────


def test_has_recent_tagged_dm_disabled_returns_false():
    """A disabled exchange cannot query; it answers False (caller may send)."""
    ex = _make_exchange(nsec="definitely-not-a-valid-nsec")
    assert ex._enabled is False
    assert ex.has_recent_tagged_dm("npub1whatever", "dpyc-dunning") is False


def test_has_recent_tagged_dm_builds_author_scoped_filter(monkeypatch):
    """The dedup query filters by our authorship + recipient + marker tag."""
    ex = _make_exchange()
    recipient_npub = PrivateKey().public_key.bech32()
    captured: dict = {}

    def fake_query(relay_url, sub_id, filt):
        captured["filt"] = filt
        return True

    monkeypatch.setattr(ex, "_query_one_relay_has_event", fake_query)

    assert ex.has_recent_tagged_dm(
        recipient_npub, "dpyc-dunning", within_seconds=600,
    ) is True

    f = captured["filt"]
    assert f["kinds"] == [4]
    assert f["authors"] == [ex._pubkey_hex]
    assert f["#t"] == ["dpyc-dunning"]
    assert f["#p"]  # recipient hex present
    assert "since" in f and f["limit"] == 1


def test_has_recent_tagged_dm_false_when_no_relay_has_event(monkeypatch):
    ex = _make_exchange(relays=["wss://a.test", "wss://b.test"])
    recipient_npub = PrivateKey().public_key.bech32()
    monkeypatch.setattr(
        ex, "_query_one_relay_has_event", lambda *a, **k: False,
    )
    assert ex.has_recent_tagged_dm(recipient_npub, "dpyc-dunning") is False


# ── dunning dispatcher ───────────────────────────────────────────────


class _FakeExchange:
    def __init__(self, has_recent: bool):
        self._has_recent = has_recent
        self.recent_args = None
        self.sent: list = []

    def has_recent_tagged_dm(self, npub, tag, *, within_seconds=600):
        self.recent_args = (npub, tag, within_seconds)
        return self._has_recent

    def send_dm(self, npub, message, *, extra_tags=None):
        self.sent.append((npub, message, extra_tags))


class _FakeCourier:
    def __init__(self, exchange):
        self._exchange = exchange


@pytest.fixture
def inline_threads(monkeypatch):
    """Run the dunning daemon thread synchronously so assertions are stable."""

    class _InlineThread:
        def __init__(self, target=None, daemon=None):
            self._target = target

        def start(self):
            if self._target:
                self._target()

    monkeypatch.setattr(rt_mod.threading, "Thread", _InlineThread)


def test_dun_sends_marker_tagged_dm_to_self_when_not_queued(inline_threads):
    ex = _FakeExchange(has_recent=False)
    rt_mod._dun_self_low_cert_balance(
        _FakeCourier(ex), "npub1self", "NorthAmerica", "optionality",
    )

    assert len(ex.sent) == 1
    npub, message, extra_tags = ex.sent[0]
    # Self-notice: the DM targets the purchasing actor's own npub, never the
    # parent Authority's.
    assert npub == "npub1self"
    assert extra_tags == [["t", rt_mod._AUTHORITY_DUNNING_TAG]]
    assert "optionality" in message
    assert "NorthAmerica" in message
    # deduped against a ~10-minute window, keyed on our own npub
    assert ex.recent_args == ("npub1self", rt_mod._AUTHORITY_DUNNING_TAG, 600)


def test_dun_skips_send_when_reminder_already_queued(inline_threads):
    ex = _FakeExchange(has_recent=True)
    rt_mod._dun_self_low_cert_balance(
        _FakeCourier(ex), "npub1self", "NorthAmerica", "optionality",
    )
    assert ex.sent == []


def test_dun_noop_without_courier(inline_threads):
    # No courier / no self npub must never raise on the patron's critical path.
    rt_mod._dun_self_low_cert_balance(None, "npub1self", "NA", "optionality")
    rt_mod._dun_self_low_cert_balance(
        _FakeCourier(_FakeExchange(False)), "", "NA", "x",
    )
