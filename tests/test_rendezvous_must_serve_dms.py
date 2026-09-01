"""A rendezvous relay must hand the DM back, not merely accept it.

``wss://nos.lol`` answers every kind-4/1059 filter with ``CLOSED
auth-required`` and then rejects every NIP-42 AUTH with "relay needs
serviceUrl to be configured" — so nobody, the recipient's own client
included, can ever read a DM out of it. It still returns ``OK ... true`` on
the write. The courier used to pin the first relay that accepted the publish,
which made that black hole the rendezvous: ``welcome_dm_sent: true``, an
honest publish ack, and a message no human could ever receive.

Every case below was previously scored as a usable rendezvous.
"""

from __future__ import annotations

import json

import pytest
from pynostr.key import PrivateKey

from tollbooth import relay_reports
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_credentials import NostrCredentialExchange

GOOD = "wss://serves.test"
BLIND = "wss://blackhole.test"


class FakeWS:
    """A websocket that replays scripted frames, then times out."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent: list[str] = []

    def send(self, msg):
        self.sent.append(msg)

    def settimeout(self, _t):
        pass

    def recv(self):
        if not self._frames:
            raise TimeoutError("timed out")
        frame = self._frames.pop(0)
        if isinstance(frame, Exception):
            raise frame
        return frame

    def close(self):
        pass


@pytest.fixture(autouse=True)
def _reset():
    relay_reports.reset_relay_reports()
    yield
    relay_reports.reset_relay_reports()


TEMPLATES = {
    "svc": CredentialTemplate(
        service="svc", version=1, fields={"api_key": FieldSpec()},
    ),
}


def _exchange(relays):
    return NostrCredentialExchange(
        nsec=PrivateKey().nsec, templates=TEMPLATES, relays=relays,
    )


def _probe(frames, relay=GOOD, monkeypatch=None):
    exchange = _exchange([relay])
    monkeypatch.setattr(
        "tollbooth.nostr_credentials.create_connection",
        lambda *a, **k: FakeWS(frames),
    )
    return exchange._relay_serves_dm_reads(relay)


class TestVerdict:
    def test_eose_proves_the_relay_will_answer(self, monkeypatch):
        serves, _ = _probe([json.dumps(["EOSE", "rvz"])], monkeypatch=monkeypatch)
        assert serves is True

    def test_a_delivered_event_proves_it_too(self, monkeypatch):
        serves, _ = _probe(
            [json.dumps(["EVENT", "rvz", {"kind": 4, "id": "a" * 64}])],
            monkeypatch=monkeypatch,
        )
        assert serves is True

    def test_auth_required_closed_is_a_refusal(self, monkeypatch):
        """The nos.lol shape: the write lands, the read never will."""
        serves, detail = _probe(
            [json.dumps(
                ["CLOSED", "rvz",
                 "ERROR: auth-required: requested filter requires authentication"],
            )],
            monkeypatch=monkeypatch,
        )
        assert serves is False
        assert "auth-required" in detail

    def test_a_notice_before_the_eose_is_read_past(self, monkeypatch):
        """A NOTICE is the relay explaining itself, not answering."""
        serves, _ = _probe(
            [json.dumps(["NOTICE", "you are noisy"]),
             json.dumps(["EOSE", "rvz"])],
            monkeypatch=monkeypatch,
        )
        assert serves is True

    def test_a_lone_notice_is_still_a_refusal(self, monkeypatch):
        """Read past, but never a substitute for the EOSE that never came."""
        serves, detail = _probe(
            [json.dumps(["NOTICE", "restricted: not authorized"])],
            monkeypatch=monkeypatch,
        )
        assert serves is False
        assert "timed out" in detail

    def test_silence_is_a_refusal(self, monkeypatch):
        """No EOSE, no CLOSED, nothing — silence is not consent here either."""
        serves, detail = _probe([], monkeypatch=monkeypatch)
        assert serves is False
        assert "timed out" in detail

    def test_an_unreachable_relay_is_a_refusal(self, monkeypatch):
        exchange = _exchange([GOOD])

        def _boom(*_a, **_k):
            raise ConnectionRefusedError("nope")

        monkeypatch.setattr(
            "tollbooth.nostr_credentials.create_connection", _boom,
        )
        serves, detail = exchange._relay_serves_dm_reads(GOOD)
        assert serves is False
        assert "connect failed" in detail

    def test_the_verdict_is_cached_not_re_probed(self, monkeypatch):
        exchange = _exchange([GOOD])
        calls = []

        def _connect(*_a, **_k):
            calls.append(1)
            return FakeWS([json.dumps(["EOSE", "rvz"])])

        monkeypatch.setattr(
            "tollbooth.nostr_credentials.create_connection", _connect,
        )
        assert exchange._relay_serves_dm_reads(GOOD)[0] is True
        assert exchange._relay_serves_dm_reads(GOOD)[0] is True
        assert len(calls) == 1

    def test_the_probe_asks_for_both_dm_kinds(self, monkeypatch):
        """The drain subscribes to kind 4 and 1059; the probe must match it."""
        ws = FakeWS([json.dumps(["EOSE", "rvz"])])
        exchange = _exchange([GOOD])
        monkeypatch.setattr(
            "tollbooth.nostr_credentials.create_connection",
            lambda *a, **k: ws,
        )
        exchange._relay_serves_dm_reads(GOOD)
        req = json.loads(ws.sent[0])
        assert req[0] == "REQ"
        assert set(req[2]["kinds"]) == {4, 1059}


class TestRendezvousSelection:
    async def test_a_read_blind_relay_is_never_pinned(self, monkeypatch):
        """It accepts the write; it must still lose the rendezvous."""
        exchange = _exchange([BLIND, GOOD])
        monkeypatch.setattr(
            NostrCredentialExchange, "_relay_serves_dm_reads",
            lambda _self, url: (url == GOOD, "" if url == GOOD else "auth-required"),
        )
        published: list[str] = []
        monkeypatch.setattr(
            NostrCredentialExchange, "_publish_to_one_relay",
            lambda _self, _msg, url: (published.append(url), (True, ""))[1],
        )

        result = await exchange.open_channel(
            "svc", greeting="hi", recipient_npub=PrivateKey().public_key.bech32(),
        )

        assert result["rendezvous_relay"] == GOOD
        assert BLIND not in published

    async def test_all_relays_read_blind_is_unreachable_not_success(
        self, monkeypatch,
    ):
        """The old code reported welcome_dm_sent: true into a black hole."""
        from tollbooth.nostr_credentials import CourierUnreachableError

        exchange = _exchange([BLIND])
        monkeypatch.setattr(
            NostrCredentialExchange, "_relay_serves_dm_reads",
            lambda _self, _url: (False, "auth-required"),
        )
        monkeypatch.setattr(
            NostrCredentialExchange, "_publish_to_one_relay",
            lambda _self, _msg, _url: (True, ""),
        )

        with pytest.raises(CourierUnreachableError) as excinfo:
            await exchange.open_channel(
                "svc", greeting="hi",
                recipient_npub=PrivateKey().public_key.bech32(),
            )
        assert "will not serve DM reads" in str(excinfo.value)

    async def test_a_read_blind_relay_is_reported_to_the_oracle(self, monkeypatch):
        exchange = _exchange([BLIND, GOOD])
        monkeypatch.setattr(
            NostrCredentialExchange, "_relay_serves_dm_reads",
            lambda _self, url: (url == GOOD, "" if url == GOOD else "auth-required"),
        )
        monkeypatch.setattr(
            NostrCredentialExchange, "_publish_to_one_relay",
            lambda _self, _msg, _url: (True, ""),
        )

        await exchange.open_channel(
            "svc", greeting="hi", recipient_npub=PrivateKey().public_key.bech32(),
        )

        assert BLIND in relay_reports.pending_relay_failures()
