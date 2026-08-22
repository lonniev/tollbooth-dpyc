"""A publish counts only when the relay says OK to OUR event.

The courier pins the first relay that "accepted" the challenge as the
per-conversation rendezvous. If a publish is wrongly scored accepted, the
courier pins a relay that never stored the DM, tells the patron to reply
there, and then drains an empty mailbox — reporting that the patron never
replied. Every case below was previously scored as SUCCESS.
"""

from __future__ import annotations

import json

import pytest
from pynostr.key import PrivateKey

from tollbooth import relay_reports
from tollbooth.nostr_credentials import NostrCredentialExchange

EVENT_ID = "a" * 64
OTHER_ID = "b" * 64
MESSAGE = json.dumps(["EVENT", {"id": EVENT_ID, "kind": 1059, "content": "x"}])


class FakeWS:
    """A websocket that replays a scripted list of frames, then times out."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.sent = []
        self.timeouts = []

    def send(self, msg):
        self.sent.append(msg)

    def settimeout(self, t):
        self.timeouts.append(t)

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


def _exchange():
    return NostrCredentialExchange(
        nsec=PrivateKey().nsec, templates={}, relays=["wss://relay.test"],
    )


def _ack(ws):
    return _exchange()._await_ok_ack(ws, EVENT_ID, "wss://relay.test")


class TestAcceptance:
    def test_matching_ok_is_accepted(self):
        ok, _ = _ack(FakeWS([json.dumps(["OK", EVENT_ID, True, ""])]))
        assert ok is True

    def test_explicit_rejection_carries_the_relay_reason(self):
        ok, detail = _ack(FakeWS([json.dumps(["OK", EVENT_ID, False, "rate-limited"])]))
        assert ok is False
        assert "rate-limited" in detail

    def test_a_notice_before_the_ok_is_read_past(self):
        """Relays explain themselves in NOTICEs; that is not a verdict."""
        ok, _ = _ack(FakeWS([
            json.dumps(["NOTICE", "you are noisy"]),
            json.dumps(["OK", EVENT_ID, True, ""]),
        ]))
        assert ok is True

    def test_non_json_before_the_ok_is_read_past(self):
        ok, _ = _ack(FakeWS(["<html>bad gateway</html>",
                             json.dumps(["OK", EVENT_ID, True, ""])]))
        assert ok is True


class TestSilenceIsNotConsent:
    def test_no_response_at_all_is_a_failure(self):
        """Previously returned True with the detail 'no response'."""
        ok, detail = _ack(FakeWS([]))
        assert ok is False
        assert "no OK" in detail

    def test_a_lone_notice_is_a_failure(self):
        """Previously returned True — the NOTICE was scored as acceptance."""
        ok, detail = _ack(FakeWS([json.dumps(["NOTICE", "blocked: spam filter"])]))
        assert ok is False
        assert "no OK" in detail

    def test_a_lone_non_json_frame_is_a_failure(self):
        ok, detail = _ack(FakeWS(["<html>502</html>"]))
        assert ok is False
        assert "502" in detail

    def test_an_ok_for_somebody_elses_event_is_a_failure(self):
        """The relay stored someone else's event, not ours."""
        ok, detail = _ack(FakeWS([json.dumps(["OK", OTHER_ID, True, ""])]))
        assert ok is False
        assert "no OK" in detail

    def test_a_dropped_socket_is_a_failure(self):
        ok, _ = _ack(FakeWS([ConnectionResetError("peer closed")]))
        assert ok is False


class TestEventIdExtraction:
    def test_id_is_read_from_the_relay_message(self):
        assert NostrCredentialExchange._event_id_of(MESSAGE) == EVENT_ID

    def test_unparseable_message_yields_no_id(self):
        assert NostrCredentialExchange._event_id_of("not json") == ""

    def test_without_an_id_any_ok_is_accepted(self):
        """Degrade to the old behaviour rather than fail every publish."""
        ex = _exchange()
        ok, _ = ex._await_ok_ack(
            FakeWS([json.dumps(["OK", OTHER_ID, True, ""])]), "", "wss://relay.test",
        )
        assert ok is True


class TestReporting:
    def test_a_failed_pinned_publish_is_reported_to_the_oracle(self):
        ex = _exchange()
        import unittest.mock as m
        with m.patch(
            "tollbooth.nostr_credentials.create_connection",
            return_value=FakeWS([json.dumps(["NOTICE", "blocked"])]),
        ):
            ok, _ = ex._publish_to_one_relay(MESSAGE, "wss://relay.test")
        assert ok is False
        assert relay_reports.pending_relay_failures() == {"wss://relay.test": "send"}
