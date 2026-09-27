"""Tests for send_bootstrap_config — strict NIP-20 OK parsing.

A relay rejection like ``["OK", id, false, "rate-limited"]`` must not
count as published: substring-matching on "ok" silently dropped relays
from the bootstrap config's coverage (observed against nos.lol on
2026-06-06, leaving a stale DM as that relay's only copy).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from pynostr.key import PrivateKey

from tollbooth.bootstrap_relay import PublishResult, send_bootstrap_config

AUTHORITY_NSEC = PrivateKey().bech32()
OPERATOR_NPUB = PrivateKey().public_key.bech32()
CONFIG = {"neon_database_url": "postgresql://x", "schema": "op_test"}


def _ws_replying(reply: str) -> MagicMock:
    ws = MagicMock()
    ws.recv.return_value = reply
    return ws


def _send(reply: str) -> PublishResult:
    with patch("websocket.create_connection", return_value=_ws_replying(reply)):
        return send_bootstrap_config(
            authority_nsec=AUTHORITY_NSEC,
            operator_npub=OPERATOR_NPUB,
            config=CONFIG,
            relays=["wss://relay.test"],
        )


class TestStrictOkParsing:
    def test_ok_true_counts_as_published(self) -> None:
        result = _send(json.dumps(["OK", "eventid", True, ""]))
        assert result == PublishResult(1, 0) and result

    def test_ok_false_rejection_does_not_count(self) -> None:
        """The bug: 'rate-limited' replies contain 'ok' as a substring."""
        result = _send(json.dumps(["OK", "eventid", False, "rate-limited"]))
        assert result == PublishResult(0, 1) and not result

    def test_notice_does_not_count(self) -> None:
        assert not _send(json.dumps(["NOTICE", "restricted: auth required"]))

    def test_garbage_reply_does_not_count(self) -> None:
        assert not _send("not json at all, but okay")

    def test_connection_failure_does_not_count(self) -> None:
        with patch(
            "websocket.create_connection", side_effect=OSError("refused")
        ):
            result = send_bootstrap_config(
                authority_nsec=AUTHORITY_NSEC,
                operator_npub=OPERATOR_NPUB,
                config=CONFIG,
                relays=["wss://relay.test"],
            )
            assert result == PublishResult(0, 1) and not result
