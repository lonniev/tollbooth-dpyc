"""send_bootstrap_config must publish via fan_out, not a serial 10 s walk.

A serial walk of 11 relays at 10 s each can take nearly two minutes; the weekly
refresh fires on the back of a tool response and must finish inside one request.
``relay_fanout.fan_out`` is the wheel's concurrent primitive — reuse it.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from pynostr.key import PrivateKey

from tollbooth.bootstrap_relay import PublishResult, send_bootstrap_config

AUTHORITY_NSEC = PrivateKey().bech32()
OPERATOR_NPUB = PrivateKey().public_key.bech32()
CONFIG = {"neon_database_url": "postgresql://x", "schema": "op_test"}


def _ok_ws() -> MagicMock:
    ws = MagicMock()
    ws.recv.return_value = json.dumps(["OK", "eventid", True, ""])
    return ws


class TestPublishUsesFanOut:
    def test_publish_reaches_every_relay_concurrently(self):
        """Every relay is asked; the return value reports accepted vs rejected."""
        relays = [f"wss://relay{i}.test" for i in range(5)]
        connects: list[str] = []

        def _connect(url, timeout=10):
            connects.append(url)
            return _ok_ws()

        with patch("websocket.create_connection", side_effect=_connect):
            result = send_bootstrap_config(
                authority_nsec=AUTHORITY_NSEC,
                operator_npub=OPERATOR_NPUB,
                config=CONFIG,
                relays=relays,
            )

        assert result == PublishResult(5, 0) and result
        assert set(connects) == set(relays)

    def test_partial_accept_still_counts_as_published(self):
        relays = ["wss://good.test", "wss://bad.test"]

        def _connect(url, timeout=10):
            ws = MagicMock()
            if "good" in url:
                ws.recv.return_value = json.dumps(["OK", "id", True, ""])
            else:
                ws.recv.return_value = json.dumps(["OK", "id", False, "rate-limited"])
            return ws

        with patch("websocket.create_connection", side_effect=_connect):
            result = send_bootstrap_config(
                authority_nsec=AUTHORITY_NSEC,
                operator_npub=OPERATOR_NPUB,
                config=CONFIG,
                relays=relays,
            )

        assert result == PublishResult(1, 1) and result

    def test_all_reject_is_not_published(self):
        def _connect(url, timeout=10):
            ws = MagicMock()
            ws.recv.return_value = json.dumps(["OK", "id", False, "no"])
            return ws

        with patch("websocket.create_connection", side_effect=_connect):
            result = send_bootstrap_config(
                authority_nsec=AUTHORITY_NSEC,
                operator_npub=OPERATOR_NPUB,
                config=CONFIG,
                relays=["wss://a.test", "wss://b.test"],
            )

        assert result == PublishResult(0, 2) and not result