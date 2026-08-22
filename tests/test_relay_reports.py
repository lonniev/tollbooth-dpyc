"""Reporting unreachable relays back to the Oracle.

The Oracle serves the fleet's relay order from a curated guess; operators are
the ones who discover a relay is down. These cover the carrying of that news —
and, just as importantly, that carrying it never costs the operator anything.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from tollbooth import relay_reports


@pytest.fixture(autouse=True)
def _reset():
    relay_reports.reset_relay_reports()
    yield
    relay_reports.reset_relay_reports()


class TestBuffering:
    def test_a_failure_is_buffered_not_sent(self):
        """note_relay_failure is bookkeeping — it must not touch the network."""
        relay_reports.note_relay_failure("wss://dead.example", "read")
        assert relay_reports.pending_relay_failures() == {"wss://dead.example": "read"}

    def test_buffering_never_raises(self):
        """A failure path is the worst place to introduce a new exception."""
        relay_reports.note_relay_failure("", "read")
        relay_reports.note_relay_failure(None, "read")  # type: ignore[arg-type]
        assert relay_reports.pending_relay_failures() == {}

    def test_the_buffer_is_bounded(self):
        """Fed from failure paths, so a runaway loop must not accumulate."""
        for i in range(relay_reports._MAX_BUFFERED + 20):
            relay_reports.note_relay_failure(f"wss://r{i}.example", "send")
        assert len(relay_reports.pending_relay_failures()) == relay_reports._MAX_BUFFERED

    def test_first_mode_wins_for_a_repeated_relay(self):
        relay_reports.note_relay_failure("wss://dead.example", "send")
        relay_reports.note_relay_failure("wss://dead.example", "read")
        assert relay_reports.pending_relay_failures() == {"wss://dead.example": "send"}


class TestFlush:
    @pytest.mark.asyncio
    async def test_nothing_buffered_means_no_oracle_call(self):
        client = AsyncMock()
        with patch("tollbooth.oracle_client.default_oracle_client", return_value=client):
            assert await relay_reports.flush_relay_failures("npub1x", "aa" * 32) == []
        client.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flush_reports_and_clears(self):
        relay_reports.note_relay_failure("wss://dead.example", "read")
        client = AsyncMock()
        client.call_tool.return_value = {"success": True, "probed": "unreachable",
                                         "order_changed": False}
        with patch("tollbooth.oracle_client.default_oracle_client", return_value=client), \
             patch.object(relay_reports, "_signed_report", return_value='{"id":"x"}'):
            responses = await relay_reports.flush_relay_failures("npub1x", "aa" * 32)

        assert len(responses) == 1
        args = client.call_tool.await_args.args
        assert args[0] == "report_relay_failure"
        assert args[1]["relay"] == "wss://dead.example"
        assert args[1]["mode"] == "read"
        assert relay_reports.pending_relay_failures() == {}

    @pytest.mark.asyncio
    async def test_an_unreachable_oracle_is_not_the_operators_problem(self):
        """Telemetry failing must never surface to a patron."""
        from tollbooth.oracle_client import OracleClientError

        relay_reports.note_relay_failure("wss://dead.example", "read")
        client = AsyncMock()
        client.call_tool.side_effect = OracleClientError("oracle down")
        with patch("tollbooth.oracle_client.default_oracle_client", return_value=client), \
             patch.object(relay_reports, "_signed_report", return_value='{"id":"x"}'):
            assert await relay_reports.flush_relay_failures("npub1x", "aa" * 32) == []

    @pytest.mark.asyncio
    async def test_a_changed_fleet_order_invalidates_the_local_cache(self):
        """Otherwise the operator keeps using the old order until the TTL."""
        relay_reports.note_relay_failure("wss://dead.example", "send")
        client = AsyncMock()
        client.call_tool.return_value = {"success": True, "probed": "unreachable",
                                         "order_changed": True}
        invalidate = patch("tollbooth.relay_registry.invalidate_relays_cache")
        with patch("tollbooth.oracle_client.default_oracle_client", return_value=client), \
             patch.object(relay_reports, "_signed_report", return_value='{"id":"x"}'), \
             invalidate as mock_invalidate:
            await relay_reports.flush_relay_failures("npub1x", "aa" * 32)
        mock_invalidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_relay_is_not_re_reported_during_its_cooldown(self):
        """A dead relay fails on every attempt; the Oracle needs to hear once."""
        client = AsyncMock()
        client.call_tool.return_value = {"success": True, "order_changed": False}
        with patch("tollbooth.oracle_client.default_oracle_client", return_value=client), \
             patch.object(relay_reports, "_signed_report", return_value='{"id":"x"}'):
            relay_reports.note_relay_failure("wss://dead.example", "send")
            await relay_reports.flush_relay_failures("npub1x", "aa" * 32)
            # It fails again moments later.
            relay_reports.note_relay_failure("wss://dead.example", "send")
            await relay_reports.flush_relay_failures("npub1x", "aa" * 32)
        assert client.call_tool.await_count == 1

    @pytest.mark.asyncio
    async def test_the_cooldown_expires(self):
        client = AsyncMock()
        client.call_tool.return_value = {"success": True, "order_changed": False}
        with patch("tollbooth.oracle_client.default_oracle_client", return_value=client), \
             patch.object(relay_reports, "_signed_report", return_value='{"id":"x"}'):
            relay_reports.note_relay_failure("wss://dead.example", "send")
            await relay_reports.flush_relay_failures("npub1x", "aa" * 32)
            relay_reports._last_sent["wss://dead.example"] = (
                time.monotonic() - relay_reports._REPORT_COOLDOWN_SECONDS - 1
            )
            relay_reports.note_relay_failure("wss://dead.example", "send")
            await relay_reports.flush_relay_failures("npub1x", "aa" * 32)
        assert client.call_tool.await_count == 2


class TestSigning:
    def test_the_signature_names_the_relay_and_verifies(self):
        """Cross-library: we sign with pynostr, the Oracle verifies with nostr_sdk."""
        pytest.importorskip("nostr_sdk")
        from nostr_sdk import Event
        from pynostr.key import PrivateKey

        key = PrivateKey()
        signed = relay_reports._signed_report("wss://dead.example", key.hex())
        assert signed is not None

        event = Event.from_json(signed)
        event.verify()
        assert event.author().to_hex() == key.public_key.hex()
        assert "wss://dead.example" in event.content()
