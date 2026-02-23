"""Tests for NostrAuditPublisher and AuditedVault."""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.nostr_audit import (
    AuditedVault,
    NostrAuditPublisher,
    _extract_audit_fields,
    _npub_to_hex,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A realistic nsec for testing (not a real key — generated for test only)
# pynostr is optional, so we mock it in tests that need signing
FAKE_NSEC = "nsec1vl029mgpspedva04g90vltkh6fvh240zqtv9k0t9af8935ke9laqsnlfe5"
FAKE_RELAYS = ["wss://relay.example.com", "wss://relay2.example.com"]

SAMPLE_LEDGER_JSON = json.dumps({
    "v": 4,
    "tranches": [
        {"remaining_sats": 3000, "initial_sats": 5000},
        {"remaining_sats": 2220, "initial_sats": 3000},
    ],
    "total_deposited_api_sats": 8000,
    "total_consumed_api_sats": 2780,
    "total_expired_api_sats": 0,
    "pending_invoices": {},
    "credited_invoices": [],
    "daily_log": {},
    "history": {},
    "invoices": {},
})


def _mock_inner_vault() -> MagicMock:
    """Create a mock VaultBackend with async methods."""
    vault = MagicMock()
    vault.store_ledger = AsyncMock(return_value="42")
    vault.fetch_ledger = AsyncMock(return_value=SAMPLE_LEDGER_JSON)
    vault.snapshot_ledger = AsyncMock(return_value="snap-1")
    return vault


def _mock_private_key() -> MagicMock:
    """Create a mock PrivateKey with expected pynostr interface."""
    pk = MagicMock()
    pk.public_key.hex.return_value = "a" * 64
    pk.hex.return_value = "b" * 64
    return pk


# ---------------------------------------------------------------------------
# _extract_audit_fields
# ---------------------------------------------------------------------------


class TestExtractAuditFields:
    def test_extracts_balance_from_tranches(self) -> None:
        fields = _extract_audit_fields(SAMPLE_LEDGER_JSON)
        assert fields["balance"] == 5220
        assert fields["deposited"] == 8000
        assert fields["consumed"] == 2780
        assert fields["expired"] == 0
        assert fields["tranches"] == 2

    def test_empty_tranches(self) -> None:
        ledger = json.dumps({"v": 4, "tranches": []})
        fields = _extract_audit_fields(ledger)
        assert fields["balance"] == 0
        assert fields["tranches"] == 0

    def test_invalid_json(self) -> None:
        fields = _extract_audit_fields("not json")
        assert fields["balance"] == 0
        assert fields["deposited"] == 0
        assert fields["tranches"] == 0

    def test_empty_string(self) -> None:
        fields = _extract_audit_fields("")
        assert fields["balance"] == 0

    def test_missing_fields_default_to_zero(self) -> None:
        ledger = json.dumps({"v": 4, "tranches": [{"remaining_sats": 100}]})
        fields = _extract_audit_fields(ledger)
        assert fields["balance"] == 100
        assert fields["deposited"] == 0
        assert fields["consumed"] == 0


# ---------------------------------------------------------------------------
# _npub_to_hex
# ---------------------------------------------------------------------------


class TestNpubToHex:
    def test_non_npub_passthrough(self) -> None:
        assert _npub_to_hex("user_01KGZ") == "user_01KGZ"

    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", False)
    def test_no_pynostr_passthrough(self) -> None:
        assert _npub_to_hex("npub1abc123") == "npub1abc123"


# ---------------------------------------------------------------------------
# NostrAuditPublisher — construction
# ---------------------------------------------------------------------------


class TestPublisherConstruction:
    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", False)
    def test_disabled_when_pynostr_missing(self) -> None:
        pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS)
        assert not pub.enabled

    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", False)
    def test_disabled_when_websocket_missing(self) -> None:
        pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS)
        assert not pub.enabled

    def test_disabled_by_flag(self) -> None:
        pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=False)
        assert not pub.enabled

    def test_disabled_with_empty_relays(self) -> None:
        pub = NostrAuditPublisher(FAKE_NSEC, [], enabled=True)
        assert not pub.enabled

    def test_disabled_with_invalid_nsec(self) -> None:
        pub = NostrAuditPublisher("invalid", FAKE_RELAYS, enabled=True)
        assert not pub.enabled


# ---------------------------------------------------------------------------
# NostrAuditPublisher — publish_ledger_update
# ---------------------------------------------------------------------------


class TestPublishLedgerUpdate:
    def test_noop_when_disabled(self) -> None:
        pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=False)
        # Should not raise
        pub.publish_ledger_update("npub1abc", SAMPLE_LEDGER_JSON, "flush")

    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    def test_constructs_event_and_publishes(self) -> None:
        """Test that a valid publisher constructs an event and fires a thread."""
        mock_pk = _mock_private_key()

        mock_event = MagicMock()
        mock_event.to_message.return_value = '["EVENT", {}]'

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True) as MockEvent,
            patch("tollbooth.nostr_audit.threading") as mock_threading,
        ):
            MockPK.from_nsec.return_value = mock_pk
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            assert pub.enabled

            pub.publish_ledger_update(
                "npub1testuser", SAMPLE_LEDGER_JSON, "flush",
            )

            # Event was constructed with kind 30078
            MockEvent.assert_called_once()
            call_kwargs = MockEvent.call_args
            assert call_kwargs.kwargs["kind"] == 30078

            # Content is valid JSON with expected fields
            content = json.loads(call_kwargs.kwargs["content"])
            assert content["user_id"] == "npub1testuser"
            assert content["balance"] == 5220
            assert content["deposited"] == 8000
            assert content["consumed"] == 2780
            assert content["event_type"] == "flush"
            assert "timestamp" in content

            # Tags include d-tag, p-tag, t-tag, L-tag
            tags = call_kwargs.kwargs["tags"]
            tag_names = [t[0] for t in tags]
            assert "d" in tag_names
            assert "p" in tag_names
            assert "t" in tag_names
            assert "L" in tag_names

            # d-tag starts with tollbooth-audit-
            d_tag = [t for t in tags if t[0] == "d"][0]
            assert d_tag[1].startswith("tollbooth-audit-")

            # t-tag is tollbooth-audit
            t_tag = [t for t in tags if t[0] == "t"][0]
            assert t_tag[1] == "tollbooth-audit"

            # Event was signed
            mock_event.sign.assert_called_once()

            # Thread was started for relay publishing
            mock_threading.Thread.assert_called_once()
            mock_threading.Thread.return_value.start.assert_called_once()

    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    def test_publish_catches_construction_errors(self) -> None:
        """If event construction fails, it logs but doesn't raise."""
        mock_pk = _mock_private_key()

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True, side_effect=RuntimeError("boom")),
        ):
            MockPK.from_nsec.return_value = mock_pk
            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            # Should not raise
            pub.publish_ledger_update("user1", SAMPLE_LEDGER_JSON)


# ---------------------------------------------------------------------------
# NostrAuditPublisher — _publish_to_relays
# ---------------------------------------------------------------------------


class TestPublishToRelays:
    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    def test_publishes_to_all_relays(self) -> None:
        mock_pk = _mock_private_key()

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.create_connection", create=True) as mock_ws,
        ):
            MockPK.from_nsec.return_value = mock_pk
            mock_conn = MagicMock()
            mock_conn.recv.return_value = '["OK"]'
            mock_ws.return_value = mock_conn

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            pub._publish_to_relays('["EVENT", {}]')

            # Called once per relay
            assert mock_ws.call_count == 2
            assert mock_conn.send.call_count == 2
            assert mock_conn.close.call_count == 2

    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    def test_relay_failure_does_not_block(self) -> None:
        mock_pk = _mock_private_key()

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch(
                "tollbooth.nostr_audit.create_connection",
                create=True,
                side_effect=ConnectionError("relay down"),
            ),
        ):
            MockPK.from_nsec.return_value = mock_pk
            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            # Should not raise
            pub._publish_to_relays('["EVENT", {}]')


# ---------------------------------------------------------------------------
# AuditedVault — delegation
# ---------------------------------------------------------------------------


class TestAuditedVaultDelegation:
    @pytest.mark.asyncio
    async def test_store_ledger_delegates_and_publishes(self) -> None:
        inner = _mock_inner_vault()
        publisher = MagicMock(spec=NostrAuditPublisher)
        vault = AuditedVault(inner, publisher)

        result = await vault.store_ledger("npub1user", SAMPLE_LEDGER_JSON)

        assert result == "42"
        inner.store_ledger.assert_awaited_once_with(
            "npub1user", SAMPLE_LEDGER_JSON,
        )
        publisher.publish_ledger_update.assert_called_once_with(
            "npub1user", SAMPLE_LEDGER_JSON, "flush",
        )

    @pytest.mark.asyncio
    async def test_fetch_ledger_passthrough(self) -> None:
        inner = _mock_inner_vault()
        publisher = MagicMock(spec=NostrAuditPublisher)
        vault = AuditedVault(inner, publisher)

        result = await vault.fetch_ledger("npub1user")

        assert result == SAMPLE_LEDGER_JSON
        inner.fetch_ledger.assert_awaited_once_with("npub1user")
        publisher.publish_ledger_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_snapshot_ledger_delegates_and_publishes(self) -> None:
        inner = _mock_inner_vault()
        publisher = MagicMock(spec=NostrAuditPublisher)
        vault = AuditedVault(inner, publisher)

        result = await vault.snapshot_ledger(
            "npub1user", SAMPLE_LEDGER_JSON, "2026-02-23T12:00:00Z",
        )

        assert result == "snap-1"
        inner.snapshot_ledger.assert_awaited_once_with(
            "npub1user", SAMPLE_LEDGER_JSON, "2026-02-23T12:00:00Z",
        )
        publisher.publish_ledger_update.assert_called_once_with(
            "npub1user", SAMPLE_LEDGER_JSON, "snapshot",
        )

    @pytest.mark.asyncio
    async def test_inner_vault_error_propagates(self) -> None:
        """AuditedVault does not swallow inner vault errors."""
        inner = _mock_inner_vault()
        inner.store_ledger = AsyncMock(side_effect=RuntimeError("db error"))
        publisher = MagicMock(spec=NostrAuditPublisher)
        vault = AuditedVault(inner, publisher)

        with pytest.raises(RuntimeError, match="db error"):
            await vault.store_ledger("npub1user", SAMPLE_LEDGER_JSON)

        # Publisher should NOT be called if inner vault fails
        publisher.publish_ledger_update.assert_not_called()


# ---------------------------------------------------------------------------
# Event content structure
# ---------------------------------------------------------------------------


class TestEventContent:
    def test_snapshot_event_type(self) -> None:
        """Snapshot events should have event_type='snapshot'."""
        mock_pk = _mock_private_key()

        with (
            patch("tollbooth.nostr_audit._HAS_PYNOSTR", True),
            patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True),
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True) as MockEvent,
            patch("tollbooth.nostr_audit.threading"),
        ):
            MockPK.from_nsec.return_value = mock_pk
            mock_event = MagicMock()
            mock_event.to_message.return_value = '["EVENT", {}]'
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            pub.publish_ledger_update("user1", SAMPLE_LEDGER_JSON, "snapshot")

            content = json.loads(MockEvent.call_args.kwargs["content"])
            assert content["event_type"] == "snapshot"
