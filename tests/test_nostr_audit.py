"""Tests for NostrAuditPublisher and AuditedVault."""

from __future__ import annotations

import json
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
    def test_skips_publishing_for_non_npub_target(self) -> None:
        """Non-npub (non-encryptable) targets must NOT be published at all.

        Audit content carries balances/financial figures. If the target can't
        be NIP-44 encrypted, the publisher refuses to broadcast plaintext — it
        skips the event entirely rather than leak cleartext to public relays.
        """
        mock_pk = _mock_private_key()

        mock_event = MagicMock()
        mock_event.to_message.return_value = '["EVENT", {}]'

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True) as MockEvent,
            patch("tollbooth.nostr_audit.threading"),
        ):
            MockPK.from_nsec.return_value = mock_pk
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            assert pub.enabled

            pub.publish_ledger_update(
                "user_01KGZY", SAMPLE_LEDGER_JSON, "flush",
            )

            # No event constructed or signed — plaintext broadcast refused.
            MockEvent.assert_not_called()
            mock_event.sign.assert_not_called()

    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    def test_encrypts_content_for_npub_patron(self) -> None:
        """npub user IDs get NIP-44 encrypted content + encrypted tag."""
        mock_pk = _mock_private_key()

        mock_event = MagicMock()
        mock_event.to_message.return_value = '["EVENT", {}]'

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True) as MockEvent,
            patch("tollbooth.nostr_audit.threading") as _mock_threading,
            patch("tollbooth.nostr_audit._HAS_NIP44", True),
            patch(
                "tollbooth.nostr_audit._nip44_encrypt",
                return_value="ENCRYPTED_BASE64_PAYLOAD",
                create=True,
            ) as mock_encrypt,
        ):
            MockPK.from_nsec.return_value = mock_pk
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)

            pub.publish_ledger_update(
                "npub1testuser", SAMPLE_LEDGER_JSON, "flush",
            )

            # NIP-44 encrypt was called with the plaintext JSON
            mock_encrypt.assert_called_once()
            encrypt_args = mock_encrypt.call_args
            plaintext_arg = encrypt_args.args[0]
            assert '"balance":5220' in plaintext_arg
            assert '"user_id":"npub1testuser"' in plaintext_arg

            # Event content is the encrypted payload, not plaintext
            MockEvent.assert_called_once()
            call_kwargs = MockEvent.call_args
            assert call_kwargs.kwargs["content"] == "ENCRYPTED_BASE64_PAYLOAD"

            # Tags include the encrypted marker
            tags = call_kwargs.kwargs["tags"]
            tag_names = [t[0] for t in tags]
            assert "encrypted" in tag_names
            enc_tag = next(t for t in tags if t[0] == "encrypted")
            assert enc_tag[1] == "nip44"

    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    @patch("tollbooth.nostr_audit._HAS_NIP44", False)
    def test_skips_publish_when_nip44_unavailable_for_npub(self) -> None:
        """Refuse plaintext fallback — skip event entirely for npub patrons."""
        mock_pk = _mock_private_key()

        with (
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True) as MockEvent,
            patch("tollbooth.nostr_audit.threading") as mock_threading,
        ):
            MockPK.from_nsec.return_value = mock_pk

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            pub.publish_ledger_update(
                "npub1testuser", SAMPLE_LEDGER_JSON, "flush",
            )

            # Event should NOT be constructed — refused plaintext fallback
            MockEvent.assert_not_called()
            mock_threading.Thread.assert_not_called()

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
        """Snapshot events should have event_type='snapshot'.

        Uses a real npub target (financial audit content is only ever published
        NIP-44-encrypted); ``_nip44_encrypt`` is patched to an identity so the
        content stays inspectable as plaintext JSON.
        """
        from pynostr.key import PrivateKey as RealPrivateKey

        mock_pk = _mock_private_key()
        patron_npub = RealPrivateKey().public_key.bech32()

        with (
            patch("tollbooth.nostr_audit._HAS_PYNOSTR", True),
            patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True),
            patch("tollbooth.nostr_audit._HAS_NIP44", True),
            patch(
                "tollbooth.nostr_audit._nip44_encrypt",
                lambda plaintext, sk_hex, pk_hex: plaintext,
            ),
            patch("tollbooth.nostr_audit.PrivateKey", create=True) as MockPK,
            patch("tollbooth.nostr_audit.Event", create=True) as MockEvent,
            patch("tollbooth.nostr_audit.threading"),
        ):
            MockPK.from_nsec.return_value = mock_pk
            mock_event = MagicMock()
            mock_event.to_message.return_value = '["EVENT", {}]'
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            pub.publish_ledger_update(patron_npub, SAMPLE_LEDGER_JSON, "snapshot")

            content = json.loads(MockEvent.call_args.kwargs["content"])
            assert content["event_type"] == "snapshot"


# ---------------------------------------------------------------------------
# SSL certificate verification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NIP-44 encryption integration — real crypto, no mocks
# ---------------------------------------------------------------------------


class TestNIP44EncryptionIntegration:
    """End-to-end tests: real NostrAuditPublisher with real NIP-44 encryption."""

    def test_real_encrypted_event_decryptable_by_patron(self) -> None:
        """Publish an encrypted event, verify patron can decrypt the content."""
        from pynostr.key import PrivateKey as RealPrivateKey

        from tollbooth.nip44 import decrypt as nip44_decrypt

        operator_sk = RealPrivateKey.from_nsec(FAKE_NSEC)
        patron_sk = RealPrivateKey()
        patron_npub = patron_sk.public_key.bech32()

        captured_events: list[dict] = []

        # Patch Event to capture rather than sign
        with (
            patch("tollbooth.nostr_audit.Event") as MockEvent,
            patch("tollbooth.nostr_audit.threading"),
        ):
            mock_event = MagicMock()
            mock_event.to_message.return_value = '["EVENT", {}]'
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            assert pub.enabled

            pub.publish_ledger_update(
                patron_npub, SAMPLE_LEDGER_JSON, "flush",
            )

            MockEvent.assert_called_once()
            call_kwargs = MockEvent.call_args.kwargs
            captured_events.append(call_kwargs)

        event_kwargs = captured_events[0]

        # Content should be NIP-44 encrypted (base64, not JSON)
        content = event_kwargs["content"]
        try:
            json.loads(content)
            pytest.fail("Content should be encrypted, not valid JSON")
        except json.JSONDecodeError:
            pass  # Expected — it's base64 ciphertext

        # Patron decrypts with their nsec + operator's pubkey
        decrypted = nip44_decrypt(
            content, patron_sk.hex(), operator_sk.public_key.hex(),
        )
        payload = json.loads(decrypted)
        assert payload["user_id"] == patron_npub
        assert payload["balance"] == 5220
        assert payload["deposited"] == 8000
        assert payload["event_type"] == "flush"

        # Encrypted tag present
        tags = event_kwargs["tags"]
        enc_tags = [t for t in tags if t[0] == "encrypted"]
        assert len(enc_tags) == 1
        assert enc_tags[0][1] == "nip44"

    def test_observer_cannot_read_encrypted_content(self) -> None:
        """A third party without the patron's nsec cannot decrypt."""
        from pynostr.key import PrivateKey as RealPrivateKey

        from tollbooth.nip44 import decrypt as nip44_decrypt

        operator_sk = RealPrivateKey.from_nsec(FAKE_NSEC)
        patron_sk = RealPrivateKey()
        observer_sk = RealPrivateKey()

        with (
            patch("tollbooth.nostr_audit.Event") as MockEvent,
            patch("tollbooth.nostr_audit.threading"),
        ):
            mock_event = MagicMock()
            mock_event.to_message.return_value = '["EVENT", {}]'
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            pub.publish_ledger_update(
                patron_sk.public_key.bech32(), SAMPLE_LEDGER_JSON,
            )

            content = MockEvent.call_args.kwargs["content"]

        # Observer tries to decrypt — should fail
        with pytest.raises(ValueError, match="decryption failed"):
            nip44_decrypt(
                content, observer_sk.hex(), operator_sk.public_key.hex(),
            )

    def test_p_tag_contains_patron_hex_pubkey(self) -> None:
        """The p-tag enables patron discovery on relays."""
        from pynostr.key import PrivateKey as RealPrivateKey

        patron_sk = RealPrivateKey()
        patron_npub = patron_sk.public_key.bech32()
        patron_hex = patron_sk.public_key.hex()

        with (
            patch("tollbooth.nostr_audit.Event") as MockEvent,
            patch("tollbooth.nostr_audit.threading"),
        ):
            mock_event = MagicMock()
            mock_event.to_message.return_value = '["EVENT", {}]'
            MockEvent.return_value = mock_event

            pub = NostrAuditPublisher(FAKE_NSEC, FAKE_RELAYS, enabled=True)
            pub.publish_ledger_update(patron_npub, SAMPLE_LEDGER_JSON)

            tags = MockEvent.call_args.kwargs["tags"]
            p_tags = [t for t in tags if t[0] == "p"]
            assert len(p_tags) == 1
            assert p_tags[0][1] == patron_hex


# ---------------------------------------------------------------------------
# SSL certificate verification
# ---------------------------------------------------------------------------


class TestSSLCertVerification:
    @patch("tollbooth.nostr_audit._HAS_PYNOSTR", True)
    @patch("tollbooth.nostr_audit._HAS_WEBSOCKET", True)
    def test_sslopt_does_not_disable_cert_verification(self) -> None:
        """Ensure _publish_to_relays does NOT set cert_reqs to CERT_NONE."""
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

            # Verify sslopt does NOT contain cert_reqs disabling verification
            for call in mock_ws.call_args_list:
                sslopt = call.kwargs.get("sslopt", {})
                assert "cert_reqs" not in sslopt, (
                    "sslopt must not disable SSL certificate verification"
                )
