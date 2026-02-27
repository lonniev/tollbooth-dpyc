"""Tests for the Nostr credential exchange (Secure Courier Service)."""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.credential_vault_backend import CredentialVaultBackend
from tollbooth.nip44 import encrypt as nip44_encrypt
from tollbooth.nip04 import _get_shared_secret
from tollbooth.nostr_credentials import (
    CourierError,
    CourierNotReady,
    CourierTimeout,
    CourierValidationError,
    NostrCredentialExchange,
    NostrProfile,
    _KIND_ENCRYPTED_DM,
    _KIND_GIFT_WRAP,
    _KIND_METADATA,
    _KIND_SEAL,
    _KIND_PRIVATE_DM,
)


# ── Fixtures ──────────────────────────────────────────────────────────

def _test_template() -> dict[str, CredentialTemplate]:
    """Single-service test template."""
    return {
        "x": CredentialTemplate(
            service="x",
            version=1,
            fields={
                "api_key": FieldSpec(required=True, sensitive=True),
                "api_secret": FieldSpec(required=True, sensitive=True),
            },
            description="Test X API credentials",
        ),
    }


def _make_exchange(
    nsec: str | None = None,
    relays: list[str] | None = None,
    templates: dict[str, CredentialTemplate] | None = None,
    **kwargs,
) -> NostrCredentialExchange:
    """Create an exchange with test defaults."""
    if nsec is None:
        nsec = PrivateKey().nsec
    if relays is None:
        relays = ["wss://relay.test.com"]
    return NostrCredentialExchange(
        nsec=nsec,
        relays=relays,
        templates=templates or _test_template(),
        **kwargs,
    )


def _make_nip04_event(
    sender_privkey: PrivateKey,
    recipient_pubkey_hex: str,
    payload: dict,
    created_at: int | None = None,
) -> dict:
    """Build a kind 4 NIP-04 event dict for testing."""
    import base64
    import os
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.padding import PKCS7

    plaintext = json.dumps(payload).encode("utf-8")

    shared_secret = _get_shared_secret(sender_privkey.hex(), recipient_pubkey_hex)
    iv = os.urandom(16)

    padder = PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    ct_b64 = base64.b64encode(ciphertext).decode()
    iv_b64 = base64.b64encode(iv).decode()

    return {
        "id": "event_nip04_test",
        "kind": _KIND_ENCRYPTED_DM,
        "pubkey": sender_privkey.public_key.hex(),
        "content": f"{ct_b64}?iv={iv_b64}",
        "tags": [["p", recipient_pubkey_hex]],
        "created_at": created_at or int(time.time()),
        "sig": "fake_sig",
    }


def _make_gift_wrap_event(
    sender_privkey: PrivateKey,
    recipient_privkey_hex: str,
    recipient_pubkey_hex: str,
    payload: dict,
    created_at: int | None = None,
) -> dict:
    """Build a kind 1059 NIP-17 gift wrap event dict for testing.

    Three layers:
    1. DM (kind 14): plaintext JSON payload
    2. Seal (kind 13): DM encrypted with NIP-44 (sender → recipient)
    3. Gift wrap (kind 1059): Seal encrypted with NIP-44 (random → recipient)
    """
    now = created_at or int(time.time())

    # Layer 3: The actual DM content
    dm_event = {
        "kind": _KIND_PRIVATE_DM,
        "content": json.dumps(payload),
        "pubkey": sender_privkey.public_key.hex(),
        "created_at": now,
        "tags": [["p", recipient_pubkey_hex]],
    }

    # Layer 2: Seal — DM encrypted to recipient with sender's key
    seal_content = nip44_encrypt(
        json.dumps(dm_event),
        sender_privkey.hex(),
        recipient_pubkey_hex,
    )
    seal_event = {
        "kind": _KIND_SEAL,
        "content": seal_content,
        "pubkey": sender_privkey.public_key.hex(),
        "created_at": now,
        "tags": [],
    }

    # Layer 1: Gift wrap — Seal encrypted to recipient with random key
    random_key = PrivateKey()
    wrap_content = nip44_encrypt(
        json.dumps(seal_event),
        random_key.hex(),
        recipient_pubkey_hex,
    )

    return {
        "id": "event_giftwrap_test",
        "kind": _KIND_GIFT_WRAP,
        "content": wrap_content,
        "pubkey": random_key.public_key.hex(),
        "created_at": now,
        "tags": [["p", recipient_pubkey_hex]],
        "sig": "fake_sig",
    }


# ── Initialization Tests ─────────────────────────────────────────────

class TestExchangeInit:
    """Tests for NostrCredentialExchange initialization."""

    def test_valid_init(self):
        """Exchange initializes with valid nsec."""
        ex = _make_exchange()
        assert ex.enabled
        assert ex.npub.startswith("npub1")
        assert len(ex.relays) == 1

    def test_invalid_nsec_disables(self):
        """Invalid nsec disables the exchange."""
        ex = _make_exchange(nsec="nsec1invalid")
        assert not ex.enabled

    def test_no_relays_disables(self):
        """Empty relay list disables the exchange."""
        ex = _make_exchange(relays=[])
        assert not ex.enabled

    @patch("tollbooth.nostr_credentials._HAS_PYNOSTR", False)
    def test_missing_pynostr_disables(self):
        """Missing pynostr disables the exchange."""
        ex = NostrCredentialExchange(
            nsec="ignored",
            relays=["wss://relay.test.com"],
            templates=_test_template(),
        )
        assert not ex.enabled

    @patch("tollbooth.nostr_credentials._HAS_WEBSOCKET", False)
    def test_missing_websocket_disables(self):
        """Missing websocket-client disables the exchange."""
        ex = _make_exchange()
        assert not ex.enabled


# ── open_channel Tests ────────────────────────────────────────────────

class TestOpenChannel:
    """Tests for open_channel()."""

    @pytest.mark.asyncio
    async def test_returns_npub_and_instructions(self):
        """open_channel returns npub, relays, and template instructions."""
        ex = _make_exchange()
        # Mock the subscription to avoid real WebSocket
        with patch.object(ex, "_start_subscription"):
            result = await ex.open_channel("x")

        assert result["success"] is True
        assert result["npub"] == ex.npub
        assert result["relays"] == ex.relays
        assert result["service"] == "x"
        assert "api_key" in result["instructions"]
        assert "api_secret" in result["instructions"]

    @pytest.mark.asyncio
    async def test_unknown_service_raises(self):
        """open_channel with unknown service raises CourierValidationError."""
        ex = _make_exchange()
        with pytest.raises(CourierValidationError, match="Unknown service"):
            await ex.open_channel("nonexistent")

    @pytest.mark.asyncio
    async def test_disabled_exchange_raises(self):
        """open_channel on disabled exchange raises CourierNotReady."""
        ex = _make_exchange(nsec="nsec1invalid")
        with pytest.raises(CourierNotReady):
            await ex.open_channel("x")


# ── receive Tests — NIP-04 ───────────────────────────────────────────

class TestReceiveNip04:
    """Tests for receive() with NIP-04 kind 4 DMs."""

    @pytest.mark.asyncio
    async def test_receive_nip04_dm(self):
        """Receive and decrypt a NIP-04 DM."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "sk-test-123", "api_secret": "secret-456"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        # Inject into buffer
        with ex._lock:
            ex._received_events.append(event)

        # Mock relay fetch (already in buffer) and deletion
        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["service"] == "x"
        assert result["fields_received"] == 2
        assert result["encryption"] == "nip04"
        assert result["credentials"]["api_key"] == "sk-test-123"

    @pytest.mark.asyncio
    async def test_nip04_rejected_when_nip44_only(self):
        """NIP-04 DM rejected when nip44_only=True."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec, nip44_only=True)

        payload = {"api_key": "key", "api_secret": "secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             pytest.raises(CourierValidationError, match="NIP-04 DMs rejected"):
            await ex.receive(sender.public_key.bech32())


# ── receive Tests — NIP-17 Gift Wrap ─────────────────────────────────

class TestReceiveNip17:
    """Tests for receive() with NIP-17 gift-wrapped DMs."""

    @pytest.mark.asyncio
    async def test_receive_gift_wrap_dm(self):
        """Receive and unwrap a NIP-17 gift-wrapped DM."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "sk-wrapped-123", "api_secret": "wrapped-secret"}
        event = _make_gift_wrap_event(
            sender, operator.hex(), operator.public_key.hex(), payload,
        )

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["service"] == "x"
        assert result["encryption"] == "nip44"
        assert result["credentials"]["api_key"] == "sk-wrapped-123"

    @pytest.mark.asyncio
    async def test_gift_wrap_wrong_sender_rejected(self):
        """Gift wrap from wrong sender is rejected."""
        operator = PrivateKey()
        sender = PrivateKey()
        impersonator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "key", "api_secret": "secret"}
        # Wrap is created by impersonator but we claim it's from sender
        event = _make_gift_wrap_event(
            impersonator, operator.hex(), operator.public_key.hex(), payload,
        )

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             pytest.raises(CourierValidationError, match="does not match"):
            await ex.receive(sender.public_key.bech32())


# ── receive Tests — Validation ────────────────────────────────────────

class TestReceiveValidation:
    """Tests for payload validation during receive."""

    @pytest.mark.asyncio
    async def test_invalid_json_rejected(self):
        """Non-JSON DM content is rejected."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        # Manually build a NIP-04 event with non-JSON content
        import base64
        import os
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7

        plaintext = b"this is not json"
        shared_secret = _get_shared_secret(
            sender.hex(), operator.public_key.hex(),
        )
        iv = os.urandom(16)
        padder = PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
        ct = cipher.encryptor().update(padded) + cipher.encryptor().finalize()
        content = f"{base64.b64encode(ct).decode()}?iv={base64.b64encode(iv).decode()}"

        event = {
            "id": "bad_json",
            "kind": _KIND_ENCRYPTED_DM,
            "pubkey": sender.public_key.hex(),
            "content": content,
            "tags": [["p", operator.public_key.hex()]],
            "created_at": int(time.time()),
            "sig": "fake",
        }

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"):
            with pytest.raises(CourierValidationError):
                await ex.receive(sender.public_key.bech32())

    @pytest.mark.asyncio
    async def test_unknown_fields_rejected(self):
        """Payload with unknown fields is rejected."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {
            "api_key": "key",
            "api_secret": "secret",
            "rogue_field": "evil",
        }
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"):
            with pytest.raises(CourierValidationError, match="Unknown fields"):
                await ex.receive(sender.public_key.bech32())

    @pytest.mark.asyncio
    async def test_missing_required_fields_rejected(self):
        """Payload missing required fields is rejected."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "key"}  # missing api_secret
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"):
            with pytest.raises(CourierValidationError, match="Missing required"):
                await ex.receive(sender.public_key.bech32())


# ── Freshness and Replay Tests ────────────────────────────────────────

class TestFreshnessAndReplay:
    """Tests for freshness window and double-pickup prevention."""

    @pytest.mark.asyncio
    async def test_stale_event_ignored(self):
        """Events older than freshness window are not matched."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec, freshness_window=60)

        payload = {"api_key": "key", "api_secret": "secret"}
        event = _make_nip04_event(
            sender, operator.public_key.hex(), payload,
            created_at=int(time.time()) - 120,  # 2 minutes ago, window is 1 min
        )

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"):
            with pytest.raises(CourierTimeout):
                await ex.receive(sender.public_key.bech32())

    @pytest.mark.asyncio
    async def test_double_pickup_prevented(self):
        """Same event cannot be received twice."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "key", "api_secret": "secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            # First receive succeeds
            result = await ex.receive(sender.public_key.bech32())
            assert result["success"]

            # Second receive fails (event consumed)
            with pytest.raises(CourierTimeout):
                await ex.receive(sender.public_key.bech32())

    @pytest.mark.asyncio
    async def test_no_dm_found_raises_timeout(self):
        """No matching DM raises CourierTimeout."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        with patch.object(ex, "_fetch_dms_from_relays"):
            with pytest.raises(CourierTimeout, match="No DM found"):
                await ex.receive(sender.public_key.bech32())


# ── Relay Subscription Tests ─────────────────────────────────────────

class TestRelaySubscription:
    """Tests for relay WebSocket subscription."""

    def test_subscribe_parses_events(self):
        """Subscription collects EVENT messages from relay."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        fake_event = {
            "id": "test123",
            "kind": 4,
            "pubkey": "abc",
            "content": "encrypted",
            "created_at": int(time.time()),
            "tags": [],
        }

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(side_effect=[
            json.dumps(["EVENT", "sub1", fake_event]),
            json.dumps(["EOSE", "sub1"]),
        ])

        with patch("tollbooth.nostr_credentials.create_connection", return_value=mock_ws):
            ex._subscribe_to_relays()

        with ex._lock:
            assert len(ex._received_events) == 1
            assert ex._received_events[0]["id"] == "test123"

    def test_relay_failure_non_fatal(self):
        """Relay connection failure doesn't crash."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        with patch(
            "tollbooth.nostr_credentials.create_connection",
            side_effect=ConnectionError("relay down"),
        ):
            # Should not raise
            ex._subscribe_to_relays()

        with ex._lock:
            assert len(ex._received_events) == 0


# ── NIP-09 Deletion Tests ────────────────────────────────────────────

class TestDeletion:
    """Tests for NIP-09 deletion requests."""

    def test_deletion_event_published(self):
        """Deletion request is published to relays."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(return_value='["OK","del1",true,""]')

        with patch(
            "tollbooth.nostr_credentials.create_connection",
            return_value=mock_ws,
        ):
            ex._publish_to_relays("test_message")

        mock_ws.send.assert_called_once_with("test_message")

    def test_deletion_failure_non_fatal(self):
        """Deletion relay failure doesn't raise."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        with patch(
            "tollbooth.nostr_credentials.create_connection",
            side_effect=ConnectionError("relay down"),
        ):
            # Should not raise
            ex._publish_to_relays("test_message")


# ── Template Matching Tests ───────────────────────────────────────────

class TestTemplateMatching:
    """Tests for template auto-matching logic."""

    @pytest.mark.asyncio
    async def test_match_by_service_field(self):
        """Payload with service field matches correct template."""
        operator = PrivateKey()
        sender = PrivateKey()

        templates = {
            "x": CredentialTemplate(
                service="x", version=1,
                fields={"api_key": FieldSpec(), "api_secret": FieldSpec()},
            ),
            "openai": CredentialTemplate(
                service="openai", version=1,
                fields={"openai_key": FieldSpec()},
            ),
        }
        ex = _make_exchange(nsec=operator.nsec, templates=templates)

        payload = {"service": "openai", "openai_key": "sk-test"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())
            assert result["service"] == "openai"

    @pytest.mark.asyncio
    async def test_single_template_auto_match(self):
        """Single template matches without service field."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "key", "api_secret": "secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())
            assert result["service"] == "x"


# ── Mock Credential Vault ────────────────────────────────────────────

class MockCredentialVault:
    """In-memory credential vault for testing."""

    def __init__(self):
        self._store: dict[str, str] = {}

    def _key(self, service: str, npub: str) -> str:
        return f"{service}:{npub}"

    async def store_credentials(
        self, service: str, npub: str, encrypted_blob: str,
    ) -> None:
        self._store[self._key(service, npub)] = encrypted_blob

    async def fetch_credentials(
        self, service: str, npub: str,
    ) -> str | None:
        return self._store.get(self._key(service, npub))

    async def delete_credentials(
        self, service: str, npub: str,
    ) -> bool:
        key = self._key(service, npub)
        if key in self._store:
            del self._store[key]
            return True
        return False


# ── Credential Vault Tests ───────────────────────────────────────────

class TestCredentialVault:
    """Tests for vault-first credential lookup and storage."""

    @pytest.mark.asyncio
    async def test_vault_store_after_first_receive(self):
        """Credentials are stored in vault after first relay pickup."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        payload = {"api_key": "sk-first-123", "api_secret": "secret-456"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["encryption"] == "nip04"
        assert result["credentials"]["api_key"] == "sk-first-123"
        # Vault should have the blob
        blob = await vault.fetch_credentials("x", sender.public_key.bech32())
        assert blob is not None

    @pytest.mark.asyncio
    async def test_vault_hit_skips_relay(self):
        """Second receive returns from vault without relay I/O."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        payload = {"api_key": "sk-cached", "api_secret": "cached-secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        # First receive — from relay
        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result1 = await ex.receive(sender.public_key.bech32())

        assert result1["encryption"] == "nip04"

        # Second receive — should come from vault
        with patch.object(ex, "_fetch_dms_from_relays") as mock_fetch, \
             patch.object(ex, "_find_dm_in_buffer") as mock_find:
            result2 = await ex.receive(sender.public_key.bech32())

        assert result2["success"] is True
        assert result2["encryption"] == "vault"
        assert result2["credentials"]["api_key"] == "sk-cached"
        assert result2["credentials"]["api_secret"] == "cached-secret"
        # Relay methods should NOT have been called
        mock_fetch.assert_not_called()
        mock_find.assert_not_called()

    @pytest.mark.asyncio
    async def test_vault_blob_is_encrypted(self):
        """Vault blob is not plaintext JSON — it's NIP-04 encrypted."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        payload = {"api_key": "sk-secret", "api_secret": "top-secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            await ex.receive(sender.public_key.bech32())

        blob = await vault.fetch_credentials("x", sender.public_key.bech32())
        assert blob is not None
        # Blob should be NIP-04 format, not plaintext
        assert "?iv=" in blob
        # Blob should NOT contain plaintext credentials
        assert "sk-secret" not in blob
        assert "top-secret" not in blob

    @pytest.mark.asyncio
    async def test_forget_clears_vault(self):
        """forget() deletes credentials from vault."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        payload = {"api_key": "key", "api_secret": "secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            await ex.receive(sender.public_key.bech32())

        # Vault should have credentials
        assert await vault.fetch_credentials("x", sender.public_key.bech32()) is not None

        # Forget them
        result = await ex.forget(sender.public_key.bech32())
        assert result["success"] is True
        assert result["deleted"] is True

        # Vault should be empty
        assert await vault.fetch_credentials("x", sender.public_key.bech32()) is None

    @pytest.mark.asyncio
    async def test_forget_nonexistent_returns_false(self):
        """forget() returns deleted=False when no credentials exist."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        result = await ex.forget(sender.public_key.bech32())
        assert result["success"] is True
        assert result["deleted"] is False

    @pytest.mark.asyncio
    async def test_forget_without_vault(self):
        """forget() without vault returns failure message."""
        ex = _make_exchange()
        sender = PrivateKey()

        result = await ex.forget(sender.public_key.bech32())
        assert result["success"] is False
        assert "No credential vault" in result["message"]

    @pytest.mark.asyncio
    async def test_no_vault_preserves_existing_behavior(self):
        """Without a vault, receive() works identically to v0.1.30."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)  # No vault

        payload = {"api_key": "key", "api_secret": "secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["encryption"] == "nip04"
        # No vault mention in message
        assert "vault" not in result["message"].lower()

    @pytest.mark.asyncio
    async def test_vault_miss_falls_through_to_relay(self):
        """Empty vault falls through to relay DM flow."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()  # Empty vault
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        payload = {"api_key": "fresh-key", "api_secret": "fresh-secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["encryption"] == "nip04"  # Came from relay, not vault
        assert result["credentials"]["api_key"] == "fresh-key"

    @pytest.mark.asyncio
    async def test_forget_then_receive_uses_relay(self):
        """After forget(), receive() falls back to relay."""
        operator = PrivateKey()
        sender = PrivateKey()
        vault = MockCredentialVault()
        ex = _make_exchange(nsec=operator.nsec, credential_vault=vault)

        payload = {"api_key": "original", "api_secret": "secret"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        # First receive stores in vault
        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            await ex.receive(sender.public_key.bech32())

        # Forget
        await ex.forget(sender.public_key.bech32())

        # New DM with rotated credentials
        new_payload = {"api_key": "rotated", "api_secret": "new-secret"}
        new_event = _make_nip04_event(
            sender, operator.public_key.hex(), new_payload,
        )
        new_event["id"] = "event_rotated"

        with ex._lock:
            ex._received_events.append(new_event)

        # Second receive should use relay (vault is empty)
        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["credentials"]["api_key"] == "rotated"
        assert result["encryption"] == "nip04"


# ── NostrProfile tests ────────────────────────────────────────────────


class TestNostrProfile:
    def test_to_metadata_full(self):
        profile = NostrProfile(
            name="excalibur-mcp",
            display_name="eXcalibur MCP",
            about="Sword-swift tweets",
            picture="https://example.com/avatar.png",
            nip05="excalibur@dpyc.community",
            website="https://github.com/lonniev/excalibur-mcp",
        )
        meta = profile.to_metadata()
        assert meta["name"] == "excalibur-mcp"
        assert meta["display_name"] == "eXcalibur MCP"
        assert meta["about"] == "Sword-swift tweets"
        assert meta["picture"] == "https://example.com/avatar.png"
        assert meta["nip05"] == "excalibur@dpyc.community"
        assert meta["website"] == "https://github.com/lonniev/excalibur-mcp"

    def test_to_metadata_minimal(self):
        profile = NostrProfile(name="test-mcp")
        meta = profile.to_metadata()
        assert meta == {"name": "test-mcp"}
        assert "display_name" not in meta
        assert "picture" not in meta

    def test_extra_fields(self):
        profile = NostrProfile(
            name="test",
            extra={"custom_field": "value"},
        )
        meta = profile.to_metadata()
        assert meta["custom_field"] == "value"


class TestPublishProfile:
    def test_publishes_kind_0_event(self):
        ex = _make_exchange()
        profile = NostrProfile(name="test-mcp", about="Test service")

        published = []
        original_publish = ex._publish_to_relays

        def capture_publish(message: str) -> None:
            published.append(message)

        with patch.object(ex, "_publish_to_relays", side_effect=capture_publish):
            ex.publish_profile(profile)

            # Wait for daemon thread
            import time
            time.sleep(0.2)

        assert len(published) == 1
        msg = json.loads(published[0])
        assert msg[0] == "EVENT"
        event = msg[1]
        assert event["kind"] == _KIND_METADATA
        content = json.loads(event["content"])
        assert content["name"] == "test-mcp"
        assert content["about"] == "Test service"

    def test_publish_profile_disabled(self):
        """No error when exchange is disabled."""
        ex = _make_exchange(relays=[])
        ex._enabled = False
        ex.publish_profile(NostrProfile(name="test"))
        # Should not raise


class TestSendDm:
    def test_sends_gift_wrapped_dm(self):
        """send_dm publishes a kind 1059 gift wrap with ephemeral pubkey."""
        ex = _make_exchange()
        patron = PrivateKey()

        published = []

        def capture_publish(message: str) -> list:
            published.append(message)
            return [("wss://relay.test.com", True, "")]

        with patch.object(ex, "_publish_to_relays", side_effect=capture_publish):
            ex.send_dm(patron.public_key.bech32(), "Hello patron!")

        assert len(published) == 1
        msg = json.loads(published[0])
        assert msg[0] == "EVENT"
        event = msg[1]
        # Gift wrap is kind 1059
        assert event["kind"] == _KIND_GIFT_WRAP
        # Pubkey should be ephemeral (NOT the operator's)
        assert event["pubkey"] != ex._pubkey_hex
        # p-tag should reference the patron (for relay routing)
        p_tags = [t for t in event["tags"] if t[0] == "p"]
        assert len(p_tags) == 1
        assert p_tags[0][1] == patron.public_key.hex()
        # Content should NOT be NIP-04 format (no ?iv= separator)
        assert "?iv=" not in event["content"]

    def test_send_dm_unwrappable(self):
        """Patron can unwrap the gift wrap to recover the plaintext."""
        from tollbooth.nip44 import decrypt as nip44_decrypt

        ex = _make_exchange()
        patron = PrivateKey()

        published = []

        def capture_publish(message: str) -> list:
            published.append(message)
            return [("wss://relay.test.com", True, "")]

        with patch.object(ex, "_publish_to_relays", side_effect=capture_publish):
            ex.send_dm(patron.public_key.bech32(), "Test message")

        wrap_event = json.loads(published[0])[1]

        # Layer 1: Decrypt gift wrap → seal JSON
        seal_json = nip44_decrypt(
            wrap_event["content"], patron.hex(), wrap_event["pubkey"],
        )
        seal = json.loads(seal_json)
        assert seal["kind"] == _KIND_SEAL
        assert seal["pubkey"] == ex._pubkey_hex  # Seal is from the real sender

        # Layer 2: Decrypt seal → rumor JSON
        dm_json = nip44_decrypt(
            seal["content"], patron.hex(), seal["pubkey"],
        )
        dm = json.loads(dm_json)
        assert dm["kind"] == _KIND_PRIVATE_DM
        assert dm["content"] == "Test message"
        assert dm["pubkey"] == ex._pubkey_hex

    def test_send_dm_invalid_npub(self):
        ex = _make_exchange()
        with pytest.raises(CourierValidationError, match="Invalid recipient"):
            ex.send_dm("not-an-npub", "hello")

    def test_send_dm_all_relays_reject(self):
        """Raises CourierError when every relay rejects the event."""
        from tollbooth.nostr_credentials import CourierError

        ex = _make_exchange()
        patron = PrivateKey()

        def reject_all(message: str) -> list:
            return [("wss://relay.test.com", False, "blocked: kind not allowed")]

        with patch.object(ex, "_publish_to_relays", side_effect=reject_all):
            with pytest.raises(CourierError, match="rejected"):
                ex.send_dm(patron.public_key.bech32(), "Rejected")


class TestOpenChannelWithWelcomeDm:
    @pytest.mark.asyncio
    async def test_sends_welcome_dm_when_npub_provided(self):
        ex = _make_exchange()
        patron = PrivateKey()

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm") as mock_send:
            result = await ex.open_channel("x", recipient_npub=patron.public_key.bech32())

        assert result["success"] is True
        assert result["welcome_dm_sent"] is True
        mock_send.assert_called_once()
        call_args = mock_send.call_args
        assert call_args[0][0] == patron.public_key.bech32()
        assert "credentials" in call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_no_welcome_dm_without_npub(self):
        ex = _make_exchange()

        with patch.object(ex, "_start_subscription"):
            result = await ex.open_channel("x")

        assert result["success"] is True
        assert result["welcome_dm_sent"] is False
        assert self._npub_in_message(result, ex.npub)

    @pytest.mark.asyncio
    async def test_fallback_on_dm_failure(self):
        ex = _make_exchange()
        patron = PrivateKey()

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm", side_effect=Exception("relay down")):
            result = await ex.open_channel("x", recipient_npub=patron.public_key.bech32())

        assert result["success"] is True
        assert result["welcome_dm_sent"] is False
        assert "could not send" in result["message"].lower()

    @staticmethod
    def _npub_in_message(result: dict, npub: str) -> bool:
        return npub in result.get("message", "")


# ── Conversational DM Flow Tests ──────────────────────────────────────────


class TestConversationalDmFlow:
    """Tests for conversational welcome, success, and error DMs."""

    @pytest.mark.asyncio
    async def test_welcome_dm_is_conversational(self):
        """Welcome message contains the new conversational text patterns."""
        ex = _make_exchange()
        patron = PrivateKey()

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm") as mock_send:
            await ex.open_channel("x", recipient_npub=patron.public_key.bech32())

        mock_send.assert_called_once()
        welcome_text = mock_send.call_args[0][1]
        assert "requested a credential channel" in welcome_text
        assert "If you didn't request this" in welcome_text

    @pytest.mark.asyncio
    async def test_success_dm_sent_after_receive(self):
        """After a successful receive, a success DM is sent to the patron."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "sk-test-123", "api_secret": "secret-456"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"), \
             patch.object(ex, "send_dm") as mock_send:
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        mock_send.assert_called_once()
        success_text = mock_send.call_args[0][1]
        assert "securely stored" in success_text

    @pytest.mark.asyncio
    async def test_error_dm_sent_on_validation_failure(self):
        """When credential parsing fails, an error DM is sent before raising."""
        import base64
        import os

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7

        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        plaintext = b"this is not json"
        shared_secret = _get_shared_secret(
            sender.hex(), operator.public_key.hex(),
        )
        iv = os.urandom(16)
        padder = PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
        ct = cipher.encryptor().update(padded) + cipher.encryptor().finalize()
        content = (
            f"{base64.b64encode(ct).decode()}"
            f"?iv={base64.b64encode(iv).decode()}"
        )

        event = {
            "id": "bad_json_dm",
            "kind": _KIND_ENCRYPTED_DM,
            "pubkey": sender.public_key.hex(),
            "content": content,
            "tags": [["p", operator.public_key.hex()]],
            "created_at": int(time.time()),
            "sig": "fake",
        }

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "send_dm") as mock_send:
            with pytest.raises(CourierValidationError):
                await ex.receive(sender.public_key.bech32())

        mock_send.assert_called_once()
        error_text = mock_send.call_args[0][1]
        assert "couldn’t process" in error_text

    @pytest.mark.asyncio
    async def test_success_dm_failure_nonfatal(self):
        """If the success DM fails to send, receive still succeeds."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "sk-test-123", "api_secret": "secret-456"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"), \
             patch.object(ex, "send_dm", side_effect=Exception("relay down")):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["credentials"]["api_key"] == "sk-test-123"

    @pytest.mark.asyncio
    async def test_error_dm_failure_nonfatal(self):
        """If the error DM fails to send, the original
        CourierValidationError still propagates."""
        import base64
        import os

        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7

        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        plaintext = b"not json at all"
        shared_secret = _get_shared_secret(
            sender.hex(), operator.public_key.hex(),
        )
        iv = os.urandom(16)
        padder = PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
        ct = cipher.encryptor().update(padded) + cipher.encryptor().finalize()
        content = (
            f"{base64.b64encode(ct).decode()}"
            f"?iv={base64.b64encode(iv).decode()}"
        )

        event = {
            "id": "bad_json_dm_2",
            "kind": _KIND_ENCRYPTED_DM,
            "pubkey": sender.public_key.hex(),
            "content": content,
            "tags": [["p", operator.public_key.hex()]],
            "created_at": int(time.time()),
            "sig": "fake",
        }

        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "send_dm", side_effect=Exception("relay down")):
            with pytest.raises(CourierValidationError, match="not valid JSON"):
                await ex.receive(sender.public_key.bech32())


# ── NIP-17 Subscription Tests ────────────────────────────────────────


class TestNip17Subscription:
    """Tests for NIP-17 gift-wrap subscription and filter handling."""

    def test_subscription_includes_both_kinds(self):
        """REQ message sent to relay includes both kind 4 and kind 1059."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sent_messages: list[str] = []
        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(side_effect=[
            json.dumps(["EOSE", "sub1"]),
        ])
        mock_ws.send = MagicMock(side_effect=lambda msg: sent_messages.append(msg))

        with patch("tollbooth.nostr_credentials.create_connection", return_value=mock_ws):
            ex._subscribe_to_relays()

        # The first send call is the REQ message
        req_msg = json.loads(sent_messages[0])
        assert req_msg[0] == "REQ"
        # Collect all kinds from all filters (elements after sub_id)
        all_kinds: set[int] = set()
        for filt in req_msg[2:]:
            for k in filt.get("kinds", []):
                all_kinds.add(k)
        assert _KIND_ENCRYPTED_DM in all_kinds, "kind 4 (NIP-04) missing from REQ"
        assert _KIND_GIFT_WRAP in all_kinds, "kind 1059 (NIP-17) missing from REQ"

    def test_gift_wrap_without_ptag_collected(self):
        """Gift wrap event without the operator’s p-tag is collected in the
        buffer thanks to the broad filter (filter 3)."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        random_ptag_pubkey = PrivateKey().public_key.hex()
        gift_wrap_event = {
            "id": "gw_no_ptag_test",
            "kind": _KIND_GIFT_WRAP,
            "pubkey": PrivateKey().public_key.hex(),
            "content": "encrypted-content-placeholder",
            "created_at": int(time.time()),
            "tags": [["p", random_ptag_pubkey]],  # random p-tag, NOT the operator
            "sig": "fake_sig",
        }

        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(side_effect=[
            json.dumps(["EVENT", "sub1", gift_wrap_event]),
            json.dumps(["EOSE", "sub1"]),
        ])

        with patch("tollbooth.nostr_credentials.create_connection", return_value=mock_ws):
            ex._subscribe_to_relays()

        with ex._lock:
            assert len(ex._received_events) == 1
            assert ex._received_events[0]["id"] == "gw_no_ptag_test"

    def test_gift_wrap_events_in_buffer_included_as_candidates(self):
        """Gift wrap events in the buffer are returned as candidates by
        _find_dm_in_buffer regardless of their pubkey (sender is hidden)."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        gift_wrap_event = {
            "id": "gw_candidate_test",
            "kind": _KIND_GIFT_WRAP,
            "pubkey": PrivateKey().public_key.hex(),  # random wrap pubkey
            "content": "encrypted-gift-wrap",
            "created_at": int(time.time()),
            "tags": [["p", PrivateKey().public_key.hex()]],  # random p-tag
            "sig": "fake_sig",
        }

        with ex._lock:
            ex._received_events.append(gift_wrap_event)

        # _find_dm_in_buffer should return the gift wrap as a candidate
        # even though the sender_hex doesn’t match the event pubkey,
        # because gift wrap sender identity is hidden until unwrap.
        result = ex._find_dm_in_buffer(sender.public_key.hex())
        assert result is not None
        assert result["id"] == "gw_candidate_test"

    def test_multiple_filters_in_req(self):
        """REQ message contains multiple filter objects (not a single merged one)."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sent_messages: list[str] = []
        mock_ws = MagicMock()
        mock_ws.recv = MagicMock(side_effect=[
            json.dumps(["EOSE", "sub1"]),
        ])
        mock_ws.send = MagicMock(side_effect=lambda msg: sent_messages.append(msg))

        with patch("tollbooth.nostr_credentials.create_connection", return_value=mock_ws):
            ex._subscribe_to_relays()

        req_msg = json.loads(sent_messages[0])
        assert req_msg[0] == "REQ"
        # sub_id is req_msg[1], filters start at req_msg[2:]
        filters = req_msg[2:]
        assert len(filters) == 3, f"Expected 3 filters, got {len(filters)}"

        # Filter 1: NIP-04 with p-tag
        assert filters[0]["kinds"] == [_KIND_ENCRYPTED_DM]
        assert "#p" in filters[0]

        # Filter 2: NIP-17 gift wrap with p-tag
        assert filters[1]["kinds"] == [_KIND_GIFT_WRAP]
        assert "#p" in filters[1]

        # Filter 3: NIP-17 gift wrap broad (no p-tag constraint)
        assert filters[2]["kinds"] == [_KIND_GIFT_WRAP]
        assert "#p" not in filters[2]
        assert filters[2]["limit"] == 20
