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
    _TIMESTAMP_FUZZ_SECONDS,
    _parse_delimited_credentials,
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


def _to_delimited(payload: dict) -> str:
    """Serialize a dict as @@@ delimited text for testing."""
    return "\n".join(f"{k} = @@@{v}@@@" for k, v in payload.items())


def _make_nip04_event(
    sender_privkey: PrivateKey,
    recipient_pubkey_hex: str,
    payload: dict | str,
    created_at: int | None = None,
) -> dict:
    """Build a kind 4 NIP-04 event dict for testing."""
    import base64
    import os
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.padding import PKCS7

    raw = payload if isinstance(payload, str) else _to_delimited(payload)
    plaintext = raw.encode("utf-8")

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
        "content": _to_delimited(payload),
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
            result = await ex.open_channel("x", greeting="Test greeting")

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
            await ex.open_channel("nonexistent", greeting="Test greeting")

    @pytest.mark.asyncio
    async def test_disabled_exchange_raises(self):
        """open_channel on disabled exchange raises CourierNotReady."""
        ex = _make_exchange(nsec="nsec1invalid")
        with pytest.raises(CourierNotReady):
            await ex.open_channel("x", greeting="Test greeting")


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
    async def test_gift_wrap_wrong_sender_skipped(self):
        """Gift wrap from wrong sender is skipped (not matching sender)."""
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

        # The impersonator's gift wrap decrypts but the seal reveals
        # a different sender — it gets skipped, resulting in timeout.
        with patch.object(ex, "_fetch_dms_from_relays"), \
             pytest.raises(CourierTimeout, match="No decryptable DM found"):
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
             patch.object(ex, "_find_dm_candidates") as mock_find:
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
    def test_sends_dual_protocol_dm(self):
        """send_dm publishes both a kind 1059 gift wrap and a kind 4 NIP-04 DM."""
        ex = _make_exchange()
        patron = PrivateKey()

        published = []

        def capture_publish(message: str) -> list:
            published.append(message)
            return [("wss://relay.test.com", True, "")]

        with patch.object(ex, "_publish_to_relays", side_effect=capture_publish):
            ex.send_dm(patron.public_key.bech32(), "Hello patron!")

        assert len(published) == 2
        events = [json.loads(m)[1] for m in published]
        kinds = {e["kind"] for e in events}
        # Both protocols sent
        assert _KIND_GIFT_WRAP in kinds
        assert _KIND_ENCRYPTED_DM in kinds

        # Gift wrap has ephemeral pubkey
        wrap = next(e for e in events if e["kind"] == _KIND_GIFT_WRAP)
        assert wrap["pubkey"] != ex._pubkey_hex
        p_tags = [t for t in wrap["tags"] if t[0] == "p"]
        assert p_tags[0][1] == patron.public_key.hex()

        # NIP-04 DM has operator pubkey and NIP-04 format
        dm = next(e for e in events if e["kind"] == _KIND_ENCRYPTED_DM)
        assert dm["pubkey"] == ex._pubkey_hex
        assert "?iv=" in dm["content"]
        p_tags = [t for t in dm["tags"] if t[0] == "p"]
        assert p_tags[0][1] == patron.public_key.hex()

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

        # Find the NIP-17 gift wrap (kind 1059) among published messages
        wrap_event = next(
            json.loads(m)[1] for m in published
            if json.loads(m)[1]["kind"] == _KIND_GIFT_WRAP
        )

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
        """Raises CourierError when every relay rejects both protocols."""
        from tollbooth.nostr_credentials import CourierError

        ex = _make_exchange()
        patron = PrivateKey()

        def reject_all(message: str) -> list:
            return [("wss://relay.test.com", False, "blocked: kind not allowed")]

        with patch.object(ex, "_publish_to_relays", side_effect=reject_all):
            with pytest.raises(CourierError, match="All relay sends failed"):
                ex.send_dm(patron.public_key.bech32(), "Rejected")


class TestOpenChannelWithWelcomeDm:
    @pytest.mark.asyncio
    async def test_sends_welcome_dm_when_npub_provided(self):
        ex = _make_exchange()
        patron = PrivateKey()

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm") as mock_send:
            result = await ex.open_channel(
                "x", greeting="Test greeting", recipient_npub=patron.public_key.bech32(),
            )

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
            result = await ex.open_channel("x", greeting="Test greeting")

        assert result["success"] is True
        assert result["welcome_dm_sent"] is False
        assert self._npub_in_message(result, ex.npub)

    @pytest.mark.asyncio
    async def test_fallback_on_dm_failure(self):
        ex = _make_exchange()
        patron = PrivateKey()

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm", side_effect=Exception("relay down")):
            result = await ex.open_channel(
                "x", greeting="Test greeting", recipient_npub=patron.public_key.bech32(),
            )

        assert result["success"] is True
        assert result["welcome_dm_sent"] is False
        assert "could not send" in result["message"].lower()

    @staticmethod
    def _npub_in_message(result: dict, npub: str) -> bool:
        return npub in result.get("message", "")


# ── Poison Slug (Anti-Replay) Tests ──────────────────────────────────────


class TestPoisonSlug:
    """Tests for the anti-replay poison token in the Secure Courier flow."""

    @pytest.mark.asyncio
    async def test_open_channel_returns_poison(self):
        """open_channel includes a poison slug in the result."""
        ex = _make_exchange()

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm"):
            result = await ex.open_channel(
                "x", greeting="Test greeting", recipient_npub="npub1test123",
            )

        assert "poison" in result
        # Format: adjective-noun-number
        parts = result["poison"].split("-")
        assert len(parts) == 3
        assert parts[2].isdigit()

    @pytest.mark.asyncio
    async def test_poison_validated_on_receive(self):
        """Receive rejects payload with wrong poison."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        # Register a poison
        sender_npub = sender.public_key.bech32()
        ex._pending_poisons[(sender_npub, "x")] = ("bold-hawk-42", time.time() + 600)

        # Build a NIP-04 event with wrong poison
        payload = {"api_key": "k", "api_secret": "s", "poison": "wrong-slug-99"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             pytest.raises(CourierValidationError, match="Anti-replay token mismatch"):
            await ex.receive(sender_npub)

    @pytest.mark.asyncio
    async def test_poison_accepted_on_match(self):
        """Receive accepts payload with correct poison."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sender_npub = sender.public_key.bech32()
        ex._pending_poisons[(sender_npub, "x")] = ("bold-hawk-42", time.time() + 600)

        payload = {"api_key": "k", "api_secret": "s", "poison": "bold-hawk-42"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender_npub)

        assert result["success"] is True
        # Poison should be consumed
        assert (sender_npub, "x") not in ex._pending_poisons

    @pytest.mark.asyncio
    async def test_poison_expired(self):
        """Receive rejects payload when poison has expired."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sender_npub = sender.public_key.bech32()
        # Expired 10 seconds ago
        ex._pending_poisons[(sender_npub, "x")] = ("bold-hawk-42", time.time() - 10)

        payload = {"api_key": "k", "api_secret": "s", "poison": "bold-hawk-42"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             pytest.raises(CourierValidationError, match="expired"):
            await ex.receive(sender_npub)

    @pytest.mark.asyncio
    async def test_no_poison_required_without_open_channel(self):
        """Receive works without poison when no open_channel was called."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        payload = {"api_key": "k", "api_secret": "s"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_poison_picks_correct_dm_from_multiple(self):
        """When multiple DMs exist, receive picks the one with matching poison."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sender_npub = sender.public_key.bech32()
        ex._pending_poisons[(sender_npub, "x")] = ("bold-hawk-42", time.time() + 600)

        # Stale DM with old/wrong poison (injected first)
        stale_payload = {"api_key": "old_k", "api_secret": "old_s", "poison": "true-quill-26"}
        stale_event = _make_nip04_event(sender, operator.public_key.hex(), stale_payload)
        stale_event["id"] = "stale_event_001"

        # Valid DM with correct poison (injected second)
        valid_payload = {"api_key": "correct_k", "api_secret": "correct_s", "poison": "bold-hawk-42"}
        valid_event = _make_nip04_event(sender, operator.public_key.hex(), valid_payload)
        valid_event["id"] = "valid_event_002"

        with ex._lock:
            ex._received_events.append(stale_event)
            ex._received_events.append(valid_event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion") as mock_delete:
            result = await ex.receive(sender_npub)

        assert result["success"] is True
        assert result["credentials"]["api_key"] == "correct_k"
        # Poison should be consumed
        assert (sender_npub, "x") not in ex._pending_poisons

    @pytest.mark.asyncio
    async def test_stale_dms_get_deletion_requests(self):
        """Stale DMs (wrong poison) get NIP-09 deletion requests."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sender_npub = sender.public_key.bech32()
        ex._pending_poisons[(sender_npub, "x")] = ("bold-hawk-42", time.time() + 600)

        # Two stale DMs
        stale1 = _make_nip04_event(
            sender, operator.public_key.hex(),
            {"api_key": "k", "api_secret": "s", "poison": "old-slug-01"},
        )
        stale1["id"] = "stale_001"
        stale2 = _make_nip04_event(
            sender, operator.public_key.hex(),
            {"api_key": "k", "api_secret": "s", "poison": "old-slug-02"},
        )
        stale2["id"] = "stale_002"

        # Valid DM
        valid = _make_nip04_event(
            sender, operator.public_key.hex(),
            {"api_key": "k", "api_secret": "s", "poison": "bold-hawk-42"},
        )
        valid["id"] = "valid_003"

        with ex._lock:
            ex._received_events.extend([stale1, stale2, valid])

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion") as mock_delete:
            result = await ex.receive(sender_npub)

        assert result["success"] is True
        # Should have deletion calls for stale events + the valid event
        deleted_ids = [call.args[0] for call in mock_delete.call_args_list]
        assert "stale_001" in deleted_ids
        assert "stale_002" in deleted_ids
        # Stale events should be in consumed set
        assert "stale_001" in ex._consumed_ids
        assert "stale_002" in ex._consumed_ids

    @pytest.mark.asyncio
    async def test_no_poison_match_reports_found_poisons(self):
        """When no DM matches the expected poison, error lists found poisons."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        sender_npub = sender.public_key.bech32()
        ex._pending_poisons[(sender_npub, "x")] = ("bold-hawk-42", time.time() + 600)

        # Two DMs, neither with the correct poison
        dm1 = _make_nip04_event(
            sender, operator.public_key.hex(),
            {"api_key": "k", "api_secret": "s", "poison": "true-quill-26"},
        )
        dm1["id"] = "dm_001"
        dm2 = _make_nip04_event(
            sender, operator.public_key.hex(),
            {"api_key": "k", "api_secret": "s", "poison": "lazy-fox-99"},
        )
        dm2["id"] = "dm_002"

        with ex._lock:
            ex._received_events.extend([dm1, dm2])

        with patch.object(ex, "_fetch_dms_from_relays"), \
             pytest.raises(CourierValidationError, match="none of 2 DM") as exc_info:
            await ex.receive(sender_npub)

        err = str(exc_info.value)
        assert "true-quill-26" in err
        assert "lazy-fox-99" in err
        assert "request_credential_channel" in err


# ── Concurrent Multi-Service Exchange Tests ──────────────────────────────


class TestConcurrentExchanges:
    """Tests for multi-service poison independence."""

    @staticmethod
    def _two_service_templates() -> dict[str, CredentialTemplate]:
        return {
            "x": CredentialTemplate(
                service="x",
                version=1,
                fields={
                    "api_key": FieldSpec(required=True, sensitive=True),
                    "api_secret": FieldSpec(required=True, sensitive=True),
                },
                description="X API credentials",
            ),
            "openai": CredentialTemplate(
                service="openai",
                version=1,
                fields={
                    "api_key": FieldSpec(required=True, sensitive=True),
                },
                description="OpenAI API credentials",
            ),
        }

    @pytest.mark.asyncio
    async def test_two_services_independent_poisons(self):
        """Two open_channel calls for different services store separate poisons."""
        ex = _make_exchange(templates=self._two_service_templates())
        npub = "npub1test123"

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm"):
            r1 = await ex.open_channel(
                "x", greeting="X greeting", recipient_npub=npub,
            )
            r2 = await ex.open_channel(
                "openai", greeting="OpenAI greeting", recipient_npub=npub,
            )

        assert r1["poison"] != r2["poison"]
        assert (npub, "x") in ex._pending_poisons
        assert (npub, "openai") in ex._pending_poisons
        assert ex._pending_poisons[(npub, "x")][0] == r1["poison"]
        assert ex._pending_poisons[(npub, "openai")][0] == r2["poison"]

    @pytest.mark.asyncio
    async def test_receive_matches_correct_service_poison(self):
        """Receive for service X uses X's poison, leaving OpenAI's intact."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(
            nsec=operator.nsec, templates=self._two_service_templates(),
        )

        sender_npub = sender.public_key.bech32()
        ex._pending_poisons[(sender_npub, "x")] = ("x-hawk-42", time.time() + 600)
        ex._pending_poisons[(sender_npub, "openai")] = ("ai-fox-99", time.time() + 600)

        payload = {"api_key": "k", "api_secret": "s", "poison": "x-hawk-42"}
        event = _make_nip04_event(sender, operator.public_key.hex(), payload)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender_npub, service="x")

        assert result["success"] is True
        # X poison consumed
        assert (sender_npub, "x") not in ex._pending_poisons
        # OpenAI poison still intact
        assert (sender_npub, "openai") in ex._pending_poisons

    @pytest.mark.asyncio
    async def test_second_open_channel_same_service_replaces_poison(self):
        """Re-opening a channel for the same service overwrites the old poison."""
        ex = _make_exchange(templates=self._two_service_templates())
        npub = "npub1test123"

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm"):
            r1 = await ex.open_channel(
                "x", greeting="First attempt", recipient_npub=npub,
            )
            r2 = await ex.open_channel(
                "x", greeting="Second attempt", recipient_npub=npub,
            )

        # Only the second poison should be stored
        assert ex._pending_poisons[(npub, "x")][0] == r2["poison"]
        assert ex._pending_poisons[(npub, "x")][0] != r1["poison"]


# ── Conversational DM Flow Tests ──────────────────────────────────────────


class TestConversationalDmFlow:
    """Tests for conversational welcome, success, and error DMs."""

    @pytest.mark.asyncio
    async def test_welcome_dm_is_conversational(self):
        """Welcome message contains the operator greeting and standard elements."""
        ex = _make_exchange()
        patron = PrivateKey()
        greeting = "Hi — I'm eXcalibur, a Tollbooth MCP service."

        with patch.object(ex, "_start_subscription"), \
             patch.object(ex, "send_dm") as mock_send:
            await ex.open_channel(
                "x", greeting=greeting, recipient_npub=patron.public_key.bech32(),
            )

        mock_send.assert_called_once()
        welcome_text = mock_send.call_args[0][1]
        assert greeting in welcome_text
        assert "If you didn't request this" in welcome_text
        assert "tollbooth-dpyc v" in welcome_text

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
            with pytest.raises(CourierValidationError, match="@@@"):
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

    def test_gift_wrap_with_operator_ptag_collected(self):
        """Gift wrap event with the operator’s p-tag is collected in the buffer."""
        operator = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        gift_wrap_event = {
            "id": "gw_ptag_test",
            "kind": _KIND_GIFT_WRAP,
            "pubkey": PrivateKey().public_key.hex(),
            "content": "encrypted-content-placeholder",
            "created_at": int(time.time()),
            "tags": [["p", operator.public_key.hex()]],  # operator p-tag
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
            assert ex._received_events[0]["id"] == "gw_ptag_test"

    def test_gift_wrap_events_in_buffer_included_as_candidates(self):
        """Gift wrap events in the buffer are returned as candidates by
        _find_dm_candidates regardless of their pubkey (sender is hidden)."""
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

        # _find_dm_candidates should include the gift wrap
        # even though the sender_hex doesn’t match the event pubkey,
        # because gift wrap sender identity is hidden until unwrap.
        results = ex._find_dm_candidates(sender.public_key.hex())
        assert len(results) == 1
        assert results[0]["id"] == "gw_candidate_test"

    def test_multiple_filters_in_req(self):
        """REQ message contains two filter objects: NIP-04 and NIP-17 (p-tagged)."""
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
        assert len(filters) == 2, f"Expected 2 filters, got {len(filters)}"

        # Filter 1: NIP-04 with p-tag
        assert filters[0]["kinds"] == [_KIND_ENCRYPTED_DM]
        assert "#p" in filters[0]

        # Filter 2: NIP-17 gift wrap with p-tag
        assert filters[1]["kinds"] == [_KIND_GIFT_WRAP]
        assert "#p" in filters[1]



class TestReceiveFormatEnforcement:
    """End-to-end tests verifying @@@ format is required."""

    @pytest.mark.asyncio
    async def test_receive_rejects_json_payload(self):
        """receive() rejects raw JSON (@@@ format required)."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        raw = '{"api_key": "sk-test", "api_secret": "secret"}'
        event = _make_nip04_event(sender, operator.public_key.hex(), raw)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "send_dm"), \
             pytest.raises(CourierValidationError, match="@@@"):
            await ex.receive(sender.public_key.bech32())

    @pytest.mark.asyncio
    async def test_receive_delimited_payload(self):
        """receive() parses @@@ delimited credentials."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec)

        raw = "api_key = @@@sk-test@@@\napi_secret = @@@secret@@@"
        event = _make_nip04_event(sender, operator.public_key.hex(), raw)
        with ex._lock:
            ex._received_events.append(event)

        with patch.object(ex, "_fetch_dms_from_relays"), \
             patch.object(ex, "_request_deletion"):
            result = await ex.receive(sender.public_key.bech32())

        assert result["success"] is True
        assert result["credentials"]["api_key"] == "sk-test"


class TestParseDelimitedCredentials:
    """Tests for _parse_delimited_credentials()."""

    def test_basic_single_field(self):
        """Extract a single field."""
        result = _parse_delimited_credentials("api_key = @@@sk-123@@@")
        assert result == {"api_key": "sk-123"}

    def test_multiline_fields(self):
        """Extract multiple fields across lines."""
        text = "api_key = @@@sk-123@@@\napi_secret = @@@sec-456@@@"
        result = _parse_delimited_credentials(text)
        assert result == {"api_key": "sk-123", "api_secret": "sec-456"}

    def test_with_preamble(self):
        """Surrounding text is ignored — only @@@ pairs extracted."""
        text = "Here are my creds:\napi_key = @@@sk-123@@@\nThanks!"
        result = _parse_delimited_credentials(text)
        assert result == {"api_key": "sk-123"}

    def test_with_poison(self):
        """Poison field extracted alongside credentials."""
        text = (
            "api_key = @@@sk-123@@@\n"
            "api_secret = @@@sec-456@@@\n"
            "poison = @@@bold-hawk-42@@@"
        )
        result = _parse_delimited_credentials(text)
        assert result is not None
        assert result["poison"] == "bold-hawk-42"
        assert result["api_key"] == "sk-123"

    def test_whitespace_stripped(self):
        """Whitespace around keys and values is stripped."""
        text = "  api_key  =  @@@ sk-123 @@@  "
        result = _parse_delimited_credentials(text)
        assert result == {"api_key": "sk-123"}

    def test_returns_none_without_markers(self):
        """Returns None when no @@@ patterns found."""
        assert _parse_delimited_credentials("just some text") is None
        assert _parse_delimited_credentials('{"json": "data"}') is None

    def test_no_spacing_around_equals(self):
        """Works without spaces around =."""
        text = "api_key=@@@sk-123@@@"
        result = _parse_delimited_credentials(text)
        assert result == {"api_key": "sk-123"}


# ── NIP-17 Timestamp Hardening Tests ─────────────────────────────────


class TestNip17TimestampHardening:
    """Tests for gift wrap timestamp fuzz handling."""

    def test_giftwrap_filter_includes_fuzz_window(self):
        """Gift wrap filter uses wider since window for NIP-17 timestamp fuzz."""
        ex = _make_exchange()
        filters_sent = []

        def capture_subscribe(relay_url, sub_id, filters):
            filters_sent.append(filters)

        with patch.object(ex, "_subscribe_one_relay", side_effect=capture_subscribe):
            ex._subscribe_to_relays()

        assert len(filters_sent) > 0
        nip04_filter, giftwrap_filter = filters_sent[0]

        # NIP-04 filter uses normal freshness window
        nip04_since = nip04_filter["since"]
        # Gift wrap filter has a wider window (48h wider for NIP-17 fuzz)
        giftwrap_since = giftwrap_filter["since"]
        assert giftwrap_since < nip04_since
        assert nip04_since - giftwrap_since == _TIMESTAMP_FUZZ_SECONDS

    def test_nip17_fuzzed_timestamp_found_by_candidate_filter(self):
        """Gift wrap with created_at fuzzed 3h into the past is still returned."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec, freshness_window=900)  # 15 min

        three_hours_ago = int(time.time()) - 3 * 60 * 60

        gift_wrap_event = {
            "id": "gw_fuzzed_ts",
            "kind": _KIND_GIFT_WRAP,
            "pubkey": PrivateKey().public_key.hex(),
            "content": "encrypted-gift-wrap",
            "created_at": three_hours_ago,
            "tags": [["p", operator.public_key.hex()]],
            "sig": "fake_sig",
        }

        with ex._lock:
            ex._received_events.append(gift_wrap_event)

        results = ex._find_dm_candidates(sender.public_key.hex())
        assert len(results) == 1
        assert results[0]["id"] == "gw_fuzzed_ts"

    def test_nip04_stale_event_still_filtered(self):
        """NIP-04 events older than freshness window are still rejected."""
        operator = PrivateKey()
        sender = PrivateKey()
        ex = _make_exchange(nsec=operator.nsec, freshness_window=60)

        stale_event = {
            "id": "nip04_stale",
            "kind": _KIND_ENCRYPTED_DM,
            "pubkey": sender.public_key.hex(),
            "content": "encrypted-content",
            "created_at": int(time.time()) - 120,  # 2 min ago, window is 1 min
            "tags": [["p", operator.public_key.hex()]],
            "sig": "fake_sig",
        }

        with ex._lock:
            ex._received_events.append(stale_event)

        results = ex._find_dm_candidates(sender.public_key.hex())
        assert len(results) == 0
