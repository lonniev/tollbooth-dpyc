"""Tests for the Nostr credential exchange (Secure Courier Service)."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nip44 import encrypt as nip44_encrypt
from tollbooth.nip04 import _get_shared_secret
from tollbooth.nostr_credentials import (
    CourierNotReady,
    CourierTimeout,
    CourierValidationError,
    NostrCredentialExchange,
    _KIND_ENCRYPTED_DM,
    _KIND_GIFT_WRAP,
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
