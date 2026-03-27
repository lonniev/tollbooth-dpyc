"""Tests for QR Credential Card — ncred encoding, encrypt/decrypt roundtrip, QR rendering."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest
from pynostr.event import Event
from pynostr.key import PrivateKey

from tollbooth.credential_card import (
    DEFAULT_EXPIRY_DAYS,
    CredentialCardExpired,
    CredentialCardInvalid,
    _ncred_decode,
    _ncred_encode,
    decode_credential_card,
    encode_credential_card,
    render_qr,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def operator_key():
    return PrivateKey()


@pytest.fixture
def user_key():
    return PrivateKey()


@pytest.fixture
def operator_nsec(operator_key):
    return operator_key.nsec


@pytest.fixture
def user_npub(user_key):
    return user_key.public_key.bech32()


@pytest.fixture
def sample_credentials():
    return {"api_key": "sk-test-abc123", "brain_id": "9e115e02-fedb-4254-a1ae-39cce16c63e6"}


# ── TestNcredEncoding ──────────────────────────────────────────────────


class TestNcredEncoding:
    """Bech32 roundtrip, invalid prefix, corrupt checksum."""

    def test_roundtrip(self):
        """Encode → decode preserves bytes exactly."""
        data = json.dumps({"hello": "world"}).encode("utf-8")
        ncred = _ncred_encode(data)
        assert ncred.startswith("ncred1")
        decoded = _ncred_decode(ncred)
        assert decoded == {"hello": "world"}

    def test_large_payload_roundtrip(self):
        """Large JSON payloads survive encoding."""
        big = {"key": "x" * 500}
        data = json.dumps(big).encode("utf-8")
        ncred = _ncred_encode(data)
        decoded = _ncred_decode(ncred)
        assert decoded == big

    def test_invalid_prefix_rejected(self):
        """Non-ncred bech32 strings are rejected."""
        # Encode with a different HRP
        from pynostr.bech32 import Encoding, bech32_encode, convertbits

        bits5 = convertbits(b"test", 8, 5)
        wrong_hrp = bech32_encode("wrong", bits5, Encoding.BECH32)
        with pytest.raises(CredentialCardInvalid, match="Invalid credential card prefix"):
            _ncred_decode(wrong_hrp)

    def test_corrupt_checksum_rejected(self):
        """Corrupted bech32 checksum fails decode."""
        data = json.dumps({"a": 1}).encode("utf-8")
        ncred = _ncred_encode(data)
        # Flip last char to corrupt checksum
        corrupted = ncred[:-1] + ("q" if ncred[-1] != "q" else "p")
        with pytest.raises(CredentialCardInvalid):
            _ncred_decode(corrupted)


# ── TestEncodeDecodeRoundtrip ──────────────────────────────────────────


class TestEncodeDecodeRoundtrip:
    """Full cycle with real keys — preserves credentials, service, user_npub."""

    def test_basic_roundtrip(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "thebrain", operator_nsec, user_npub,
        )
        assert ncred.startswith("ncred1")

        result = decode_credential_card(ncred, operator_nsec)
        assert result["service"] == "thebrain"
        assert result["credentials"] == sample_credentials
        assert result["user_npub"] == user_npub
        assert result["issued_at"] > 0
        assert result["expires_at"] > result["issued_at"]

    def test_custom_expiry(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "myservice", operator_nsec, user_npub,
            expiry_days=7,
        )
        result = decode_credential_card(ncred, operator_nsec)
        assert result["expires_at"] - result["issued_at"] == 7 * 86400

    def test_default_expiry(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        result = decode_credential_card(ncred, operator_nsec)
        assert result["expires_at"] - result["issued_at"] == DEFAULT_EXPIRY_DAYS * 86400

    def test_different_services(self, operator_nsec, user_npub):
        for svc in ("thebrain", "myapp", "test-service"):
            ncred = encode_credential_card(
                {"key": "val"}, svc, operator_nsec, user_npub,
            )
            result = decode_credential_card(ncred, operator_nsec)
            assert result["service"] == svc

    def test_unicode_credentials(self, operator_nsec, user_npub):
        creds = {"name": "José García", "token": "日本語テスト"}
        ncred = encode_credential_card(creds, "test", operator_nsec, user_npub)
        result = decode_credential_card(ncred, operator_nsec)
        assert result["credentials"] == creds


# ── TestExpiration ─────────────────────────────────────────────────────


class TestExpiration:
    """Expired cards rejected, valid cards accepted."""

    def test_expired_card_rejected(self, operator_nsec, user_npub, sample_credentials):
        # Create card with 1-day expiry, then simulate time passing
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub, expiry_days=1,
        )
        # Fast-forward 2 days
        future = time.time() + 2 * 86400
        with patch("tollbooth.credential_card.time") as mock_time:
            mock_time.time.return_value = future
            with pytest.raises(CredentialCardExpired, match="expired"):
                decode_credential_card(ncred, operator_nsec)

    def test_valid_card_accepted(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub, expiry_days=90,
        )
        # Should not raise
        result = decode_credential_card(ncred, operator_nsec)
        assert result["credentials"] == sample_credentials


# ── TestSignatureVerification ──────────────────────────────────────────


class TestSignatureVerification:
    """Tampered content, wrong operator, forged sig."""

    def test_tampered_content_rejected(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        # Decode the bech32, tamper with content, re-encode
        event_dict = _ncred_decode(ncred)
        event_dict["content"] = "tampered"
        tampered_json = json.dumps(event_dict, separators=(",", ":")).encode("utf-8")
        tampered_ncred = _ncred_encode(tampered_json)

        with pytest.raises(CredentialCardInvalid, match="signature verification failed"):
            decode_credential_card(tampered_ncred, operator_nsec)

    def test_wrong_operator_rejected(self, user_npub, sample_credentials):
        operator1 = PrivateKey()
        operator2 = PrivateKey()

        ncred = encode_credential_card(
            sample_credentials, "x", operator1.nsec, user_npub,
        )
        with pytest.raises(CredentialCardInvalid, match="not signed by this operator"):
            decode_credential_card(ncred, operator2.nsec)

    def test_forged_sig_rejected(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        # Replace sig with a different key's signature
        event_dict = _ncred_decode(ncred)
        _forger = PrivateKey()
        # Create a valid event from forger but with our operator's pubkey
        event_dict["sig"] = "0" * 128  # invalid sig
        forged_json = json.dumps(event_dict, separators=(",", ":")).encode("utf-8")
        forged_ncred = _ncred_encode(forged_json)

        with pytest.raises(CredentialCardInvalid, match="signature verification failed"):
            decode_credential_card(forged_ncred, operator_nsec)


# ── TestDecryption ─────────────────────────────────────────────────────


class TestDecryption:
    """Uses tagged p-tag for shared secret, missing p-tag rejected."""

    def test_uses_p_tag_for_decryption(self, operator_nsec, user_npub, sample_credentials):
        """Decryption uses the p-tag user pubkey, not a hardcoded value."""
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        result = decode_credential_card(ncred, operator_nsec)
        assert result["user_npub"] == user_npub
        assert result["credentials"] == sample_credentials

    def test_missing_p_tag_rejected(self, operator_nsec, user_npub, sample_credentials):
        """Card with p-tag removed should fail."""
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        event_dict = _ncred_decode(ncred)
        # Remove p-tag
        event_dict["tags"] = [t for t in event_dict["tags"] if t[0] != "p"]
        # Re-sign with the operator key (so sig is valid but p-tag missing)
        pk = PrivateKey.from_nsec(operator_nsec)
        event = Event(
            kind=event_dict["kind"],
            content=event_dict["content"],
            tags=event_dict["tags"],
            pubkey=pk.public_key.hex(),
            created_at=event_dict["created_at"],
        )
        event.sign(pk.hex())
        modified_json = json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8")
        modified_ncred = _ncred_encode(modified_json)

        with pytest.raises(CredentialCardInvalid, match="missing 'p' tag"):
            decode_credential_card(modified_ncred, operator_nsec)


# ── TestQRRendering ────────────────────────────────────────────────────


class TestQRRendering:
    """Produces valid PNG, realistic payload renders, scale parameter."""

    def test_produces_valid_png(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        png = render_qr(ncred)
        # PNG magic bytes
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_realistic_payload_renders(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "thebrain", operator_nsec, user_npub,
        )
        png = render_qr(ncred)
        assert len(png) > 100  # non-trivial image

    def test_scale_parameter(self, operator_nsec, user_npub, sample_credentials):
        ncred = encode_credential_card(
            sample_credentials, "x", operator_nsec, user_npub,
        )
        small = render_qr(ncred, scale=2)
        large = render_qr(ncred, scale=10)
        assert len(large) > len(small)


# ── TestCapacity ───────────────────────────────────────────────────────


class TestCapacity:
    """2-field payload under 1500 chars, 5-field payload under 4296 chars."""

    def test_two_field_under_1500(self, operator_nsec, user_npub):
        creds = {
            "api_key": "sk-test-" + "a" * 40,
            "brain_id": "9e115e02-fedb-4254-a1ae-39cce16c63e6",
        }
        ncred = encode_credential_card(creds, "thebrain", operator_nsec, user_npub)
        assert len(ncred) < 1500, f"2-field ncred too long: {len(ncred)} chars"

    def test_five_field_under_4296(self, operator_nsec, user_npub):
        creds = {
            "api_key": "sk-test-" + "a" * 40,
            "brain_id": "9e115e02-fedb-4254-a1ae-39cce16c63e6",
            "secret_token": "tok-" + "b" * 60,
            "endpoint": "https://api.example.com/v1/long/path/endpoint",
            "org_id": "org-" + "c" * 40,
        }
        ncred = encode_credential_card(creds, "thebrain", operator_nsec, user_npub)
        assert len(ncred) < 4296, f"5-field ncred too long: {len(ncred)} chars"
