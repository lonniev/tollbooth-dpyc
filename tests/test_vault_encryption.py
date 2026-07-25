"""Tests for vault_encryption — nsec-derived AES-256-GCM encryption."""

import json

import pytest

from tollbooth.vault_encryption import VaultCipher

# A test nsec (hex) — NOT a real key
TEST_NSEC_HEX = "a" * 64  # 32 bytes of 0xaa


def test_encrypt_decrypt_roundtrip():
    cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    original = '{"balance_api_sats": 1000, "tranches": []}'
    encrypted = cipher.encrypt(original)
    decrypted = cipher.decrypt(encrypted)
    assert decrypted == original


def test_different_nsecs_produce_different_ciphertext():
    cipher1 = VaultCipher(nsec_hex="a" * 64)
    cipher2 = VaultCipher(nsec_hex="b" * 64)
    plaintext = "secret data"
    ct1 = cipher1.encrypt(plaintext)
    ct2 = cipher2.encrypt(plaintext)
    assert ct1 != ct2  # Different keys → different ciphertext


def test_wrong_key_fails_decrypt():
    cipher1 = VaultCipher(nsec_hex="a" * 64)
    cipher2 = VaultCipher(nsec_hex="b" * 64)
    encrypted = cipher1.encrypt("secret")
    with pytest.raises(Exception):  # InvalidTag from AES-GCM  # noqa: B017
        cipher2.decrypt(encrypted)


def test_same_plaintext_different_ciphertext():
    """Random nonce ensures same plaintext encrypts differently each time."""
    cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    ct1 = cipher.encrypt("same data")
    ct2 = cipher.encrypt("same data")
    assert ct1 != ct2  # Different nonces


def test_deterministic_key_derivation():
    """Same nsec always derives the same encryption key."""
    cipher1 = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    cipher2 = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    encrypted = cipher1.encrypt("test")
    decrypted = cipher2.decrypt(encrypted)  # Different instance, same key
    assert decrypted == "test"


def test_is_encrypted_detection():
    cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    encrypted = cipher.encrypt("hello")
    plain_json = '{"balance": 100}'

    assert cipher.is_encrypted(encrypted) is True
    assert cipher.is_encrypted(plain_json) is False
    assert cipher.is_encrypted("") is False
    assert cipher.is_encrypted("[1,2,3]") is False


def test_large_payload():
    """Encrypt/decrypt a realistic ledger JSON."""
    cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    ledger = {
        "balance_api_sats": 5000,
        "total_deposited_api_sats": 10000,
        "total_consumed_api_sats": 4500,
        "total_expired_api_sats": 500,
        "tranches": [
            {"granted_at": "2026-03-24T10:00:00Z", "original_sats": 5000,
             "remaining_sats": 2500, "invoice_id": "inv_123",
             "expires_at": "2026-03-31T10:00:00Z"},
            {"granted_at": "2026-03-25T10:00:00Z", "original_sats": 5000,
             "remaining_sats": 2500, "invoice_id": "inv_456",
             "expires_at": "2026-04-01T10:00:00Z"},
        ],
        "credited_invoices": ["inv_123", "inv_456"],
    }
    original = json.dumps(ledger)
    encrypted = cipher.encrypt(original)
    decrypted = cipher.decrypt(encrypted)
    assert json.loads(decrypted) == ledger


def test_tampered_ciphertext_fails():
    cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
    encrypted = cipher.encrypt("sensitive data")
    import base64
    raw = bytearray(base64.b64decode(encrypted))
    raw[-1] ^= 0xFF  # Flip last byte (in the GCM tag)
    tampered = base64.b64encode(bytes(raw)).decode()
    with pytest.raises(Exception):  # noqa: B017
        cipher.decrypt(tampered)


def test_migration_detection():
    """is_encrypted distinguishes old plaintext from new encrypted values."""
    cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)

    # Old plaintext ledger (what's currently in Neon)
    old_plain = '{"balance_api_sats": 100, "tranches": []}'
    assert cipher.is_encrypted(old_plain) is False

    # New encrypted ledger
    new_encrypted = cipher.encrypt(old_plain)
    assert cipher.is_encrypted(new_encrypted) is True


# ---------------------------------------------------------------------------
# AAD enforcement — the cross-slot ciphertext-swap defense
# ---------------------------------------------------------------------------


class TestAADEnforcement:
    """Verify that AAD binds ciphertext to its context, with no silent
    fallback that would let one slot's ciphertext be decrypted as another's.

    A pre-v0.17.5 shim retried decrypt without AAD on tag failure, which
    made AAD purely advisory. That shim was removed once production
    rotation had aged out the pre-AAD ciphertext population. These tests
    pin the now-enforced behavior so the shim doesn't quietly come back.
    """

    def test_same_aad_roundtrip(self):
        """The normal case — same AAD on encrypt and decrypt → success."""
        cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
        ct = cipher.encrypt("access-token-data", aad="oauth/access_token")
        pt = cipher.decrypt(ct, aad="oauth/access_token")
        assert pt == "access-token-data"

    def test_cross_slot_swap_rejected(self):
        """Ciphertext written for one slot cannot be decrypted as another.
        This is the entire point of AAD — without enforcement, an attacker
        with DB access could swap rows between slots."""
        cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
        ct = cipher.encrypt("refresh-token-data", aad="oauth/refresh_token")
        with pytest.raises(Exception):  # InvalidTag from AES-GCM  # noqa: B017
            cipher.decrypt(ct, aad="oauth/access_token")

    def test_aad_required_on_decrypt_when_encrypt_used_aad(self):
        """If encrypt used AAD, decrypt without AAD must fail. Pre-v0.17.5
        a fallback would have made this succeed — this test pins that
        the fallback is gone."""
        cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
        ct = cipher.encrypt("sensitive", aad="some-context")
        with pytest.raises(Exception):  # noqa: B017
            cipher.decrypt(ct, aad="")

    def test_aad_required_on_encrypt_when_decrypt_uses_aad(self):
        """Symmetric: if encrypt was called without AAD, decrypt-with-AAD
        must fail. Pre-v0.17.5 the fallback would have masked this — the
        test exists to keep that fallback removed."""
        cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
        ct = cipher.encrypt("sensitive")  # no aad
        with pytest.raises(Exception):  # noqa: B017
            cipher.decrypt(ct, aad="some-context")

    def test_empty_aad_unchanged_behavior(self):
        """When neither side passes AAD, behavior is unchanged from
        pre-AAD — the no-AAD path still roundtrips. This is the bulk
        of existing call sites the wheel ships today."""
        cipher = VaultCipher(nsec_hex=TEST_NSEC_HEX)
        ct = cipher.encrypt("data")
        pt = cipher.decrypt(ct)
        assert pt == "data"
