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
    with pytest.raises(Exception):  # InvalidTag from AES-GCM
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
    with pytest.raises(Exception):
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
