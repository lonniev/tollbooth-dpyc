"""Tests for NIP-04 decryption."""

import base64
import os

import pytest
from pynostr.key import PrivateKey

from tollbooth.nip04 import _get_shared_secret, decrypt, encrypt


class TestNip04Decrypt:
    """NIP-04 AES-256-CBC decryption tests."""

    def _encrypt_nip04(
        self, plaintext: str, privkey_hex: str, pubkey_hex: str,
    ) -> str:
        """Encrypt a message in NIP-04 format for testing."""
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives.padding import PKCS7

        shared_secret = _get_shared_secret(privkey_hex, pubkey_hex)
        iv = os.urandom(16)

        padder = PKCS7(128).padder()
        padded = padder.update(plaintext.encode("utf-8")) + padder.finalize()

        cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()

        ct_b64 = base64.b64encode(ciphertext).decode("ascii")
        iv_b64 = base64.b64encode(iv).decode("ascii")
        return f"{ct_b64}?iv={iv_b64}"

    def test_round_trip(self):
        """Encrypt then decrypt returns original plaintext."""
        sender = PrivateKey()
        recipient = PrivateKey()

        plaintext = '{"api_key": "sk-test-123", "api_secret": "secret-456"}'
        encrypted = self._encrypt_nip04(
            plaintext, sender.hex(), recipient.public_key.hex(),
        )
        decrypted = decrypt(
            encrypted, recipient.hex(), sender.public_key.hex(),
        )
        assert decrypted == plaintext

    def test_symmetric_ecdh(self):
        """ECDH shared secret is symmetric (sender/recipient order doesn't matter)."""
        a = PrivateKey()
        b = PrivateKey()
        secret_ab = _get_shared_secret(a.hex(), b.public_key.hex())
        secret_ba = _get_shared_secret(b.hex(), a.public_key.hex())
        assert secret_ab == secret_ba

    def test_wrong_key_fails(self):
        """Decrypting with the wrong key raises ValueError."""
        sender = PrivateKey()
        recipient = PrivateKey()
        wrong = PrivateKey()

        encrypted = self._encrypt_nip04(
            "secret data", sender.hex(), recipient.public_key.hex(),
        )
        with pytest.raises(ValueError, match="decryption failed"):
            decrypt(encrypted, wrong.hex(), sender.public_key.hex())

    def test_invalid_format_no_iv(self):
        """Missing ?iv= separator raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NIP-04 format"):
            decrypt("justbase64data", "a" * 64, "b" * 64)

    def test_invalid_base64(self):
        """Invalid base64 raises ValueError."""
        with pytest.raises(ValueError):
            decrypt("not-valid-b64?iv=also-not-valid", "a" * 64, "b" * 64)

    def test_wrong_iv_length(self):
        """IV that's not 16 bytes raises ValueError."""
        ct = base64.b64encode(b"ciphertext").decode()
        iv = base64.b64encode(b"short").decode()
        with pytest.raises(ValueError, match="IV must be 16 bytes"):
            decrypt(f"{ct}?iv={iv}", "a" * 64, "b" * 64)

    def test_unicode_content(self):
        """Non-ASCII content survives round-trip."""
        sender = PrivateKey()
        recipient = PrivateKey()

        plaintext = '{"name": "日本語テスト", "emoji": "🔐📬"}'
        encrypted = self._encrypt_nip04(
            plaintext, sender.hex(), recipient.public_key.hex(),
        )
        decrypted = decrypt(
            encrypted, recipient.hex(), sender.public_key.hex(),
        )
        assert decrypted == plaintext


    def test_decrypt_unpadded_base64(self):
        """Decrypt tolerates base64 with stripped '=' padding (Primal, etc.)."""
        sender = PrivateKey()
        recipient = PrivateKey()

        encrypted = encrypt("hello world", sender.hex(), recipient.public_key.hex())
        # Strip trailing '=' from both ciphertext and IV parts
        ct_part, iv_part = encrypted.split("?iv=", 1)
        stripped = f"{ct_part.rstrip('=')}?iv={iv_part.rstrip('=')}"
        decrypted = decrypt(stripped, recipient.hex(), sender.public_key.hex())
        assert decrypted == "hello world"


class TestNip04Encrypt:
    """NIP-04 AES-256-CBC encryption tests."""

    def test_encrypt_round_trip(self):
        """encrypt() → decrypt() round-trip succeeds."""
        sender = PrivateKey()
        recipient = PrivateKey()

        plaintext = '{"api_key": "sk-test", "api_secret": "secret"}'
        encrypted = encrypt(plaintext, sender.hex(), recipient.public_key.hex())
        decrypted = decrypt(encrypted, recipient.hex(), sender.public_key.hex())
        assert decrypted == plaintext

    def test_encrypt_produces_nip04_format(self):
        """encrypt() output has NIP-04 format with ?iv= separator."""
        sender = PrivateKey()
        recipient = PrivateKey()

        encrypted = encrypt("test", sender.hex(), recipient.public_key.hex())
        assert "?iv=" in encrypted

    def test_self_encryption(self):
        """Self-encryption (same key as sender and recipient) works."""
        key = PrivateKey()

        plaintext = '{"credential": "value"}'
        encrypted = encrypt(plaintext, key.hex(), key.public_key.hex())
        decrypted = decrypt(encrypted, key.hex(), key.public_key.hex())
        assert decrypted == plaintext
