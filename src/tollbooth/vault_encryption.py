"""Nsec-derived field encryption for NeonVault.

Derives a symmetric encryption key from the operator's nsec via HKDF.
Uses AES-256-GCM for authenticated encryption — tamper-evident, fast.

The nsec is the root secret. Without it, vault contents are gibberish.
The Authority, Neon infrastructure, and database admins see only ciphertext.

Usage:
    cipher = VaultCipher(nsec_hex="<operator's private key hex>")
    ciphertext = cipher.encrypt("plaintext ledger json")
    plaintext = cipher.decrypt(ciphertext)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os


# AES-256-GCM via cryptography library (already a transitive dep)
def _get_aesgcm():
    """Lazy import to avoid hard dependency at module level."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM


class VaultCipher:
    """Symmetric encryption using a key derived from the operator's nsec."""

    _SALT = b"tollbooth-vault-v1"
    _INFO_LEDGER = b"vault-ledger-encryption"
    _NONCE_SIZE = 12  # AES-GCM standard nonce

    def __init__(self, nsec_hex: str) -> None:
        """Derive a 256-bit encryption key from the nsec via HKDF-SHA256.

        Args:
            nsec_hex: The operator's private key in hex (64 chars).
        """
        ikm = bytes.fromhex(nsec_hex)
        # HKDF-Extract
        prk = hmac.new(self._SALT, ikm, hashlib.sha256).digest()
        # HKDF-Expand (single round — 32 bytes < hash output)
        expanded = hmac.new(prk, self._INFO_LEDGER + b"\x01", hashlib.sha256).digest()
        self._key = expanded  # 32 bytes = AES-256

    def encrypt(self, plaintext: str, aad: str = "") -> str:
        """Encrypt a string. Returns base64-encoded (nonce + ciphertext + tag).

        The nonce is randomly generated per encryption — same plaintext
        produces different ciphertext each time.

        ``aad`` (Additional Authenticated Data) binds the ciphertext to
        its context (e.g., vault key). Prevents cross-entry ciphertext
        swapping. Must be the same on decrypt.
        """
        AESGCM = _get_aesgcm()
        nonce = os.urandom(self._NONCE_SIZE)
        aes = AESGCM(self._key)
        aad_bytes = aad.encode("utf-8") if aad else None
        ct = aes.encrypt(nonce, plaintext.encode("utf-8"), aad_bytes)
        payload = nonce + ct
        return base64.b64encode(payload).decode("ascii")

    def decrypt(self, ciphertext_b64: str, aad: str = "") -> str:
        """Decrypt a base64-encoded payload. Raises on tamper or wrong key.

        ``aad`` must match the value used during encryption.
        For backward compatibility, empty ``aad`` matches ciphertext
        encrypted without AAD.
        """
        AESGCM = _get_aesgcm()
        payload = base64.b64decode(ciphertext_b64)
        if len(payload) < self._NONCE_SIZE + 16:
            raise ValueError("Ciphertext too short")
        nonce = payload[:self._NONCE_SIZE]
        ct = payload[self._NONCE_SIZE:]
        aes = AESGCM(self._key)
        aad_bytes = aad.encode("utf-8") if aad else None
        try:
            plaintext = aes.decrypt(nonce, ct, aad_bytes)
        except Exception:
            if aad:
                # Retry without AAD for backward compatibility with
                # ciphertext encrypted before AAD was introduced
                plaintext = aes.decrypt(nonce, ct, None)
            else:
                raise
        return plaintext.decode("utf-8")

    def is_encrypted(self, value: str) -> bool:
        """Heuristic: check if a value looks like our encrypted format.

        Encrypted values are base64 and decode to >= 28 bytes (12 nonce + 16 tag).
        Plain JSON starts with '{' or '['.
        """
        if not value:
            return False
        if value.startswith(("{", "[")):
            return False
        try:
            raw = base64.b64decode(value)
            return len(raw) >= self._NONCE_SIZE + 16
        except Exception:
            return False
