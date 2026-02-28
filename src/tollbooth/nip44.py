"""NIP-44v2 encryption — ECDH + HKDF + ChaCha20 stream cipher + HMAC-SHA256.

Implements the NIP-44 version 2 payload encryption scheme for Nostr events.
Used by the audit publisher to encrypt patron balance data so only the
patron's nsec can decrypt it.

Payload format: version(1) + nonce(32) + ciphertext(N) + hmac(32)

Dependencies: ``coincurve`` (secp256k1 ECDH) and ``cryptography``
(HKDF-SHA256, ChaCha20) — both already transitive deps of pynostr.

Reference: https://github.com/nostr-protocol/nips/blob/master/44.md
"""

from __future__ import annotations

import hashlib
import hmac
import os
import struct

from coincurve import PrivateKey as _CoinPrivateKey  # type: ignore[import-untyped]
from coincurve import PublicKey as _CoinPublicKey  # type: ignore[import-untyped]
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDFExpand

# NIP-44v2 constants
_VERSION = 2
_MIN_PLAINTEXT_SIZE = 1
_MAX_PLAINTEXT_SIZE = 65535


def _calc_padded_len(unpadded_len: int) -> int:
    """Calculate NIP-44v2 padded length.

    Padding ensures ciphertext length reveals minimal information about
    plaintext length. Uses a power-of-two bucket scheme.
    """
    if unpadded_len <= 32:
        return 32
    next_power = 1
    while next_power < unpadded_len:
        next_power *= 2
    if next_power <= 256:
        return next_power
    chunk = next_power // 8
    return chunk * ((unpadded_len + chunk - 1) // chunk)


def _pad(plaintext: bytes) -> bytes:
    """Pad plaintext per NIP-44v2: 2-byte BE length prefix + zero-padding."""
    unpadded_len = len(plaintext)
    if unpadded_len < _MIN_PLAINTEXT_SIZE or unpadded_len > _MAX_PLAINTEXT_SIZE:
        raise ValueError(
            f"Plaintext length {unpadded_len} outside valid range "
            f"[{_MIN_PLAINTEXT_SIZE}, {_MAX_PLAINTEXT_SIZE}]"
        )
    padded_len = _calc_padded_len(unpadded_len)
    prefix = struct.pack(">H", unpadded_len)
    padding = b"\x00" * (padded_len - unpadded_len)
    return prefix + plaintext + padding


def _unpad(padded: bytes) -> bytes:
    """Remove NIP-44v2 padding: read 2-byte BE length, extract plaintext."""
    if len(padded) < 2:
        raise ValueError("Padded data too short")
    (unpadded_len,) = struct.unpack(">H", padded[:2])
    if unpadded_len < _MIN_PLAINTEXT_SIZE or unpadded_len > _MAX_PLAINTEXT_SIZE:
        raise ValueError(f"Invalid unpadded length: {unpadded_len}")
    if 2 + unpadded_len > len(padded):
        raise ValueError("Padded data shorter than declared length")
    # Verify padding is all zeros
    tail = padded[2 + unpadded_len :]
    if tail != b"\x00" * len(tail):
        raise ValueError("Non-zero padding bytes detected")
    return padded[2 : 2 + unpadded_len]


def _get_conversation_key(
    private_key_hex: str, public_key_hex: str,
) -> bytes:
    """Derive NIP-44v2 conversation key via secp256k1 ECDH + HKDF-extract.

    1. Scalar-multiply recipient pubkey by sender privkey → shared point.
    2. Take x-coordinate (32 bytes) of the shared point.
    3. HKDF-extract(salt="nip44-v2", IKM=shared_x) → 32-byte conversation key.

    Args:
        private_key_hex: 32-byte private key as hex string.
        public_key_hex: 32-byte x-only public key as hex string.

    Returns:
        32-byte conversation key.
    """
    sk = _CoinPrivateKey(bytes.fromhex(private_key_hex))
    # x-only pubkey needs 02 prefix for coincurve (assume even Y)
    pk = _CoinPublicKey(b"\x02" + bytes.fromhex(public_key_hex))

    # Scalar multiplication → shared point, then extract x-coordinate
    shared_point = pk.multiply(sk.secret)
    shared_x = shared_point.format(compressed=False)[1:33]

    # HKDF-extract is HMAC-SHA256(salt, IKM)
    return hmac.new(b"nip44-v2", shared_x, hashlib.sha256).digest()


def _get_message_keys(
    conversation_key: bytes, nonce: bytes,
) -> tuple[bytes, bytes, bytes]:
    """Derive per-message chacha_key, chacha_nonce, and hmac_key via HKDF-expand.

    HKDF-expand(PRK=conversation_key, info=nonce, L=76)
    → chacha_key(32) + chacha_nonce(12) + hmac_key(32)

    Returns:
        (chacha_key: 32 bytes, chacha_nonce: 12 bytes, hmac_key: 32 bytes)
    """
    expanded = HKDFExpand(
        algorithm=SHA256(),
        length=76,
        info=nonce,
    ).derive(conversation_key)

    chacha_key = expanded[:32]
    chacha_nonce = expanded[32:44]
    hmac_key = expanded[44:76]
    return chacha_key, chacha_nonce, hmac_key


def encrypt(
    plaintext: str,
    private_key_hex: str,
    public_key_hex: str,
) -> str:
    """Encrypt a plaintext string per NIP-44v2.

    Args:
        plaintext: UTF-8 string to encrypt.
        private_key_hex: Sender's 32-byte private key as hex.
        public_key_hex: Recipient's 32-byte x-only public key as hex.

    Returns:
        Base64-encoded NIP-44v2 payload: version(1) + nonce(32) + ciphertext + hmac(32).
    """
    import base64

    plaintext_bytes = plaintext.encode("utf-8")
    padded = _pad(plaintext_bytes)

    conversation_key = _get_conversation_key(private_key_hex, public_key_hex)
    nonce = os.urandom(32)
    chacha_key, chacha_nonce, hmac_key = _get_message_keys(conversation_key, nonce)

    # ChaCha20 stream cipher (NOT AEAD). NIP-44 uses a 16-byte nonce:
    # 4 zero bytes (counter) + 12-byte chacha_nonce.
    nonce16 = b"\x00" * 4 + chacha_nonce
    cipher = Cipher(algorithms.ChaCha20(chacha_key, nonce16), mode=None)
    ciphertext = cipher.encryptor().update(padded)

    # Separate HMAC-SHA256 over (nonce || ciphertext)
    mac = hmac.new(hmac_key, nonce + ciphertext, hashlib.sha256).digest()

    # NIP-44v2 payload: version(1) + nonce(32) + ciphertext(N) + mac(32)
    payload = bytes([_VERSION]) + nonce + ciphertext + mac
    return base64.b64encode(payload).decode("ascii")


def decrypt(
    payload_b64: str,
    private_key_hex: str,
    public_key_hex: str,
) -> str:
    """Decrypt a NIP-44v2 base64 payload.

    Args:
        payload_b64: Base64-encoded NIP-44v2 payload.
        private_key_hex: Recipient's 32-byte private key as hex.
        public_key_hex: Sender's 32-byte x-only public key as hex.

    Returns:
        Decrypted UTF-8 plaintext string.

    Raises:
        ValueError: On invalid version, HMAC failure, or padding error.
    """
    import base64

    # Normalize base64 padding — some Nostr clients (Primal) strip trailing '='
    payload_b64 += "=" * (-len(payload_b64) % 4)
    payload = base64.b64decode(payload_b64)
    # Minimum: 1 version + 32 nonce + 34 min ciphertext (2 pad-prefix + 32 padded) + 32 mac = 99
    if len(payload) < 99:
        raise ValueError("NIP-44 payload too short")

    version = payload[0]
    if version != _VERSION:
        raise ValueError(f"Unsupported NIP-44 version: {version}")

    nonce = payload[1:33]
    ciphertext = payload[33:-32]
    mac = payload[-32:]

    conversation_key = _get_conversation_key(private_key_hex, public_key_hex)
    chacha_key, chacha_nonce, hmac_key = _get_message_keys(conversation_key, nonce)

    # Verify HMAC-SHA256 (constant-time comparison)
    expected_mac = hmac.new(hmac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected_mac):
        raise ValueError("NIP-44 decryption failed: HMAC verification failed")

    # ChaCha20 stream cipher decrypt
    nonce16 = b"\x00" * 4 + chacha_nonce
    cipher = Cipher(algorithms.ChaCha20(chacha_key, nonce16), mode=None)
    padded = cipher.decryptor().update(ciphertext)

    plaintext_bytes = _unpad(padded)
    return plaintext_bytes.decode("utf-8")
