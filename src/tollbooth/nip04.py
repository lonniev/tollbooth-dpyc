"""NIP-04 decryption — legacy AES-256-CBC encrypted direct messages.

Implements the NIP-04 decryption scheme used by older Nostr clients for
kind 4 encrypted DMs.  The content format is ``<base64_ciphertext>?iv=<base64_iv>``.

The shared secret is derived via secp256k1 ECDH (same curve as NIP-44), but
the symmetric cipher is AES-256-CBC instead of ChaCha20-Poly1305.

Dependencies: ``coincurve`` (secp256k1 ECDH) and ``cryptography``
(AES-256-CBC) — both already transitive deps of pynostr.

Reference: https://github.com/nostr-protocol/nips/blob/master/04.md
"""

from __future__ import annotations

import base64

from coincurve import PrivateKey as _CoinPrivateKey  # type: ignore[import-untyped]
from coincurve import PublicKey as _CoinPublicKey  # type: ignore[import-untyped]
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.padding import PKCS7


def _get_shared_secret(private_key_hex: str, public_key_hex: str) -> bytes:
    """Derive NIP-04 shared secret via secp256k1 ECDH.

    Returns the SHA-256 hash of the x-coordinate of the ECDH shared point,
    used as the AES-256-CBC key.
    """
    sk = _CoinPrivateKey(bytes.fromhex(private_key_hex))
    pk = _CoinPublicKey(b"\x02" + bytes.fromhex(public_key_hex))
    shared_point = pk.multiply(sk.secret)
    shared_x = shared_point.format(compressed=False)[1:33]
    return shared_x


def encrypt(
    plaintext: str,
    private_key_hex: str,
    public_key_hex: str,
) -> str:
    """Encrypt a message using NIP-04 AES-256-CBC.

    Args:
        plaintext: UTF-8 string to encrypt.
        private_key_hex: Sender's 32-byte private key as hex.
        public_key_hex: Recipient's 32-byte x-only public key as hex.

    Returns:
        NIP-04 format ``<base64_ciphertext>?iv=<base64_iv>``.
    """
    import os

    shared_secret = _get_shared_secret(private_key_hex, public_key_hex)
    iv = os.urandom(16)

    padder = PKCS7(128).padder()
    padded = padder.update(plaintext.encode("utf-8")) + padder.finalize()

    cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    ct_b64 = base64.b64encode(ciphertext).decode()
    iv_b64 = base64.b64encode(iv).decode()
    return f"{ct_b64}?iv={iv_b64}"


def decrypt(
    ciphertext_with_iv: str,
    private_key_hex: str,
    public_key_hex: str,
) -> str:
    """Decrypt a NIP-04 message.

    Args:
        ciphertext_with_iv: NIP-04 format ``<base64_ciphertext>?iv=<base64_iv>``.
        private_key_hex: Recipient's 32-byte private key as hex.
        public_key_hex: Sender's 32-byte x-only public key as hex.

    Returns:
        Decrypted UTF-8 plaintext string.

    Raises:
        ValueError: On invalid format, decryption failure, or padding error.
    """
    if "?iv=" not in ciphertext_with_iv:
        raise ValueError(
            "Invalid NIP-04 format: expected <base64>?iv=<base64>"
        )

    parts = ciphertext_with_iv.split("?iv=", 1)
    try:
        # Normalize base64 padding — some Nostr clients strip trailing '='
        ct_b64 = parts[0] + "=" * (-len(parts[0]) % 4)
        iv_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
        ciphertext = base64.b64decode(ct_b64)
        iv = base64.b64decode(iv_b64)
    except Exception as e:
        raise ValueError(f"NIP-04 base64 decode failed: {e}") from e

    if len(iv) != 16:
        raise ValueError(f"NIP-04 IV must be 16 bytes, got {len(iv)}")

    shared_secret = _get_shared_secret(private_key_hex, public_key_hex)

    try:
        cipher = Cipher(algorithms.AES(shared_secret), modes.CBC(iv))
        decryptor = cipher.decryptor()
        padded = decryptor.update(ciphertext) + decryptor.finalize()

        unpadder = PKCS7(128).unpadder()
        plaintext_bytes = unpadder.update(padded) + unpadder.finalize()
        return plaintext_bytes.decode("utf-8")
    except Exception as e:
        raise ValueError(f"NIP-04 decryption failed: {e}") from e
