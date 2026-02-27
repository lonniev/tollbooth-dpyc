"""Tests for NIP-44v2 encryption module."""

from __future__ import annotations

import base64

import pytest
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.nip44 import (
    _calc_padded_len,
    _get_conversation_key,
    _pad,
    _unpad,
    decrypt,
    encrypt,
)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


class TestCalcPaddedLen:
    def test_short_texts_pad_to_32(self) -> None:
        for n in (1, 15, 31, 32):
            assert _calc_padded_len(n) == 32

    def test_power_of_two_buckets(self) -> None:
        assert _calc_padded_len(33) == 64
        assert _calc_padded_len(64) == 64
        assert _calc_padded_len(65) == 128
        assert _calc_padded_len(128) == 128
        assert _calc_padded_len(129) == 256
        assert _calc_padded_len(256) == 256

    def test_chunk_based_above_256(self) -> None:
        # 257 → next_power=512, chunk=64, ceil(257/64)*64 = 5*64 = 320
        assert _calc_padded_len(257) == 320
        # 512 → next_power=512, chunk=64, 512/64 * 64 = 512
        assert _calc_padded_len(512) == 512

    def test_always_at_least_input_size(self) -> None:
        for n in range(1, 1000):
            assert _calc_padded_len(n) >= n


class TestPadUnpad:
    def test_round_trip(self) -> None:
        for text in (b"x", b"hello", b"a" * 100, b"z" * 1000):
            padded = _pad(text)
            assert _unpad(padded) == text

    def test_pad_adds_length_prefix(self) -> None:
        padded = _pad(b"hello")
        # First 2 bytes: big-endian length (5)
        assert padded[0] == 0
        assert padded[1] == 5
        # Followed by the plaintext
        assert padded[2:7] == b"hello"
        # Total padded length = 2 + padded_len(5) = 2 + 32 = 34
        assert len(padded) == 34

    def test_padding_is_zero_filled(self) -> None:
        padded = _pad(b"hi")
        # After prefix (2 bytes) + plaintext (2 bytes) = zeros to fill to 32
        tail = padded[4:]
        assert tail == b"\x00" * len(tail)

    def test_empty_plaintext_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside valid range"):
            _pad(b"")

    def test_oversized_plaintext_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside valid range"):
            _pad(b"x" * 65536)

    def test_unpad_short_data_rejected(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            _unpad(b"\x00")

    def test_unpad_bad_length_rejected(self) -> None:
        # Declare length 100 but only provide 10 bytes
        bad = b"\x00\x64" + b"x" * 10
        with pytest.raises(ValueError, match="shorter than declared"):
            _unpad(bad)

    def test_unpad_nonzero_padding_rejected(self) -> None:
        padded = _pad(b"hi")
        # Corrupt a padding byte
        corrupted = padded[:-1] + b"\xff"
        with pytest.raises(ValueError, match="Non-zero padding"):
            _unpad(corrupted)


# ---------------------------------------------------------------------------
# Conversation key
# ---------------------------------------------------------------------------


class TestConversationKey:
    def test_symmetric(self) -> None:
        """Both parties derive the same conversation key."""
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ck1 = _get_conversation_key(sk1.hex(), sk2.public_key.hex())
        ck2 = _get_conversation_key(sk2.hex(), sk1.public_key.hex())
        assert ck1 == ck2

    def test_different_pairs_differ(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()
        sk3 = PrivateKey()

        ck12 = _get_conversation_key(sk1.hex(), sk2.public_key.hex())
        ck13 = _get_conversation_key(sk1.hex(), sk3.public_key.hex())
        assert ck12 != ck13

    def test_deterministic(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ck_a = _get_conversation_key(sk1.hex(), sk2.public_key.hex())
        ck_b = _get_conversation_key(sk1.hex(), sk2.public_key.hex())
        assert ck_a == ck_b


# ---------------------------------------------------------------------------
# Encrypt / Decrypt
# ---------------------------------------------------------------------------


class TestEncryptDecrypt:
    def test_round_trip_short(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        msg = "hello"
        ct = encrypt(msg, sk1.hex(), sk2.public_key.hex())
        pt = decrypt(ct, sk2.hex(), sk1.public_key.hex())
        assert pt == msg

    def test_round_trip_json(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        msg = '{"user_id":"npub1abc","balance":5220,"deposited":8000}'
        ct = encrypt(msg, sk1.hex(), sk2.public_key.hex())
        pt = decrypt(ct, sk2.hex(), sk1.public_key.hex())
        assert pt == msg

    def test_round_trip_long(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        msg = "x" * 10000
        ct = encrypt(msg, sk1.hex(), sk2.public_key.hex())
        pt = decrypt(ct, sk2.hex(), sk1.public_key.hex())
        assert pt == msg

    def test_ciphertext_is_base64(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ct = encrypt("test", sk1.hex(), sk2.public_key.hex())
        # Should not raise
        payload = base64.b64decode(ct)
        # Version byte should be 2
        assert payload[0] == 2

    def test_plaintext_not_in_ciphertext(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        msg = "sensitive_balance_data_12345"
        ct = encrypt(msg, sk1.hex(), sk2.public_key.hex())
        payload = base64.b64decode(ct)
        assert msg.encode() not in payload

    def test_wrong_key_fails(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()
        sk3 = PrivateKey()

        ct = encrypt("secret", sk1.hex(), sk2.public_key.hex())
        with pytest.raises(ValueError, match="decryption failed"):
            decrypt(ct, sk3.hex(), sk1.public_key.hex())

    def test_tampered_ciphertext_fails(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ct = encrypt("secret", sk1.hex(), sk2.public_key.hex())
        payload = bytearray(base64.b64decode(ct))
        # Flip a byte in the ciphertext portion
        payload[-5] ^= 0xFF
        tampered = base64.b64encode(bytes(payload)).decode()
        with pytest.raises(ValueError, match="decryption failed"):
            decrypt(tampered, sk2.hex(), sk1.public_key.hex())

    def test_wrong_version_rejected(self) -> None:
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ct = encrypt("test", sk1.hex(), sk2.public_key.hex())
        payload = bytearray(base64.b64decode(ct))
        payload[0] = 99  # wrong version
        bad = base64.b64encode(bytes(payload)).decode()
        with pytest.raises(ValueError, match="Unsupported NIP-44 version"):
            decrypt(bad, sk2.hex(), sk1.public_key.hex())

    def test_payload_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            decrypt(base64.b64encode(b"\x02" + b"\x00" * 10).decode(), "aa" * 32, "bb" * 32)

    def test_decrypt_unpadded_base64(self) -> None:
        """Decrypt tolerates base64 with stripped '=' padding (Primal, etc.)."""
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ct = encrypt("hello world", sk1.hex(), sk2.public_key.hex())
        # Strip trailing '=' to simulate what Primal sends
        ct_stripped = ct.rstrip("=")
        assert ct_stripped != ct  # sanity: we actually stripped something
        pt = decrypt(ct_stripped, sk2.hex(), sk1.public_key.hex())
        assert pt == "hello world"

    def test_each_encryption_differs(self) -> None:
        """Random nonce means same plaintext encrypts differently each time."""
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        ct1 = encrypt("same", sk1.hex(), sk2.public_key.hex())
        ct2 = encrypt("same", sk1.hex(), sk2.public_key.hex())
        assert ct1 != ct2

        # But both decrypt to the same plaintext
        assert decrypt(ct1, sk2.hex(), sk1.public_key.hex()) == "same"
        assert decrypt(ct2, sk2.hex(), sk1.public_key.hex()) == "same"
