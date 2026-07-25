"""Tests for NIP-44v2 encryption module."""

from __future__ import annotations

import base64

import pytest
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.nip44 import (
    _calc_padded_len,
    _get_conversation_key,
    _get_message_keys,
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
            decrypt(base64.b64encode(b"\x02" + b"\x00" * 60).decode(), "aa" * 32, "bb" * 32)

    def test_decrypt_unpadded_base64(self) -> None:
        """Decrypt tolerates base64 with stripped '=' padding (Primal, etc.)."""
        sk1 = PrivateKey()
        sk2 = PrivateKey()

        # Use 33-byte plaintext → padded_len=64 → payload=131 bytes → 131%3=2 → base64 has '='
        msg = "x" * 33
        ct = encrypt(msg, sk1.hex(), sk2.public_key.hex())
        # Strip trailing '=' to simulate what Primal sends
        ct_stripped = ct.rstrip("=")
        assert ct_stripped != ct  # sanity: we actually stripped something
        pt = decrypt(ct_stripped, sk2.hex(), sk1.public_key.hex())
        assert pt == msg

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


# ---------------------------------------------------------------------------
# Official NIP-44 test vectors (from paulmillr/nip44)
# ---------------------------------------------------------------------------


class TestConversationKeyVectors:
    """Validate conversation key derivation against official NIP-44 vectors."""

    VECTORS = [  # noqa: RUF012
        (
            "315e59ff51cb9209768cf7da80791ddcaae56ac9775eb25b6dee1234bc5d2268",
            "c2f9d9948dc8c7c38321e4b85c8558872eafa0641cd269db76848a6073e69133",
            "3dfef0ce2a4d80a25e7a328accf73448ef67096f65f79588e358d9a0eb9013f1",
        ),
        (
            "a1e37752c9fdc1273be53f68c5f74be7c8905728e8de75800b94262f9497c86e",
            "03bb7947065dde12ba991ea045132581d0954f042c84e06d8c00066e23c1a800",
            "4d14f36e81b8452128da64fe6f1eae873baae2f444b02c950b90e43553f2178b",
        ),
        (
            "98a5902fd67518a0c900f0fb62158f278f94a21d6f9d33d30cd3091195500311",
            "aae65c15f98e5e677b5050de82e3aba47a6fe49b3dab7863cf35d9478ba9f7d1",
            "9c00b769d5f54d02bf175b7284a1cbd28b6911b06cda6666b2243561ac96bad7",
        ),
    ]

    @pytest.mark.parametrize("sec1,pub2,expected", VECTORS)
    def test_conversation_key_vector(self, sec1: str, pub2: str, expected: str) -> None:
        ck = _get_conversation_key(sec1, pub2)
        assert ck.hex() == expected


class TestMessageKeyVectors:
    """Validate message key derivation against official NIP-44 vectors."""

    def test_message_keys_vector(self) -> None:
        conv_key = bytes.fromhex(
            "a1a3d60f3470a8612633924e91febf96dc5366ce130f658b1f0fc652c20b3b54"
        )
        nonce = bytes.fromhex(
            "e1e6f880560d6d149ed83dcc7e5861ee62a5ee051f7fde9975fe5d25d2a02d72"
        )
        chacha_key, chacha_nonce, hmac_key = _get_message_keys(conv_key, nonce)
        assert chacha_key.hex() == "f145f3bed47cb70dbeaac07f3a3fe683e822b3715edb7c4fe310829014ce7d76"
        assert chacha_nonce.hex() == "c4ad129bb01180c0933a160c"
        assert hmac_key.hex() == "027c1db445f05e2eee864a0975b0ddef5b7110583c8c192de3732571ca5838c4"


class TestEncryptDecryptVectors:
    """Validate encrypt/decrypt against official NIP-44 vector (sec1=0x01, sec2=0x02)."""

    SEC1 = "0000000000000000000000000000000000000000000000000000000000000001"
    SEC2 = "0000000000000000000000000000000000000000000000000000000000000002"
    CONV_KEY = "c41c775356fd92eadc63ff5a0dc1da211b268cbea22316767095b2871ea1412d"
    NONCE_HEX = "0000000000000000000000000000000000000000000000000000000000000001"
    PLAINTEXT = "a"
    EXPECTED_PAYLOAD_B64 = (
        "AgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABee0G5VSK0/9YypIObAtDKfYEAjD35uVkHyB0F4DwrcNaCXlCWZKaArsGrY6M9wnuTMxWfp1RTN9Xga8no+kF5Vsb"
    )

    def test_decrypt_official_vector(self) -> None:
        """Decrypt the official test payload and verify plaintext."""
        from coincurve import PrivateKey as CoinPriv

        # sec2 is the recipient — derive pub1 from sec1
        pub1 = CoinPriv(bytes.fromhex(self.SEC1)).public_key.format(compressed=True)[1:].hex()
        pt = decrypt(self.EXPECTED_PAYLOAD_B64, self.SEC2, pub1)
        assert pt == self.PLAINTEXT

    def test_encrypt_matches_official_vector(self) -> None:
        """Encrypt with the official nonce and verify the payload matches exactly."""
        import os
        from unittest.mock import patch

        from coincurve import PrivateKey as CoinPriv

        pub2 = CoinPriv(bytes.fromhex(self.SEC2)).public_key.format(compressed=True)[1:].hex()

        fixed_nonce = bytes.fromhex(self.NONCE_HEX)
        with patch.object(os, "urandom", return_value=fixed_nonce):
            payload_b64 = encrypt(self.PLAINTEXT, self.SEC1, pub2)

        assert payload_b64 == self.EXPECTED_PAYLOAD_B64
