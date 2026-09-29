"""Secure Courier proof grant — kind-30080 Operator-signed bearer (#267).

The challenge nonce selects a grant; it never unlocks one. These tests pin:
  - sign/verify round-trip and rejection reasons
  - require_proof accepts a grant envelope and refuses a bare phrase
  - wrong nonce / wrong npub / expired / revoked are distinct reasons
  - Tactic-2 kind-27235 behaviour is unchanged
"""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.constants import ErrorCode
from tollbooth.identity_credential import (
    IDENTITY_CREDENTIAL_KIND,
    PROOF_GRANT_TAG,
    IdentityCredentialError,
    sign_proof_grant,
    verify_proof_grant,
)
from tollbooth.identity_proof import PROOF_EVENT_KIND, require_proof
from tollbooth.proven_npub import ProvenNpubCache


@pytest.fixture()
def operator():
    pk = PrivateKey()
    return pk, pk.bech32(), pk.public_key.hex(), pk.public_key.bech32()


@pytest.fixture()
def patron():
    pk = PrivateKey()
    return pk, pk.public_key.bech32(), pk.public_key.hex()


def _challenge(nonce: str = "bold-hawk-42") -> tuple[str, str]:
    return nonce, hashlib.sha256(nonce.encode()).hexdigest()


def _envelope(grant_json: str, nonce: str) -> str:
    return json.dumps(
        {"grant": json.loads(grant_json), "nonce": nonce},
        separators=(",", ":"),
    )


def _signed_27235(pk: PrivateKey, tool: str = "tool") -> str:
    event = Event(
        kind=PROOF_EVENT_KIND,
        content="",
        tags=[["u", tool], ["nonce", "a1"]],
        pubkey=pk.public_key.hex(),
    )
    event.created_at = int(time.time())
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


# ---------------------------------------------------------------------------
# sign + verify
# ---------------------------------------------------------------------------


class TestSignVerifyGrant:
    def test_round_trip(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        nonce, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig-abc",
            patron_event_id="eid-1",
            ttl_seconds=3600,
        )
        claims = verify_proof_grant(
            grant,
            expected_patron_npub=patron_npub,
            challenge_hash=ch,
            expected_operator_hex=op_hex,
        )
        assert claims["patron_npub"] == patron_npub
        assert claims["challenge"] == ch
        assert claims["patron_sig"] == "sig-abc"
        assert claims["jti"]
        assert claims["expiration"] > time.time()
        assert json.loads(grant)["kind"] == IDENTITY_CREDENTIAL_KIND
        tags = {t[0]: t[1] for t in json.loads(grant)["tags"]}
        assert tags["t"] == PROOF_GRANT_TAG
        assert "challenge" in tags
        # Raw nonce never appears in the grant.
        assert nonce not in grant

    def test_wrong_nonce_challenge_mismatch(self, operator, patron):
        _, op_nsec, _, _ = operator
        _, patron_npub, _ = patron
        _, ch = _challenge("bold-hawk-42")
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        with pytest.raises(IdentityCredentialError, match="challenge_mismatch"):
            verify_proof_grant(
                grant,
                expected_patron_npub=patron_npub,
                challenge_hash=hashlib.sha256(b"other-nonce-1").hexdigest(),
            )

    def test_wrong_npub(self, operator, patron):
        _, op_nsec, _, _ = operator
        _, patron_npub, _ = patron
        other = PrivateKey().public_key.bech32()
        _, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        with pytest.raises(IdentityCredentialError, match="wrong_npub"):
            verify_proof_grant(
                grant, expected_patron_npub=other, challenge_hash=ch,
            )

    def test_expired(self, operator, patron):
        _, op_nsec, _, _ = operator
        _, patron_npub, _ = patron
        _, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=-10,
        )
        with pytest.raises(IdentityCredentialError, match="expired"):
            verify_proof_grant(
                grant, expected_patron_npub=patron_npub, challenge_hash=ch,
            )

    def test_operator_mismatch(self, operator, patron):
        _, op_nsec, _, _ = operator
        _, patron_npub, _ = patron
        _, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        other_hex = PrivateKey().public_key.hex()
        with pytest.raises(IdentityCredentialError, match="operator_mismatch"):
            verify_proof_grant(
                grant,
                expected_patron_npub=patron_npub,
                challenge_hash=ch,
                expected_operator_hex=other_hex,
            )


# ---------------------------------------------------------------------------
# require_proof gate
# ---------------------------------------------------------------------------


class TestRequireProofGrant:
    @pytest.mark.asyncio
    async def test_phrase_alone_refused(self, patron):
        _, patron_npub, _ = patron
        cache = SimpleNamespace(
            is_proven=AsyncMock(return_value=True),
            is_grant_revoked=AsyncMock(return_value=False),
        )
        r = await require_proof(
            patron_npub, "bold-hawk-42", "tool", proven_cache=cache,
        )
        assert r["error_code"] == ErrorCode.PROOF_REFRESH_NEEDED

    @pytest.mark.asyncio
    async def test_grant_envelope_accepted(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        nonce, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        cache = SimpleNamespace(is_grant_revoked=AsyncMock(return_value=False))
        r = await require_proof(
            patron_npub,
            _envelope(grant, nonce),
            "tool",
            proven_cache=cache,
            operator_hex=op_hex,
        )
        assert r is None

    @pytest.mark.asyncio
    async def test_grant_wrong_nonce_reason(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        _nonce, ch = _challenge("bold-hawk-42")
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        r = await require_proof(
            patron_npub,
            _envelope(grant, "calm-reef-77"),
            "tool",
            operator_hex=op_hex,
        )
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "challenge_mismatch"

    @pytest.mark.asyncio
    async def test_grant_wrong_npub_reason(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        other = PrivateKey().public_key.bech32()
        nonce, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        r = await require_proof(
            other, _envelope(grant, nonce), "tool", operator_hex=op_hex,
        )
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "wrong_npub"

    @pytest.mark.asyncio
    async def test_grant_expired_reason(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        nonce, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=-5,
        )
        r = await require_proof(
            patron_npub, _envelope(grant, nonce), "tool", operator_hex=op_hex,
        )
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "expired"

    @pytest.mark.asyncio
    async def test_grant_revoked_reason(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        nonce, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        cache = SimpleNamespace(is_grant_revoked=AsyncMock(return_value=True))
        r = await require_proof(
            patron_npub,
            _envelope(grant, nonce),
            "tool",
            proven_cache=cache,
            operator_hex=op_hex,
        )
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "revoked"

    @pytest.mark.asyncio
    async def test_bare_grant_without_nonce_refused(self, operator, patron):
        _, op_nsec, op_hex, _ = operator
        _, patron_npub, _ = patron
        _, ch = _challenge()
        grant = sign_proof_grant(
            patron_npub=patron_npub,
            operator_nsec=op_nsec,
            challenge_hash=ch,
            patron_sig="sig",
            ttl_seconds=3600,
        )
        r = await require_proof(
            patron_npub, grant, "tool", operator_hex=op_hex,
        )
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "missing_nonce"

    @pytest.mark.asyncio
    async def test_tactic2_inline_unchanged(self):
        pk = PrivateKey()
        r = await require_proof(
            pk.public_key.bech32(), _signed_27235(pk, "tool"), "tool",
        )
        assert r is None


# ---------------------------------------------------------------------------
# revocation store
# ---------------------------------------------------------------------------


class TestGrantRevocation:
    @pytest.mark.asyncio
    async def test_revoke_and_check(self):
        cache = ProvenNpubCache(ttl_seconds=3600)
        assert not await cache.is_grant_revoked("jti-1")
        await cache.revoke_grant("jti-1")
        assert await cache.is_grant_revoked("jti-1")
