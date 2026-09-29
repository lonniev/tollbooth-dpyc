"""Secure Courier proof grant — kind-30080 Operator-signed bearer (#267).

The challenge nonce selects a grant; it never unlocks one. These tests pin:
  - sign/verify round-trip and every rejection reason
  - require_proof accepts a grant envelope and refuses a bare phrase
  - the gate FAILS CLOSED: unknown issuer, foreign issuer, unreadable
    revocation store, and a patron's forget_credentials all refuse
  - Tactic-2 kind-27235 behaviour is unchanged
"""

from __future__ import annotations

import hashlib
import json
import time
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
from tollbooth.proven_npub import ProofGrantRevocations, RevocationStoreUnavailable

NONCE = "bold-hawk-42"
CHALLENGE = hashlib.sha256(NONCE.encode()).hexdigest()


@pytest.fixture()
def operator():
    pk = PrivateKey()
    return pk.bech32(), pk.public_key.hex()


@pytest.fixture()
def patron():
    return PrivateKey().public_key.bech32()


def _grant(op_nsec: str, patron_npub: str, *, ttl: int = 3600, challenge: str = CHALLENGE) -> str:
    return sign_proof_grant(
        patron_npub=patron_npub,
        operator_nsec=op_nsec,
        challenge_hash=challenge,
        patron_sig="ab" * 64,
        patron_event_id="seal-1",
        ttl_seconds=ttl,
    )


def _envelope(grant_json: str, nonce: str = NONCE) -> str:
    return json.dumps({"grant": json.loads(grant_json), "nonce": nonce}, separators=(",", ":"))


def _signed_27235(pk: PrivateKey, tool: str = "tool") -> str:
    event = Event(
        kind=PROOF_EVENT_KIND, content="", tags=[["u", tool], ["nonce", "a1"]],
        pubkey=pk.public_key.hex(),
    )
    event.created_at = int(time.time())
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


class _NoRevocations:
    async def is_revoked(self, npub: str, created_at: int) -> bool:
        return False


# ---------------------------------------------------------------------------
# sign + verify
# ---------------------------------------------------------------------------


class TestSignVerifyGrant:
    def test_round_trip(self, operator, patron):
        op_nsec, op_hex = operator
        grant = _grant(op_nsec, patron)
        claims = verify_proof_grant(
            grant, expected_patron_npub=patron, challenge_hash=CHALLENGE,
            expected_operator_hex=op_hex,
        )
        assert claims["patron_npub"] == patron
        assert claims["operator_hex"] == op_hex
        assert claims["challenge"] == CHALLENGE
        assert claims["patron_sig"] == "ab" * 64
        assert claims["patron_event_id"] == "seal-1"
        assert claims["jti"]
        assert claims["created_at"] <= time.time() <= claims["expiration"]
        raw = json.loads(grant)
        assert raw["kind"] == IDENTITY_CREDENTIAL_KIND
        assert {t[0]: t[1] for t in raw["tags"]}["t"] == PROOF_GRANT_TAG
        assert NONCE not in grant  # the raw nonce never appears

    def test_patron_signature_and_event_id_are_required(self, operator, patron):
        op_nsec, _ = operator
        with pytest.raises(IdentityCredentialError, match="patron_sig"):
            sign_proof_grant(patron_npub=patron, operator_nsec=op_nsec,
                             challenge_hash=CHALLENGE, patron_sig="", patron_event_id="x")
        with pytest.raises(IdentityCredentialError, match="patron_event_id"):
            sign_proof_grant(patron_npub=patron, operator_nsec=op_nsec,
                             challenge_hash=CHALLENGE, patron_sig="ab", patron_event_id="")

    @pytest.mark.parametrize("override,reason", [
        ({"challenge_hash": hashlib.sha256(b"other").hexdigest()}, "challenge_mismatch"),
        ({"expected_patron_npub": PrivateKey().public_key.bech32()}, "wrong_npub"),
        ({"expected_operator_hex": PrivateKey().public_key.hex()}, "operator_mismatch"),
    ])
    def test_rejection_reasons(self, operator, patron, override, reason):
        op_nsec, op_hex = operator
        kwargs = {
            "expected_patron_npub": patron,
            "challenge_hash": CHALLENGE,
            "expected_operator_hex": op_hex,
            **override,
        }
        with pytest.raises(IdentityCredentialError, match=reason):
            verify_proof_grant(_grant(op_nsec, patron), **kwargs)

    def test_expired(self, operator, patron):
        op_nsec, op_hex = operator
        with pytest.raises(IdentityCredentialError, match="expired"):
            verify_proof_grant(_grant(op_nsec, patron, ttl=-10), expected_patron_npub=patron,
                               challenge_hash=CHALLENGE, expected_operator_hex=op_hex)

    def test_a_plain_identity_credential_is_not_a_grant(self, operator, patron):
        from tollbooth.identity_credential import sign_identity_credential
        op_nsec, op_hex = operator
        cred = sign_identity_credential(citizen_npub=patron, operator_nsec=op_nsec)
        with pytest.raises(IdentityCredentialError, match="not_proof_grant"):
            verify_proof_grant(cred, expected_patron_npub=patron, expected_operator_hex=op_hex)


# ---------------------------------------------------------------------------
# require_proof gate
# ---------------------------------------------------------------------------


class TestRequireProofGrant:
    @pytest.mark.asyncio
    async def test_phrase_alone_refused(self, patron):
        r = await require_proof(patron, NONCE, "tool", revocations=_NoRevocations(), operator_hex="ab" * 32)
        assert r["error_code"] == ErrorCode.PROOF_REFRESH_NEEDED

    @pytest.mark.asyncio
    async def test_grant_envelope_accepted(self, operator, patron):
        op_nsec, op_hex = operator
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron)), "tool",
                                revocations=_NoRevocations(), operator_hex=op_hex)
        assert r is None

    @pytest.mark.asyncio
    async def test_grant_accepted_without_a_revocation_store(self, operator, patron):
        op_nsec, op_hex = operator
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron)), "tool", operator_hex=op_hex)
        assert r is None

    @pytest.mark.asyncio
    async def test_unknown_issuer_refuses_fail_closed(self, operator, patron):
        # A runtime that cannot name its own pubkey must not accept ANY grant.
        op_nsec, _ = operator
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron)), "tool",
                                revocations=_NoRevocations(), operator_hex=None)
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "operator_unknown"

    @pytest.mark.asyncio
    async def test_foreign_operator_grant_refused(self, operator, patron):
        op_nsec, _ = operator
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron)), "tool",
                                revocations=_NoRevocations(), operator_hex=PrivateKey().public_key.hex())
        assert r["reason"] == "operator_mismatch"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_nonce,expected", [("calm-reef-77", "challenge_mismatch")])
    async def test_wrong_nonce_reason(self, operator, patron, bad_nonce, expected):
        op_nsec, op_hex = operator
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron), bad_nonce), "tool", operator_hex=op_hex)
        assert r["error_code"] == ErrorCode.PROOF_INVALID and r["reason"] == expected

    @pytest.mark.asyncio
    async def test_wrong_npub_reason(self, operator, patron):
        op_nsec, op_hex = operator
        other = PrivateKey().public_key.bech32()
        r = await require_proof(other, _envelope(_grant(op_nsec, patron)), "tool", operator_hex=op_hex)
        assert r["reason"] == "wrong_npub"

    @pytest.mark.asyncio
    async def test_expired_reason(self, operator, patron):
        op_nsec, op_hex = operator
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron, ttl=-5)), "tool", operator_hex=op_hex)
        assert r["reason"] == "expired"

    @pytest.mark.asyncio
    async def test_forget_credentials_revokes_every_grant(self, operator, patron):
        op_nsec, op_hex = operator
        revocations = ProofGrantRevocations()
        before = _envelope(_grant(op_nsec, patron))
        assert await require_proof(patron, before, "tool", revocations=revocations, operator_hex=op_hex) is None

        await revocations.revoke_all(patron)  # what forget_credentials does

        r = await require_proof(patron, before, "tool", revocations=revocations, operator_hex=op_hex)
        assert r["error_code"] == ErrorCode.PROOF_INVALID and r["reason"] == "revoked"

    @pytest.mark.asyncio
    async def test_unreadable_revocation_store_refuses_fail_closed(self, operator, patron):
        op_nsec, op_hex = operator
        revocations = AsyncMock()
        revocations.is_revoked = AsyncMock(side_effect=RevocationStoreUnavailable("neon down"))
        r = await require_proof(patron, _envelope(_grant(op_nsec, patron)), "tool",
                                revocations=revocations, operator_hex=op_hex)
        assert r["error_code"] == ErrorCode.PROOF_INVALID
        assert r["reason"] == "revocations_unavailable"
        assert "Retry" in r["error"]

    @pytest.mark.asyncio
    async def test_bare_grant_without_nonce_refused(self, operator, patron):
        op_nsec, op_hex = operator
        r = await require_proof(patron, _grant(op_nsec, patron), "tool", operator_hex=op_hex)
        assert r["reason"] == "missing_nonce"

    @pytest.mark.asyncio
    async def test_tactic2_inline_unchanged(self):
        pk = PrivateKey()
        r = await require_proof(pk.public_key.bech32(), _signed_27235(pk, "tool"), "tool")
        assert r is None

    @pytest.mark.asyncio
    async def test_denials_never_carry_the_grant_or_key_material(self, operator, patron):
        op_nsec, op_hex = operator
        grant = _grant(op_nsec, patron)
        r = await require_proof(patron, _envelope(grant, "calm-reef-77"), "tool", operator_hex=op_hex)
        blob = json.dumps(r)
        assert json.loads(grant)["sig"] not in blob
        assert op_nsec not in blob
