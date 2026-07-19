"""Tests for identity_proof.require_proof — the canonical proof gate.

Focus: the S4 fix (a dpop_token-shaped token that can't be validated as a cached
proof gets clear "refresh" feedback instead of a confusing malformed-Schnorr
error) plus the surrounding accept/deny paths.
"""

import base64
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

from tollbooth.constants import ErrorCode
from tollbooth.identity_proof import PROOF_EVENT_KIND, require_proof

NPUB = "npub1d999638gqpn8c594teklxtxva0uvxdng80q3ycyqvldjdl457c7qcrq64z"
DPOP_TOKEN = "glad-blade-13"  # matches the <word>-<word>-<n> dpop_token shape

TOOL = "excalibur_check_oauth_status"


def _signed_proof(
    pk: PrivateKey,
    *,
    tool_name: str = TOOL,
    kind: int = PROOF_EVENT_KIND,
    created_at: int | None = None,
    nonce: str = "a1",
) -> str:
    """Mint a raw-JSON, signed kind-27235 inline proof."""
    event = Event(
        kind=kind,
        content="",
        tags=[["u", tool_name], ["nonce", nonce]],
        pubkey=pk.public_key.hex(),
    )
    event.created_at = int(time.time()) if created_at is None else created_at
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


@pytest.mark.asyncio
async def test_empty_proof_required():
    r = await require_proof(NPUB, "", "tool")
    assert r["error_code"] == ErrorCode.PROOF_REQUIRED


@pytest.mark.asyncio
async def test_invalid_npub_rejected():
    r = await require_proof("npub1short", DPOP_TOKEN, "tool")
    assert r["error_code"] == ErrorCode.NPUB_INVALID


@pytest.mark.asyncio
async def test_dpop_token_shaped_token_without_cache_gives_refresh_feedback():
    # S4: no proven_cache wired, but the token looks like a dpop_token. Must
    # not fall through to "Invalid identity proof" — give refresh guidance.
    r = await require_proof(NPUB, DPOP_TOKEN, "tool", proven_cache=None)
    assert r["error_code"] == ErrorCode.PROOF_REFRESH_NEEDED
    assert "dpop_token" in r["error"]
    assert r["next_steps"]


@pytest.mark.asyncio
async def test_cached_dpop_token_hit_passes():
    cache = SimpleNamespace(is_proven=AsyncMock(return_value=True))
    r = await require_proof(NPUB, DPOP_TOKEN, "tool", proven_cache=cache)
    assert r is None  # success → caller proceeds


@pytest.mark.asyncio
async def test_cached_dpop_token_miss_gives_refresh_feedback():
    cache = SimpleNamespace(is_proven=AsyncMock(return_value=False))
    r = await require_proof(NPUB, DPOP_TOKEN, "tool", proven_cache=cache)
    assert r["error_code"] == ErrorCode.PROOF_REFRESH_NEEDED


@pytest.mark.asyncio
async def test_non_dpop_token_garbage_is_invalid_proof():
    # A non-dpop_token, non-Schnorr string still fails as a malformed inline proof.
    r = await require_proof(NPUB, "garbage", "tool")
    assert r["error_code"] == ErrorCode.PROOF_INVALID


# ---------------------------------------------------------------------------
# Issue #137: Tactic-2 inline-proof denials must be diagnosable — carry a
# machine-readable ``reason`` (and ``expected_u`` on tool_mismatch) instead of
# one opaque "Invalid identity proof." Accept/reject logic is unchanged; only
# the explanation is added.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_inline_proof_still_passes_unchanged():
    # Happy path: a correctly-shaped, fresh, correctly-signed proof → None.
    pk = PrivateKey()
    r = await require_proof(pk.public_key.bech32(), _signed_proof(pk), TOOL)
    assert r is None


@pytest.mark.asyncio
async def test_tool_mismatch_reason_and_expected_u():
    # The reporting agent's mistake: endpoint URL in the u tag, not the tool name.
    pk = PrivateKey()
    proof = _signed_proof(pk, tool_name="https://operator.example/mcp")
    r = await require_proof(pk.public_key.bech32(), proof, TOOL)
    assert r["error_code"] == ErrorCode.PROOF_INVALID
    assert r["reason"] == "tool_mismatch"
    assert r["expected_u"] == TOOL  # names the tool the caller already invokes


@pytest.mark.asyncio
async def test_base64_wrapped_event_reason_malformed_json():
    # The reporting agent's other mistake: base64-encoding the event.
    pk = PrivateKey()
    b64 = base64.b64encode(_signed_proof(pk).encode()).decode()
    r = await require_proof(pk.public_key.bech32(), b64, TOOL)
    assert r["reason"] == "malformed_json"
    assert "expected_u" not in r  # only tool_mismatch carries expected_u


@pytest.mark.asyncio
async def test_expired_reason():
    pk = PrivateKey()
    proof = _signed_proof(pk, created_at=int(time.time()) - 120)
    r = await require_proof(pk.public_key.bech32(), proof, TOOL)
    assert r["reason"] == "expired"


@pytest.mark.asyncio
async def test_replayed_reason():
    pk = PrivateKey()
    proof = _signed_proof(pk, nonce="replaytest137")
    npub = pk.public_key.bech32()
    first = await require_proof(npub, proof, TOOL)
    assert first is None  # first use consumes the event id
    second = await require_proof(npub, proof, TOOL)
    assert second["reason"] == "replayed"


@pytest.mark.asyncio
async def test_npub_mismatch_reason():
    signer = PrivateKey()
    other = PrivateKey()
    proof = _signed_proof(signer)
    r = await require_proof(other.public_key.bech32(), proof, TOOL)
    assert r["reason"] == "npub_mismatch"


@pytest.mark.asyncio
async def test_wrong_kind_reason():
    pk = PrivateKey()
    proof = _signed_proof(pk, kind=1)
    r = await require_proof(pk.public_key.bech32(), proof, TOOL)
    assert r["reason"] == "wrong_kind"


@pytest.mark.asyncio
async def test_tampered_signature_reason():
    pk = PrivateKey()
    event_dict = json.loads(_signed_proof(pk))
    event_dict["content"] = "tampered"
    r = await require_proof(
        pk.public_key.bech32(), json.dumps(event_dict), TOOL
    )
    assert r["reason"] == "signature_invalid"
