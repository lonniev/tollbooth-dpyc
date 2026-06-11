"""Tests for tollbooth.authority.tools (audit M2.4b).

The authority tool surface was ~12% covered. This targets the pure helpers
(_parse_oracle_commit_url, _resolve_npub_or_operator) and the core money/cert
tool certify_credits (fee/net, Schnorr certificate signing, replay record,
membership gate + fee refund). The remaining lifecycle/onboarding tools are
heavier orchestration left for later.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

import tollbooth.authority.tools as at
from tollbooth.authority.nostr_signing import AuthorityNostrSigner
from tollbooth.authority.replay import ReplayTracker
from tollbooth.authority.tools import OracleRegistryError, _parse_oracle_commit_url
from tollbooth.registry import RegistryError


# ── _parse_oracle_commit_url (pure) ───────────────────────────────────

def _result(*texts):
    return SimpleNamespace(content=[SimpleNamespace(text=t) for t in texts])


def test_parse_commit_url_success():
    r = _result(json.dumps({"success": True, "commit_url": "https://github.com/x/y"}))
    assert _parse_oracle_commit_url(r) == "https://github.com/x/y"


def test_parse_commit_url_success_false_raises():
    r = _result(json.dumps({"success": False, "error": "service_url is required"}))
    with pytest.raises(OracleRegistryError, match="service_url is required") as ei:
        _parse_oracle_commit_url(r)
    assert ei.value.raw["error"] == "service_url is required"


def test_parse_commit_url_missing_commit_url_is_empty_string():
    r = _result(json.dumps({"success": True}))
    assert _parse_oracle_commit_url(r) == ""


def test_parse_commit_url_non_json_text_returned_raw():
    r = _result("just text, not json")
    assert _parse_oracle_commit_url(r) == "just text, not json"


def test_parse_commit_url_no_content_attr():
    assert _parse_oracle_commit_url("raw-string") == "raw-string"


# ── _resolve_npub_or_operator ─────────────────────────────────────────

def test_resolve_npub_passthrough_for_valid():
    real = PrivateKey().public_key.bech32()
    assert at._resolve_npub_or_operator(real) == real


def test_resolve_npub_falls_back_to_operator_on_invalid():
    rt = MagicMock()
    rt.operator_npub.return_value = "npub1operatorfallback"
    with patch.object(at, "_get_runtime", return_value=rt):
        assert at._resolve_npub_or_operator("") == "npub1operatorfallback"


# ── certify_credits ───────────────────────────────────────────────────

@contextmanager
def _authority_tools(runtime, *, settings, signer, replay, registry):
    tools: dict = {}

    def fake_slug_tool(_m, _s):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch.dict("os.environ"), patch.multiple(
        at,
        make_slug_tool=MagicMock(side_effect=fake_slug_tool),
        _get_settings=MagicMock(return_value=settings),
        _get_nostr_signer=MagicMock(return_value=signer),
        _get_replay_tracker=MagicMock(return_value=replay),
        _get_dpyc_registry=MagicMock(return_value=registry),
        _maybe_refresh_bootstrap_dm=AsyncMock(),
    ):
        os.environ.pop("NEON_DATABASE_URL", None)
        at.register_authority_tools(MagicMock(), runtime)
        yield tools


def _passthrough_paid_tool(*_a, **_k):
    """Stand-in for runtime.paid_tool: register the tool body unwrapped so the
    test drives _last_debit_cost directly (the real gate is tested elsewhere)."""
    def deco(fn):
        return fn
    return deco


def _fake_runtime(fee=20):
    ledger = SimpleNamespace(
        balance_api_sats=500, total_deposited_api_sats=1000, total_consumed_api_sats=500,
    )
    cache = SimpleNamespace(
        flush_user=AsyncMock(return_value=True),
        get=AsyncMock(return_value=ledger),
        health=MagicMock(return_value="ok"),
    )
    return SimpleNamespace(
        _last_debit_cost=fee,
        rollback_debit=AsyncMock(),
        ledger_cache=AsyncMock(return_value=cache),
        paid_tool=_passthrough_paid_tool,
        operator_npub=MagicMock(return_value="npub1self"),
    )


def _settings():
    return SimpleNamespace(
        certificate_ttl_seconds=3600,
        dpyc_enforce_membership=False,
        neon_database_url="postgresql://x",
        dpyc_registry_cache_ttl_seconds=60,
    )


@pytest.mark.asyncio
async def test_certify_credits_rejects_nonpositive_amount():
    rt = _fake_runtime()
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),
                          replay=ReplayTracker(), registry=None) as tools:
        r = await tools["certify_credits"](npub="npub1op", proof="p", amount_sats=0)
    assert r == {"success": False, "error": "amount_sats must be positive."}


@pytest.mark.asyncio
async def test_certify_credits_signs_a_verifiable_certificate():
    signer = AuthorityNostrSigner(PrivateKey().nsec)
    replay = ReplayTracker()
    rt = _fake_runtime(fee=20)
    operator_npub = PrivateKey().public_key.bech32()

    with _authority_tools(rt, settings=_settings(), signer=signer,
                          replay=replay, registry=None) as tools:
        r = await tools["certify_credits"](npub=operator_npub, proof="p", amount_sats=1000)
        await asyncio.sleep(0)  # let the fire-and-forget bootstrap task settle

    assert r["success"] is True
    assert r["amount_sats"] == 1000 and r["fee_sats"] == 20 and r["net_sats"] == 980
    # the returned certificate is a real Schnorr-signed event
    event = Event.from_dict(json.loads(r["certificate"]))
    assert event.verify() is True and event.pubkey == signer.pubkey_hex
    claims = json.loads(event.content)
    assert claims["amount_sats"] == 1000 and claims["fee_sats"] == 20 and claims["net_sats"] == 980
    # jti recorded for replay protection; fee debit flushed
    assert replay.size == 1
    rt.ledger_cache.return_value.flush_user.assert_awaited_once()


@pytest.mark.asyncio
async def test_certify_credits_membership_failure_refunds_fee():
    registry = SimpleNamespace(check_membership=AsyncMock(side_effect=RegistryError("not a member")))
    rt = _fake_runtime(fee=20)
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),
                          replay=ReplayTracker(), registry=registry) as tools:
        r = await tools["certify_credits"](npub="npub1op", proof="p", amount_sats=1000)

    assert r["success"] is False
    assert r["error_code"] == "dpyc_membership_required"
    rt.rollback_debit.assert_awaited_once()  # certification fee refunded


# ── operator_status / check_dpyc_membership / register_authority_npub ──

@pytest.mark.asyncio
async def test_operator_status_self_inspection_skips_proof():
    rt = _fake_runtime()
    signer = SimpleNamespace(npub="npub1authority")
    with _authority_tools(rt, settings=_settings(), signer=signer,
                          replay=ReplayTracker(), registry=None) as tools:
        r = await tools["operator_status"](npub="", proof="")

    assert r["npub"] == "npub1self"          # fell back to operator identity
    assert r["balance_sats"] == 500
    assert r["total_deposited_sats"] == 1000 and r["total_consumed_sats"] == 500
    assert r["authority_npub"] == "npub1authority"
    assert r["vault_configured"] is True and r["vault_backend"] == "neon"
    assert r["cache_health"] == "ok"


@pytest.mark.asyncio
async def test_check_dpyc_membership_success_and_error():
    rt = _fake_runtime()
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),
                          replay=ReplayTracker(), registry=None) as tools:
        ok_reg = SimpleNamespace(check_membership=AsyncMock(return_value={"role": "operator"}), close=AsyncMock())
        with patch.object(at, "DPYCRegistry", return_value=ok_reg):
            r = await tools["check_dpyc_membership"](npub="npub1op")
        assert r == {"success": True, "member": {"role": "operator"}}
        ok_reg.close.assert_awaited_once()    # registry closed even on success

        bad_reg = SimpleNamespace(check_membership=AsyncMock(side_effect=RegistryError("not found")), close=AsyncMock())
        with patch.object(at, "DPYCRegistry", return_value=bad_reg):
            r = await tools["check_dpyc_membership"](npub="npub1op")
        assert r == {"success": False, "error": "not found"}
        bad_reg.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_register_authority_npub_rejects_bad_format():
    rt = _fake_runtime()
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),
                          replay=ReplayTracker(), registry=None) as tools:
        r = await tools["register_authority_npub"](candidate_npub="bad")
    assert r["success"] is False and "Invalid npub format" in r["error"]


@pytest.mark.asyncio
async def test_register_authority_npub_blocks_when_curator_exists():
    rt = _fake_runtime()
    valid = "npub1" + "q" * 59  # passes the format gate (starts npub1, len>=60)
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),
                          replay=ReplayTracker(), registry=None) as tools:
        with patch.object(at, "_get_authority_npub", AsyncMock(return_value="npub1existingcurator")):
            r = await tools["register_authority_npub"](candidate_npub=valid)
    assert r["success"] is False and "already has a curator" in r["error"]
