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
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.event import Event  # type: ignore[import-untyped]
from pynostr.key import PrivateKey  # type: ignore[import-untyped]

import tollbooth.authority.tools as at
from tollbooth.authority.nostr_signing import AuthorityNostrSigner
from tollbooth.authority.onboarding import OnboardingState
from tollbooth.authority.replay import ReplayTracker
from tollbooth.authority.tools import OracleRegistryError, _parse_oracle_commit_url
from tollbooth.constants import ErrorCode
from tollbooth.registry import RegistryError
from tollbooth.runtime import OperatorRuntime

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

    def fake_slug_tool(_m, _s, _identify):
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
    cache.mark_dirty = MagicMock()
    # Provisioning materializes the operator's ledger row write-through, so the
    # fake must answer mutate(); run the fn against the stand-in ledger the way
    # the real cache runs it against fresh vault state.
    cache.mutate = AsyncMock(side_effect=lambda npub, fn: fn(ledger))
    rt = SimpleNamespace(
        identity_for=MagicMock(),
        _last_debit_cost=fee,
        rollback_debit=AsyncMock(),
        ledger_cache=AsyncMock(return_value=cache),
        paid_tool=_passthrough_paid_tool,
        operator_npub=MagicMock(return_value="npub1self"),
        runtime_name=MagicMock(side_effect=lambda cap: f"authority_{cap}"),
        proof_grant_revocations=AsyncMock(return_value=MagicMock()),
        operator_pubkey_hex=MagicMock(return_value="ab" * 32),
        vault=AsyncMock(return_value=SimpleNamespace(_cipher=None)),
        holder_auth_enabled=False,
    )
    # The real gate, so a patched tollbooth.runtime.require_held_grant decides.
    rt.prove_caller = MethodType(OperatorRuntime.prove_caller, rt)
    return rt


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
        r = await tools["certify_credits"](npub="npub1op", amount_sats=0)
    assert r == {"success": False, "error": "amount_sats must be positive."}


@pytest.mark.asyncio
async def test_certify_credits_signs_a_verifiable_certificate():
    signer = AuthorityNostrSigner(PrivateKey().nsec)
    replay = ReplayTracker()
    rt = _fake_runtime(fee=20)
    operator_npub = PrivateKey().public_key.bech32()

    with _authority_tools(rt, settings=_settings(), signer=signer,
                          replay=replay, registry=None) as tools:
        r = await tools["certify_credits"](npub=operator_npub, amount_sats=1000)
        await asyncio.sleep(0)  # let the fire-and-forget bootstrap task settle

    assert r["success"] is True
    assert r["amount_sats"] == 1000 and r["fee_sats"] == 20 and r["net_sats"] == 980
    # the returned certificate is a real Schnorr-signed event
    event = Event.from_dict(json.loads(r["certificate"]))
    assert event.verify() is True and event.pubkey == signer.pubkey_hex
    claims = json.loads(event.content)
    assert claims["amount_sats"] == 1000 and claims["fee_sats"] == 20 and claims["net_sats"] == 980
    # jti recorded for replay protection. The fee debit needs no assertion here:
    # a non-zero charge is CAS-written by LedgerCache.mutate() before this tool
    # body runs, so there is no flush left for certify_credits to perform.
    assert replay.size == 1


@pytest.mark.asyncio
async def test_certify_credits_membership_failure_refunds_fee():
    registry = SimpleNamespace(check_membership=AsyncMock(side_effect=RegistryError("not a member")))
    rt = _fake_runtime(fee=20)
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),
                          replay=ReplayTracker(), registry=registry) as tools:
        r = await tools["certify_credits"](npub="npub1op", amount_sats=1000)

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
        r = await tools["operator_status"](npub="")

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
    with _authority_tools(rt, settings=_settings(), signer=MagicMock(),  # noqa: SIM117
                          replay=ReplayTracker(), registry=None) as tools:
        with patch.object(at, "_get_authority_npub", AsyncMock(return_value="npub1existingcurator")):
            r = await tools["register_authority_npub"](candidate_npub=valid)
    assert r["success"] is False and "already has a curator" in r["error"]


# ── update / deregister / get_config / the owner's request queue ──────

def _tools(signer=None, rt=None):
    """Shorthand: enter the authority-tools context with sane defaults."""
    return _authority_tools(
        rt or _fake_runtime(), settings=_settings(),
        signer=signer or SimpleNamespace(npub="npub1authority"),
        replay=ReplayTracker(), registry=None,
    )


_PASS = patch("tollbooth.runtime.require_held_grant", AsyncMock(return_value=None))


def _refuse(npubs_refused: set[str]):
    """require_held_grant that proves every npub but the refused ones."""
    async def gate(npub, envelope, **_kw):
        if npub in npubs_refused:
            return {"success": False, "error_code": ErrorCode.PROOF_REQUIRED, "error": "not held"}
        return None
    return patch("tollbooth.runtime.require_held_grant", AsyncMock(side_effect=gate))


OP = PrivateKey().public_key.bech32()
OTHER = PrivateKey().public_key.bech32()


def _runtime_with_store(rows=None):
    from tests.test_operator_requests import FakeVault

    rt = _fake_runtime()
    vault = FakeVault(rows=rows)
    vault._cipher = None
    rt.vault = AsyncMock(return_value=vault)
    return rt, vault


@pytest.mark.asyncio
async def test_update_operator_nothing_to_update():
    with _tools() as tools, _PASS:
        r = await tools["update_operator"](npub=OP)
    assert "Nothing to update" in r["error"]


@pytest.mark.asyncio
async def test_update_operator_records_a_pending_request_and_tells_the_owner():
    rt, vault = _runtime_with_store()
    dm = AsyncMock()
    with _tools(rt=rt) as tools, _PASS, patch.object(at, "_dm_owner", dm), \
            patch.object(at, "_update_operator_via_oracle", AsyncMock()) as oracle:
        r = await tools["update_operator"](npub=OP, service_url="https://new")
    assert r["success"] is True and r["status"] == "pending"
    insert = next(sql for sql, _ in vault.calls if sql.startswith("INSERT"))
    assert "operator_update_requests" in insert
    oracle.assert_not_awaited()  # nothing changes until the owner approves
    dm.assert_awaited_once()


@pytest.mark.asyncio
async def test_update_operator_needs_the_operators_own_proof():
    with _tools() as tools, _refuse({OP}):
        r = await tools["update_operator"](npub=OP, service_url="https://new")
    assert r["error_code"] == ErrorCode.PROOF_REQUIRED


@pytest.mark.asyncio
async def test_approving_an_update_applies_it_through_the_oracle():
    row = {"operator_npub": OP, "service_url": "https://new", "display_name": "", "status": "pending"}
    rt, vault = _runtime_with_store(rows=[row])
    with _tools(rt=rt) as tools, _PASS, patch.object(at, "_resend_bootstrap_dm", AsyncMock()), \
            patch.object(at, "_update_operator_via_oracle", AsyncMock(return_value="https://c")) as oracle:
        r = await tools["approve_operator_request"](operator_npub=OP, kind="update")
    assert r["success"] is True and r["commit_url"] == "https://c"
    assert oracle.await_args.kwargs["service_url"] == "https://new"
    update = next(params for sql, params in vault.calls if sql.startswith("UPDATE"))
    assert update == [OP, "applied"]


@pytest.mark.asyncio
async def test_approving_an_adoption_provisions_with_the_requested_name():
    row = {"operator_npub": OP, "service_url": "https://svc", "display_name": "my-service", "status": "pending"}
    rt, vault = _runtime_with_store(rows=[row])
    oracle = AsyncMock(return_value="https://commit")
    with _tools(rt=rt) as tools, _PASS, patch.multiple(
        at, _resend_bootstrap_dm=AsyncMock(), _register_operator_via_oracle=oracle,
    ), patch.multiple(
        "tollbooth.authority.tenant_provisioner",
        ensure_bootstrap_table=AsyncMock(),
        provision_operator_schema=AsyncMock(return_value=("op_schema", "pw")),
        store_operator_config=AsyncMock(),
        neon_url_for_operator=MagicMock(return_value="postgresql://op_schema:pw@h/db"),
    ):
        r = await tools["approve_operator_request"](operator_npub=OP, kind="adoption")
    assert r["success"] is True and r["commit_url"] == "https://commit"
    assert oracle.await_args.kwargs["display_name"] == "my-service"
    assert [OP, "provisioned"] in [params for sql, params in vault.calls if sql.startswith("UPDATE")]


@pytest.mark.asyncio
async def test_an_approved_request_is_not_approved_twice():
    rt, _ = _runtime_with_store(rows=[{"operator_npub": OP, "status": "applied"}])
    with _tools(rt=rt) as tools, _PASS:
        r = await tools["approve_operator_request"](operator_npub=OP, kind="update")
    assert r["error_code"] == ErrorCode.ADOPTION_ALREADY_PROVISIONED


@pytest.mark.asyncio
async def test_only_the_owner_works_the_queue():
    with _tools() as tools, _refuse({"npub1self"}):
        for tool, kwargs in (
            ("list_operator_requests", {}),
            ("approve_operator_request", {"operator_npub": OP, "kind": "update"}),
            ("reject_operator_request", {"operator_npub": OP, "kind": "adoption"}),
        ):
            r = await tools[tool](**kwargs)
            assert r["error_code"] == ErrorCode.AUTHORITY_CONSENT_REQUIRED, tool


@pytest.mark.asyncio
async def test_list_operator_requests_names_each_kind():
    rt, _ = _runtime_with_store(rows=[{"operator_npub": OP, "requested_at": "2026-10-17"}])
    with _tools(rt=rt) as tools, _PASS:
        r = await tools["list_operator_requests"]()
    assert r["success"] is True
    assert sorted(x["kind"] for x in r["requests"]) == ["adoption", "update"]


@pytest.mark.asyncio
async def test_the_operator_alone_may_leave():
    rt, vault = _runtime_with_store()
    with _tools(rt=rt) as tools, _refuse({"npub1self"}), \
            patch.object(at, "_deregister_operator_via_oracle", AsyncMock(return_value="https://c")):
        r = await tools["deregister_operator"](npub=OP)
    assert r["success"] is True and r["commit_url"] == "https://c"
    assert [OP, "closed"] in [params for sql, params in vault.calls if sql.startswith("UPDATE")]


@pytest.mark.asyncio
async def test_the_authority_alone_may_disown_an_operator():
    rt, _ = _runtime_with_store()
    with _tools(rt=rt) as tools, _refuse({OP}), \
            patch.object(at, "_deregister_operator_via_oracle", AsyncMock(return_value="https://c")):
        r = await tools["deregister_operator"](npub=OP)
    assert r["success"] is True


@pytest.mark.asyncio
async def test_a_third_party_may_not_deregister():
    with _tools() as tools, _refuse({OP, "npub1self"}), \
            patch.object(at, "_deregister_operator_via_oracle", AsyncMock()) as oracle:
        r = await tools["deregister_operator"](npub=OP)
    assert r["error_code"] == ErrorCode.AUTHORITY_CONSENT_REQUIRED
    oracle.assert_not_awaited()


@pytest.mark.asyncio
async def test_deregister_reports_an_oracle_failure():
    with _tools() as tools, _PASS, \
            patch.object(at, "_deregister_operator_via_oracle", AsyncMock(side_effect=RuntimeError("x"))):
        bad = await tools["deregister_operator"](npub=OP)
    assert bad["success"] is False and "Deregistration failed" in bad["error"]


@pytest.mark.asyncio
async def test_receive_adoption_request_needs_the_operators_signature():
    rt, vault = _runtime_with_store()
    with _tools(rt=rt) as tools, patch.object(at, "_dm_owner", AsyncMock()):
        with patch.object(at, "_signed_by", MagicMock(return_value=False)):
            refused = await tools["receive_adoption_request"](operator_npub=OP)
        with patch.object(at, "_signed_by", MagicMock(return_value=True)):
            ok = await tools["receive_adoption_request"](
                operator_npub=OP, service_url="https://svc", display_name="Svc", note="hi",
            )
    assert refused["error_code"] == ErrorCode.PROOF_INVALID
    assert ok["success"] is True and ok["status"] == "pending"
    params = next(params for sql, params in vault.calls if sql.startswith("INSERT"))
    assert params == [OP, "https://svc", "Svc", "hi"]


@pytest.mark.asyncio
async def test_get_operator_config_no_config_and_filters_password():
    with _tools() as tools:
        with _PASS, patch("tollbooth.authority.tenant_provisioner.get_all_operator_config", AsyncMock(return_value={})):
            empty = await tools["get_operator_config"](npub="npub1op")
        assert empty["success"] is False and "No configuration found" in empty["error"]

        cfg = {"schema": "op_s", "role_password": "SECRET", "neon_database_url": "url"}
        with _PASS, patch.object(at, "_resend_bootstrap_dm", AsyncMock()), \
             patch("tollbooth.authority.tenant_provisioner.get_all_operator_config", AsyncMock(return_value=cfg)):
            r = await tools["get_operator_config"](npub="npub1op")
        assert r["success"] is True
        assert "role_password" not in r["config"]          # secret filtered out
        assert r["config"]["schema"] == "op_s" and r["config"]["neon_database_url"] == "url"


@pytest.mark.asyncio
async def test_dm_owner_sends_a_plain_dm_and_never_raises():
    rt = _fake_runtime()
    exchange = SimpleNamespace(send_dm=MagicMock())
    with patch.object(at, "_nostr_exchange", AsyncMock(return_value=exchange)):
        await at._dm_owner(rt, "check your queue")
    exchange.send_dm.assert_called_once_with("npub1self", "check your queue")

    broken = SimpleNamespace(send_dm=MagicMock(side_effect=RuntimeError("relay down")))
    with patch.object(at, "_nostr_exchange", AsyncMock(return_value=broken)):
        await at._dm_owner(rt, "x")  # logged, not raised


# ── 3-step Authority onboarding (confirm_authority_claim / check_approval)

class FakeExchange:
    """The courier exchange as onboarding uses it. ``receive`` has the real
    signature, so a call that leaves out the nonce fails as it would live."""

    def __init__(self, nonce: str = "bold-hawk-42"):
        self.nonce = nonce
        self.opened: list[str] = []
        self.received: list[tuple[str, str, str]] = []

    async def open_channel(self, service, *, greeting, recipient_npub=None):
        self.opened.append(service)
        return {"success": True, "nonce": self.nonce, "message": "DM sent"}

    async def receive(self, sender_npub, *, service, nonce):
        self.received.append((sender_npub, service, nonce))
        return {"success": True}


def _exchange(fake):
    return patch.object(at, "_nostr_exchange", AsyncMock(return_value=fake))


@pytest.mark.asyncio
async def test_confirm_claim_gates():
    with _tools() as tools:
        with patch.object(at, "_onboarding", OnboardingState()):  # nothing active
            r = await tools["confirm_authority_claim"](candidate_npub="npub1cand")
        assert "No active onboarding" in r["error"]

        wrong = OnboardingState()
        wrong.start_claim("npub1other")
        with patch.object(at, "_onboarding", wrong):
            r = await tools["confirm_authority_claim"](candidate_npub="npub1cand")
        assert "Active onboarding is for" in r["error"]


@pytest.mark.asyncio
async def test_confirm_claim_happy_escalates_to_parent():
    state = OnboardingState()
    state.start_claim("npub1cand").nonce = "claim-nonce-1"
    exchange = FakeExchange(nonce="approval-nonce-2")
    with _tools(signer=SimpleNamespace(npub="npub1auth")) as tools, patch.object(at, "_onboarding", state), \
             _exchange(exchange), \
             patch.object(at, "resolve_my_parent_npub", AsyncMock(return_value="npub1parent")):
        r = await tools["confirm_authority_claim"](candidate_npub="npub1cand")
    assert r["success"] is True and r["phase"] == "approval" and r["parent_npub"] == "npub1parent"
    # The claim reply is found by the nonce its channel was opened with …
    assert exchange.received == [("npub1cand", "authority_claim", "claim-nonce-1")]
    # … and the approval channel's nonce is kept for step 3.
    assert exchange.opened == ["authority_approval"]
    assert state.get().phase == "approval" and state.get().nonce == "approval-nonce-2"


@pytest.mark.asyncio
async def test_check_approval_wrong_phase():
    state = OnboardingState()
    state.start_claim("npub1cand")  # still in claim, not approval
    with _tools() as tools, patch.object(at, "_onboarding", state):
        r = await tools["check_authority_approval"](candidate_npub="npub1cand")
    assert "not 'approval'" in r["error"]


@pytest.mark.asyncio
async def test_register_authority_npub_sends_challenge():
    valid = "npub1" + "q" * 59
    exchange = FakeExchange(nonce="claim-nonce-1")
    state = OnboardingState()
    with _tools() as tools, patch.object(at, "_get_authority_npub", AsyncMock(return_value=None)), \
             patch.object(at, "_onboarding", state), _exchange(exchange):
        r = await tools["register_authority_npub"](candidate_npub=valid)
    assert r["success"] is True and r["phase"] == "claim"
    assert r["candidate_npub"] == valid
    assert exchange.opened == ["authority_claim"]
    assert state.get().nonce == "claim-nonce-1"  # kept to find the reply


@pytest.mark.asyncio
async def test_require_owner_pass_and_fail():
    rt = _fake_runtime()
    rt.operator_npub = MagicMock(return_value="npub1authorityXXXXXXXXX")
    # the request proves the Authority's npub → None (caller proceeds)
    with patch("tollbooth.runtime.require_held_grant", AsyncMock(return_value=None)):
        assert await at._require_owner(rt) is None
    # it does not → structured AUTHORITY_CONSENT_REQUIRED error
    with patch("tollbooth.runtime.require_held_grant", AsyncMock(return_value={"err": 1})):
        err = await at._require_owner(rt)
    assert err["error_code"] == ErrorCode.AUTHORITY_CONSENT_REQUIRED
    assert err["authority_npub"] == "npub1authorityXXXXXXXXX"


@pytest.mark.asyncio
async def test_check_approval_happy_activates_and_completes():
    state = OnboardingState()
    state.start_claim("npub1cand")
    state.promote_to_approval("npub1parent").nonce = "approval-nonce-2"
    exchange = FakeExchange()
    with _tools() as tools, patch.object(at, "_onboarding", state), \
             _exchange(exchange), \
             patch.object(at, "_set_authority_npub", AsyncMock()) as set_npub, \
             patch.object(at, "_resolve_own_service_url", AsyncMock(return_value="https://svc")), \
             patch.object(at, "_register_via_oracle", AsyncMock(return_value="https://commit")):
        r = await tools["check_authority_approval"](candidate_npub="npub1cand")
    assert r["success"] is True and r["activated"] is True
    assert exchange.received == [("npub1parent", "authority_approval", "approval-nonce-2")]
    set_npub.assert_awaited_once_with("npub1cand")
    assert r["commit_url"] == "https://commit"
    assert state.get() is None  # onboarding completed and cleared


@pytest.mark.asyncio
async def test_check_approval_aborts_when_authority_npub_persist_fails():
    # A vault-write failure persisting the authority npub must abort activation
    # cleanly — never report a phantom "activated" with the cert-critical key
    # absent from the vault (it would vanish on restart, breaking certificate
    # verification). Regression for the M1.4 §2 re-raise fix.
    state = OnboardingState()
    state.start_claim("npub1cand")
    state.promote_to_approval("npub1parent")
    exchange = FakeExchange()
    with _tools() as tools, patch.object(at, "_onboarding", state), \
             _exchange(exchange), \
             patch.object(at, "_set_authority_npub",
                      AsyncMock(side_effect=RuntimeError("neon unreachable"))):
        r = await tools["check_authority_approval"](candidate_npub="npub1cand")
    assert r["success"] is False
    assert "persist authority npub" in r["error"]
    assert "neon unreachable" in r["error"]
    assert r.get("activated") is not True
    assert state.get() is not None  # onboarding NOT completed — retryable


@pytest.mark.asyncio
async def test_set_authority_npub_caches_only_after_successful_write():
    vault = SimpleNamespace(set_config=AsyncMock())
    rt = SimpleNamespace(vault=AsyncMock(return_value=vault))
    with patch.object(at, "_get_runtime", MagicMock(return_value=rt)), \
         patch.object(at, "_cached_authority_npub", None):
        await at._set_authority_npub("npub1xyz")
        vault.set_config.assert_awaited_once_with("authority_npub", "npub1xyz")
        assert at._cached_authority_npub == "npub1xyz"


@pytest.mark.asyncio
async def test_set_authority_npub_propagates_write_failure_without_caching():
    vault = SimpleNamespace(set_config=AsyncMock(side_effect=RuntimeError("neon down")))
    rt = SimpleNamespace(vault=AsyncMock(return_value=vault))
    with patch.object(at, "_get_runtime", MagicMock(return_value=rt)), \
         patch.object(at, "_cached_authority_npub", None):
        with pytest.raises(RuntimeError, match="neon down"):
            await at._set_authority_npub("npub1xyz")
        assert at._cached_authority_npub is None  # never cached on failed write
