"""Characterization net for OperatorRuntime.debit_or_deny (audit M2.2 / M0.3).

debit_or_deny is THE money gate — every paid tool call flows through its
10-stage pipeline (identity → proof → restricted-access → pricing →
caller-resolve → constraints → coupons → no-charge / credit / billing). It was
only exercised indirectly via the paid_tool decorator, and the constraint /
pricing-failure branches were unexercised (the decorator's fake resolver lacks
get_chain / last_error_*). This pins the per-stage behavior before the Phase 2
decomposition. Proof verification is patched out so the other stages can be
characterized in isolation; the proof stage is pinned by its own test.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.constants import ErrorCode
from tollbooth.runtime import OperatorRuntime
from tollbooth.tool_identity import ToolIdentity, capability_uuid

PATRON = PrivateKey().public_key.bech32()
OTHER = PrivateKey().public_key.bech32()

_PASS_PROOF = patch("tollbooth.runtime.require_proof", AsyncMock(return_value=None))


class FakeResolver:
    def __init__(self, *, cost=10, chain=(), neon=True, permanent=False, quota=False, priced=True, has=True):
        self._cost = cost
        self._chain = list(chain)
        self._neon = neon
        self.last_error_permanent = permanent
        self.last_error_quota = quota
        self.last_error_summary = "neon rejected the query"
        self._priced = priced
        self._has = has

    @property
    def neon_available(self):
        return self._neon

    async def _ensure_fresh(self):
        pass

    async def has_tool(self, tid):
        return self._has

    async def is_priced(self, tid):
        return self._priced

    async def get_tool_pricing(self, tid):
        from tollbooth.pricing import ToolPricing
        return ToolPricing(fixed=self._cost)

    async def get_chain(self, tid):
        return self._chain


class FakeGate:
    """Stand-in ConstraintGate: returns a canned (denial, effective, consumed)."""

    def __init__(self, *, denial=None, effective=10, consumed=()):
        self._denial = denial
        self._effective = effective
        self._consumed = list(consumed)

    def collect_coupon_ids(self, chain):
        return []

    def evaluate_chain(self, **kw):
        return self._denial, self._effective, self._consumed


class FakeLedgerCache:
    def __init__(self, balance=1000):
        from tollbooth.ledger import Tranche
        from datetime import datetime, timezone
        self._ledgers: dict = {}
        self._balance = balance
        self._Tranche = Tranche
        self._now = datetime.now(timezone.utc).isoformat()

    async def get(self, npub):
        if npub not in self._ledgers:
            from tollbooth.ledger import UserLedger
            led = UserLedger()
            led.tranches.append(self._Tranche(
                granted_at=self._now, original_sats=self._balance,
                remaining_sats=self._balance, invoice_id="seed",
            ))
            self._ledgers[npub] = led
        return self._ledgers[npub]

    async def get_fresh(self, npub):
        return await self.get(npub)

    async def mutate(self, npub, fn, *, retries=6):
        led = await self.get(npub)
        return fn(led)

    async def debit(self, npub, tool_name, cost):
        led = await self.get(npub)
        return led.debit(tool_name, cost)

    async def credit(self, npub, api_sats, invoice_id, *, ttl_seconds=None):
        led = await self.get(npub)
        led.credit_deposit(api_sats, invoice_id, ttl_seconds=ttl_seconds)

    def mark_dirty(self, npub):
        pass

    async def flush_user(self, npub):
        return True


def _registry(name="read_tool", category="read"):
    tid = capability_uuid(name)
    return {tid: ToolIdentity(tool_id=tid, capability=name, category=category, intent="t")}, tid


def _runtime(registry, *, balance=1000, resolver=None, gate=None, operator_npub=PATRON):
    rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
    rt._pricing_resolver = resolver if resolver is not None else FakeResolver()
    rt._ledger_cache = FakeLedgerCache(balance)
    rt._proven_npub_cache = MagicMock()
    rt._operator_npub = operator_npub
    rt.get_global_demand = AsyncMock(return_value={})
    rt.resolve_tranche_lifetime = AsyncMock(return_value=None)
    rt.coupons_vault = AsyncMock(return_value=MagicMock(burn_use=AsyncMock()))
    if gate is not None:
        rt._constraint_gate = gate
    return rt


# ── identity / proof / access ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_tool_denied():
    rt = _runtime({})
    r = await rt.debit_or_deny("not-a-real-uuid", PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.TOOL_NOT_REGISTERED


@pytest.mark.asyncio
async def test_proof_failure_is_returned_verbatim():
    registry, tid = _registry()
    rt = _runtime(registry)
    err = {"success": False, "error_code": ErrorCode.PROOF_REQUIRED, "error": "need proof"}
    with patch("tollbooth.runtime.require_proof", AsyncMock(return_value=err)):
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="")
    assert r is err


@pytest.mark.asyncio
async def test_restricted_allows_operator_free():
    registry, tid = _registry(category="restricted")
    rt = _runtime(registry, operator_npub=PATRON)
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r == 0  # operator calling its own restricted tool — free


@pytest.mark.asyncio
async def test_restricted_denies_non_operator():
    registry, tid = _registry(category="restricted")
    rt = _runtime(registry, operator_npub=OTHER)
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.RESTRICTED


# ── pricing ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pricing_warming_up_when_neon_unavailable():
    registry, tid = _registry()
    rt = _runtime(registry, resolver=FakeResolver(neon=False, permanent=False))
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.WARMING_UP


@pytest.mark.asyncio
async def test_pricing_misconfigured_when_permanent_error():
    registry, tid = _registry()
    rt = _runtime(registry, resolver=FakeResolver(neon=False, permanent=True))
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.PERSISTENCE_MISCONFIGURED


@pytest.mark.asyncio
async def test_pricing_quota_exceeded_is_not_warming_up():
    """A Neon 402 must classify as quota_exceeded — NOT warming_up — so the
    patron is never told to 'retry' through a provider outage, and the
    Authority is alerted."""
    registry, tid = _registry()
    rt = _runtime(registry, resolver=FakeResolver(neon=False, quota=True))
    alert = AsyncMock(return_value=None)
    with _PASS_PROOF, patch.object(rt, "_alert_authority_quota_exceeded", alert):
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.PERSISTENCE_QUOTA_EXCEEDED
    # The honest message states non-transience — it must NOT tell the patron to
    # simply retry (the old warming_up copy did exactly that through an outage).
    assert "will not help" in r["error"].lower()
    assert "warming up" not in r["error"].lower()
    # The Authority was notified out of band.
    alert.assert_awaited_once()


@pytest.mark.asyncio
async def test_tool_not_priced():
    registry, tid = _registry()
    rt = _runtime(registry, resolver=FakeResolver(priced=False))
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.TOOL_NOT_PRICED


@pytest.mark.asyncio
async def test_free_tool_returns_zero():
    registry, tid = _registry(category="free")
    rt = _runtime(registry)
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r == 0


# ── billing ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_happy_path_debits_and_returns_cost():
    registry, tid = _registry()
    rt = _runtime(registry, balance=1000, resolver=FakeResolver(cost=10))
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r == 10
    ledger = await (await rt.ledger_cache()).get(PATRON)
    assert ledger.balance_api_sats == 990  # debited


@pytest.mark.asyncio
async def test_insufficient_balance_denied():
    registry, tid = _registry()
    rt = _runtime(registry, balance=5, resolver=FakeResolver(cost=100))
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.INSUFFICIENT_BALANCE
    assert r["balance_sats"] == 5 and r["cost_sats"] == 100


# ── constraints ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_constraint_denial_returns_constraint_denied():
    registry, tid = _registry()
    resolver = FakeResolver(cost=10, chain=["step"])
    gate = FakeGate(denial={"success": False, "constraint_reason": "rate limited"})
    rt = _runtime(registry, resolver=resolver, gate=gate)
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r["error_code"] == ErrorCode.CONSTRAINT_DENIED
    assert r["constraint_reason"] == "rate limited"


@pytest.mark.asyncio
async def test_constraint_discount_debits_effective_cost():
    registry, tid = _registry()
    resolver = FakeResolver(cost=100, chain=["step"])
    gate = FakeGate(effective=30)  # chain discounted 100 → 30
    rt = _runtime(registry, balance=1000, resolver=resolver, gate=gate)
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r == 30
    ledger = await (await rt.ledger_cache()).get(PATRON)
    assert ledger.balance_api_sats == 970


@pytest.mark.asyncio
async def test_invalid_npub_denied_for_paid_tool():
    registry, tid = _registry()
    rt = _runtime(registry)
    r = await rt.debit_or_deny(tid, "not-an-npub", dpop_token="p")
    assert r["error_code"] == ErrorCode.NPUB_INVALID


@pytest.mark.asyncio
async def test_consumed_coupons_are_burned():
    registry, tid = _registry()
    resolver = FakeResolver(cost=10, chain=["step"])
    gate = FakeGate(effective=0, consumed=["coupon-1"])  # discounted to free
    rt = _runtime(registry, resolver=resolver, gate=gate)
    burn = AsyncMock()
    rt.coupons_vault = AsyncMock(return_value=MagicMock(burn_use=burn))
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r == 0
    burn.assert_awaited_once_with("coupon-1", PATRON)


# ── resolve_tranche_lifetime (regression: was silently broken 2026-03-31 →
#    2026-06-11; called a non-existent ensure_pricing_store, masked by bare except) ──

@pytest.mark.asyncio
async def test_resolve_tranche_lifetime_reads_ttl_from_pricing_model():
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    rt._operator_npub = PATRON
    rt.vault = AsyncMock(return_value=MagicMock())
    with patch("tollbooth.pricing_store.PricingModelStore"), \
         patch("tollbooth.tools.pricing.get_pricing_model_tool",
               new=AsyncMock(return_value={"status": "ok", "tranche_lifetime": {"ttl_days": 7}})):
        ttl = await rt.resolve_tranche_lifetime()
    assert ttl == 7 * 86400  # honors the pricing model's tranche lifetime


@pytest.mark.asyncio
async def test_resolve_tranche_lifetime_none_when_unset():
    rt = OperatorRuntime(tool_registry={}, nsec_env_var="__UNUSED__")
    rt._operator_npub = PATRON
    rt.vault = AsyncMock(return_value=MagicMock())
    with patch("tollbooth.pricing_store.PricingModelStore"), \
         patch("tollbooth.tools.pricing.get_pricing_model_tool",
               new=AsyncMock(return_value={"status": "ok"})):  # no tranche_lifetime
        ttl = await rt.resolve_tranche_lifetime()
    assert ttl is None


@pytest.mark.asyncio
async def test_constraint_credit_deposits_and_returns_signed():
    registry, tid = _registry()
    resolver = FakeResolver(cost=10, chain=["step"])
    gate = FakeGate(effective=-50)  # chain drove price negative → credit
    rt = _runtime(registry, balance=1000, resolver=resolver, gate=gate)
    with _PASS_PROOF:
        r = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert r == -50  # signed — caller detects the credit case
    ledger = await (await rt.ledger_cache()).get(PATRON)
    assert ledger.balance_api_sats == 1050  # credited 50
