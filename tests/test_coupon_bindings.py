"""A coupon owns its binding, so re-pricing a tool cannot drop the discount.

The trap this closes (Good Earth, 2026-10-07; Bee's Knees, 2026-09-09): a
coupon bound by a ``coupon`` step in the pricing model's chain vanished the
moment a price push arrived without the chain, and every paid call went out
at list price with nothing to say so. Here the model has NO chain at all and
the discount still applies, because the gate derives the step from the
coupon's own ``tool_ids``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.coupons.models import Coupon, CouponRedemption
from tollbooth.pricing_model import PipelineStep
from tollbooth.runtime import OperatorRuntime
from tollbooth.tool_identity import ToolIdentity, capability_uuid

from .test_credit_gate import PATRON, FakeLedgerCache, FakeResolver

OPERATOR = PrivateKey().public_key.bech32()
NOW = datetime.now(UTC)
CID = "11111111-1111-4111-8111-111111111111"
_PASS_PROOF = patch("tollbooth.runtime.require_proof", AsyncMock(return_value=None))


def _registry(*names):
    reg = {}
    for n in names:
        tid = capability_uuid(n)
        reg[tid] = ToolIdentity(tool_id=tid, capability=n, category="read", intent="t")
    return reg


def _coupon(tool_ids, percent=100.0, cid=CID):
    return Coupon(
        id=cid, operator=OPERATOR, name="PIONEER", discount_percent=percent,
        valid_from=NOW - timedelta(days=1), valid_until=NOW + timedelta(days=30),
        uses_per_patron=None, tool_ids=list(tool_ids),
    )


def _redemption(c: Coupon) -> CouponRedemption:
    return CouponRedemption(
        coupon_id=c.id, name=c.name, discount_percent=c.discount_percent,
        valid_from=c.valid_from, valid_until=c.valid_until,
        uses_per_patron=None, total_uses=None, times_redeemed=0, use_count=0,
    )


class FakeCoupons:
    """The coupons vault as the gate sees it: bound coupons, redemptions, burns."""

    def __init__(self, bound):
        self.bound = list(bound)
        self.bound_calls = 0
        self.burned = []

    async def bound_for_operator(self, operator):
        assert operator == OPERATOR
        self.bound_calls += 1
        return list(self.bound)

    async def fetch_redemptions_for_chain(self, npub, coupon_ids):
        return {c.id: _redemption(c) for c in self.bound if c.id in coupon_ids}

    async def burn_use(self, coupon_id, npub):
        self.burned.append(coupon_id)
        return True


def _runtime(registry, *, resolver, coupons):
    rt = OperatorRuntime(tool_registry=registry, nsec_env_var="__UNUSED__")
    rt._pricing_resolver = resolver
    rt._ledger_cache = FakeLedgerCache(1000)
    rt._proof_grant_revocations = MagicMock()
    rt._operator_npub = OPERATOR
    rt.get_global_demand = AsyncMock(return_value={})
    rt.resolve_tranche_lifetime = AsyncMock(return_value=None)
    rt.coupons_vault = AsyncMock(return_value=coupons)
    return rt


@pytest.mark.asyncio
async def test_a_model_pushed_without_chains_still_honours_the_bound_coupon():
    registry = _registry("read_tool")
    tid = next(iter(registry))
    coupons = FakeCoupons([_coupon([tid])])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[]), coupons=coupons)
    with _PASS_PROOF:
        charged = await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert charged == 0, "the model has no chain and the coupon still applies"
    assert coupons.burned == [CID]
    ledger = await (await rt.ledger_cache()).get(PATRON)
    assert ledger.balance_api_sats == 1000


@pytest.mark.asyncio
async def test_star_binds_every_paid_tool_and_an_empty_binding_binds_none():
    registry = _registry("a_tool", "b_tool")
    a, b = registry
    star = FakeCoupons([_coupon(["*"], percent=50.0)])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[]), coupons=star)
    with _PASS_PROOF:
        assert await rt.debit_or_deny(a, PATRON, dpop_token="p") == 10
        assert await rt.debit_or_deny(b, PATRON, dpop_token="p") == 10

    none = FakeCoupons([_coupon([], percent=50.0)])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[]), coupons=none)
    with _PASS_PROOF:
        assert await rt.debit_or_deny(a, PATRON, dpop_token="p") == 20
    assert none.burned == []


@pytest.mark.asyncio
async def test_a_coupon_authored_in_the_chain_and_bound_on_the_row_applies_once():
    registry = _registry("read_tool")
    tid = next(iter(registry))
    authored = PipelineStep(id="s1", type="coupon", params={"coupon_id": CID})
    coupons = FakeCoupons([_coupon([tid], percent=50.0)])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[authored]), coupons=coupons)
    chain, _ = await rt._effective_chain(tid, "read_tool", PATRON)
    assert [s.id for s in chain] == ["s1"], "the authored step is where the operator put it; no second one"
    with _PASS_PROOF:
        assert await rt.debit_or_deny(tid, PATRON, dpop_token="p") == 10  # 50 % once, not 75 %
    assert coupons.burned == [CID]


@pytest.mark.asyncio
async def test_bound_coupons_are_read_once_per_ttl_and_again_after_a_coupon_write():
    registry = _registry("read_tool")
    tid = next(iter(registry))
    coupons = FakeCoupons([_coupon([tid])])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[]), coupons=coupons)
    with _PASS_PROOF:
        await rt.debit_or_deny(tid, PATRON, dpop_token="p")
        await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert coupons.bound_calls == 1
    rt._forget_bound_coupons()
    with _PASS_PROOF:
        await rt.debit_or_deny(tid, PATRON, dpop_token="p")
    assert coupons.bound_calls == 2


@pytest.mark.asyncio
async def test_a_vault_outage_keeps_the_last_bindings_rather_than_dropping_them():
    registry = _registry("read_tool")
    tid = next(iter(registry))
    coupons = FakeCoupons([_coupon([tid])])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[]), coupons=coupons)
    assert len(await rt._bound_coupons()) == 1
    rt._bound_coupons_memo = (rt._bound_coupons_memo[0] - 10_000, rt._bound_coupons_memo[1])  # expired

    async def down(_op):
        raise RuntimeError("neon down")

    coupons.bound_for_operator = down
    assert len(await rt._bound_coupons()) == 1


def test_coupon_is_not_offered_as_an_authorable_step_but_still_evaluates():
    from tollbooth.constraints import CONSTRAINT_REGISTRY
    from tollbooth.tools.pricing import list_constraint_types

    assert "coupon" not in {s["type"] for s in list_constraint_types()}
    assert "coupon" in CONSTRAINT_REGISTRY


def test_set_pricing_model_counts_the_coupon_steps_it_carries():
    import asyncio
    import json

    from tollbooth.tools.pricing import set_pricing_model_tool

    store = MagicMock()
    store.fetch_active_model = AsyncMock(return_value=None)
    store.create_model = AsyncMock(return_value="m1")
    store.activate_model = AsyncMock()
    tid = capability_uuid("read_tool")
    model = {"name": "m", "tools": [
        {"tool_id": tid, "tool_name": "read_tool", "price_sats": 20, "category": "read",
         "chain": [{"id": "s1", "type": "coupon", "params": {"coupon_id": CID}}]},
    ]}
    r = asyncio.run(set_pricing_model_tool(store, OPERATOR, json.dumps(model)))
    assert r["status"] == "ok" and r["coupon_steps_in_model"] == 1


@pytest.mark.asyncio
async def test_check_price_previews_the_bound_coupon_exactly_as_the_debit_will_charge():
    """The probe that found the bug now answers it: a bound coupon shows in the preview."""
    from tollbooth.runtime import register_standard_tools

    registry = _registry("read_tool")
    tid = next(iter(registry))
    coupons = FakeCoupons([_coupon([tid])])
    rt = _runtime(registry, resolver=FakeResolver(cost=20, chain=[]), coupons=coupons)
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")

    r = await tools["check_price"](tool_id=tid, npub=PATRON)
    assert r["success"] and r["constraints_enabled"] is True
    assert r["effective_cost_api_sats"] == 0 and r["base_cost_api_sats"] == 20
    assert coupons.burned == [], "a preview burns nothing"

    r = await tools["check_price"](tool_id=tid)
    assert r["constraints_enabled"] is True and r["effective_cost_api_sats"] == 20
    assert any("npub required" in e["message"] for e in r["constraint_effects"])


def test_resolve_tool_ids_accepts_ids_or_names_and_refuses_strangers():
    from tollbooth.tools.coupons import resolve_tool_ids

    known = {"id-a": "svc_a_tool", "id-b": "svc_b_tool"}
    assert resolve_tool_ids(["id-a", "svc_b_tool", "id-a"], known) == (["id-a", "id-b"], None)
    assert resolve_tool_ids(["*"], known) == (["*"], None)
    assert resolve_tool_ids([], known) == ([], None)
    _, why = resolve_tool_ids(["id-a", "nobody_tool"], known)
    assert why and "nobody_tool" in why
    _, why = resolve_tool_ids(["*", "id-a"], known)
    assert why and "stands alone" in why
    _, why = resolve_tool_ids("id-a", known)
    assert why and "list" in why
    _, why = resolve_tool_ids([1, 2], known)
    assert why


@pytest.mark.asyncio
async def test_mint_and_update_carry_the_binding_and_name_a_refused_tool():
    from tollbooth.tools import coupons as ct

    class _CV:
        def __init__(self):
            self.kw = {}

        async def mint(self, **kw):
            self.kw = kw
            return _coupon(kw["tool_ids"])

        async def update(self, cid, op, **kw):
            self.kw = kw
            return _coupon(kw.get("tool_ids", ["old"]))

    cv = _CV()
    known = {"id-a": "svc_a_tool"}
    r = await ct.mint_coupon_tool(
        cv, OPERATOR, name="X", discount_percent=10, valid_from="2026-01-01T00:00:00Z",
        valid_until="2026-02-01T00:00:00Z", uses_per_patron=None, total_uses=None,
        tool_ids=["svc_a_tool"], known_tools=known,
    )
    assert r["success"] and cv.kw["tool_ids"] == ["id-a"] and r["coupon"]["applies_to"] == "1 tool"

    r = await ct.mint_coupon_tool(
        cv, OPERATOR, name="X", discount_percent=10, valid_from="2026-01-01T00:00:00Z",
        valid_until="2026-02-01T00:00:00Z", uses_per_patron=None, total_uses=None,
        tool_ids=["ghost"], known_tools=known,
    )
    assert not r["success"] and "ghost" in r["error"]

    r = await ct.update_coupon_tool(
        cv, OPERATOR, CID, name=None, discount_percent=None, valid_from=None, valid_until=None,
        uses_per_patron=None, total_uses=None, clear_uses_per_patron=False, clear_total_uses=False,
        tool_ids=["*"], known_tools=known,
    )
    assert r["success"] and cv.kw["tool_ids"] == ["*"] and r["coupon"]["applies_to"] == "every paid tool"

    r = await ct.update_coupon_tool(
        cv, OPERATOR, CID, name="Y", discount_percent=None, valid_from=None, valid_until=None,
        uses_per_patron=None, total_uses=None, clear_uses_per_patron=False, clear_total_uses=False,
    )
    assert r["success"] and "tool_ids" not in cv.kw, "omitted = the binding is left alone"
