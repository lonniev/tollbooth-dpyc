"""Per-patron constraints decide by the patron's real calls of the tool."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from tollbooth.constraints.gate import ConstraintGate, reads_patron_usage
from tollbooth.ledger import UserLedger
from tollbooth.ledger_cache import LedgerCache
from tollbooth.pricing_model import PipelineStep
from tollbooth.runtime import OperatorRuntime

NPUB = "npub1patron"
TOOL = "svc_paid"


def _step(type_, **params):
    return PipelineStep(id="s1", type=type_, params=params)


class _Vault:
    """The ledger store, reduced to what LedgerCache asks of it."""

    def __init__(self, ledger=None, readable=True):
        self.text = ledger.to_json() if ledger else None
        self.readable = readable

    async def fetch_ledger(self, _user):
        if not self.readable:
            raise RuntimeError("store down")
        return (self.text, 1) if self.text else None

    async def store_ledger(self, _user, text, _version=None):
        self.text = text


def _runtime(chain, vault):
    rt = OperatorRuntime(tool_registry={}, service_name="Test Operator")
    cache = LedgerCache(vault)
    rt.ledger_cache = AsyncMock(return_value=cache)
    gate = ConstraintGate()
    rt._effective_chain = AsyncMock(return_value=(chain, None, gate))
    rt.get_global_demand = AsyncMock(return_value={})
    return rt, cache


def _used(calls: int) -> UserLedger:
    ledger = UserLedger()
    ledger.record_usage(TOOL, calls)
    return ledger


async def _cost(rt, base=50):
    return await rt._evaluate_constraints("tid", TOOL, NPUB, base)


@pytest.mark.asyncio
async def test_a_free_trial_ends_after_its_calls():
    rt, _ = _runtime([_step("free_trial", first_n_free=3)], _Vault(_used(2)))
    assert (await _cost(rt))[0] == 0
    rt, _ = _runtime([_step("free_trial", first_n_free=3)], _Vault(_used(3)))
    assert (await _cost(rt))[0] == 50


@pytest.mark.asyncio
async def test_free_calls_not_yet_folded_still_use_up_the_trial():
    rt, cache = _runtime([_step("free_trial", first_n_free=3)], _Vault(_used(2)))
    cache.note_usage(NPUB, TOOL)
    assert (await _cost(rt))[0] == 50


@pytest.mark.asyncio
async def test_a_per_patron_cap_denies_and_another_tools_calls_do_not_count():
    chain = [_step("finite_supply", max_invocations=5, scope="per_patron")]
    rt, _ = _runtime(chain, _Vault(_used(5)))
    assert (await _cost(rt))[2]["constraint_reason"] == "supply_exhausted"

    other = UserLedger()
    other.record_usage("svc_other", 9)
    rt, _ = _runtime(chain, _Vault(other))
    assert (await _cost(rt))[2] is None


@pytest.mark.asyncio
async def test_a_per_patron_window_denies_within_it_and_opens_again_after():
    chain = [_step("periodic_refresh", max_per_window=2, window_duration="P1D", scope="per_patron")]
    rt, _ = _runtime(chain, _Vault(_used(2)))
    assert (await _cost(rt))[2]["constraint_reason"] == "rate_limited"

    # The same two calls made long ago: a lifetime count would lock the patron out for good.
    old = _used(2)
    long_ago = (date.today() - timedelta(days=5)).isoformat()
    old.daily_log = {long_ago: old.daily_log[date.today().isoformat()]}
    rt, _ = _runtime(chain, _Vault(old))
    assert (await _cost(rt))[2] is None


@pytest.mark.asyncio
async def test_an_unreadable_ledger_refuses_without_charging_when_the_chain_needs_counts():
    rt, _ = _runtime([_step("free_trial", first_n_free=3)], _Vault(readable=False))
    _, consumed, denial = await _cost(rt)
    assert denial["error_code"] == "vault_unavailable"
    assert consumed == []


@pytest.mark.asyncio
async def test_a_chain_that_needs_no_counts_does_not_reread_the_ledger():
    rt, cache = _runtime([_step("coupon", coupon_id="c1")], _Vault(UserLedger()))
    cache.get_fresh = AsyncMock()
    await _cost(rt)
    cache.get_fresh.assert_not_awaited()


def test_only_usage_steps_read_usage():
    reads = reads_patron_usage
    assert reads([_step("free_trial", first_n_free=1)])
    assert reads([_step("finite_supply", max_invocations=1, scope="per_patron")])
    assert reads([_step("expression", rule={"field": "env.invocation_count", "op": "<", "value": 3})])
    assert not reads([_step("finite_supply", max_invocations=1, scope="global")])
    assert not reads([_step("coupon", coupon_id="c1")])


def test_a_window_counts_each_day_it_touches():
    from tollbooth.constraints.base import EnvironmentSnapshot

    env = EnvironmentSnapshot(
        utc_now=datetime(2026, 11, 1, 2, tzinfo=UTC),
        daily_calls=(("2026-10-30", 7), ("2026-10-31", 2), ("2026-11-01", 1)),
    )
    assert env.calls_since(datetime(2026, 10, 31, 21, tzinfo=UTC)) == 3
