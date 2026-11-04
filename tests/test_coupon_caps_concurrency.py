"""Concurrent calls cannot spend a capped coupon past its caps (scan F36/F37).

Every call prices from a snapshot of the coupon's counts. Before the fix the
use was burned after the debit, with no cap, so 20 calls fired at once all read
use_count=0 and all got the discount. Now a call keeps the discount only after
reserving a use under the caps. A miscount may under-honor a coupon, never
over-honor it.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from tollbooth.constants import ErrorCode
from tollbooth.coupons.models import Coupon, CouponRedemption

from .test_coupon_bindings import _PASS_PROOF, OPERATOR, _registry, _runtime
from .test_credit_gate import PATRON, FakeLedgerCache, FakeResolver

NOW = datetime.now(UTC)
CID = "22222222-2222-4222-8222-222222222222"


class CountingCoupons:
    """A coupons vault whose counters and caps behave like the capped SQL."""

    def __init__(self, *, percent=100.0, uses_per_patron=None, total_uses=None, fail_reserve=False):
        self.coupon = Coupon(
            id=CID, operator=OPERATOR, name="LAUNCH", discount_percent=percent,
            valid_from=NOW - timedelta(days=1), valid_until=NOW + timedelta(days=30),
            uses_per_patron=uses_per_patron, total_uses=total_uses, tool_ids=["*"],
        )
        self.use_count: dict[str, int] = {}
        self.times_redeemed = 0
        self.fail_reserve = fail_reserve

    async def bound_for_operator(self, operator):
        return [self.coupon]

    async def fetch_redemptions_for_chain(self, npub, coupon_ids):
        c = self.coupon
        view = CouponRedemption(
            coupon_id=c.id, name=c.name, discount_percent=c.discount_percent,
            valid_from=c.valid_from, valid_until=c.valid_until,
            uses_per_patron=c.uses_per_patron, total_uses=c.total_uses,
            times_redeemed=self.times_redeemed, use_count=self.use_count.get(npub, 0),
        )
        await asyncio.sleep(0)  # let every concurrent call read the same counts
        return {c.id: view} if c.id in coupon_ids else {}

    async def reserve_use(self, coupon_id, npub):
        await asyncio.sleep(0)
        if self.fail_reserve:
            raise RuntimeError("neon down")
        c = self.coupon
        if c.uses_per_patron is not None and self.use_count.get(npub, 0) >= c.uses_per_patron:
            return False
        if c.total_uses is not None and self.times_redeemed >= c.total_uses:
            return False
        self.use_count[npub] = self.use_count.get(npub, 0) + 1
        self.times_redeemed += 1
        return True

    async def release_use(self, coupon_id, npub):
        self.use_count[npub] -= 1
        self.times_redeemed -= 1
        return True


async def _burst(coupons, *, calls=20, cost=20, balance=1000):
    registry = _registry("read_tool")
    tid = next(iter(registry))
    rt = _runtime(registry, resolver=FakeResolver(cost=cost, chain=[]), coupons=coupons)
    rt._ledger_cache = FakeLedgerCache(balance)
    with _PASS_PROOF:
        return await asyncio.gather(*(rt.debit_or_deny(tid, PATRON) for _ in range(calls)))


@pytest.mark.asyncio
async def test_a_single_use_coupon_discounts_one_of_twenty_concurrent_calls():
    coupons = CountingCoupons(uses_per_patron=1)
    charged = await _burst(coupons)
    assert sorted(charged) == [0] + [20] * 19
    assert coupons.use_count[PATRON] == 1 and coupons.times_redeemed == 1


@pytest.mark.asyncio
async def test_a_total_cap_of_three_discounts_three_of_twenty_concurrent_calls():
    coupons = CountingCoupons(total_uses=3)
    charged = await _burst(coupons)
    assert charged.count(0) == 3 and charged.count(20) == 17
    assert coupons.times_redeemed == 3


@pytest.mark.asyncio
async def test_a_reservation_error_charges_full_fare():
    coupons = CountingCoupons(uses_per_patron=1, fail_reserve=True)
    charged = await _burst(coupons, calls=3)
    assert charged == [20, 20, 20]
    assert coupons.times_redeemed == 0


@pytest.mark.asyncio
async def test_a_call_that_cannot_pay_gives_its_use_back():
    coupons = CountingCoupons(percent=50.0, uses_per_patron=1)
    [denied] = await _burst(coupons, calls=1, cost=20, balance=5)
    assert denied["error_code"] == ErrorCode.INSUFFICIENT_BALANCE
    assert coupons.use_count[PATRON] == 0 and coupons.times_redeemed == 0
