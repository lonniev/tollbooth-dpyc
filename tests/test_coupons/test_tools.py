"""Direct unit tests for tollbooth.tools.coupons (audit M2.1a).

These exercise the coupon tool bodies extracted out of register_standard_tools.
Before extraction this validation/branching logic was trapped in closures and
could not be tested without a full runtime + FastMCP. Here a lightweight fake
CouponsVault stands in for the real Neon-backed vault.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tollbooth.coupons.vault import CouponAlreadyExists, CouponNotFound
from tollbooth.tools import coupons as ct

OP = "npub1operator"
PATRON = "npub1patron"
_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _coupon(**over):
    """A fake coupon row exposing the attributes the tool bodies read.

    Default window brackets the real ``datetime.now`` the redeem path uses,
    so an unmodified fixture reads as currently active.
    """
    now = datetime.now(timezone.utc)
    base = dict(
        id="c-1",
        name="SUMMER",
        discount_percent=10.0,
        valid_from=now - timedelta(days=1),
        valid_until=now + timedelta(days=1),
        uses_per_patron=1,
        total_uses=None,
        times_redeemed=0,
    )
    base.update(over)
    ns = SimpleNamespace(**base)
    ns.to_dict = lambda: {k: base[k] for k in ("id", "name", "discount_percent")}
    return ns


class FakeVault:
    """Records calls and returns canned results / raises canned errors."""

    def __init__(self, **handlers):
        self._h = handlers
        self.calls = []

    async def _run(self, key, *args, **kwargs):
        self.calls.append((key, args, kwargs))
        h = self._h.get(key)
        if isinstance(h, Exception):
            raise h
        if callable(h):
            return h(*args, **kwargs)
        return h

    async def mint(self, **kw): return await self._run("mint", **kw)
    async def list_for_operator(self, op): return await self._run("list_for_operator", op)
    async def update(self, cid, op, **kw): return await self._run("update", cid, op, **kw)
    async def delete(self, cid, op): return await self._run("delete", cid, op)
    async def find_by_name(self, op, code): return await self._run("find_by_name", op, code)
    async def redeem(self, cid, patron): return await self._run("redeem", cid, patron)
    async def list_redemptions_for_patron(self, patron, operator=None):
        return await self._run("list_redemptions_for_patron", patron, operator=operator)
    async def forget(self, cid, patron): return await self._run("forget", cid, patron)


# ── mint ──────────────────────────────────────────────────────────────

async def _mint(cv, **over):
    kw = dict(
        name="SUMMER", discount_percent=10.0,
        valid_from="2026-06-01T00:00:00Z", valid_until="2026-07-01T00:00:00Z",
        uses_per_patron=1, total_uses=None,
    )
    kw.update(over)
    return await ct.mint_coupon_tool(cv, OP, **kw)


@pytest.mark.asyncio
async def test_mint_success():
    cv = FakeVault(mint=lambda **kw: _coupon(name=kw["name"], total_uses=5, times_redeemed=2))
    r = await _mint(cv)
    assert r["success"] is True
    assert r["coupon"]["name"] == "SUMMER"
    assert r["coupon"]["progress"] == "2 / 5"
    # validated + coerced values reached the vault
    _, _, kw = cv.calls[0]
    assert kw["operator"] == OP and kw["discount_percent"] == 10.0


@pytest.mark.asyncio
async def test_mint_unlimited_total_uses_progress_infinity():
    cv = FakeVault(mint=lambda **kw: _coupon(total_uses=None, times_redeemed=3))
    r = await _mint(cv)
    assert r["coupon"]["progress"] == "3 / ∞"


@pytest.mark.asyncio
@pytest.mark.parametrize("over,frag", [
    ({"name": "  "}, "name is required"),
    ({"discount_percent": 0}, "discount_percent must be in (0, 100]"),
    ({"discount_percent": 150}, "discount_percent must be in (0, 100]"),
    ({"valid_from": "not-a-date"}, "valid_from must be ISO-8601"),
    ({"valid_from": "2026-07-01T00:00:00Z", "valid_until": "2026-06-01T00:00:00Z"},
     "valid_until must be strictly later"),
    ({"uses_per_patron": 0}, "uses_per_patron must be a positive integer"),
    ({"total_uses": -1}, "total_uses must be a positive integer"),
])
async def test_mint_validation_rejects(over, frag):
    cv = FakeVault(mint=lambda **kw: _coupon())
    r = await _mint(cv, **over)
    assert r["success"] is False
    assert frag in r["error"]
    assert cv.calls == []  # never reached the vault


@pytest.mark.asyncio
async def test_mint_already_exists():
    cv = FakeVault(mint=CouponAlreadyExists("dupe SUMMER"))
    r = await _mint(cv)
    assert r["success"] is False and "dupe SUMMER" in r["error"]


@pytest.mark.asyncio
async def test_mint_generic_failure_stringified():
    cv = FakeVault(mint=RuntimeError("neon down"))
    r = await _mint(cv)
    assert r["success"] is False and r["error"] == "mint failed: neon down"


# ── list / delete / forget ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_coupons_success():
    cv = FakeVault(list_for_operator=[_coupon(name="A"), _coupon(name="B", total_uses=2)])
    r = await ct.list_coupons_tool(cv, OP)
    assert r["success"] and r["count"] == 2
    assert [c["name"] for c in r["coupons"]] == ["A", "B"]


@pytest.mark.asyncio
async def test_list_coupons_failure():
    cv = FakeVault(list_for_operator=RuntimeError("boom"))
    r = await ct.list_coupons_tool(cv, OP)
    assert r["success"] is False and r["error"] == "list failed: boom"


@pytest.mark.asyncio
async def test_delete_missing_id():
    cv = FakeVault()
    r = await ct.delete_coupon_tool(cv, OP, "")
    assert r["success"] is False and "coupon_id is required" in r["error"]
    assert cv.calls == []


@pytest.mark.asyncio
async def test_delete_not_found_vs_success():
    assert (await ct.delete_coupon_tool(FakeVault(delete=False), OP, "c-1"))["success"] is False
    ok = await ct.delete_coupon_tool(FakeVault(delete=True), OP, "c-1")
    assert ok == {"success": True, "coupon_id": "c-1"}


@pytest.mark.asyncio
async def test_forget_not_found_vs_success():
    assert (await ct.forget_coupon_tool(FakeVault(forget=False), PATRON, "c-1"))["success"] is False
    ok = await ct.forget_coupon_tool(FakeVault(forget=True), PATRON, "c-1")
    assert ok == {"success": True, "coupon_id": "c-1"}


# ── update ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_update_clear_flags_pass_none():
    cv = FakeVault(update=lambda cid, op, **kw: _coupon())
    r = await ct.update_coupon_tool(
        cv, OP, "c-1",
        name=None, discount_percent=None, valid_from=None, valid_until=None,
        uses_per_patron=None, total_uses=None,
        clear_uses_per_patron=True, clear_total_uses=True,
    )
    assert r["success"] is True
    _, _, kw = cv.calls[0]
    assert kw["uses_per_patron"] is None and kw["total_uses"] is None


@pytest.mark.asyncio
async def test_update_not_found():
    cv = FakeVault(update=CouponNotFound())
    r = await ct.update_coupon_tool(
        cv, OP, "c-x",
        name="X", discount_percent=None, valid_from=None, valid_until=None,
        uses_per_patron=None, total_uses=None,
        clear_uses_per_patron=False, clear_total_uses=False,
    )
    assert r["success"] is False and "No coupon 'c-x'" in r["error"]


@pytest.mark.asyncio
async def test_update_requires_coupon_id():
    cv = FakeVault()
    r = await ct.update_coupon_tool(
        cv, OP, "",
        name=None, discount_percent=None, valid_from=None, valid_until=None,
        uses_per_patron=None, total_uses=None,
        clear_uses_per_patron=False, clear_total_uses=False,
    )
    assert r["success"] is False and "coupon_id is required" in r["error"]


def _update_kw(**over):
    kw = dict(
        name=None, discount_percent=None, valid_from=None, valid_until=None,
        uses_per_patron=None, total_uses=None,
        clear_uses_per_patron=False, clear_total_uses=False,
    )
    kw.update(over)
    return kw


@pytest.mark.asyncio
@pytest.mark.parametrize("over,frag", [
    ({"discount_percent": 0}, "discount_percent must be in (0, 100]"),
    ({"discount_percent": 101}, "discount_percent must be in (0, 100]"),
    ({"valid_from": "nope"}, "valid_from must be ISO-8601"),
    ({"valid_until": "nope"}, "valid_until must be ISO-8601"),
    ({"uses_per_patron": 0}, "uses_per_patron must be a positive integer"),
    ({"total_uses": -2}, "total_uses must be a positive integer"),
])
async def test_update_validation_rejects(over, frag):
    cv = FakeVault(update=lambda cid, op, **kw: _coupon())
    r = await ct.update_coupon_tool(cv, OP, "c-1", **_update_kw(**over))
    assert r["success"] is False and frag in r["error"]
    assert cv.calls == []  # rejected before the vault


@pytest.mark.asyncio
async def test_update_success_passes_coerced_values():
    cv = FakeVault(update=lambda cid, op, **kw: _coupon(name="RENAMED", total_uses=8, times_redeemed=1))
    r = await ct.update_coupon_tool(
        cv, OP, "c-1",
        **_update_kw(name=" RENAMED ", discount_percent=25, uses_per_patron=4),
    )
    assert r["success"] is True and r["coupon"]["name"] == "RENAMED"
    _, _, kw = cv.calls[0]
    assert kw["name"] == "RENAMED" and kw["discount_percent"] == 25.0 and kw["uses_per_patron"] == 4


@pytest.mark.asyncio
async def test_update_already_exists_on_rename():
    cv = FakeVault(update=CouponAlreadyExists("name taken"))
    r = await ct.update_coupon_tool(cv, OP, "c-1", **_update_kw(name="DUPE"))
    assert r["success"] is False and "name taken" in r["error"]


@pytest.mark.asyncio
async def test_update_generic_failure_stringified():
    cv = FakeVault(update=RuntimeError("neon down"))
    r = await ct.update_coupon_tool(cv, OP, "c-1", **_update_kw(name="X"))
    assert r["error"] == "update failed: neon down"


@pytest.mark.asyncio
async def test_redeem_generic_failure_stringified():
    coupon = _coupon()
    cv = FakeVault(find_by_name=coupon, redeem=RuntimeError("write fail"))
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "SUMMER")
    assert r["error"] == "redeem failed: write fail"


@pytest.mark.asyncio
async def test_delete_and_forget_failures_and_missing_id():
    assert (await ct.delete_coupon_tool(FakeVault(delete=RuntimeError("x")), OP, "c-1"))["error"] == "delete failed: x"
    assert (await ct.forget_coupon_tool(FakeVault(forget=RuntimeError("y")), PATRON, "c-1"))["error"] == "forget failed: y"
    miss = await ct.forget_coupon_tool(FakeVault(), PATRON, "")
    assert miss["success"] is False and "coupon_id is required" in miss["error"]


# ── redeem ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_redeem_success():
    coupon = _coupon(id="c-9", uses_per_patron=3, total_uses=100, times_redeemed=4)
    cv = FakeVault(find_by_name=coupon, redeem=SimpleNamespace(use_count=1))
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "SUMMER")
    assert r["success"] is True
    assert r["coupon_id"] == "c-9"
    assert r["uses_remaining"] == 2  # 3 - 1


@pytest.mark.asyncio
async def test_redeem_unlimited_per_patron_remaining_none():
    coupon = _coupon(uses_per_patron=None)
    cv = FakeVault(find_by_name=coupon, redeem=SimpleNamespace(use_count=9))
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "SUMMER")
    assert r["uses_remaining"] is None


@pytest.mark.asyncio
async def test_redeem_missing_code():
    cv = FakeVault()
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "   ")
    assert r["success"] is False and "code is required" in r["error"]
    assert cv.calls == []


@pytest.mark.asyncio
async def test_redeem_unknown_code():
    cv = FakeVault(find_by_name=None)
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "NOPE")
    assert r["success"] is False and "No coupon named 'NOPE'" in r["error"]


@pytest.mark.asyncio
async def test_redeem_not_yet_active():
    coupon = _coupon(valid_from=datetime.now(timezone.utc) + timedelta(days=2),
                     valid_until=datetime.now(timezone.utc) + timedelta(days=3))
    cv = FakeVault(find_by_name=coupon)
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "SOON")
    assert r["success"] is False and "isn't active yet" in r["error"]


@pytest.mark.asyncio
async def test_redeem_window_closed():
    coupon = _coupon(valid_from=datetime.now(timezone.utc) - timedelta(days=3),
                     valid_until=datetime.now(timezone.utc) - timedelta(days=1))
    cv = FakeVault(find_by_name=coupon)
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "OLD")
    assert r["success"] is False and "Window is closed" in r["error"]


@pytest.mark.asyncio
async def test_redeem_fully_claimed():
    coupon = _coupon(valid_from=datetime.now(timezone.utc) - timedelta(days=1),
                     valid_until=datetime.now(timezone.utc) + timedelta(days=1),
                     total_uses=5, times_redeemed=5)
    cv = FakeVault(find_by_name=coupon)
    r = await ct.redeem_coupon_tool(cv, OP, PATRON, "FULL")
    assert r["success"] is False and "fully claimed" in r["error"]


# ── list_my_coupons ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_my_coupons_status_mapping():
    def _view(usable, reason):
        return SimpleNamespace(
            coupon_id="c-1", name="A", discount_percent=10.0,
            valid_from=_NOW, valid_until=_NOW + timedelta(days=1),
            uses_per_patron=2, use_count=1, total_uses=None,
            is_usable=lambda now, u=usable, r=reason: (u, r),
            uses_remaining=lambda: 1, total_remaining=lambda: None,
        )
    cv = FakeVault(list_redemptions_for_patron=[
        _view(True, ""), _view(False, "window_closed"),
    ])
    r = await ct.list_my_coupons_tool(cv, OP, PATRON)
    assert r["success"] and r["count"] == 2
    assert [row["status"] for row in r["coupons"]] == ["active", "window_closed"]


@pytest.mark.asyncio
async def test_list_my_coupons_failure():
    cv = FakeVault(list_redemptions_for_patron=RuntimeError("x"))
    r = await ct.list_my_coupons_tool(cv, OP, PATRON)
    assert r["success"] is False and r["error"] == "list failed: x"
