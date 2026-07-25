"""Tests for tollbooth.coupons.models — Coupon / PatronCoupon / CouponRedemption."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tollbooth.coupons.models import (
    Coupon,
    CouponRedemption,
    CouponRedemptionMap,
    PatronCoupon,
    _parse_dt,
    _to_iso,
)

# ---------------------------------------------------------------------------
# Coupon round-trip
# ---------------------------------------------------------------------------


def _coupon() -> Coupon:
    return Coupon(
        id="11111111-1111-4111-8111-111111111111",
        operator="npub1abc",
        name="FRESHMAN",
        discount_percent=50.0,
        valid_from=datetime(2026, 5, 1, tzinfo=UTC),
        valid_until=datetime(2026, 6, 1, tzinfo=UTC),
        uses_per_patron=1,
        total_uses=100,
        times_redeemed=4,
    )


class TestCoupon:
    def test_round_trip_from_row(self) -> None:
        c = _coupon()
        row = {
            "id": c.id,
            "operator": c.operator,
            "name": c.name,
            "discount_percent": c.discount_percent,
            "valid_from": _to_iso(c.valid_from),
            "valid_until": _to_iso(c.valid_until),
            "uses_per_patron": c.uses_per_patron,
            "total_uses": c.total_uses,
            "times_redeemed": c.times_redeemed,
        }
        restored = Coupon.from_row(row)
        assert restored.id == c.id
        assert restored.name == c.name
        assert restored.discount_percent == c.discount_percent
        assert restored.uses_per_patron == c.uses_per_patron
        assert restored.total_uses == c.total_uses
        assert restored.times_redeemed == c.times_redeemed

    def test_from_row_handles_null_caps(self) -> None:
        """uses_per_patron and total_uses are both nullable (unlimited)."""
        row = {
            "id": "abc", "operator": "npub", "name": "FREE",
            "discount_percent": 10.0,
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": "2027-01-01T00:00:00+00:00",
            "uses_per_patron": None, "total_uses": None,
            "times_redeemed": 0,
        }
        c = Coupon.from_row(row)
        assert c.uses_per_patron is None
        assert c.total_uses is None

    def test_to_dict_includes_iso_dates(self) -> None:
        c = _coupon()
        d = c.to_dict()
        assert d["valid_from"].endswith("+00:00")
        assert d["valid_until"].endswith("+00:00")


# ---------------------------------------------------------------------------
# PatronCoupon
# ---------------------------------------------------------------------------


class TestPatronCoupon:
    def test_round_trip(self) -> None:
        pc = PatronCoupon(
            id="22222222-2222-4222-8222-222222222222",
            coupon_id="11111111-1111-4111-8111-111111111111",
            npub="npub1pat",
            use_count=2,
            redeemed_at=datetime(2026, 5, 15, tzinfo=UTC),
        )
        restored = PatronCoupon.from_row({
            "id": pc.id,
            "coupon_id": pc.coupon_id,
            "npub": pc.npub,
            "use_count": pc.use_count,
            "redeemed_at": _to_iso(pc.redeemed_at),
        })
        assert restored == pc


# ---------------------------------------------------------------------------
# CouponRedemption.is_usable
# ---------------------------------------------------------------------------


def _view(
    *,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    uses_per_patron: int | None = 1,
    total_uses: int | None = None,
    times_redeemed: int = 0,
    use_count: int = 0,
) -> CouponRedemption:
    return CouponRedemption(
        coupon_id="cid",
        name="X",
        discount_percent=25.0,
        valid_from=valid_from or datetime(2026, 1, 1, tzinfo=UTC),
        valid_until=valid_until or datetime(2027, 1, 1, tzinfo=UTC),
        uses_per_patron=uses_per_patron,
        total_uses=total_uses,
        times_redeemed=times_redeemed,
        use_count=use_count,
    )


NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


class TestCouponRedemptionUsable:
    def test_active(self) -> None:
        ok, reason = _view().is_usable(NOW)
        assert ok is True
        assert reason == ""

    def test_window_not_started(self) -> None:
        v = _view(
            valid_from=datetime(2027, 1, 1, tzinfo=UTC),
            valid_until=datetime(2028, 1, 1, tzinfo=UTC),
        )
        ok, reason = v.is_usable(NOW)
        assert ok is False
        assert reason == "window_not_started"

    def test_window_closed(self) -> None:
        v = _view(
            valid_from=datetime(2025, 1, 1, tzinfo=UTC),
            valid_until=datetime(2025, 12, 1, tzinfo=UTC),
        )
        ok, reason = v.is_usable(NOW)
        assert ok is False
        assert reason == "window_closed"

    def test_patron_limit(self) -> None:
        v = _view(uses_per_patron=3, use_count=3)
        ok, reason = v.is_usable(NOW)
        assert ok is False
        assert reason == "patron_limit"

    def test_total_limit(self) -> None:
        v = _view(total_uses=100, times_redeemed=100)
        ok, reason = v.is_usable(NOW)
        assert ok is False
        assert reason == "total_limit"

    def test_unlimited_uses_when_none(self) -> None:
        v = _view(uses_per_patron=None, use_count=9999, total_uses=None)
        ok, reason = v.is_usable(NOW)
        assert ok is True
        assert reason == ""

    def test_uses_remaining_unlimited_is_none(self) -> None:
        assert _view(uses_per_patron=None).uses_remaining() is None

    def test_uses_remaining_decrements(self) -> None:
        assert _view(uses_per_patron=5, use_count=2).uses_remaining() == 3

    def test_total_remaining_unlimited_is_none(self) -> None:
        assert _view(total_uses=None).total_remaining() is None

    def test_total_remaining_decrements(self) -> None:
        assert _view(total_uses=10, times_redeemed=3).total_remaining() == 7


# ---------------------------------------------------------------------------
# CouponRedemptionMap
# ---------------------------------------------------------------------------


class TestCouponRedemptionMap:
    def test_get_returns_match(self) -> None:
        m = CouponRedemptionMap(entries=(("a", _view()),))
        v = m.get("a")
        assert v is not None
        assert v.coupon_id == "cid"

    def test_get_missing_returns_none(self) -> None:
        m = CouponRedemptionMap(entries=())
        assert m.get("missing") is None

    def test_truthy_when_populated(self) -> None:
        assert bool(CouponRedemptionMap(entries=(("a", _view()),))) is True

    def test_falsy_when_empty(self) -> None:
        assert bool(CouponRedemptionMap()) is False


# ---------------------------------------------------------------------------
# _parse_dt edge cases
# ---------------------------------------------------------------------------


class TestParseDt:
    def test_string_with_z_suffix(self) -> None:
        dt = _parse_dt("2026-06-01T12:00:00Z")
        assert dt.tzinfo is not None

    def test_naive_string_treated_as_utc(self) -> None:
        dt = _parse_dt("2026-06-01T12:00:00")
        assert dt.tzinfo == UTC

    def test_datetime_passthrough(self) -> None:
        original = datetime(2026, 6, 1, tzinfo=UTC)
        assert _parse_dt(original) == original

    def test_naive_datetime_gets_utc(self) -> None:
        naive = datetime(2026, 6, 1)  # noqa: DTZ001
        assert _parse_dt(naive).tzinfo == UTC

    def test_invalid_raises(self) -> None:
        with pytest.raises(TypeError):
            _parse_dt(12345)
