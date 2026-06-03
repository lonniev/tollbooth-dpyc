"""Tests for tollbooth.constraints.pricing — all pricing constraints."""

from datetime import datetime, timezone


from tollbooth.constraints.base import (
    ConstraintContext,
    EnvironmentSnapshot,
    LedgerSnapshot,
    PatronIdentity,
)
from tollbooth.constraints.pricing import (
    BulkBonusConstraint,
    CouponConstraint,
    FreeTrialConstraint,
    HappyHourConstraint,
    LoyaltyDiscountConstraint,
)
from tollbooth.coupons.models import CouponRedemption, CouponRedemptionMap


def _ctx(
    utc_now=None,
    invocation_count=0,
    total_consumed=0,
    balance=0,
):
    if utc_now is None:
        utc_now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    return ConstraintContext(
        ledger=LedgerSnapshot(
            balance_api_sats=balance,
            total_consumed_api_sats=total_consumed,
        ),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(
            utc_now=utc_now,
            invocation_count=invocation_count,
        ),
    )


# ---------------------------------------------------------------------------
# CouponConstraint — references an operator-owned coupon by id
# ---------------------------------------------------------------------------


def _redemption(
    coupon_id: str = "11111111-1111-4111-8111-111111111111",
    *,
    discount_percent: float = 50.0,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    uses_per_patron: int | None = 1,
    total_uses: int | None = None,
    times_redeemed: int = 0,
    use_count: int = 0,
    name: str = "FRESHMAN",
) -> CouponRedemption:
    if valid_from is None:
        valid_from = datetime(2026, 1, 1, tzinfo=timezone.utc)
    if valid_until is None:
        valid_until = datetime(2027, 1, 1, tzinfo=timezone.utc)
    return CouponRedemption(
        coupon_id=coupon_id,
        name=name,
        discount_percent=discount_percent,
        valid_from=valid_from,
        valid_until=valid_until,
        uses_per_patron=uses_per_patron,
        total_uses=total_uses,
        times_redeemed=times_redeemed,
        use_count=use_count,
    )


def _ctx_with_redemptions(*redemptions: CouponRedemption, utc_now=None) -> ConstraintContext:
    if utc_now is None:
        utc_now = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
    rmap = CouponRedemptionMap(
        entries=tuple((r.coupon_id, r) for r in redemptions),
    )
    return ConstraintContext(
        ledger=LedgerSnapshot(),
        patron=PatronIdentity(),
        env=EnvironmentSnapshot(utc_now=utc_now),
        coupon_redemptions=rmap,
    )


class TestCouponConstraint:
    COUPON_ID = "11111111-1111-4111-8111-111111111111"

    def test_no_pre_load_is_neutral(self):
        """Loader didn't run → constraint stays neutral (no discount)."""
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        result = c.evaluate(_ctx())  # coupon_redemptions is None
        assert result.allowed is True
        assert result.price_modifier is None

    def test_patron_has_not_redeemed_is_neutral(self):
        """Pre-load ran but this patron hasn't redeemed → neutral."""
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        result = c.evaluate(_ctx_with_redemptions())  # empty map
        assert result.allowed is True
        assert result.price_modifier is None

    def test_active_redemption_applies_discount(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        rmap = _ctx_with_redemptions(
            _redemption(self.COUPON_ID, discount_percent=50.0)
        )
        result = c.evaluate(rmap)
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 50.0
        assert result.metadata["consume_coupon_id"] == self.COUPON_ID
        assert result.metadata["coupon_name"] == "FRESHMAN"

    def test_window_not_started_is_neutral(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        ctx = _ctx_with_redemptions(
            _redemption(
                self.COUPON_ID,
                valid_from=datetime(2030, 1, 1, tzinfo=timezone.utc),
                valid_until=datetime(2031, 1, 1, tzinfo=timezone.utc),
            )
        )
        result = c.evaluate(ctx)
        assert result.allowed is True
        assert result.price_modifier is None

    def test_window_closed_is_neutral(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        ctx = _ctx_with_redemptions(
            _redemption(
                self.COUPON_ID,
                valid_from=datetime(2025, 1, 1, tzinfo=timezone.utc),
                valid_until=datetime(2025, 12, 1, tzinfo=timezone.utc),
            )
        )
        result = c.evaluate(ctx)
        assert result.allowed is True
        assert result.price_modifier is None

    def test_per_patron_exhausted_is_neutral(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        ctx = _ctx_with_redemptions(
            _redemption(self.COUPON_ID, uses_per_patron=3, use_count=3)
        )
        result = c.evaluate(ctx)
        assert result.allowed is True
        assert result.price_modifier is None

    def test_total_cap_reached_is_neutral(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        ctx = _ctx_with_redemptions(
            _redemption(self.COUPON_ID, total_uses=100, times_redeemed=100)
        )
        result = c.evaluate(ctx)
        assert result.allowed is True
        assert result.price_modifier is None

    def test_unknown_coupon_id_in_chain_is_neutral(self):
        """Orphan ref (deleted coupon) → neutral, no denial."""
        c = CouponConstraint(coupon_id="22222222-2222-4222-8222-222222222222")
        ctx = _ctx_with_redemptions(
            _redemption(self.COUPON_ID)  # different id present
        )
        result = c.evaluate(ctx)
        assert result.allowed is True
        assert result.price_modifier is None


class TestCouponSerialization:
    COUPON_ID = "33333333-3333-4333-8333-333333333333"

    def test_to_dict(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        d = c.to_dict()
        assert d == {"type": "coupon", "coupon_id": self.COUPON_ID}

    def test_round_trip(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        restored = CouponConstraint.from_dict(c.to_dict())
        assert restored.coupon_id == self.COUPON_ID

    def test_from_dict_requires_coupon_id(self):
        import pytest
        with pytest.raises(ValueError):
            CouponConstraint.from_dict({"type": "coupon"})

    def test_describe(self):
        c = CouponConstraint(coupon_id=self.COUPON_ID)
        assert "33333333" in c.describe()


# ---------------------------------------------------------------------------
# FreeTrialConstraint
# ---------------------------------------------------------------------------


class TestFreeTrialConstraint:
    def test_within_trial(self):
        c = FreeTrialConstraint(first_n_free=5)
        result = c.evaluate(_ctx(invocation_count=3))
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.free is True
        assert result.metadata["trial_remaining"] == 2

    def test_first_invocation(self):
        c = FreeTrialConstraint(first_n_free=3)
        result = c.evaluate(_ctx(invocation_count=0))
        assert result.allowed is True
        assert result.price_modifier.free is True
        assert result.metadata["trial_remaining"] == 3

    def test_trial_exhausted(self):
        c = FreeTrialConstraint(first_n_free=3)
        result = c.evaluate(_ctx(invocation_count=3))
        assert result.allowed is True  # Still allowed, just not free
        assert result.price_modifier is None
        assert result.metadata["trial_remaining"] == 0

    def test_well_past_trial(self):
        c = FreeTrialConstraint(first_n_free=3)
        result = c.evaluate(_ctx(invocation_count=100))
        assert result.allowed is True
        assert result.price_modifier is None


class TestFreeTrialSerialization:
    def test_to_dict(self):
        c = FreeTrialConstraint(first_n_free=10)
        d = c.to_dict()
        assert d["type"] == "free_trial"
        assert d["first_n_free"] == 10

    def test_round_trip(self):
        c = FreeTrialConstraint(first_n_free=7)
        restored = FreeTrialConstraint.from_dict(c.to_dict())
        assert restored.first_n_free == 7

    def test_describe(self):
        c = FreeTrialConstraint(first_n_free=5)
        assert "5" in c.describe()


# ---------------------------------------------------------------------------
# LoyaltyDiscountConstraint
# ---------------------------------------------------------------------------


class TestLoyaltyDiscountConstraint:
    def test_qualified(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=1000,
            discount_percent=15.0,
        )
        result = c.evaluate(_ctx(total_consumed=1500))
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 15.0
        assert result.metadata["loyalty_qualified"] is True

    def test_at_threshold(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=1000,
            discount_percent=10.0,
        )
        result = c.evaluate(_ctx(total_consumed=1000))
        assert result.price_modifier is not None

    def test_not_qualified(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=1000,
            discount_percent=10.0,
        )
        result = c.evaluate(_ctx(total_consumed=500))
        assert result.allowed is True
        assert result.price_modifier is None
        assert result.metadata["loyalty_qualified"] is False
        assert result.metadata["loyalty_remaining"] == 500


class TestLoyaltySerialization:
    def test_round_trip(self):
        c = LoyaltyDiscountConstraint(
            threshold_consumed_api_sats=5000,
            discount_percent=20.0,
        )
        restored = LoyaltyDiscountConstraint.from_dict(c.to_dict())
        assert restored.threshold_consumed_api_sats == 5000
        assert restored.discount_percent == 20.0


# ---------------------------------------------------------------------------
# BulkBonusConstraint
# ---------------------------------------------------------------------------


class TestBulkBonusConstraint:
    def test_best_tier_matched(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
            {"min_consumed": 500, "bonus_multiplier": 1.25},
            {"min_consumed": 1000, "bonus_multiplier": 1.5},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=750))
        assert result.price_modifier is not None
        assert result.price_modifier.bonus_multiplier == 1.25

    def test_top_tier(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
            {"min_consumed": 1000, "bonus_multiplier": 1.5},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=5000))
        assert result.price_modifier.bonus_multiplier == 1.5

    def test_no_tier_matched(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=50))
        assert result.price_modifier is None

    def test_at_tier_boundary(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=100))
        assert result.price_modifier.bonus_multiplier == 1.1

    def test_single_tier(self):
        tiers = [{"min_consumed": 0, "bonus_multiplier": 1.05}]
        c = BulkBonusConstraint(tiers=tiers)
        result = c.evaluate(_ctx(total_consumed=0))
        assert result.price_modifier.bonus_multiplier == 1.05


class TestBulkBonusSerialization:
    def test_round_trip(self):
        tiers = [
            {"min_consumed": 100, "bonus_multiplier": 1.1},
            {"min_consumed": 500, "bonus_multiplier": 1.25},
        ]
        c = BulkBonusConstraint(tiers=tiers)
        d = c.to_dict()
        assert d["type"] == "bulk_bonus"
        # Should be sorted by min_consumed ascending
        assert d["tiers"][0]["min_consumed"] == 100
        assert d["tiers"][1]["min_consumed"] == 500

        restored = BulkBonusConstraint.from_dict(d)
        result = restored.evaluate(_ctx(total_consumed=200))
        assert result.price_modifier.bonus_multiplier == 1.1


# ---------------------------------------------------------------------------
# HappyHourConstraint
# ---------------------------------------------------------------------------


class TestHappyHourConstraint:
    def test_during_happy_hour(self):
        c = HappyHourConstraint(
            in_effect="11:00", until="14:00",
            percent_off=50.0,
        )
        result = c.evaluate(_ctx())  # 12:00 UTC
        assert result.allowed is True
        assert result.price_modifier is not None
        assert result.price_modifier.discount_percent == 50.0
        assert result.metadata["happy_hour_active"] is True

    def test_outside_happy_hour(self):
        c = HappyHourConstraint(
            in_effect="14:00", until="16:00",
            percent_off=50.0,
        )
        result = c.evaluate(_ctx())  # 12:00 UTC
        assert result.allowed is True  # Still allowed
        assert result.price_modifier is None
        assert result.metadata["happy_hour_active"] is False

    def test_free_happy_hour(self):
        c = HappyHourConstraint(
            in_effect="11:00", until="14:00",
            free=True,
        )
        result = c.evaluate(_ctx())
        assert result.price_modifier.free is True

    def test_with_timezone(self):
        # 12:00 UTC = 07:00 US/Eastern (EST)
        c = HappyHourConstraint(
            in_effect="08:00", until="17:00",
            timezone="US/Eastern",
            percent_off=25.0,
        )
        result = c.evaluate(_ctx())  # 07:00 Eastern -> outside
        assert result.metadata["happy_hour_active"] is False

    def test_with_days_of_week(self):
        # 2026-03-01 is Sunday (weekday=6)
        c = HappyHourConstraint(
            in_effect="00:00", until="23:59",
            days_of_week=[0, 1, 2, 3, 4],  # Mon-Fri only
            percent_off=30.0,
        )
        result = c.evaluate(_ctx())  # Sunday
        assert result.allowed is True  # Still allowed, just no discount
        assert result.metadata["happy_hour_active"] is False


class TestHappyHourSerialization:
    def test_to_dict(self):
        c = HappyHourConstraint(
            in_effect="11:00", until="14:00",
            timezone="US/Eastern",
            percent_off=25.0,
        )
        d = c.to_dict()
        assert d["type"] == "happy_hour"
        assert d["in_effect"] == "11:00"
        assert d["until"] == "14:00"
        assert d["percent_off"] == 25.0

    def test_round_trip(self):
        c = HappyHourConstraint(
            in_effect="18:00", until="22:00",
            timezone="Europe/Berlin",
            days_of_week=[4, 5],
            max_discount=50,
        )
        restored = HappyHourConstraint.from_dict(c.to_dict())
        assert restored.in_effect == "18:00"
        assert restored.until == "22:00"
        assert restored.timezone == "Europe/Berlin"
        assert restored.days_of_week == [4, 5]
        assert restored.max_discount == 50

    def test_describe(self):
        c = HappyHourConstraint(in_effect="17:00", until="19:00", percent_off=40.0)
        desc = c.describe()
        assert "17:00-19:00" in desc
        assert "40" in desc
