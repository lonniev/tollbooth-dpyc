"""Coupon offers — operator-owned discount codes redeemed by patrons.

A coupon is a first-class operator object with a catchy ``name`` (the
code the patron redeems), a percentage discount, a calendar window,
and optional caps (per-patron uses, total uses).

Per-tool constraint chains reference a coupon **by id** via
``CouponConstraint(coupon_id=...)``.  The wheel's runtime pre-loads the
caller's coupon redemptions before walking the chain so the constraint
can read state synchronously.
"""

from tollbooth.coupons.models import (
    Coupon,
    CouponRedemption,
    CouponRedemptionMap,
    PatronCoupon,
)
from tollbooth.coupons.vault import CouponsVault

__all__ = [
    "Coupon",
    "CouponRedemption",
    "CouponRedemptionMap",
    "CouponsVault",
    "PatronCoupon",
]
