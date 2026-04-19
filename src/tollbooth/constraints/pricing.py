"""Pricing constraints — coupons, free trials, loyalty, bulk bonus, happy hour."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tollbooth.constraints.base import (
    ConstraintContext,
    ConstraintResult,
    ConstraintSchema,
    ParamSchema,
    PriceModifier,
    ToolConstraint,
)
from tollbooth.constraints.temporal import TemporalWindowConstraint


# ---------------------------------------------------------------------------
# CouponConstraint
# ---------------------------------------------------------------------------


class CouponConstraint(ToolConstraint):
    """Apply a discount when a valid coupon code is supplied.

    The coupon code itself must be passed via ``context.env`` metadata or
    matched externally. This constraint checks expiry, redemption caps,
    and per-patron caps.

    Parameters
    ----------
    code:
        The coupon code string.
    discount_percent:
        Percentage discount (0-100).
    discount_sats:
        Absolute discount in api-sats.
    free:
        If ``True``, the tool call is free.
    expires_at:
        Optional ISO-8601 expiry datetime.
    max_redemptions:
        Optional global cap on total redemptions.
    max_per_patron:
        Optional per-patron redemption cap.
    current_redemptions:
        Externally tracked total redemptions so far.
    patron_redemptions:
        Externally tracked redemptions for the current patron.
    """

    def __init__(
        self,
        code: str,
        discount_percent: float = 0.0,
        discount_sats: int = 0,
        free: bool = False,
        expires_at: str | None = None,
        max_redemptions: int | None = None,
        max_per_patron: int | None = None,
        current_redemptions: int = 0,
        patron_redemptions: int = 0,
    ) -> None:
        self.code = code
        self.discount_percent = discount_percent
        self.discount_sats = discount_sats
        self.free = free
        self.expires_at = expires_at
        self.max_redemptions = max_redemptions
        self.max_per_patron = max_per_patron
        self.current_redemptions = current_redemptions
        self.patron_redemptions = patron_redemptions

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        now = context.env.utc_now

        # Check expiry
        if self.expires_at is not None:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=timezone.utc)
            if now >= exp:
                return ConstraintResult(
                    allowed=False,
                    reason="coupon_expired",
                    message=f"Coupon {self.code!r} has expired.",
                )

        # Check global cap
        if (
            self.max_redemptions is not None
            and self.current_redemptions >= self.max_redemptions
        ):
            return ConstraintResult(
                allowed=False,
                reason="coupon_exhausted",
                message=f"Coupon {self.code!r} has been fully redeemed.",
            )

        # Check per-patron cap
        if (
            self.max_per_patron is not None
            and self.patron_redemptions >= self.max_per_patron
        ):
            return ConstraintResult(
                allowed=False,
                reason="coupon_patron_limit",
                message=f"You have already used coupon {self.code!r} the maximum number of times.",
            )

        # Valid — build modifier
        modifier = PriceModifier(
            discount_percent=self.discount_percent,
            discount_sats=self.discount_sats,
            free=self.free,
        )
        return ConstraintResult(
            allowed=True,
            price_modifier=modifier,
            metadata={"coupon_code": self.code},
        )

    def describe(self) -> str:
        parts = [f"Coupon {self.code!r}:"]
        if self.free:
            parts.append("free")
        elif self.discount_percent:
            parts.append(f"{self.discount_percent}% off")
        elif self.discount_sats:
            parts.append(f"{self.discount_sats} sats off")
        if self.expires_at:
            parts.append(f"(expires {self.expires_at})")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": "coupon", "code": self.code}
        if self.discount_percent:
            d["discount_percent"] = self.discount_percent
        if self.discount_sats:
            d["discount_sats"] = self.discount_sats
        if self.free:
            d["free"] = True
        if self.expires_at is not None:
            d["expires_at"] = self.expires_at
        if self.max_redemptions is not None:
            d["max_redemptions"] = self.max_redemptions
        if self.max_per_patron is not None:
            d["max_per_patron"] = self.max_per_patron
        if self.current_redemptions:
            d["current_redemptions"] = self.current_redemptions
        if self.patron_redemptions:
            d["patron_redemptions"] = self.patron_redemptions
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CouponConstraint:
        return cls(
            code=data["code"],
            discount_percent=float(data.get("discount_percent", 0.0)),
            discount_sats=int(data.get("discount_sats", 0)),
            free=bool(data.get("free", False)),
            expires_at=data.get("expires_at"),
            max_redemptions=data.get("max_redemptions"),
            max_per_patron=data.get("max_per_patron"),
            current_redemptions=int(data.get("current_redemptions", 0)),
            patron_redemptions=int(data.get("patron_redemptions", 0)),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="coupon",
            category="Pricing",
            description="Apply a discount when a valid coupon code is present. Supports expiry, global caps, and per-patron caps.",
            params=[
                ParamSchema(name="code", type="string", description="The coupon code string."),
                ParamSchema(name="discount_percent", type="float", required=False, default=0.0, description="Percentage discount (0-100)."),
                ParamSchema(name="discount_sats", type="int", required=False, default=0, description="Absolute discount in api-sats."),
                ParamSchema(name="free", type="bool", required=False, default=False, description="If true, the tool call is free."),
                ParamSchema(name="expires_at", type="string", required=False, description="ISO-8601 expiry datetime."),
                ParamSchema(name="max_redemptions", type="int", required=False, description="Global cap on total redemptions."),
                ParamSchema(name="max_per_patron", type="int", required=False, description="Per-patron redemption cap."),
                ParamSchema(name="current_redemptions", type="int", required=False, default=0, description="Total redemptions so far (externally tracked)."),
                ParamSchema(name="patron_redemptions", type="int", required=False, default=0, description="Current patron's redemptions (externally tracked)."),
            ],
        )


# ---------------------------------------------------------------------------
# FreeTrialConstraint
# ---------------------------------------------------------------------------


class FreeTrialConstraint(ToolConstraint):
    """Grant the first *n* invocations for free.

    Uses ``context.env.invocation_count`` to determine whether the patron
    is still within the trial window.
    """

    def __init__(self, first_n_free: int) -> None:
        self.first_n_free = first_n_free

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        used = context.env.invocation_count
        if used < self.first_n_free:
            remaining = self.first_n_free - used
            return ConstraintResult(
                allowed=True,
                price_modifier=PriceModifier(free=True),
                metadata={
                    "trial_remaining": remaining,
                    "trial_total": self.first_n_free,
                },
            )
        # Trial exhausted — allow the call but with no discount
        return ConstraintResult(
            allowed=True,
            metadata={
                "trial_remaining": 0,
                "trial_total": self.first_n_free,
            },
        )

    def describe(self) -> str:
        return f"First {self.first_n_free} invocations are free."

    def to_dict(self) -> dict[str, Any]:
        return {"type": "free_trial", "first_n_free": self.first_n_free}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FreeTrialConstraint:
        return cls(first_n_free=int(data["first_n_free"]))

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="free_trial",
            category="Pricing",
            description="Grant the first N invocations for free.",
            params=[
                ParamSchema(name="first_n_free", type="int", description="Number of free invocations per patron."),
            ],
        )


# ---------------------------------------------------------------------------
# LoyaltyDiscountConstraint
# ---------------------------------------------------------------------------


class LoyaltyDiscountConstraint(ToolConstraint):
    """Reward patrons who have consumed at least *threshold* api-sats."""

    def __init__(
        self,
        threshold_consumed_api_sats: int,
        discount_percent: float,
    ) -> None:
        self.threshold_consumed_api_sats = threshold_consumed_api_sats
        self.discount_percent = discount_percent

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        consumed = context.ledger.total_consumed_api_sats
        if consumed >= self.threshold_consumed_api_sats:
            return ConstraintResult(
                allowed=True,
                price_modifier=PriceModifier(
                    discount_percent=self.discount_percent
                ),
                metadata={"loyalty_qualified": True},
            )
        return ConstraintResult(
            allowed=True,
            metadata={
                "loyalty_qualified": False,
                "loyalty_remaining": self.threshold_consumed_api_sats - consumed,
            },
        )

    def describe(self) -> str:
        return (
            f"{self.discount_percent}% loyalty discount after "
            f"{self.threshold_consumed_api_sats} api-sats consumed."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "loyalty_discount",
            "threshold_consumed_api_sats": self.threshold_consumed_api_sats,
            "discount_percent": self.discount_percent,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LoyaltyDiscountConstraint:
        return cls(
            threshold_consumed_api_sats=int(data["threshold_consumed_api_sats"]),
            discount_percent=float(data["discount_percent"]),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="loyalty_discount",
            category="Pricing",
            description="Reward patrons who have consumed at least a threshold of api-sats with a percentage discount.",
            params=[
                ParamSchema(name="threshold_consumed_api_sats", type="int", description="Minimum api-sats consumed to qualify."),
                ParamSchema(name="discount_percent", type="float", description="Percentage discount (0-100) for qualifying patrons."),
            ],
        )


# ---------------------------------------------------------------------------
# BulkBonusConstraint
# ---------------------------------------------------------------------------


class BulkBonusConstraint(ToolConstraint):
    """Tiered bonus multiplier based on total consumption.

    Each tier specifies ``min_consumed`` (api-sats threshold) and
    ``bonus_multiplier``.  The patron receives the best matching tier.
    """

    def __init__(self, tiers: list[dict[str, Any]]) -> None:
        # Sort by min_consumed descending for easy best-match lookup
        self.tiers = sorted(tiers, key=lambda t: t["min_consumed"], reverse=True)

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        consumed = context.ledger.total_consumed_api_sats

        for tier in self.tiers:
            if consumed >= tier["min_consumed"]:
                return ConstraintResult(
                    allowed=True,
                    price_modifier=PriceModifier(
                        bonus_multiplier=float(tier["bonus_multiplier"])
                    ),
                    metadata={
                        "bulk_tier_min": tier["min_consumed"],
                        "bulk_multiplier": tier["bonus_multiplier"],
                    },
                )

        # No tier matched
        return ConstraintResult(allowed=True)

    def describe(self) -> str:
        parts = ["Bulk bonus tiers:"]
        for tier in sorted(self.tiers, key=lambda t: t["min_consumed"]):
            parts.append(
                f"  {tier['min_consumed']}+ sats -> {tier['bonus_multiplier']}x"
            )
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "bulk_bonus",
            "tiers": [
                {"min_consumed": t["min_consumed"], "bonus_multiplier": t["bonus_multiplier"]}
                for t in sorted(self.tiers, key=lambda t: t["min_consumed"])
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BulkBonusConstraint:
        return cls(tiers=list(data["tiers"]))

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="bulk_bonus",
            category="Pricing",
            description="Tiered bonus multiplier based on total consumption. Each tier has min_consumed and bonus_multiplier.",
            params=[
                ParamSchema(name="tiers", type="tiers", description="List of {min_consumed: int, bonus_multiplier: float} tier objects."),
            ],
        )


# ---------------------------------------------------------------------------
# HappyHourConstraint
# ---------------------------------------------------------------------------


class HappyHourConstraint(ToolConstraint):
    """During a temporal window, apply a pricing discount.

    This is a composition of :class:`TemporalWindowConstraint` (for window
    detection) with a :class:`PriceModifier`.  Outside the window the
    call is still allowed — just without the discount.
    """

    # Valid recurrence values
    RECURRENCE_OPTIONS = ("never", "weekly", "biweekly", "monthly", "annually")

    def __init__(
        self,
        in_effect: str,
        until: str,
        timezone: str = "UTC",
        days_of_week: list[int] | None = None,
        percent_off: float = 0.0,
        max_discount: int = 0,
        free: bool = False,
        repeats: str = "weekly",
        apply_on: str | None = None,
    ) -> None:
        self.in_effect = in_effect
        self.until = until
        self.timezone = timezone
        self.days_of_week = days_of_week
        self.percent_off = percent_off
        self.max_discount = max_discount
        self.free = free
        self.repeats = repeats if repeats in self.RECURRENCE_OPTIONS else "weekly"
        self.apply_on = apply_on
        self._window = TemporalWindowConstraint(
            schedule_start=in_effect,
            schedule_end=until,
            timezone=timezone,
            days_of_week=days_of_week,
        )

    def _is_active_recurrence(self, context: ConstraintContext) -> bool:
        """Check if today falls within the recurrence schedule."""
        if self.repeats == "weekly":
            return True
        if self.repeats == "never":
            if not self.apply_on:
                return True
            today = context.env.utc_now.astimezone(
                __import__("zoneinfo").ZoneInfo(self.timezone)
            ).date().isoformat()
            return today == self.apply_on
        if not self.apply_on:
            return True
        from datetime import date as _date
        start = _date.fromisoformat(self.apply_on)
        local_date = context.env.utc_now.astimezone(
            __import__("zoneinfo").ZoneInfo(self.timezone)
        ).date()
        if self.repeats == "biweekly":
            return ((local_date - start).days // 7) % 2 == 0
        if self.repeats == "monthly":
            return local_date.day == start.day
        if self.repeats == "annually":
            return local_date.month == start.month and local_date.day == start.day
        return True

    def evaluate(self, context: ConstraintContext) -> ConstraintResult:
        if not self._is_active_recurrence(context):
            return ConstraintResult(
                allowed=True,
                metadata={"happy_hour_active": False, "recurrence_skip": True},
            )
        window_result = self._window.evaluate(context)
        if window_result.allowed:
            return ConstraintResult(
                allowed=True,
                price_modifier=PriceModifier(
                    discount_percent=self.percent_off,
                    discount_sats=self.max_discount,
                    free=self.free,
                ),
                metadata={"happy_hour_active": True},
            )
        return ConstraintResult(
            allowed=True,
            metadata={"happy_hour_active": False},
        )

    def describe(self) -> str:
        parts = [f"Happy hour {self.in_effect}-{self.until} {self.timezone}:"]
        if self.free:
            parts.append("free")
        elif self.percent_off:
            parts.append(f"{self.percent_off}% off")
        elif self.max_discount:
            parts.append(f"{self.max_discount} sats off")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "type": "happy_hour",
            "in_effect": self.in_effect,
            "until": self.until,
            "timezone": self.timezone,
        }
        if self.days_of_week is not None:
            d["days_of_week"] = self.days_of_week
        if self.percent_off:
            d["percent_off"] = self.percent_off
        if self.max_discount:
            d["max_discount"] = self.max_discount
        if self.free:
            d["free"] = True
        if self.repeats != "weekly":
            d["repeats"] = self.repeats
        if self.apply_on:
            d["apply_on"] = self.apply_on
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HappyHourConstraint:
        return cls(
            in_effect=data["in_effect"],
            until=data["until"],
            timezone=data.get("timezone", "UTC"),
            days_of_week=data.get("days_of_week"),
            percent_off=float(data.get("percent_off", 0.0)),
            max_discount=int(data.get("max_discount", 0)),
            free=bool(data.get("free", False)),
            repeats=data.get("repeats", "weekly"),
            apply_on=data.get("apply_on"),
        )

    @classmethod
    def schema(cls) -> ConstraintSchema:
        return ConstraintSchema(
            type="happy_hour",
            category="Pricing",
            description="Apply a pricing discount during a temporal window.",
            params=[
                ParamSchema(name="in_effect", type="schedule", description="Start time HH:MM (24-hour)."),
                ParamSchema(name="until", type="schedule", description="End time HH:MM (24-hour)."),
                ParamSchema(name="timezone", type="timezone", required=False, default="UTC", description="IANA timezone."),
                ParamSchema(name="days_of_week", type="string", required=False, description="Weekday numbers (0=Mon..6=Sun)."),
                ParamSchema(name="apply_on", type="string", required=False, description="Recurrence start date (YYYY-MM-DD)."),
                ParamSchema(name="repeats", type="string", required=False, default="weekly", description="never, weekly, biweekly, monthly, annually."),
                ParamSchema(name="free", type="bool", required=False, default=False, description="Free during this window."),
                ParamSchema(name="percent_off", type="float", required=False, default=0.0, description="Discount percentage (0-100)."),
                ParamSchema(name="max_discount", type="int", required=False, default=0, description="Discount cap in api-sats. 0 = unlimited."),
            ],
        )
