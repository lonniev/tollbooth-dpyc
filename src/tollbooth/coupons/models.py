"""Coupon dataclasses — Coupon, PatronCoupon, and the joined redemption view."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _parse_dt(value: Any) -> datetime:
    """Coerce a Neon-returned timestamp (str or datetime) to UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError(f"Cannot parse datetime from {type(value).__name__}: {value!r}")


def _to_iso(value: datetime) -> str:
    """ISO-8601 string with explicit UTC offset for Neon TIMESTAMPTZ binds."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


@dataclass
class Coupon:
    """Operator-owned discount coupon row."""

    id: str
    operator: str
    name: str
    discount_percent: float
    valid_from: datetime
    valid_until: datetime
    uses_per_patron: int | None = 1
    total_uses: int | None = None
    times_redeemed: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "operator": self.operator,
            "name": self.name,
            "discount_percent": self.discount_percent,
            "valid_from": _to_iso(self.valid_from),
            "valid_until": _to_iso(self.valid_until),
            "uses_per_patron": self.uses_per_patron,
            "total_uses": self.total_uses,
            "times_redeemed": self.times_redeemed,
        }
        if self.created_at is not None:
            d["created_at"] = _to_iso(self.created_at)
        if self.updated_at is not None:
            d["updated_at"] = _to_iso(self.updated_at)
        return d

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Coupon:
        return cls(
            id=str(row["id"]),
            operator=str(row["operator"]),
            name=str(row["name"]),
            discount_percent=float(row["discount_percent"]),
            valid_from=_parse_dt(row["valid_from"]),
            valid_until=_parse_dt(row["valid_until"]),
            uses_per_patron=(
                int(row["uses_per_patron"])
                if row.get("uses_per_patron") is not None else None
            ),
            total_uses=(
                int(row["total_uses"])
                if row.get("total_uses") is not None else None
            ),
            times_redeemed=int(row.get("times_redeemed") or 0),
            created_at=_parse_dt(row["created_at"]) if row.get("created_at") else None,
            updated_at=_parse_dt(row["updated_at"]) if row.get("updated_at") else None,
        )


@dataclass
class PatronCoupon:
    """One patron's redemption of one coupon."""

    id: str
    coupon_id: str
    npub: str
    use_count: int = 0
    redeemed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "coupon_id": self.coupon_id,
            "npub": self.npub,
            "use_count": self.use_count,
        }
        if self.redeemed_at is not None:
            d["redeemed_at"] = _to_iso(self.redeemed_at)
        return d

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> PatronCoupon:
        return cls(
            id=str(row["id"]),
            coupon_id=str(row["coupon_id"]),
            npub=str(row["npub"]),
            use_count=int(row.get("use_count") or 0),
            redeemed_at=_parse_dt(row["redeemed_at"]) if row.get("redeemed_at") else None,
        )


@dataclass(frozen=True)
class CouponRedemption:
    """Joined snapshot of (coupons + patron_coupons) for constraint eval.

    Frozen because it rides on the frozen ``ConstraintContext``.  Built
    by ``CouponsVault.fetch_redemptions_for_chain`` before the gate walks
    the chain.
    """

    coupon_id: str
    name: str
    discount_percent: float
    valid_from: datetime
    valid_until: datetime
    uses_per_patron: int | None
    total_uses: int | None
    times_redeemed: int
    use_count: int  # this patron's use_count

    def is_usable(self, now: datetime) -> tuple[bool, str]:
        """Return ``(True, "")`` if the redemption can apply right now.

        Otherwise ``(False, reason)`` with a short machine-readable code
        — ``"window_not_started"``, ``"window_closed"``, ``"patron_limit"``,
        or ``"total_limit"``.  Constraints translate the reason into a
        neutral evaluation (no discount, chain continues at base price).
        """
        if now < self.valid_from:
            return False, "window_not_started"
        if now >= self.valid_until:
            return False, "window_closed"
        if self.uses_per_patron is not None and self.use_count >= self.uses_per_patron:
            return False, "patron_limit"
        if self.total_uses is not None and self.times_redeemed >= self.total_uses:
            return False, "total_limit"
        return True, ""

    def uses_remaining(self) -> int | None:
        """Per-patron uses left; ``None`` if unlimited within window."""
        if self.uses_per_patron is None:
            return None
        return max(0, self.uses_per_patron - self.use_count)

    def total_remaining(self) -> int | None:
        """Aggregate uses left across all patrons; ``None`` if unlimited."""
        if self.total_uses is None:
            return None
        return max(0, self.total_uses - self.times_redeemed)


@dataclass(frozen=True)
class CouponRedemptionMap:
    """Immutable dict-like wrapper carried on ``ConstraintContext``.

    Frozen dataclasses can't hold a mutable dict; this wrapper holds a
    tuple of (coupon_id, CouponRedemption) pairs and exposes a ``get``
    accessor.  Constructed only by the gate / runtime.
    """

    entries: tuple[tuple[str, CouponRedemption], ...] = field(default_factory=tuple)

    def get(self, coupon_id: str) -> CouponRedemption | None:
        for cid, view in self.entries:
            if cid == coupon_id:
                return view
        return None

    def __bool__(self) -> bool:
        return bool(self.entries)
