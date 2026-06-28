"""Coupon CRUD + redemption tool logic.

Extracted from ``register_standard_tools`` (audit M2.1). Each function is the
testable body behind a thin ``@tool`` shim in ``runtime.py``: the shim handles
the proof gate, npub resolution, and acquiring the ``CouponsVault`` / operator
npub from the runtime, then delegates here. These functions take only resolved
primitives (a ``CouponsVault`` and plain values) so they unit-test without a
runtime or FastMCP — mirroring ``tools/credits.py``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _format_coupon(c: Any) -> dict[str, Any]:
    """Serialize a coupon row, adding a human "redeemed of N" progress string."""
    d = c.to_dict()
    if c.total_uses is not None:
        d["progress"] = f"{c.times_redeemed} / {c.total_uses}"
    else:
        d["progress"] = f"{c.times_redeemed} / ∞"
    return d


def _parse_window(value: str, label: str) -> datetime:
    """Parse an ISO-8601 datetime, defaulting a naive value to UTC.

    Raises ``ValueError`` with an operator-facing message on bad input.
    """
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{label} must be ISO-8601 (e.g. '2026-06-01T00:00:00Z'); got {value!r}."
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


async def mint_coupon_tool(
    cv: Any,
    operator: str,
    *,
    name: str,
    discount_percent: float,
    valid_from: str,
    valid_until: str,
    uses_per_patron: int | None,
    total_uses: int | None,
) -> dict[str, Any]:
    """Validate inputs and mint a new operator-owned coupon."""
    if not name or not name.strip():
        return {"success": False, "error": "name is required and must be non-empty."}
    if not (0 < float(discount_percent) <= 100):
        return {"success": False, "error": "discount_percent must be in (0, 100]."}

    try:
        vf = _parse_window(valid_from, "valid_from")
        vu = _parse_window(valid_until, "valid_until")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    if vu <= vf:
        return {
            "success": False,
            "error": "valid_until must be strictly later than valid_from.",
        }
    if uses_per_patron is not None and int(uses_per_patron) <= 0:
        return {
            "success": False,
            "error": "uses_per_patron must be a positive integer or null for unlimited.",
        }
    if total_uses is not None and int(total_uses) <= 0:
        return {
            "success": False,
            "error": "total_uses must be a positive integer or null for unlimited.",
        }

    from tollbooth.coupons.vault import CouponAlreadyExists
    try:
        coupon = await cv.mint(
            operator=operator,
            name=name.strip(),
            discount_percent=float(discount_percent),
            valid_from=vf,
            valid_until=vu,
            uses_per_patron=int(uses_per_patron) if uses_per_patron is not None else None,
            total_uses=int(total_uses) if total_uses is not None else None,
        )
    except CouponAlreadyExists as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        return {"success": False, "error": f"mint failed: {exc}"}

    return {"success": True, "coupon": _format_coupon(coupon)}


async def list_coupons_tool(cv: Any, operator: str) -> dict[str, Any]:
    """List every coupon this operator has minted (newest first)."""
    try:
        coupons = await cv.list_for_operator(operator)
    except Exception as exc:
        return {"success": False, "error": f"list failed: {exc}"}

    return {
        "success": True,
        "count": len(coupons),
        "coupons": [_format_coupon(c) for c in coupons],
    }


async def update_coupon_tool(
    cv: Any,
    operator: str,
    coupon_id: str,
    *,
    name: str | None,
    discount_percent: float | None,
    valid_from: str | None,
    valid_until: str | None,
    uses_per_patron: int | None,
    total_uses: int | None,
    clear_uses_per_patron: bool,
    clear_total_uses: bool,
) -> dict[str, Any]:
    """Patch a coupon's editable fields; pass clear_* to set a cap unlimited."""
    if not coupon_id:
        return {"success": False, "error": "coupon_id is required."}
    if discount_percent is not None and not (0 < float(discount_percent) <= 100):
        return {"success": False, "error": "discount_percent must be in (0, 100]."}

    vf = vu = None
    try:
        if valid_from is not None:
            vf = _parse_window(valid_from, "valid_from")
        if valid_until is not None:
            vu = _parse_window(valid_until, "valid_until")
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    # Caps are tri-state: clear_* -> set unlimited (None); an explicit value
    # -> set it; otherwise OMIT the kwarg entirely so the vault's own
    # "leave alone" sentinel applies. Never pass a sentinel from here: the
    # vault forwards unrecognised values straight into the SQL params, where
    # a stray Ellipsis would blow up Neon's JSON encoder ("Object of type
    # ellipsis is not JSON serializable").
    cap_kwargs: dict[str, Any] = {}
    if clear_uses_per_patron:
        cap_kwargs["uses_per_patron"] = None
    elif uses_per_patron is not None:
        if int(uses_per_patron) <= 0:
            return {"success": False, "error": "uses_per_patron must be a positive integer."}
        cap_kwargs["uses_per_patron"] = int(uses_per_patron)

    if clear_total_uses:
        cap_kwargs["total_uses"] = None
    elif total_uses is not None:
        if int(total_uses) <= 0:
            return {"success": False, "error": "total_uses must be a positive integer."}
        cap_kwargs["total_uses"] = int(total_uses)

    from tollbooth.coupons.vault import CouponAlreadyExists, CouponNotFound
    try:
        updated = await cv.update(
            coupon_id,
            operator,
            name=name.strip() if name else None,
            discount_percent=float(discount_percent) if discount_percent is not None else None,
            valid_from=vf,
            valid_until=vu,
            **cap_kwargs,
        )
    except CouponAlreadyExists as exc:
        return {"success": False, "error": str(exc)}
    except CouponNotFound:
        return {"success": False, "error": f"No coupon {coupon_id!r} owned by this operator."}
    except Exception as exc:
        return {"success": False, "error": f"update failed: {exc}"}

    return {"success": True, "coupon": _format_coupon(updated)}


async def delete_coupon_tool(cv: Any, operator: str, coupon_id: str) -> dict[str, Any]:
    """Delete a coupon. Cascades to all patron redemptions."""
    if not coupon_id:
        return {"success": False, "error": "coupon_id is required."}

    try:
        deleted = await cv.delete(coupon_id, operator)
    except Exception as exc:
        return {"success": False, "error": f"delete failed: {exc}"}

    if not deleted:
        return {"success": False, "error": f"No coupon {coupon_id!r} owned by this operator."}
    return {"success": True, "coupon_id": coupon_id}


async def redeem_coupon_tool(
    cv: Any, operator: str, patron_npub: str, code: str,
) -> dict[str, Any]:
    """Claim a coupon by name: validate window + total cap, record redemption."""
    if not code or not code.strip():
        return {"success": False, "error": "code is required and must be non-empty."}

    try:
        coupon = await cv.find_by_name(operator, code.strip())
        if coupon is None:
            return {
                "success": False,
                "error": f"No coupon named {code!r}. Check the spelling and try again.",
            }

        now = datetime.now(timezone.utc)
        if now < coupon.valid_from:
            return {
                "success": False,
                "error": f"That coupon isn't active yet. It opens at {coupon.valid_from.isoformat()}.",
            }
        if now >= coupon.valid_until:
            return {
                "success": False,
                "error": (
                    f"That coupon was active until {coupon.valid_until.isoformat()}. "
                    "Window is closed."
                ),
            }
        if coupon.total_uses is not None and coupon.times_redeemed >= coupon.total_uses:
            return {"success": False, "error": "That coupon has been fully claimed."}

        pc = await cv.redeem(coupon.id, patron_npub)
    except Exception as exc:
        return {"success": False, "error": f"redeem failed: {exc}"}

    uses_remaining: Any
    if coupon.uses_per_patron is None:
        uses_remaining = None
    else:
        uses_remaining = max(0, coupon.uses_per_patron - pc.use_count)

    return {
        "success": True,
        "coupon_id": coupon.id,
        "name": coupon.name,
        "discount_percent": coupon.discount_percent,
        "valid_until": coupon.valid_until.isoformat(),
        "uses_remaining": uses_remaining,
        "uses_per_patron": coupon.uses_per_patron,
    }


async def list_my_coupons_tool(
    cv: Any, operator: str, patron_npub: str,
) -> dict[str, Any]:
    """List this patron's redemptions on this operator, with per-row status."""
    try:
        views = await cv.list_redemptions_for_patron(patron_npub, operator=operator)
    except Exception as exc:
        return {"success": False, "error": f"list failed: {exc}"}

    now = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for v in views:
        ok, reason = v.is_usable(now)
        rows.append({
            "coupon_id": v.coupon_id,
            "name": v.name,
            "discount_percent": v.discount_percent,
            "valid_from": v.valid_from.isoformat(),
            "valid_until": v.valid_until.isoformat(),
            "uses_per_patron": v.uses_per_patron,
            "use_count": v.use_count,
            "uses_remaining": v.uses_remaining(),
            "total_uses": v.total_uses,
            "total_remaining": v.total_remaining(),
            "status": "active" if ok else reason,
        })

    return {"success": True, "count": len(rows), "coupons": rows}


async def forget_coupon_tool(
    cv: Any, patron_npub: str, coupon_id: str,
) -> dict[str, Any]:
    """Remove a coupon from this patron's redemption list (cosmetic only)."""
    if not coupon_id:
        return {"success": False, "error": "coupon_id is required."}

    try:
        removed = await cv.forget(coupon_id, patron_npub)
    except Exception as exc:
        return {"success": False, "error": f"forget failed: {exc}"}

    if not removed:
        return {"success": False, "error": "You don't have that coupon to forget."}
    return {"success": True, "coupon_id": coupon_id}
