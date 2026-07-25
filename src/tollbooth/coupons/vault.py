"""Coupons vault — CRUD + redemption + atomic burn against Neon.

Wraps a :class:`NeonVault` the same way :class:`PricingModelStore`
does.  Two tables live in the operator's schema:

* ``coupons`` — operator-owned offers (name, discount %, window, caps)
* ``patron_coupons`` — per-patron redemption rows (use_count)

``ensure_schema`` is idempotent (``CREATE TABLE IF NOT EXISTS``) and is
called once from :meth:`NeonVault.ensure_schema` so upgrading operators
pick up the tables on first paid call.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from tollbooth.coupons.models import (
    Coupon,
    CouponRedemption,
    PatronCoupon,
    _to_iso,
)

logger = logging.getLogger(__name__)

# Sentinel for "leave this column alone" in update() — distinct from None,
# which is a meaningful value ("clear this column"). Typed Any so it can be
# the default for int | None parameters without tripping the type checker.
_UNSET: Any = object()


class CouponAlreadyExists(Exception):
    """Operator tried to mint two coupons with the same name."""


class CouponNotFound(Exception):
    """Lookup by id (operator-scoped) returned no row."""


class CouponsVault:
    """CRUD + redemption + atomic burn for the coupons feature."""

    def __init__(self, *, neon_vault: Any) -> None:
        self._neon = neon_vault

    def _t(self, table: str) -> str:
        return self._neon._t(table)

    # -- Schema -----------------------------------------------------------

    async def ensure_schema(self) -> None:
        """Create the coupons + patron_coupons tables.  Idempotent."""
        idx_prefix = ""
        if getattr(self._neon, "_schema_prefix", ""):
            idx_prefix = self._neon._schema_prefix.rstrip(".") + "_"

        await self._neon._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('coupons')} ("
            "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            "    operator TEXT NOT NULL,"
            "    name TEXT NOT NULL,"
            "    discount_percent NUMERIC(5,2) NOT NULL,"
            "    valid_from TIMESTAMPTZ NOT NULL,"
            "    valid_until TIMESTAMPTZ NOT NULL,"
            "    uses_per_patron INTEGER,"
            "    total_uses INTEGER,"
            "    times_redeemed INTEGER NOT NULL DEFAULT 0,"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    UNIQUE (operator, name)"
            ")"
        )
        await self._neon._execute(
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_coupons_operator "
            f"ON {self._t('coupons')}(operator)"
        )
        await self._neon._execute(
            f"CREATE TABLE IF NOT EXISTS {self._t('patron_coupons')} ("
            "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),"
            f"    coupon_id UUID NOT NULL REFERENCES {self._t('coupons')}(id) ON DELETE CASCADE,"
            "    npub TEXT NOT NULL,"
            "    use_count INTEGER NOT NULL DEFAULT 0,"
            "    redeemed_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    UNIQUE (coupon_id, npub)"
            ")"
        )
        await self._neon._execute(
            f"CREATE INDEX IF NOT EXISTS {idx_prefix}idx_patron_coupons_npub "
            f"ON {self._t('patron_coupons')}(npub)"
        )

    # -- Operator CRUD ----------------------------------------------------

    _SELECT = (
        "id, operator, name, discount_percent, valid_from, valid_until, "
        "uses_per_patron, total_uses, times_redeemed, created_at, updated_at"
    )

    async def mint(
        self,
        *,
        operator: str,
        name: str,
        discount_percent: float,
        valid_from: datetime,
        valid_until: datetime,
        uses_per_patron: int | None = 1,
        total_uses: int | None = None,
    ) -> Coupon:
        """Insert a new coupon row.  Raises ``CouponAlreadyExists`` on
        the ``UNIQUE(operator, name)`` collision."""
        from tollbooth.vaults.neon import NeonQueryError

        try:
            result = await self._neon._execute(
                f"INSERT INTO {self._t('coupons')} "
                "(operator, name, discount_percent, valid_from, valid_until, "
                " uses_per_patron, total_uses) "
                "VALUES ($1, $2, $3, $4::timestamptz, $5::timestamptz, $6, $7) "
                f"RETURNING {self._SELECT}",
                [
                    operator,
                    name,
                    discount_percent,
                    _to_iso(valid_from),
                    _to_iso(valid_until),
                    uses_per_patron,
                    total_uses,
                ],
            )
        except NeonQueryError as exc:
            if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
                raise CouponAlreadyExists(
                    f"Operator already has a coupon named {name!r}."
                ) from exc
            raise

        rows = result.get("rows", [])
        if not rows:
            raise RuntimeError("INSERT coupon returned no rows")
        return Coupon.from_row(rows[0])

    async def get(self, coupon_id: str, *, operator: str | None = None) -> Coupon | None:
        """Fetch one coupon by id.  Optionally enforce operator scope."""
        if operator:
            result = await self._neon._execute(
                f"SELECT {self._SELECT} FROM {self._t('coupons')} "
                "WHERE id = $1::uuid AND operator = $2 LIMIT 1",
                [coupon_id, operator],
            )
        else:
            result = await self._neon._execute(
                f"SELECT {self._SELECT} FROM {self._t('coupons')} "
                "WHERE id = $1::uuid LIMIT 1",
                [coupon_id],
            )
        rows = result.get("rows", [])
        return Coupon.from_row(rows[0]) if rows else None

    async def find_by_name(self, operator: str, name: str) -> Coupon | None:
        """Lookup a coupon by its operator-unique name (case-sensitive)."""
        result = await self._neon._execute(
            f"SELECT {self._SELECT} FROM {self._t('coupons')} "
            "WHERE operator = $1 AND name = $2 LIMIT 1",
            [operator, name],
        )
        rows = result.get("rows", [])
        return Coupon.from_row(rows[0]) if rows else None

    async def list_for_operator(self, operator: str) -> list[Coupon]:
        result = await self._neon._execute(
            f"SELECT {self._SELECT} FROM {self._t('coupons')} "
            "WHERE operator = $1 ORDER BY created_at DESC",
            [operator],
        )
        return [Coupon.from_row(r) for r in result.get("rows", [])]

    async def update(
        self,
        coupon_id: str,
        operator: str,
        *,
        name: str | None = None,
        discount_percent: float | None = None,
        valid_from: datetime | None = None,
        valid_until: datetime | None = None,
        uses_per_patron: int | None = _UNSET,  # sentinel for "leave alone"
        total_uses: int | None = _UNSET,
    ) -> Coupon:
        """Patch a coupon's editable fields.  Operator-scoped — refuses
        to touch another operator's row."""
        sets: list[str] = []
        params: list[Any] = []

        if name is not None:
            params.append(name)
            sets.append(f"name = ${len(params)}")
        if discount_percent is not None:
            params.append(discount_percent)
            sets.append(f"discount_percent = ${len(params)}")
        if valid_from is not None:
            params.append(_to_iso(valid_from))
            sets.append(f"valid_from = ${len(params)}::timestamptz")
        if valid_until is not None:
            params.append(_to_iso(valid_until))
            sets.append(f"valid_until = ${len(params)}::timestamptz")
        if uses_per_patron is not _UNSET:
            params.append(uses_per_patron)
            sets.append(f"uses_per_patron = ${len(params)}")
        if total_uses is not _UNSET:
            params.append(total_uses)
            sets.append(f"total_uses = ${len(params)}")

        if not sets:
            existing = await self.get(coupon_id, operator=operator)
            if existing is None:
                raise CouponNotFound(coupon_id)
            return existing

        sets.append("updated_at = now()")
        params.append(coupon_id)
        params.append(operator)
        sql = (
            f"UPDATE {self._t('coupons')} SET {', '.join(sets)} "
            f"WHERE id = ${len(params) - 1}::uuid AND operator = ${len(params)} "
            f"RETURNING {self._SELECT}"
        )

        from tollbooth.vaults.neon import NeonQueryError

        try:
            result = await self._neon._execute(sql, params)
        except NeonQueryError as exc:
            if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
                raise CouponAlreadyExists(
                    f"Operator already has a coupon named {name!r}."
                ) from exc
            raise

        rows = result.get("rows", [])
        if not rows:
            raise CouponNotFound(coupon_id)
        return Coupon.from_row(rows[0])

    async def delete(self, coupon_id: str, operator: str) -> bool:
        """Delete a coupon and (via ON DELETE CASCADE) all redemptions."""
        result = await self._neon._execute(
            f"DELETE FROM {self._t('coupons')} "
            "WHERE id = $1::uuid AND operator = $2",
            [coupon_id, operator],
        )
        return int(result.get("rowCount") or 0) > 0

    # -- Patron flow ------------------------------------------------------

    async def redeem(self, coupon_id: str, npub: str) -> PatronCoupon:
        """Idempotently insert a ``patron_coupons`` row.

        The window / cap checks happen at the caller — this method's job
        is just to record the redemption.  Repeated calls return the
        existing row (no error)."""
        result = await self._neon._execute(
            f"INSERT INTO {self._t('patron_coupons')} (coupon_id, npub) "
            "VALUES ($1::uuid, $2) "
            "ON CONFLICT (coupon_id, npub) DO UPDATE "
            "SET use_count = patron_coupons.use_count "  # no-op refresh
            "RETURNING id, coupon_id, npub, use_count, redeemed_at",
            [coupon_id, npub],
        )
        rows = result.get("rows", [])
        if not rows:
            raise RuntimeError("INSERT patron_coupon returned no rows")
        return PatronCoupon.from_row(rows[0])

    async def forget(self, coupon_id: str, npub: str) -> bool:
        """Patron-initiated removal of a redemption.  Pure cosmetic — the
        coupon itself still exists, the patron may re-redeem later."""
        result = await self._neon._execute(
            f"DELETE FROM {self._t('patron_coupons')} "
            "WHERE coupon_id = $1::uuid AND npub = $2",
            [coupon_id, npub],
        )
        return int(result.get("rowCount") or 0) > 0

    async def list_redemptions_for_patron(
        self,
        npub: str,
        *,
        operator: str | None = None,
    ) -> list[CouponRedemption]:
        """Joined view of a patron's redemptions — drives the FE's
        Profile → My Coupons section."""
        sql = (
            f"SELECT c.id, c.name, c.discount_percent, c.valid_from, "
            f"c.valid_until, c.uses_per_patron, c.total_uses, "
            f"c.times_redeemed, pc.use_count "
            f"FROM {self._t('patron_coupons')} pc "
            f"JOIN {self._t('coupons')} c ON c.id = pc.coupon_id "
            "WHERE pc.npub = $1"
        )
        params: list[Any] = [npub]
        if operator:
            params.append(operator)
            sql += f" AND c.operator = ${len(params)}"
        sql += " ORDER BY pc.redeemed_at DESC"

        result = await self._neon._execute(sql, params)
        return [self._row_to_redemption(r) for r in result.get("rows", [])]

    async def fetch_redemptions_for_chain(
        self,
        npub: str,
        coupon_ids: list[str],
    ) -> dict[str, CouponRedemption]:
        """Batch fetch — used by the gate before walking a chain.

        Returns a dict of ``{coupon_id: CouponRedemption}`` containing
        only ids the patron has actually redeemed.  Empty / missing ids
        in the input are silently filtered out (no rows for them).
        """
        if not coupon_ids or not npub:
            return {}

        # Defensive: filter for UUID-looking strings before casting in SQL
        clean = [cid for cid in coupon_ids if cid and isinstance(cid, str)]
        if not clean:
            return {}

        # We pass coupon_ids as a JSON array Postgres can unnest. Using a
        # native UUID[] parameter avoids per-id string interpolation.
        placeholders = ", ".join(f"${i + 2}::uuid" for i in range(len(clean)))
        sql = (
            f"SELECT c.id, c.name, c.discount_percent, c.valid_from, "
            f"c.valid_until, c.uses_per_patron, c.total_uses, "
            f"c.times_redeemed, pc.use_count "
            f"FROM {self._t('patron_coupons')} pc "
            f"JOIN {self._t('coupons')} c ON c.id = pc.coupon_id "
            f"WHERE pc.npub = $1 AND c.id IN ({placeholders})"
        )
        params: list[Any] = [npub, *clean]
        try:
            result = await self._neon._execute(sql, params)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_redemptions_for_chain failed: %s", exc)
            return {}
        out: dict[str, CouponRedemption] = {}
        for row in result.get("rows", []):
            view = self._row_to_redemption(row)
            out[view.coupon_id] = view
        return out

    async def burn_use(self, coupon_id: str, npub: str) -> bool:
        """Atomically increment per-patron + aggregate counters.

        Single SQL statement using a CTE so PostgreSQL guarantees both
        UPDATEs commit together.  Returns ``True`` if a row was touched,
        ``False`` if the redemption row didn't exist (defensive — chain
        shouldn't mark consume on a non-redeemed coupon)."""
        result = await self._neon._execute(
            f"WITH pc AS ("
            f"  UPDATE {self._t('patron_coupons')} "
            f"  SET use_count = use_count + 1 "
            f"  WHERE coupon_id = $1::uuid AND npub = $2 "
            f"  RETURNING coupon_id"
            f"), c AS ("
            f"  UPDATE {self._t('coupons')} "
            f"  SET times_redeemed = times_redeemed + 1, updated_at = now() "
            f"  WHERE id = (SELECT coupon_id FROM pc) "
            f"  RETURNING id"
            f") SELECT (SELECT count(*) FROM pc)::int AS pc_count, "
            f"         (SELECT count(*) FROM c)::int AS c_count",
            [coupon_id, npub],
        )
        rows = result.get("rows", [])
        if not rows:
            return False
        return int(rows[0].get("pc_count") or 0) > 0

    # -- Helpers ----------------------------------------------------------

    @staticmethod
    def _row_to_redemption(row: dict[str, Any]) -> CouponRedemption:
        from tollbooth.coupons.models import _parse_dt
        return CouponRedemption(
            coupon_id=str(row["id"]),
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
            use_count=int(row.get("use_count") or 0),
        )

    @staticmethod
    def now_utc() -> datetime:
        """Convenience helper — UTC ``datetime`` for window checks."""
        return datetime.now(UTC)
