"""Tests for tollbooth.coupons.vault — CouponsVault CRUD + redemption.

Mirrors the pattern in tests/test_pricing_store.py: patches
``vault._client.post`` with AsyncMock so we don't need a live Neon.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest

from tollbooth.coupons.vault import (
    CouponAlreadyExists,
    CouponNotFound,
    CouponsVault,
)
from tollbooth.vaults.neon import NeonQueryError, NeonVault

DATABASE_URL = "postgresql://user:password@ep-test.us-east-2.aws.neon.tech/testdb"
HTTP_ENDPOINT = "https://ep-test.us-east-2.aws.neon.tech/sql"


def _vault() -> NeonVault:
    return NeonVault(database_url=DATABASE_URL)


def _cvault(v: NeonVault | None = None) -> CouponsVault:
    return CouponsVault(neon_vault=v or _vault())


def _response(status_code: int = 200, json_data: dict | list | None = None) -> httpx.Response:
    return httpx.Response(
        status_code=status_code,
        json=json_data,
        request=httpx.Request("POST", HTTP_ENDPOINT),
    )


def _row(**overrides) -> dict:
    base = {
        "id": "11111111-1111-4111-8111-111111111111",
        "operator": "npub1abc",
        "name": "FRESHMAN",
        "discount_percent": 50.0,
        "valid_from": "2026-05-01T00:00:00+00:00",
        "valid_until": "2026-06-01T00:00:00+00:00",
        "uses_per_patron": 1,
        "total_uses": 100,
        "times_redeemed": 0,
        "created_at": "2026-04-30T00:00:00+00:00",
        "updated_at": "2026-04-30T00:00:00+00:00",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ensure_schema
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    @pytest.mark.asyncio
    async def test_creates_tables_and_indexes(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "CREATE", "rows": []})
        )
        await _cvault(v).ensure_schema()
        # 2 CREATE TABLE + 2 CREATE INDEX
        assert v._client.post.call_count == 4


# ---------------------------------------------------------------------------
# mint
# ---------------------------------------------------------------------------


class TestMint:
    @pytest.mark.asyncio
    async def test_returns_inserted_row(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "INSERT", "rows": [_row()]})
        )
        c = await _cvault(v).mint(
            operator="npub1abc",
            name="FRESHMAN",
            discount_percent=50.0,
            valid_from=datetime(2026, 5, 1, tzinfo=UTC),
            valid_until=datetime(2026, 6, 1, tzinfo=UTC),
            uses_per_patron=1,
            total_uses=100,
        )
        assert c.name == "FRESHMAN"
        assert c.discount_percent == 50.0
        v._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_name_raises(self) -> None:
        v = _vault()
        # Simulate the UNIQUE collision Neon returns
        v._client.post = AsyncMock(
            return_value=_response(
                400, {"message": 'duplicate key value violates unique constraint'},
            )
        )
        with pytest.raises(CouponAlreadyExists):
            await _cvault(v).mint(
                operator="npub1abc",
                name="DUP",
                discount_percent=10.0,
                valid_from=datetime(2026, 5, 1, tzinfo=UTC),
                valid_until=datetime(2026, 6, 1, tzinfo=UTC),
            )

    @pytest.mark.asyncio
    async def test_other_neon_error_propagates(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(400, {"message": "some other error"})
        )
        with pytest.raises(NeonQueryError):
            await _cvault(v).mint(
                operator="npub1abc",
                name="X",
                discount_percent=10.0,
                valid_from=datetime(2026, 5, 1, tzinfo=UTC),
                valid_until=datetime(2026, 6, 1, tzinfo=UTC),
            )


# ---------------------------------------------------------------------------
# get / find_by_name / list_for_operator
# ---------------------------------------------------------------------------


class TestRead:
    @pytest.mark.asyncio
    async def test_get_returns_row(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "SELECT", "rows": [_row()]})
        )
        c = await _cvault(v).get("uuid")
        assert c is not None
        assert c.name == "FRESHMAN"

    @pytest.mark.asyncio
    async def test_get_returns_none_on_empty(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "SELECT", "rows": []})
        )
        c = await _cvault(v).get("missing")
        assert c is None

    @pytest.mark.asyncio
    async def test_find_by_name_matches(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "SELECT", "rows": [_row()]})
        )
        c = await _cvault(v).find_by_name("npub1abc", "FRESHMAN")
        assert c is not None

    @pytest.mark.asyncio
    async def test_list_for_operator(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(
                200, {"command": "SELECT", "rows": [_row(), _row(name="OTHER")]}
            )
        )
        out = await _cvault(v).list_for_operator("npub1abc")
        assert len(out) == 2
        assert {c.name for c in out} == {"FRESHMAN", "OTHER"}


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_discount_only(self) -> None:
        v = _vault()
        updated = _row(discount_percent=75.0)
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "UPDATE", "rows": [updated]})
        )
        c = await _cvault(v).update(
            "uuid", "npub1abc", discount_percent=75.0,
        )
        assert c.discount_percent == 75.0

    @pytest.mark.asyncio
    async def test_no_changes_returns_existing(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "SELECT", "rows": [_row()]})
        )
        c = await _cvault(v).update("uuid", "npub1abc")
        assert c.name == "FRESHMAN"

    @pytest.mark.asyncio
    async def test_not_found_raises(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "UPDATE", "rows": []})
        )
        with pytest.raises(CouponNotFound):
            await _cvault(v).update(
                "uuid", "npub1abc", discount_percent=10.0,
            )

    @pytest.mark.asyncio
    async def test_clear_to_unlimited(self) -> None:
        v = _vault()
        updated = _row(uses_per_patron=None)
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "UPDATE", "rows": [updated]})
        )
        c = await _cvault(v).update(
            "uuid", "npub1abc", uses_per_patron=None,
        )
        assert c.uses_per_patron is None


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_returns_true(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(
                200, {"command": "DELETE", "rowCount": 1, "rows": []},
            )
        )
        out = await _cvault(v).delete("uuid", "npub1abc")
        assert out is True

    @pytest.mark.asyncio
    async def test_delete_not_found_returns_false(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(
                200, {"command": "DELETE", "rowCount": 0, "rows": []},
            )
        )
        out = await _cvault(v).delete("uuid", "npub1abc")
        assert out is False


# ---------------------------------------------------------------------------
# redeem / forget / burn_use
# ---------------------------------------------------------------------------


class TestPatronFlow:
    @pytest.mark.asyncio
    async def test_redeem_returns_row(self) -> None:
        v = _vault()
        pc_row = {
            "id": "pc-uuid",
            "coupon_id": "c-uuid",
            "npub": "npub1pat",
            "use_count": 0,
            "redeemed_at": "2026-06-01T12:00:00+00:00",
        }
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "INSERT", "rows": [pc_row]})
        )
        pc = await _cvault(v).redeem("c-uuid", "npub1pat")
        assert pc.coupon_id == "c-uuid"
        assert pc.use_count == 0

    @pytest.mark.asyncio
    async def test_forget_returns_true(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(
                200, {"command": "DELETE", "rowCount": 1, "rows": []},
            )
        )
        out = await _cvault(v).forget("c-uuid", "npub1pat")
        assert out is True

    @pytest.mark.asyncio
    async def test_forget_missing_returns_false(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(
                200, {"command": "DELETE", "rowCount": 0, "rows": []},
            )
        )
        assert await _cvault(v).forget("c-uuid", "npub1pat") is False

    @pytest.mark.asyncio
    async def test_burn_use_atomic_cte(self) -> None:
        v = _vault()
        # Single SQL round-trip with CTE returning counters
        v._client.post = AsyncMock(
            return_value=_response(
                200, {
                    "command": "SELECT",
                    "rows": [{"pc_count": 1, "c_count": 1}],
                },
            )
        )
        out = await _cvault(v).burn_use("c-uuid", "npub1pat")
        assert out is True
        # Confirm it's a single round-trip
        assert v._client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_burn_use_missing_row_returns_false(self) -> None:
        v = _vault()
        v._client.post = AsyncMock(
            return_value=_response(
                200, {
                    "command": "SELECT",
                    "rows": [{"pc_count": 0, "c_count": 0}],
                },
            )
        )
        assert await _cvault(v).burn_use("c-uuid", "npub1pat") is False


# ---------------------------------------------------------------------------
# fetch_redemptions_for_chain
# ---------------------------------------------------------------------------


class TestFetchRedemptionsForChain:
    @pytest.mark.asyncio
    async def test_returns_keyed_map(self) -> None:
        v = _vault()
        joined_row = {
            "id": "c-uuid-A",
            "name": "FRESHMAN",
            "discount_percent": 50.0,
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": "2027-01-01T00:00:00+00:00",
            "uses_per_patron": 1,
            "total_uses": None,
            "times_redeemed": 0,
            "use_count": 0,
        }
        v._client.post = AsyncMock(
            return_value=_response(200, {"command": "SELECT", "rows": [joined_row]})
        )
        out = await _cvault(v).fetch_redemptions_for_chain(
            "npub1pat", ["c-uuid-A", "c-uuid-B"],
        )
        assert "c-uuid-A" in out
        assert out["c-uuid-A"].discount_percent == 50.0

    @pytest.mark.asyncio
    async def test_empty_list_short_circuits(self) -> None:
        v = _vault()
        v._client.post = AsyncMock()
        out = await _cvault(v).fetch_redemptions_for_chain("npub1pat", [])
        assert out == {}
        v._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_npub_short_circuits(self) -> None:
        v = _vault()
        v._client.post = AsyncMock()
        out = await _cvault(v).fetch_redemptions_for_chain("", ["c-uuid"])
        assert out == {}
        v._client.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_neon_failure_returns_empty_map(self) -> None:
        """Pre-load failure can't block the chain — just degrades to no
        coupon discount."""
        v = _vault()
        v._client.post = AsyncMock(side_effect=Exception("boom"))
        out = await _cvault(v).fetch_redemptions_for_chain(
            "npub1pat", ["c-uuid"],
        )
        assert out == {}
