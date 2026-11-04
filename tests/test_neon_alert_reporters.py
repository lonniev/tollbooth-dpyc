"""Only an operator this Authority provisioned may file a Neon 402 alert (scan F35).

Anyone can make a Nostr key and sign a request with it. Without this check each
fresh key wrote an alert row and sent the owner a "paid tools are down" DM.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

import tollbooth.authority.tools as at
from tests.test_authority_tools import OP, _runtime_with_store, _tools
from tollbooth.constants import ErrorCode


class _AlertVault:
    """bootstrap_config answers for the provisioned npubs; records every write."""

    def __init__(self, provisioned: set[str], fail_lookup: bool = False) -> None:
        self.provisioned = provisioned
        self.fail_lookup = fail_lookup
        self.writes: list[str] = []

    def _t(self, table: str) -> str:
        return f"authority.{table}"

    async def _execute(self, sql: str, params: list[Any] | None = None) -> dict[str, Any]:
        if "bootstrap_config" in sql:
            if self.fail_lookup:
                raise RuntimeError("Neon HTTP 402")
            npub, key = params or ["", ""]
            hit = key == "neon_database_url" and npub in self.provisioned
            return {"rows": [["postgresql://…"]] if hit else []}
        self.writes.append(sql.strip().split()[0].upper())
        return {"rows": [], "rowCount": 1}


def _call(vault: _AlertVault, signer: str | None, npub: str = OP):
    rt, _ = _runtime_with_store()
    rt.vault = AsyncMock(return_value=vault)
    dm = AsyncMock()

    async def run():
        with _tools(rt=rt) as tools, patch.object(at, "_dm_owner", dm), \
                patch("tollbooth.holder_auth.signed_npub", return_value=signer):
            return await tools["receive_neon_402_alert"](npub=npub, detail="Neon HTTP 402")
    return run, dm


@pytest.mark.asyncio
async def test_a_provisioned_operator_records_and_tells_the_owner():
    vault = _AlertVault({OP})
    run, dm = _call(vault, OP)
    r = await run()
    assert r["success"] is True
    assert "INSERT" in vault.writes
    dm.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_fresh_key_is_refused_with_no_row_and_no_dm():
    vault = _AlertVault(set())
    run, dm = _call(vault, OP)
    r = await run()
    assert r["success"] is False
    assert r["error_code"] == ErrorCode.OPERATOR_NOT_REGISTERED
    assert vault.writes == []
    dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unchecked_reporter_is_refused_while_the_store_is_down():
    vault = _AlertVault({OP}, fail_lookup=True)
    run, dm = _call(vault, OP)
    r = await run()
    assert r["success"] is False
    assert "402" not in r["error"]
    assert vault.writes == []
    dm.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unsigned_alert_is_refused_before_the_lookup():
    vault = _AlertVault({OP})
    run, dm = _call(vault, None)
    r = await run()
    assert r["error_code"] == ErrorCode.PROOF_INVALID
    assert vault.writes == []
    dm.assert_not_awaited()
