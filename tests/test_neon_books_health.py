"""Neon books health — Authority alert (B) + proactive compute watch (C).

Covers:
  * AuthorityCertifier.report_neon_quota_exceeded — operator → Authority alert.
  * neon_admin.ProjectUsage status ladder + NeonAdminClient.project_usage.
  * neon_alert_store record/list/clear roundtrip against a fake vault.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.authority_client import AuthorityCertifier, AuthorityCertifyError
from tollbooth.authority import neon_alert_store
from tollbooth.authority.neon_admin import (
    DEFAULT_ALLOWANCE_SECONDS,
    NeonAdminClient,
    ProjectUsage,
)


def _text_block(data: dict) -> MagicMock:
    block = MagicMock()
    block.text = json.dumps(data)
    return block


# --------------------------------------------------------------------------
# B — operator reports its 402 to the Authority
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_report_neon_quota_signs_for_wire_name():
    """The alert calls authority_receive_neon_402_alert with an injected proof."""
    mock_client = AsyncMock()
    mock_client.call_tool = AsyncMock(return_value=[_text_block({"success": True})])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("tollbooth.authority_client.Client", return_value=mock_client):
        certifier = AuthorityCertifier(
            authority_url="https://authority.example.com/mcp",
            operator_npub="npub1operator",
        )
        await certifier.report_neon_quota_exceeded("Neon HTTP 402: quota exhausted")

    tool_name, args = mock_client.call_tool.await_args.args
    assert tool_name == "authority_receive_neon_402_alert"
    assert args["npub"] == "npub1operator"
    assert "dpop_token" in args            # proof injected by the signer
    assert args["detail"].startswith("Neon HTTP 402")


@pytest.mark.asyncio
async def test_report_neon_quota_raises_on_transport_failure():
    """A transport failure surfaces as AuthorityCertifyError (caller logs it)."""
    failing = AsyncMock()
    failing.__aenter__ = AsyncMock(side_effect=RuntimeError("relay down"))
    failing.__aexit__ = AsyncMock(return_value=False)
    with patch("tollbooth.authority_client.Client", return_value=failing):
        certifier = AuthorityCertifier("https://a.example/mcp", "npub1operator")
        with pytest.raises(AuthorityCertifyError):
            await certifier.report_neon_quota_exceeded("boom")


# --------------------------------------------------------------------------
# C — ProjectUsage status ladder
# --------------------------------------------------------------------------


def _usage(hours_used: float | None) -> ProjectUsage:
    return ProjectUsage(
        project_id="p1",
        name="ancient-water",
        compute_seconds_used=None if hours_used is None else int(hours_used * 3600),
        allowance_seconds=DEFAULT_ALLOWANCE_SECONDS,  # ~191.9 h
        quota_reset_at="2026-08-01T00:00:00Z",
    )


@pytest.mark.parametrize(
    "hours,expected",
    [
        (10.0, "ok"),          # ~5%
        (155.0, "warning"),    # ~81%
        (185.0, "critical"),   # ~96%
        (200.0, "exhausted"),  # >100%
        (None, "unknown"),     # no data from the API
    ],
)
def test_project_usage_status_ladder(hours, expected):
    assert _usage(hours).status == expected


def test_project_usage_to_dict_shape():
    d = _usage(200.0).to_dict()
    assert d["status"] == "exhausted"
    assert d["used_pct"] == pytest.approx(104.2, abs=0.2)
    assert d["compute_hours_used"] == pytest.approx(200.0, abs=0.1)
    assert d["quota_reset_at"] == "2026-08-01T00:00:00Z"


@pytest.mark.asyncio
async def test_neon_admin_client_parses_projects():
    """project_usage maps the Neon projects list into ProjectUsage rows,
    tolerating a project with no consumption field (status unknown)."""
    payload = {
        "projects": [
            {"id": "p1", "name": "ancient-water", "compute_time_seconds": 700000,
             "quota_reset_at": "2026-08-01T00:00:00Z"},
            {"id": "p2", "name": "quiet-forest"},  # no consumption reported
        ]
    }
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=payload)

    http = AsyncMock()
    http.get = AsyncMock(return_value=resp)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        usage = await client.project_usage()

    assert [u.project_id for u in usage] == ["p1", "p2"]
    assert usage[0].compute_seconds_used == 700000
    assert usage[1].status == "unknown"
    # org_id and bearer auth were passed.
    _, kwargs = http.get.await_args
    assert kwargs["params"]["org_id"] == "org-1"
    assert kwargs["headers"]["Authorization"] == "Bearer neon_api_key_xxx"


# --------------------------------------------------------------------------
# C — durable alert store roundtrip against a fake vault
# --------------------------------------------------------------------------


class _FakeVault:
    """Records SQL and answers list/roundtrip like Neon's HTTP helper."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def _t(self, table: str) -> str:
        return f"authority.{table}"

    async def _execute(self, sql: str, params=None):
        params = params or []
        s = sql.strip().upper()
        if s.startswith("CREATE TABLE"):
            return {"rows": [], "rowCount": 0}
        if s.startswith("INSERT"):
            npub, detail = params[0], params[1]
            if npub in self.rows:
                self.rows[npub]["seen_count"] += 1
                self.rows[npub]["detail"] = detail
            else:
                self.rows[npub] = {"operator_npub": npub, "detail": detail,
                                   "seen_count": 1, "first_seen_at": "t", "last_seen_at": "t"}
            return {"rows": [], "rowCount": 1}
        if s.startswith("SELECT"):
            return {"rows": list(self.rows.values())}
        if s.startswith("DELETE"):
            hit = params[0] in self.rows
            self.rows.pop(params[0], None)
            return {"rows": [], "rowCount": 1 if hit else 0}
        return {"rows": [], "rowCount": 0}


@pytest.mark.asyncio
async def test_alert_store_roundtrip():
    vault = _FakeVault()
    await neon_alert_store.ensure_schema(vault)
    await neon_alert_store.record(vault, "npub1op", "Neon HTTP 402")
    await neon_alert_store.record(vault, "npub1op", "Neon HTTP 402 again")  # idempotent bump

    alerts = await neon_alert_store.list_all(vault)
    assert len(alerts) == 1
    assert alerts[0]["operator_npub"] == "npub1op"
    assert alerts[0]["seen_count"] == 2
    assert alerts[0]["detail"] == "Neon HTTP 402 again"

    assert await neon_alert_store.clear(vault, "npub1op") is True
    assert await neon_alert_store.list_all(vault) == []
    assert await neon_alert_store.clear(vault, "npub1op") is False
