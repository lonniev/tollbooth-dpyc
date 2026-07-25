"""Neon persistence health — Authority alert (B) + proactive compute watch (C).

Covers:
  * AuthorityCertifier.report_neon_quota_exceeded — operator → Authority alert.
  * neon_admin.ProjectUsage status ladder + NeonAdminClient.project_usage.
  * neon_alert_store record/list/clear roundtrip against a fake vault.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.authority import neon_alert_store
from tollbooth.authority.neon_admin import (
    DEFAULT_ALLOWANCE_SECONDS,
    NeonAdminClient,
    ProjectUsage,
)
from tollbooth.authority_client import AuthorityCertifier, AuthorityCertifyError


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
    # /projects LIST carries ids/names/quota reset — NOT compute (Neon omits it there).
    projects_payload = {
        "projects": [
            {"id": "p1", "name": "ancient-water", "quota_reset_at": "2026-08-01T00:00:00Z"},
            {"id": "p2", "name": "quiet-forest"},  # absent from consumption below
        ]
    }
    # Per-project compute comes from the consumption_history endpoint.
    consumption_payload = {
        "projects": [
            {"project_id": "p1", "periods": [{"consumption": [{"compute_time_seconds": 700000}]}]},
        ]
    }

    def _resp(payload):
        r = MagicMock()
        r.is_error = False
        r.json = MagicMock(return_value=payload)
        return r

    def _get(url, **kwargs):
        return _resp(consumption_payload) if "consumption_history" in url else _resp(projects_payload)

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        usage = await client.project_usage()

    assert [u.project_id for u in usage] == ["p1", "p2"]
    assert usage[0].compute_seconds_used == 700000  # from consumption_history
    assert usage[1].status == "unknown"             # absent from consumption
    assert client.last_usage_note == ""             # clean read → no note
    # org_id and bearer auth were passed on the /projects call.
    first = http.get.await_args_list[0]
    assert first.kwargs["params"]["org_id"] == "org-1"
    assert first.kwargs["headers"]["Authorization"] == "Bearer neon_api_key_xxx"
    # consumption_history is queried at DAILY granularity (a monthly bucket for an
    # in-progress month can come back empty — the exact 'unknown' failure mode).
    ch_call = next(c for c in http.get.await_args_list if "consumption_history" in c.args[0])
    assert ch_call.kwargs["params"]["granularity"] == "daily"


@pytest.mark.asyncio
async def test_usage_note_explains_http_failure():
    """When consumption_history HTTP-fails, usage is 'unknown' but NOT mute —
    last_usage_note carries the status + Neon's own message."""
    projects_payload = {"projects": [{"id": "p1", "name": "ancient-water"}]}

    def _get(url, **kwargs):
        r = MagicMock()
        if "consumption_history" in url:
            r.is_error = True
            r.status_code = 403
            r.text = "This feature requires a paid plan"
        else:
            r.is_error = False
            r.json = MagicMock(return_value=projects_payload)
        return r

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        usage = await client.project_usage()

    assert usage[0].status == "unknown"
    assert "403" in client.last_usage_note
    assert "paid plan" in client.last_usage_note


@pytest.mark.asyncio
async def test_usage_note_explains_empty_rows():
    """A successful consumption_history call with no compute rows still explains
    itself rather than showing a blank 'unknown'."""
    projects_payload = {"projects": [{"id": "p1", "name": "ancient-water"}]}
    consumption_payload = {"projects": []}  # call worked, but no usage rows

    def _get(url, **kwargs):
        r = MagicMock()
        r.is_error = False
        r.json = MagicMock(
            return_value=consumption_payload if "consumption_history" in url else projects_payload
        )
        return r

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        usage = await client.project_usage()

    assert usage[0].status == "unknown"
    assert "no compute rows" in client.last_usage_note


@pytest.mark.asyncio
async def test_free_plan_heartbeat_from_projects_list():
    """On Free (consumption_history 403s) the /projects list still yields a
    heartbeat: last-active + storage read straight off each project object."""
    projects_payload = {
        "projects": [
            {
                "id": "p1",
                "name": "ancient-water",
                "quota_reset_at": "2026-08-01T00:00:00Z",
                "compute_last_active_at": "2026-07-25T09:30:00Z",
                "synthetic_storage_size": 42_000_000,
            }
        ]
    }

    def _get(url, **kwargs):
        r = MagicMock()
        if "consumption_history" in url:
            r.is_error = True
            r.status_code = 403
            r.text = "This endpoint is not available. It is included with Scale plans and above."
        else:
            r.is_error = False
            r.json = MagicMock(return_value=projects_payload)
        return r

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        usage = await client.project_usage()

    d = usage[0].to_dict()
    assert d["last_active_at"] == "2026-07-25T09:30:00Z"
    assert d["storage_mb"] == pytest.approx(42.0, abs=0.1)
    assert d["storage_pct"] == pytest.approx(7.8, abs=0.2)   # 42 MB of 512 MiB
    assert d["used_pct"] is None                             # compute honestly unknown
    assert d["status"] == "ok"                               # but storage drives a real status
    assert "Scale plans" in client.last_usage_note
    # storage WAS surfaced, so the field-name breadcrumb must NOT fire.
    assert "fields available" not in client.last_usage_note


def test_storage_drives_status_when_compute_unknown():
    """A near-full Free project reads warning/critical from storage alone."""
    from tollbooth.authority.neon_admin import FREE_STORAGE_BYTES

    near_full = ProjectUsage(
        project_id="p", name="brimming",
        compute_seconds_used=None, allowance_seconds=DEFAULT_ALLOWANCE_SECONDS,
        quota_reset_at=None,
        storage_bytes=int(FREE_STORAGE_BYTES * 0.97),  # 97% of the 0.5 GiB cap
    )
    assert near_full.used_pct is None          # compute unknown
    assert near_full.storage_pct == pytest.approx(97.0, abs=0.5)
    assert near_full.status == "critical"      # storage ≥ 95%


@pytest.mark.asyncio
async def test_breadcrumb_names_fields_when_guess_misses():
    """If our heartbeat field names miss (no storage surfaced) while compute is
    unavailable, the note names the real /projects keys for the next iteration."""
    projects_payload = {"projects": [{"id": "p1", "name": "x", "some_other_size": 5}]}

    def _get(url, **kwargs):
        r = MagicMock()
        if "consumption_history" in url:
            r.is_error = True
            r.status_code = 403
            r.text = "Scale plans and above."
        else:
            r.is_error = False
            r.json = MagicMock(return_value=projects_payload)
        return r

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        await client.project_usage()

    assert "fields available" in client.last_usage_note
    assert "some_other_size" in client.last_usage_note


@pytest.mark.asyncio
async def test_own_db_host_narrows_to_one_project():
    """Given the Authority's DSN host, only the project whose endpoint matches is
    returned — the other projects the org key can see are dropped."""
    projects_payload = {
        "projects": [
            {"id": "mine-123", "name": "Authority", "synthetic_storage_size": 36_000_000},
            {"id": "other-456", "name": "shortlinks", "synthetic_storage_size": 31_000_000},
        ]
    }
    endpoints = {
        "mine-123": {"endpoints": [{"host": "ep-billowing-brook-a1b2c3.us-east-2.aws.neon.tech"}]},
        "other-456": {"endpoints": [{"host": "ep-quiet-forest-z9y8x7.us-east-2.aws.neon.tech"}]},
    }

    def _get(url, **kwargs):
        r = MagicMock()
        r.is_error = False
        if "consumption_history" in url:
            r.is_error = True
            r.status_code = 403
            r.text = "Scale plans and above."
        elif "/endpoints" in url:
            pid = url.split("/projects/")[1].split("/endpoints")[0]
            r.json = MagicMock(return_value=endpoints[pid])
        else:
            r.json = MagicMock(return_value=projects_payload)
        return r

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        # DSN uses the pooler host — the -pooler suffix must still match.
        usage = await client.project_usage(
            own_db_host="ep-billowing-brook-a1b2c3-pooler.us-east-2.aws.neon.tech"
        )

    assert [u.project_id for u in usage] == ["mine-123"]
    assert "could not match" not in client.last_usage_note


@pytest.mark.asyncio
async def test_own_db_host_no_match_shows_all_with_note():
    """An unmatchable host falls back to all projects, never an empty panel."""
    projects_payload = {"projects": [{"id": "p1", "name": "a"}, {"id": "p2", "name": "b"}]}

    def _get(url, **kwargs):
        r = MagicMock()
        r.is_error = False
        if "consumption_history" in url:
            r.is_error = True
            r.status_code = 403
            r.text = "Scale plans and above."
        elif "/endpoints" in url:
            r.json = MagicMock(return_value={"endpoints": [{"host": "ep-nomatch.x.neon.tech"}]})
        else:
            r.json = MagicMock(return_value=projects_payload)
        return r

    http = AsyncMock()
    http.get = AsyncMock(side_effect=_get)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=http):
        client = NeonAdminClient("neon_api_key_xxx", org_id="org-1")
        usage = await client.project_usage(own_db_host="ep-something-else.x.neon.tech")

    assert len(usage) == 2                       # never an empty panel
    assert "could not match" in client.last_usage_note


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
