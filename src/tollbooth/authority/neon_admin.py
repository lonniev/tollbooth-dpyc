"""Neon control-plane client — proactive compute-quota watch for the Authority.

Neon is the DPYC economy's accounting books, and the books are the Authority's
responsibility. This module lets an Authority read, PROACTIVELY, how much of
each Neon project's compute allowance has been consumed and when it resets —
so it can raise a warning before a project 402s, instead of learning from a
patron who was told to "retry" through the outage.

It needs a Neon **API key** (org-scoped, personal or org key) — a NEW credential
that the ecosystem did not previously carry. The Authority delivers it via
Secure Courier like any other operator secret; until it arrives, the proactive
watch reports ``configured=false`` and the Authority falls back to reactive
self-detection (a 402 on its own vault) and to operator-reported alerts.

The compute allowance is not reliably exposed by the API for the Free plan, so
it is a parameter (``allowance_seconds``), defaulting to Neon Free's ~191.9
compute-hours. Override it per plan. All network shape is tolerant: missing
fields degrade to ``None`` rather than raising, because a health probe must
never itself become an outage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC
from typing import Any

logger = logging.getLogger(__name__)

NEON_API_BASE = "https://console.neon.tech/api/v2"

# Neon Free plan compute allowance ≈ 191.9 compute-hours per billing period.
FREE_COMPUTE_HOURS = 191.9
DEFAULT_ALLOWANCE_SECONDS = int(FREE_COMPUTE_HOURS * 3600)

# Fraction-of-allowance thresholds for the health status ladder.
WARNING_AT = 0.80
CRITICAL_AT = 0.95


@dataclass
class ProjectUsage:
    """One Neon project's compute posture for the current billing period."""

    project_id: str
    name: str
    compute_seconds_used: int | None
    allowance_seconds: int
    quota_reset_at: str | None
    # Free-plan heartbeat: read straight from the /projects list (no Scale-only
    # consumption endpoint). When compute-hours can't be read (Free 403s the
    # consumption API), these still prove the project is alive and how full it is.
    last_active_at: str | None = None
    storage_bytes: int | None = None

    @property
    def used_fraction(self) -> float | None:
        if self.compute_seconds_used is None or self.allowance_seconds <= 0:
            return None
        return self.compute_seconds_used / self.allowance_seconds

    @property
    def used_pct(self) -> float | None:
        frac = self.used_fraction
        return None if frac is None else round(frac * 100.0, 1)

    @property
    def status(self) -> str:
        """ok | warning | critical | exhausted | unknown."""
        frac = self.used_fraction
        if frac is None:
            return "unknown"
        if frac >= 1.0:
            return "exhausted"
        if frac >= CRITICAL_AT:
            return "critical"
        if frac >= WARNING_AT:
            return "warning"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "compute_seconds_used": self.compute_seconds_used,
            "compute_hours_used": (
                None if self.compute_seconds_used is None
                else round(self.compute_seconds_used / 3600.0, 1)
            ),
            "allowance_hours": round(self.allowance_seconds / 3600.0, 1),
            "used_pct": self.used_pct,
            "quota_reset_at": self.quota_reset_at,
            "status": self.status,
            # Free-plan heartbeat fields (present even when compute is 'unknown').
            "last_active_at": self.last_active_at,
            "storage_mb": (
                None if self.storage_bytes is None
                else round(self.storage_bytes / 1_000_000.0, 1)
            ),
        }


class NeonAdminClient:
    """Thin read-only client over the Neon control-plane REST API.

    Only reads project consumption — it never mutates Neon. One short-lived
    ``httpx.AsyncClient`` per call set; the API key is Bearer-auth and is never
    logged.
    """

    def __init__(
        self,
        api_key: str,
        *,
        org_id: str = "",
        allowance_seconds: int = DEFAULT_ALLOWANCE_SECONDS,
        base_url: str = NEON_API_BASE,
    ) -> None:
        self._api_key = api_key
        self._org_id = org_id
        self._allowance_seconds = allowance_seconds
        self._base_url = base_url.rstrip("/")
        # Diagnostic breadcrumb from the last consumption_history probe: "" when
        # compute numbers came back clean, otherwise a short, credential-free
        # reason the caller can surface so "unknown" is never mute.
        self.last_usage_note: str = ""

    async def project_usage(self) -> list[ProjectUsage]:
        """List every project's compute posture for the current period.

        Raises on transport/auth failure so the caller can distinguish "the
        key is bad" from "everything's healthy"; individual missing fields do
        NOT raise — they degrade to ``None`` (status ``unknown``).
        """
        import httpx

        params: dict[str, str] = {"limit": "400"}
        if self._org_id:
            params["org_id"] = self._org_id
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                f"{self._base_url}/projects", params=params, headers=headers
            )
            if resp.is_error:
                # Surface Neon's OWN message, not just the bare status line. The most
                # common 400 on an org-scoped key is a missing org_id — /projects needs
                # it for organization keys — so front-load that actionable hint.
                hint = (
                    " (an org-scoped Neon key needs neon_org_id — deliver it via Secure "
                    "Courier alongside neon_api_key)"
                    if resp.status_code == 400 and not self._org_id
                    else ""
                )
                raise RuntimeError(
                    f"Neon /projects {resp.status_code}{hint}: {resp.text[:200].strip()}"
                )
            data = resp.json()

            projects = data.get("projects") if isinstance(data, dict) else None
            projects = projects or []
            # Per-project COMPUTE-HOUR consumption lives in Neon's consumption_history
            # endpoint — a Scale-plan feature. On Free it 403s, and used_pct/status
            # stay "unknown"; last_usage_note explains why. Best-effort either way.
            used_by_id, self.last_usage_note = (
                await self._compute_seconds_by_project(client, headers)
            )

        out: list[ProjectUsage] = []
        for p in projects:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id") or "")
            # Heartbeat fields live on the /projects list object itself (Free-OK).
            last_active = p.get("compute_last_active_at") or p.get("updated_at")
            storage = p.get("synthetic_storage_size")
            out.append(
                ProjectUsage(
                    project_id=pid,
                    name=str(p.get("name") or pid or "unnamed"),
                    compute_seconds_used=used_by_id.get(pid),
                    allowance_seconds=self._allowance_seconds,
                    quota_reset_at=(
                        str(p["quota_reset_at"]) if p.get("quota_reset_at") else None
                    ),
                    last_active_at=str(last_active) if last_active else None,
                    storage_bytes=(
                        int(storage) if isinstance(storage, (int, float)) else None
                    ),
                )
            )

        # Self-verifying breadcrumb: if compute is unavailable AND we surfaced no
        # storage for any project, our field-name guess missed — name the real
        # keys once so the next read designs against ground truth, not a guess.
        if (
            not used_by_id
            and out
            and all(u.storage_bytes is None for u in out)
            and isinstance(projects[0], dict)
        ):
            keys = ", ".join(sorted(projects[0].keys()))
            self.last_usage_note = (
                f"{self.last_usage_note} | /projects fields available: {keys}"
            ).strip(" |")
        return out

    async def _compute_seconds_by_project(
        self, client: Any, headers: dict[str, str]
    ) -> tuple[dict[str, int], str]:
        """Sum each project's ``compute_time_seconds`` for the current billing
        period via Neon's ``consumption_history`` endpoint (the /projects list
        omits it, so used_pct would otherwise always be 'unknown').

        Returns ``(used_by_project_id, note)``. ``note`` is ``""`` on a clean read
        and otherwise a short, credential-free reason usage is unavailable — an
        HTTP status + Neon's own message, a transport error, or "no rows in
        range". Best-effort and tolerant: any failure returns an empty map (usage
        degrades to None) but NEVER a mute one — a health probe that can't read
        usage should say why, not just show "unknown".

        Granularity is **daily**, not monthly: a monthly bucket for an
        *in-progress* month can come back empty (the period hasn't closed), which
        is precisely how usage silently read as "unknown". Daily buckets exist
        from the 1st onward and sum to month-to-date compute.
        """
        from datetime import datetime

        try:
            now = datetime.now(UTC)
            # Current period ≈ the calendar month — Neon Free resets on the 1st,
            # which is what quota_reset_at reflects. Day-granular buckets from the
            # 1st to now sum to month-to-date (≤ 31 buckets/project, under limit).
            start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            params: dict[str, str] = {
                "from": start.isoformat().replace("+00:00", "Z"),
                "to": now.isoformat().replace("+00:00", "Z"),
                "granularity": "daily",
                "limit": "100",
            }
            if self._org_id:
                params["org_id"] = self._org_id
            resp = await client.get(
                f"{self._base_url}/consumption_history/projects",
                params=params,
                headers=headers,
            )
            if resp.is_error:
                msg = resp.text[:180].strip()
                logger.info("Neon consumption_history %s: %s", resp.status_code, msg)
                hint = ""
                if resp.status_code in (401, 403):
                    hint = (
                        " — the Neon key may lack consumption-metrics scope, or the"
                        " plan doesn't expose per-project consumption"
                    )
                return {}, f"consumption_history {resp.status_code}{hint}: {msg}"
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 — best-effort probe; must never raise
            logger.info("Neon consumption_history read failed: %s", exc)
            return {}, f"consumption_history read error: {str(exc)[:160]}"

        out: dict[str, int] = {}
        rows = (data.get("projects") if isinstance(data, dict) else None) or []
        for proj in rows:
            if not isinstance(proj, dict):
                continue
            pid = str(proj.get("project_id") or "")
            if not pid:
                continue
            total = 0.0
            seen = False
            for period in proj.get("periods") or []:
                if not isinstance(period, dict):
                    continue
                for c in period.get("consumption") or []:
                    v = c.get("compute_time_seconds") if isinstance(c, dict) else None
                    if isinstance(v, (int, float)):
                        total += v
                        seen = True
            if seen:
                out[pid] = int(total)
        if not out:
            # The call SUCCEEDED but carried no compute rows for the window —
            # distinct from an HTTP failure, and worth saying so.
            return {}, (
                "consumption_history returned no compute rows for the current "
                f"period ({len(rows)} project entries, none with usage)"
            )
        return out, ""
