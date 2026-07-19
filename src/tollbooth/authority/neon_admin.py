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
            resp.raise_for_status()
            data = resp.json()

        projects = data.get("projects") if isinstance(data, dict) else None
        out: list[ProjectUsage] = []
        for p in projects or []:
            if not isinstance(p, dict):
                continue
            used = p.get("compute_time_seconds")
            out.append(
                ProjectUsage(
                    project_id=str(p.get("id") or ""),
                    name=str(p.get("name") or p.get("id") or "unnamed"),
                    compute_seconds_used=int(used) if isinstance(used, (int, float)) else None,
                    allowance_seconds=self._allowance_seconds,
                    quota_reset_at=(
                        str(p["quota_reset_at"]) if p.get("quota_reset_at") else None
                    ),
                )
            )
        return out
