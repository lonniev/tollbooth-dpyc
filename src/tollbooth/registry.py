"""DPYC community registry — cached membership lookup and authority resolution."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/lonniev/dpyc-community/main/members/read-only-lookup-cache.json"
)


class RegistryError(Exception):
    """Raised when a registry lookup fails (fail closed)."""


class DPYCRegistry:
    """Cached DPYC community membership lookup via HTTP.

    Fetches the community lookup cache from the registry URL and caches it
    with a monotonic-clock TTL. Any HTTP, parse, or structure error raises
    ``RegistryError`` (fail closed — no silent pass-through).
    """

    def __init__(self, url: str, cache_ttl_seconds: int = 300) -> None:
        self._url = url
        self._ttl = cache_ttl_seconds
        self._client = httpx.AsyncClient(timeout=10.0)
        self._cache: list[dict[str, Any]] | None = None
        self._cache_time: float = 0.0

    async def check_membership(self, npub: str) -> dict[str, Any]:
        """Return the member record for *npub* or raise ``RegistryError``."""
        members = await self._fetch()

        for member in members:
            if member.get("npub") == npub:
                if member.get("status") != "active":
                    raise RegistryError(
                        f"Member {npub} is not active (status={member.get('status')})."
                    )
                return member

        raise RegistryError(f"npub {npub} not found in DPYC registry.")

    async def resolve_oracle_service(self, operator_npub: str) -> dict[str, str]:
        """Walk *operator_npub*'s authority chain to Prime Authority and return Oracle service.

        The Oracle is a community service operated by the Prime Authority.
        Resolution: operator → upstream authority → ... → Prime Authority (no
        ``upstream_authority_npub``) → find service named ``dpyc-oracle``.

        Returns ``{"npub": prime_npub, "url": service_url, "name": service_name}``.
        Raises ``RegistryError`` if no Oracle service is found.
        """
        # Walk upstream to Prime Authority
        current_npub = operator_npub
        for _ in range(10):  # guard against cycles
            member = await self.check_membership(current_npub)
            upstream = member.get("upstream_authority_npub")
            if not upstream:
                # This is the Prime Authority — look for Oracle service
                services = member.get("services") or []
                for svc in services:
                    if svc.get("name") == "dpyc-oracle":
                        return {
                            "npub": current_npub,
                            "url": svc["url"],
                            "name": svc.get("name", "dpyc-oracle"),
                        }
                raise RegistryError(
                    f"Prime Authority {current_npub} has no dpyc-oracle service registered."
                )
            current_npub = upstream

        raise RegistryError("Authority chain too deep (possible cycle).")

    async def _fetch(self) -> list[dict[str, Any]]:
        """Return the cached member list, refreshing if stale."""
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_time) < self._ttl:
            return self._cache

        try:
            resp = await self._client.get(self._url)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as e:
            raise RegistryError(f"Registry fetch failed: {e}") from e
        except Exception as e:
            raise RegistryError(f"Registry parse failed: {e}") from e

        # Handle both bare list and {"members": [...]} wrapper formats
        if isinstance(data, dict):
            if "members" in data and isinstance(data["members"], list):
                data = data["members"]
            else:
                raise RegistryError(
                    "Registry JSON object missing 'members' list."
                )
        elif not isinstance(data, list):
            raise RegistryError("Registry JSON is not a list or object.")

        self._cache = data
        self._cache_time = now
        return data

    def invalidate_cache(self) -> None:
        """Force the next lookup to re-fetch."""
        self._cache = None
        self._cache_time = 0.0

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


async def resolve_oracle_service(
    operator_npub: str,
    registry_url: str = DEFAULT_REGISTRY_URL,
    cache_ttl_seconds: int = 300,
) -> dict[str, str]:
    """Convenience function: resolve the Oracle service via authority chain.

    Returns ``{"npub": prime_npub, "url": service_url, "name": service_name}``.
    Creates a one-shot ``DPYCRegistry``, performs the lookup, and closes.
    """
    registry = DPYCRegistry(url=registry_url, cache_ttl_seconds=cache_ttl_seconds)
    try:
        return await registry.resolve_oracle_service(operator_npub)
    finally:
        await registry.close()


async def resolve_my_parent_npub(
    own_npub: str,
    registry_url: str = DEFAULT_REGISTRY_URL,
    cache_ttl_seconds: int = 300,
) -> str:
    """Resolve the parent Authority npub for this Authority's own npub.

    Reads ``own_npub``'s entry from dpyc-community and returns its
    ``upstream_authority_npub``. Used by Authority onboarding flows to
    escalate approval requests to the candidate's *registered* parent
    rather than hardcoding Prime — letting a sub-Authority (e.g.,
    Tollbooth-Authority-NewEngland) escalate to its actual sponsor
    (Tollbooth-Authority-NorthAmerica) instead of always to Prime.

    Raises:
        RegistryError: when the registry can't be fetched.
        ValueError: when ``own_npub`` is not in the registry, or is
            registered without an upstream (Prime itself has no parent).
    """
    registry = DPYCRegistry(url=registry_url, cache_ttl_seconds=cache_ttl_seconds)
    try:
        members = await registry._fetch()
        for m in members:
            if m.get("npub") == own_npub:
                parent = m.get("upstream_authority_npub", "")
                if not parent:
                    raise ValueError(
                        f"Authority {own_npub[:16]}… has no upstream_authority_npub "
                        "in the dpyc-community registry. Prime is the trust root and "
                        "has no parent; for any other Authority, add an "
                        "upstream_authority_npub to its registry entry."
                    )
                return parent
        raise ValueError(
            f"Authority {own_npub[:16]}… is not in the dpyc-community registry. "
            "Add members/authorities/{own_npub}.json with the desired "
            "upstream_authority_npub before initiating an onboarding claim."
        )
    finally:
        await registry.close()
