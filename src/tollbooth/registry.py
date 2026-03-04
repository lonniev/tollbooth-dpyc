"""DPYC community registry — cached membership lookup and authority resolution."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/lonniev/dpyc-community/main/members.json"
)


class RegistryError(Exception):
    """Raised when a registry lookup fails (fail closed)."""


class DPYCRegistry:
    """Cached DPYC community membership lookup via HTTP.

    Fetches the community members.json from the registry URL and caches it
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

    async def resolve_authority_npub(self, operator_npub: str) -> str:
        """Look up *operator_npub* and return its ``upstream_authority_npub``.

        Raises ``RegistryError`` if the operator is not found, not active,
        or has no upstream authority configured (i.e. is a Prime Authority).
        """
        member = await self.check_membership(operator_npub)
        upstream = member.get("upstream_authority_npub")
        if not upstream:
            raise RegistryError(
                f"Member {operator_npub} has no upstream authority "
                f"(role={member.get('role')})."
            )
        return upstream

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

    async def resolve_authority_service(self, operator_npub: str) -> dict[str, str]:
        """Look up *operator_npub*'s upstream authority and return its service URL.

        Returns ``{"npub": authority_npub, "url": service_url, "name": service_name}``.
        Raises ``RegistryError`` if the authority has no services registered.
        """
        authority_npub = await self.resolve_authority_npub(operator_npub)
        authority_member = await self.check_membership(authority_npub)

        services = authority_member.get("services") or []
        if not services:
            raise RegistryError(
                f"Authority {authority_npub} has no services registered in the registry."
            )

        svc = services[0]
        return {
            "npub": authority_npub,
            "url": svc["url"],
            "name": svc.get("name", "unknown"),
        }

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


async def resolve_authority_npub(
    operator_npub: str,
    registry_url: str = DEFAULT_REGISTRY_URL,
    cache_ttl_seconds: int = 300,
) -> str:
    """Convenience function: resolve an operator's upstream authority npub.

    Creates a one-shot ``DPYCRegistry``, performs the lookup, and closes.
    For repeated lookups, prefer creating a ``DPYCRegistry`` instance and
    reusing it.
    """
    registry = DPYCRegistry(url=registry_url, cache_ttl_seconds=cache_ttl_seconds)
    try:
        return await registry.resolve_authority_npub(operator_npub)
    finally:
        await registry.close()


async def resolve_authority_service(
    operator_npub: str,
    registry_url: str = DEFAULT_REGISTRY_URL,
    cache_ttl_seconds: int = 300,
) -> dict[str, str]:
    """Convenience function: resolve an operator's upstream authority service.

    Returns ``{"npub": authority_npub, "url": service_url, "name": service_name}``.
    Creates a one-shot ``DPYCRegistry``, performs the lookup, and closes.
    """
    registry = DPYCRegistry(url=registry_url, cache_ttl_seconds=cache_ttl_seconds)
    try:
        return await registry.resolve_authority_service(operator_npub)
    finally:
        await registry.close()


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
