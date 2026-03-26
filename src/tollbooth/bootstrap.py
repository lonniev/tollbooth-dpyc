"""Operator bootstrap — discover config from Authority using only nsec.

The bootstrap sequence:
1. Derive npub from nsec
2. Call Authority's get_operator_config(npub, proof) → Neon URL + config
3. If "not registered" → register via Authority → retry
4. Connect to Neon with encryption
5. Check for stored service config (BTCPay, etc.)
6. If missing → request via Secure Courier

This module provides the BootstrapClient that tollbooth operators
call at startup to discover their persistence and config.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of the bootstrap process."""
    success: bool = False
    neon_database_url: str | None = None
    encryption_nsec_hex: str | None = None
    npub: str = ""
    authority_npub: str = ""
    authority_url: str = ""
    config: dict[str, str] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Lazy singleton — call from any tool's initialization path
# ---------------------------------------------------------------------------

_cached_result: BootstrapResult | None = None


async def ensure_bootstrapped() -> BootstrapResult:
    """Run bootstrap once, cache the result for process lifetime.

    Call this from the first tool invocation. Returns immediately
    on subsequent calls.

    Reads ``TOLLBOOTH_NOSTR_OPERATOR_NSEC`` from the environment.
    """
    import os

    global _cached_result
    if _cached_result is not None:
        return _cached_result

    nsec = os.environ.get("TOLLBOOTH_NOSTR_OPERATOR_NSEC", "")
    if not nsec:
        result = BootstrapResult(error="TOLLBOOTH_NOSTR_OPERATOR_NSEC not set")
        _cached_result = result
        return result

    client = BootstrapClient(nsec_hex=nsec)
    _cached_result = await client.bootstrap()
    return _cached_result


class BootstrapClient:
    """Discovers operator config from the Authority using only the nsec.

    Usage::

        client = BootstrapClient(nsec_hex="<operator private key hex>")
        result = await client.bootstrap()
        if result.success:
            vault = NeonVault(
                database_url=result.neon_database_url,
                encryption_nsec_hex=result.encryption_nsec_hex,
            )
    """

    def __init__(self, nsec_hex: str) -> None:
        self._nsec_hex = nsec_hex
        self._npub: str | None = None
        self._pubkey_hex: str | None = None

    @property
    def npub(self) -> str:
        if self._npub is None:
            self._derive_identity()
        return self._npub  # type: ignore[return-value]

    @property
    def pubkey_hex(self) -> str:
        if self._pubkey_hex is None:
            self._derive_identity()
        return self._pubkey_hex  # type: ignore[return-value]

    def _derive_identity(self) -> None:
        """Derive npub and pubkey hex from nsec (hex or bech32 nsec1...)."""
        from pynostr.key import PrivateKey  # type: ignore[import-untyped]
        nsec = self._nsec_hex
        if nsec.startswith("nsec1"):
            pk = PrivateKey.from_nsec(nsec)
        else:
            pk = PrivateKey(bytes.fromhex(nsec))
        self._npub = pk.public_key.bech32()
        self._pubkey_hex = pk.public_key.hex()
        logger.info("Bootstrap identity: %s", self._npub[:16])

    async def bootstrap(self) -> BootstrapResult:
        """Run the full bootstrap sequence.

        Optimistic: tries get_operator_config first (fast path for
        already-registered operators). Falls back to registration
        if not found.
        """
        result = BootstrapResult(
            npub=self.npub,
            encryption_nsec_hex=self._nsec_hex,
        )

        # Step 1: Resolve Authority URL from registry
        try:
            authority_info = await self._resolve_authority()
            result.authority_npub = authority_info.get("npub", "")
            result.authority_url = authority_info.get("url", "")
        except Exception as e:
            result.error = f"Cannot resolve Authority: {e}"
            logger.warning("Bootstrap: %s", result.error)
            return result

        if not result.authority_url:
            result.error = "No Authority URL found in registry"
            return result

        # Step 2: Try get_operator_config (optimistic — already registered)
        config = await self._get_config(result.authority_url)

        if config is None:
            # Step 3: Not registered — register first, then retry
            logger.info("Bootstrap: not registered, requesting registration...")
            registered = await self._register(result.authority_url)
            if not registered:
                result.error = "Registration failed"
                return result
            # Retry config after registration
            config = await self._get_config(result.authority_url)

        if config is None:
            result.error = "Config not available after registration"
            return result

        result.config = config
        result.neon_database_url = config.get("neon_database_url")
        result.success = result.neon_database_url is not None

        if result.success:
            logger.info(
                "Bootstrap complete: npub=%s, authority=%s, neon=%s",
                self.npub[:16],
                result.authority_npub[:16] if result.authority_npub else "?",
                "configured" if result.neon_database_url else "missing",
            )
        else:
            result.error = "Neon URL not in config — Authority may not have provisioned it"

        return result

    async def _resolve_authority(self) -> dict[str, str]:
        """Resolve this operator's Authority URL from the registry."""
        from tollbooth.registry import resolve_authority_service
        return await resolve_authority_service(self.npub)

    async def _get_config(self, authority_url: str) -> dict[str, str] | None:
        """Call get_operator_config on the Authority with Schnorr proof."""
        try:
            from tollbooth.operator_proof import create_operator_proof
            proof = create_operator_proof(self._nsec_hex, "get_operator_config")

            from fastmcp import Client
            async with Client(authority_url, auth="oauth") as client:
                result = await client.call_tool(
                    "get_operator_config",
                    {"npub": self.npub, "operator_proof": proof},
                )
                # Parse response
                for block in getattr(result, "content", []):
                    if hasattr(block, "text"):
                        try:
                            data = json.loads(block.text)
                            if data.get("success"):
                                return data.get("config", {})
                            logger.info("get_operator_config: %s", data.get("error", "unknown"))
                            return None
                        except (json.JSONDecodeError, TypeError):
                            pass
        except Exception as e:
            logger.warning("get_operator_config failed: %s", e)
        return None

    async def _register(self, authority_url: str) -> bool:
        """Call register_operator on the Authority."""
        try:
            from fastmcp import Client
            async with Client(authority_url, auth="oauth") as client:
                result = await client.call_tool(
                    "register_operator",
                    {"npub": self.npub},
                )
                for block in getattr(result, "content", []):
                    if hasattr(block, "text"):
                        try:
                            data = json.loads(block.text)
                            if data.get("success"):
                                logger.info("Registration successful: %s", data.get("message", ""))
                                return True
                            logger.warning("Registration failed: %s", data.get("error", ""))
                            return False
                        except (json.JSONDecodeError, TypeError):
                            pass
        except Exception as e:
            logger.warning("register_operator failed: %s", e)
        return False
