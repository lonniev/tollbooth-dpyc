"""Operator bootstrap — discover config from Authority using only nsec.

The bootstrap sequence:
1. Derive npub from nsec
2. Look up Authority's npub from the community registry
3. Poll Nostr relays for bootstrap config DM from the Authority
4. Extract Neon URL from the encrypted config
5. Connect to Neon with encryption

The Authority sends the bootstrap config as a NIP-04 encrypted DM
at registration time. The operator reads it on cold start — no OAuth,
no MCP-to-MCP calls, no additional env vars beyond the nsec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of the bootstrap process."""
    success: bool = False
    neon_database_url: str | None = None
    encryption_nsec_hex: str | None = None
    npub: str = ""
    authority_npub: str = ""
    config: dict[str, str] = field(default_factory=dict)
    error: str | None = None


# ---------------------------------------------------------------------------
# Lazy singleton — call from any tool's initialization path
# ---------------------------------------------------------------------------

_cached_result: BootstrapResult | None = None


async def ensure_bootstrapped(
    relays: list[str] | None = None,
) -> BootstrapResult:
    """Run bootstrap once, cache the result for process lifetime.

    Call this from the first tool invocation. Returns immediately
    on subsequent calls.

    Args:
        relays: Optional relay URLs to search for the Authority's
            bootstrap config DM. Falls back to ``BOOTSTRAP_RELAYS``
            if not provided.

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

    client = BootstrapClient(nsec_hex=nsec, relays=relays)
    _cached_result = await client.bootstrap()
    return _cached_result


class BootstrapClient:
    """Discovers operator config from Nostr relays using only the nsec.

    The Authority sends a NIP-04 encrypted DM containing the operator's
    Neon URL at registration time. This client reads it on cold start.

    Usage::

        client = BootstrapClient(nsec_hex="<operator private key hex>")
        result = await client.bootstrap()
        if result.success:
            vault = NeonVault(
                database_url=result.neon_database_url,
                encryption_nsec_hex=result.encryption_nsec_hex,
            )
    """

    def __init__(self, nsec_hex: str, relays: list[str] | None = None) -> None:
        self._nsec_hex = nsec_hex
        self._relays = relays
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

        1. Resolve Authority npub from registry
        2. Poll relays for bootstrap config DM from Authority
        3. Extract Neon URL
        """
        # Convert nsec to hex for vault encryption
        from pynostr.key import PrivateKey as _PK  # type: ignore[import-untyped]
        nsec = self._nsec_hex
        if nsec.startswith("nsec1"):
            nsec_hex = _PK.from_nsec(nsec).hex()
        else:
            nsec_hex = nsec

        result = BootstrapResult(
            npub=self.npub,
            encryption_nsec_hex=nsec_hex,
        )

        # Step 1: Resolve Authority npub from registry
        try:
            authority_info = await self._resolve_authority()
            result.authority_npub = authority_info.get("npub", "")
        except Exception as e:
            result.error = f"Cannot resolve Authority: {e}"
            logger.warning("Bootstrap: %s", result.error)
            return result

        if not result.authority_npub:
            result.error = "No Authority npub found in registry"
            return result

        # Step 2: Read bootstrap config from Nostr relays
        config = self._read_config_from_relays(result.authority_npub)

        if config is None:
            diag = getattr(self, '_relay_diag', 'no diag')
            logger.warning("Bootstrap relay diagnostics: %s", diag)
            result.error = (
                f"No bootstrap config on relays from authority "
                f"{result.authority_npub[:20]}..."
            )
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
            result.error = "Neon URL not in bootstrap config from Authority"

        return result

    async def _resolve_authority(self) -> dict[str, str]:
        """Resolve this operator's Authority npub from the registry."""
        from tollbooth.registry import resolve_authority_service
        return await resolve_authority_service(self.npub)

    def _read_config_from_relays(self, authority_npub: str) -> dict[str, str] | None:
        """Poll Nostr relays for bootstrap config DM from the Authority."""
        try:
            from pynostr.key import PublicKey  # type: ignore[import-untyped]
            from tollbooth.bootstrap_relay import receive_bootstrap_config

            if authority_npub.startswith("npub1"):
                authority_hex = PublicKey.from_npub(authority_npub).hex()
            else:
                authority_hex = authority_npub

            config, diag = receive_bootstrap_config(
                operator_nsec=self._nsec_hex,
                authority_pubkey_hex=authority_hex,
                relays=self._relays,
            )
            self._relay_diag = diag
            return config
        except Exception as exc:
            self._relay_diag = str(exc)
            return None
