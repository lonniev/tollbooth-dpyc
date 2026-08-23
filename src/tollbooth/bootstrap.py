"""Operator bootstrap — discover config from Nostr using only nsec.

The bootstrap sequence (no direct GitHub access — operators are nsec-only):
1. Derive npub from nsec
2. Seed the Nostr relay set from the Oracle (one MCP call to the one fixed
   anchor, ``DPYC_ORACLE_MCP_URL``)
3. Poll those relays for THIS operator's config event by its own ``d`` tag —
   the operator does not need to know its Authority to find it; the Authority
   npub is discovered from the event's author
4. Extract Neon URL from the encrypted config
5. Connect to Neon with encryption

The Authority publishes the bootstrap config as a NIP-33 parameterized-
replaceable event (kind 30078) at registration time; relays keep the latest,
so it does not age off. The operator reads it on cold start — no OAuth, no
GitHub reads, no additional env vars beyond the nsec. The Oracle is asked who
the Authority is (to accept only its event, and/or to verify the discovered
author), but a working Neon URL is the final backstop either way.
"""

from __future__ import annotations

import asyncio
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
    # True when the failure was about reachability rather than configuration.
    # A missing nsec is a fact about this deployment and will not change; an
    # unreachable relay is a fact about the last few seconds and will.
    transient: bool = False


# ---------------------------------------------------------------------------
# Lazy singleton — call from any tool's initialization path
# ---------------------------------------------------------------------------

_cached_result: BootstrapResult | None = None

# Seconds to wait after each failed relay poll; the final 0 means "no more
# waiting, this was the last attempt". Roughly 75s of coverage in total, against
# a job budget measured in minutes — sized for a relay outage lasting seconds,
# not for one lasting long enough that a human should hear about it.
_BOOTSTRAP_RETRY_BACKOFF = (2, 5, 10, 20, 38, 0)


async def ensure_bootstrapped(
    relays: list[str] | None = None,
) -> BootstrapResult:
    """Run bootstrap once, cache the result for process lifetime.

    Call this from the first tool invocation. Returns immediately
    on subsequent calls.

    Args:
        relays: Optional relay URLs to search for the Authority's
            bootstrap config DM. Falls back to the DPYC community relay
            registry (``relay_registry.get_relays``) if not provided.

    Reads ``TOLLBOOTH_NOSTR_OPERATOR_NSEC`` from the environment.
    """
    import os

    global _cached_result
    if _cached_result is not None:
        return _cached_result

    nsec = os.environ.get("TOLLBOOTH_NOSTR_OPERATOR_NSEC", "")
    if not nsec:
        # Definitive: no amount of retrying produces an nsec. Cache it.
        result = BootstrapResult(error="TOLLBOOTH_NOSTR_OPERATOR_NSEC not set")
        _cached_result = result
        return result

    client = BootstrapClient(nsec_hex=nsec, relays=relays)
    result = await client.bootstrap()

    # Cache success, and cache a definitive failure. Do NOT cache a transient
    # one: this result is memoised for the whole process, so caching "the
    # relays were down a second ago" would pin a front to broken until it
    # recycles, and pin every later tool call to the same stale verdict. The
    # same lesson was learned one layer up at 0.62.3, where
    # _ensure_async_executor cached its resolution before loading credentials
    # and a cold-vault blip pinned a container to in-process for life.
    if result.success or not result.transient:
        _cached_result = result
    else:
        logger.info(
            "Bootstrap failed transiently (%s); not cached, next call retries.",
            result.error,
        )
    return result


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
        """Run the full bootstrap sequence — nsec + Oracle + Nostr, no GitHub.

        1. Seed relays from the Oracle (or use injected ``relays``)
        2. Ask the Oracle who our Authority is (best-effort — used to accept
           only its config event; falls back to discover-from-event)
        3. Poll relays for our config event by our own ``d`` tag
        4. Extract Neon URL; the Authority npub is the event's author
        """
        from pynostr.key import PrivateKey as _PK  # type: ignore[import-untyped]
        from pynostr.key import PublicKey

        from tollbooth.bootstrap_relay import receive_bootstrap_config
        from tollbooth.oracle_client import default_oracle_client

        # Convert nsec to hex for vault encryption
        nsec = self._nsec_hex
        nsec_hex = _PK.from_nsec(nsec).hex() if nsec.startswith("nsec1") else nsec

        result = BootstrapResult(npub=self.npub, encryption_nsec_hex=nsec_hex)

        oracle = default_oracle_client()

        # Step 1: relay set. Injected relays win (tests / callers); otherwise the
        # Oracle is the one fixed anchor an nsec-only operator may know a priori.
        relays = self._relays
        if relays is None:
            try:
                relays = await oracle.get_relays()
            except Exception as e:  # noqa: BLE001
                result.error = f"Cannot reach Oracle for relay set: {e}"
                logger.warning("Bootstrap: %s", result.error)
                return result
            # Warm the process-wide cache so synchronous consumers (courier,
            # profile, audit) reuse this set instead of re-fetching.
            from tollbooth.relay_registry import seed_relays
            seed_relays(relays)

        # Step 2: ask the Oracle who our Authority is. When it answers, we accept
        # ONLY that author's config event (spoof guard). When it can't (operator
        # unknown/new, or Oracle briefly unreachable), we proceed by our own
        # d-tag and discover the author from the event — a working Neon URL is
        # the backstop, and the author can be re-verified later.
        expected_authority_hex: str | None = None
        try:
            auth = await oracle.resolve_authority_for(self.npub)
            if auth and auth.get("npub"):
                result.authority_npub = auth["npub"]
                expected_authority_hex = PublicKey.from_npub(auth["npub"]).hex()
        except Exception as e:  # noqa: BLE001 — verification is best-effort
            logger.info(
                "Bootstrap: Oracle authority pre-resolve unavailable (%s); "
                "accepting config by operator d-tag, discovering author from event.",
                e,
            )

        # Step 3: read our own config from Nostr using only our nsec.
        #
        # Retried, because one pass is not evidence the config is unreachable.
        # Relays flap on the order of seconds: on 2026-08-23 the two relays
        # carrying one operator's config both refused within the same window
        # (502 and 503) and were serving again about 110s later. A single poll
        # turned that into a failed drill.
        #
        # A detached runner feels this where a warm front does not. Horizon
        # bootstraps once per process and keeps the result; a Modal container
        # cold-boots and bootstraps on EVERY job, so it meets whatever relay
        # weather exists at that moment. The job already holds a multi-minute
        # budget, so spending a fraction of it here is close to free — and
        # giving up on the first pass spends none of it and discards the work.
        config = author_hex = diag = None
        for attempt, pause in enumerate(_BOOTSTRAP_RETRY_BACKOFF, start=1):
            config, author_hex, diag = receive_bootstrap_config(
                operator_nsec=self._nsec_hex,
                relays=relays,
                expected_authority_hex=expected_authority_hex,
            )
            if config is not None:
                if attempt > 1:
                    logger.info("Bootstrap config found on relay attempt %d", attempt)
                break
            if pause:
                logger.info(
                    "Bootstrap: no config on attempt %d/%d (%s); retrying in %ss",
                    attempt, len(_BOOTSTRAP_RETRY_BACKOFF), diag, pause,
                )
                await asyncio.sleep(pause)
        self._relay_diag = diag

        if config is None:
            logger.warning("Bootstrap relay diagnostics: %s", diag)
            result.error = (
                "No bootstrap config on relays for this operator"
                + (
                    f" from authority {result.authority_npub[:20]}..."
                    if result.authority_npub
                    else ""
                )
            )
            # Reachability is a moment-in-time fact, so this verdict is not
            # durable — see ensure_bootstrapped, which declines to cache it.
            result.transient = True
            return result

        # Discover the Authority from the event when the Oracle didn't pre-resolve.
        if author_hex and not result.authority_npub:
            try:
                result.authority_npub = PublicKey(bytes.fromhex(author_hex)).bech32()
            except Exception as e:  # noqa: BLE001
                logger.debug("Could not encode author %s as npub: %s", author_hex[:16], e)

        result.config = config
        result.neon_database_url = config.get("neon_database_url")
        result.success = result.neon_database_url is not None

        if result.success:
            logger.info(
                "Bootstrap complete: npub=%s, authority=%s, neon=configured",
                self.npub[:16],
                result.authority_npub[:16] if result.authority_npub else "?",
            )
        else:
            result.error = "Neon URL not in bootstrap config from Authority"

        return result
