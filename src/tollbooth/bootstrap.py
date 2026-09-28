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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tollbooth.oracle_client import OracleClient

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
#
# Detached runners (Modal cold-boot per job) keep this long ladder: the job
# already holds a multi-minute budget. Live agent fronts must NOT — see
# ``FRONT_BOOTSTRAP_RETRY_BACKOFF``.
_BOOTSTRAP_RETRY_BACKOFF = (2, 5, 10, 20, 38, 0)

# Front ladder: at most two tries, then answer quickly. A miss is not cached
# (see ensure_bootstrapped), so the next call retries; spending ~80 s on four
# concurrent cold first-calls is what pinned eXcalibur on 2026-09-27.
FRONT_BOOTSTRAP_RETRY_BACKOFF = (2, 0)

# Single-flight: concurrent first calls on a cold process share one read.
# Lazily built so importing this module never needs a running event loop.
_bootstrap_lock: asyncio.Lock | None = None


def _get_bootstrap_lock() -> asyncio.Lock:
    global _bootstrap_lock
    if _bootstrap_lock is None:
        _bootstrap_lock = asyncio.Lock()
    return _bootstrap_lock


async def ensure_bootstrapped(
    relays: list[str] | None = None,
    *,
    retry_backoff: tuple[int, ...] | None = None,
) -> BootstrapResult:
    """Run bootstrap once, cache the result for process lifetime.

    Call this from the first tool invocation. Returns immediately
    on subsequent calls. Concurrent first calls share a single in-flight
    read (one relay poll serves every waiter).

    Args:
        relays: Optional relay URLs to search for the Authority's
            bootstrap config DM. Falls back to the DPYC community relay
            registry (``relay_registry.get_relays``) if not provided.
        retry_backoff: Pause schedule after each failed poll (final 0 =
            last attempt). Defaults to :data:`FRONT_BOOTSTRAP_RETRY_BACKOFF`
            — short enough for a live front. Detached runners that cold-boot
            per job should pass :data:`_BOOTSTRAP_RETRY_BACKOFF`.

    Reads ``TOLLBOOTH_NOSTR_OPERATOR_NSEC`` from the environment.
    """
    import os

    global _cached_result
    if _cached_result is not None:
        return _cached_result

    # One read serves every waiter. Hold the lock for the whole bootstrap so
    # four concurrent cold first-calls (the 2026-09-27 eXcalibur shape) cannot
    # each construct a client and each walk the full retry ladder.
    async with _get_bootstrap_lock():
        if _cached_result is not None:
            return _cached_result

        nsec = os.environ.get("TOLLBOOTH_NOSTR_OPERATOR_NSEC", "")
        if not nsec:
            # Definitive: no amount of retrying produces an nsec. Cache it.
            result = BootstrapResult(error="TOLLBOOTH_NOSTR_OPERATOR_NSEC not set")
            _cached_result = result
            return result

        client = BootstrapClient(nsec_hex=nsec, relays=relays)
        result = await client.bootstrap(
            retry_backoff=retry_backoff
            if retry_backoff is not None
            else FRONT_BOOTSTRAP_RETRY_BACKOFF,
        )

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
        # Relays already reported this process, so one container's repeated
        # bootstraps do not ask the Oracle to re-probe the same dead relay.
        self._reported_relays: set[str] = set()

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

    async def bootstrap(
        self,
        *,
        retry_backoff: tuple[int, ...] | None = None,
    ) -> BootstrapResult:
        """Run the full bootstrap sequence — nsec + Oracle + Nostr, no GitHub.

        1. Seed relays from the Oracle (or use injected ``relays``)
        2. Ask the Oracle who our Authority is (best-effort — used to accept
           only its config event; falls back to discover-from-event)
        3. Poll relays for our config event by our own ``d`` tag
        4. Extract Neon URL; the Authority npub is the event's author

        ``retry_backoff`` defaults to the long detached-runner ladder
        (:data:`_BOOTSTRAP_RETRY_BACKOFF`). Callers that serve live agents
        should pass :data:`FRONT_BOOTSTRAP_RETRY_BACKOFF` (or let
        :func:`ensure_bootstrapped` do so).
        """
        from pynostr.key import PrivateKey as _PK  # type: ignore[import-untyped]
        from pynostr.key import PublicKey

        from tollbooth.bootstrap_relay import receive_bootstrap_config
        from tollbooth.oracle_client import OracleClientError, default_oracle_client

        ladder = retry_backoff if retry_backoff is not None else _BOOTSTRAP_RETRY_BACKOFF

        # Convert nsec to hex for vault encryption
        nsec = self._nsec_hex
        nsec_hex = _PK.from_nsec(nsec).hex() if nsec.startswith("nsec1") else nsec

        result = BootstrapResult(npub=self.npub, encryption_nsec_hex=nsec_hex)

        # Steps 1–2 are two independent questions for the Oracle, so they share
        # one connection and are asked at once; the relay read below waits on
        # both, since the Authority's hex is its spoof guard.
        #
        # Step 1: relay set. Injected relays win (tests / callers); otherwise the
        # Oracle is the one fixed anchor an nsec-only operator may know a priori.
        #
        # Step 2: who is our Authority? When the Oracle answers, we accept ONLY
        # that author's config event (spoof guard). When it can't (operator
        # unknown/new, or Oracle briefly unreachable), we proceed by our own
        # d-tag and discover the author from the event — a working Neon URL is
        # the backstop, and the author can be re-verified later.
        try:
            async with default_oracle_client().session() as oracle:
                relays, authority = await asyncio.gather(
                    self._relays_from(oracle),
                    oracle.resolve_authority_for(self.npub),
                    return_exceptions=True,
                )
        except OracleClientError as e:
            # No connection: the injected relays still stand; only the guard is lost.
            relays = self._relays if self._relays is not None else e
            authority = e

        if isinstance(relays, BaseException):
            result.error = f"Cannot reach Oracle for relay set: {relays}"
            logger.warning("Bootstrap: %s", result.error)
            return result

        expected_authority_hex: str | None = None
        if isinstance(authority, BaseException):
            logger.info(
                "Bootstrap: Oracle authority pre-resolve unavailable (%s); "
                "accepting config by operator d-tag, discovering author from event.",
                authority,
            )
        elif authority and authority.get("npub"):
            result.authority_npub = authority["npub"]
            expected_authority_hex = PublicKey.from_npub(authority["npub"]).hex()

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
        # Live fronts pass a short ladder via ensure_bootstrapped.
        #
        # The poll is synchronous websocket I/O, so it runs on a worker thread:
        # every other session on this process keeps moving while we read.
        config = author_hex = diag = None
        for attempt, pause in enumerate(ladder, start=1):
            config, author_hex, diag = await asyncio.to_thread(
                receive_bootstrap_config,
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
                    attempt, len(ladder), diag, pause,
                )
                await asyncio.sleep(pause)
        self._relay_diag = diag

        if config is None:
            logger.warning("Bootstrap relay diagnostics: %s", diag)
            # Tell the Oracle which relays refused us, so fleet ordering is
            # corrected by measurement instead of staying a curated guess. A
            # detached runner is the caller most likely to know: it cold-boots
            # and re-bootstraps on EVERY job, so it meets the weather as it is.
            #
            # Deliberately AFTER the retry loop, not inside it: a relay that
            # flaps for one pass and serves on the next is not news, and
            # reporting per attempt would multiply one outage into a burst.
            await self._report_unreachable_relays(relays, diag)
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

    async def _relays_from(self, oracle: OracleClient) -> list[str]:
        """The injected relays, else the Oracle's set — which then warms the
        process-wide cache so synchronous consumers (courier, profile, audit)
        reuse it instead of re-fetching."""
        if self._relays is not None:
            return self._relays
        from tollbooth.relay_registry import seed_relays

        relays = await oracle.get_relays()
        seed_relays(relays)
        return relays

    async def _report_unreachable_relays(
        self, relays: list[str], diag: str | None,
    ) -> None:
        """Tell the Oracle which curated relays refused us. Best-effort, always.

        Nothing here may raise. This runs on a path that has ALREADY failed, and a
        failed report must not turn a transient bootstrap miss into a crash — the
        caller still has a verdict to return.

        Only relays named in the diagnostics' ``errors=[...]`` are reported. A relay
        that connected and simply held no config for us is reachable, and saying
        otherwise would ask the Oracle to demote a healthy relay over our own empty
        mailbox.
        """
        if not diag or "errors=[" not in diag:
            return
        errors = diag.split("errors=[", 1)[1]
        failed = [
            r for r in relays
            if f"{r}:" in errors and r not in self._reported_relays
        ]
        if not failed:
            return

        try:
            from nostr_sdk import EventBuilder, Keys, Kind

            from tollbooth.oracle_client import default_oracle_client

            keys = Keys.parse(self._nsec_hex)
            npub = keys.public_key().to_bech32()
            oracle = default_oracle_client()
        except Exception as exc:  # noqa: BLE001 — reporting is never load-bearing
            logger.debug("Relay reporting unavailable (%s); skipping.", exc)
            return

        for relay in failed:
            # Marked before the attempt, not after: a report that throws should not
            # be retried on the next bootstrap of this same process either.
            self._reported_relays.add(relay)
            try:
                # Signed per relay, immediately before sending. The Oracle binds
                # the signature to a moment, and requires the content to name the
                # relay so one blob cannot be replayed against a different one.
                event = (
                    EventBuilder(Kind(1), f"DPYC-RELAY-UNREACHABLE: {relay}")
                    .finalize(keys)
                    .as_json()
                )
                answer = await oracle.report_relay_failure(
                    relay=relay,
                    reporter_npub=npub,
                    signed_event=event,
                    mode="read",
                )
                logger.info(
                    "Reported %s unreachable: accepted=%s probed=%s",
                    relay, answer.get("accepted"), answer.get("probed"),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not report %s to the Oracle: %s", relay, exc)
