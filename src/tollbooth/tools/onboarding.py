"""Vault credential loaders for the Secure Courier system.

Onboarding status is now template-driven via ``OperatorRuntime.onboarding_status()``
which checks credential templates against vault contents directly — no Settings
introspection needed.

This module provides the low-level vault access helpers used by OperatorRuntime.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generic vault credential loader
# ---------------------------------------------------------------------------


async def load_vault_blob(
    courier_service: Any,
    service: str,
    operator_npub: str,
) -> tuple[dict[str, Any] | None, str]:
    """Load the raw credential blob (including reserved ``__meta__``).

    Same ``(value, situation)`` contract as ``load_vault_credentials`` —
    ``({}, "")`` means the vault answered and holds nothing; ``(None, situation)``
    means we could not ask.  Use this when the caller needs delivery
    timestamps (``__meta__.delivered_at``); prefer ``load_vault_credentials``
    for plain field values.
    """
    from tollbooth.constants import ErrorCode
    from tollbooth.persistence_errors import classify_persistence_failure

    if courier_service is None:
        return None, ErrorCode.SECURE_COURIER_UNAVAILABLE

    vault = getattr(
        getattr(courier_service, "_exchange", None), "_credential_vault", None,
    )
    if vault is None:
        # The courier exists but its vault hasn't been attached yet —
        # ``OperatorRuntime.courier()`` retries this attachment on each call, so
        # it is a cold-start state, not an absence of credentials.
        return None, ErrorCode.VAULT_BOOTSTRAPPING

    try:
        blob = await vault.fetch_credentials(service, operator_npub)
    except Exception as exc:  # noqa: BLE001 — classified, never swallowed
        situation = classify_persistence_failure(exc)
        logger.warning(
            "Vault credential fetch failed for %s (%s): %s — reporting %s",
            service, type(exc).__name__, exc, situation,
        )
        return None, situation

    if blob is None:
        return {}, ""  # the vault answered: nothing is stored here

    try:
        plaintext = courier_service._exchange._vault_decrypt(blob)
        decoded = json.loads(plaintext)
        if not isinstance(decoded, dict):
            logger.warning(
                "Vault credential decode for %s produced non-dict; treating as empty",
                service,
            )
            return {}, ""
        return decoded, ""
    except Exception as exc:  # noqa: BLE001 — a stored blob we cannot read
        # Reaching here means the vault HELD something and we could not open it:
        # a wrong key or a corrupt record. Never a cold start, and never
        # "nothing stored" — waiting will not fix either one.
        logger.warning(
            "Vault credential decode failed for %s (%s): %s",
            service, type(exc).__name__, exc,
        )
        return None, ErrorCode.PERSISTENCE_MISCONFIGURED


async def load_vault_credentials(
    courier_service: Any,
    service: str,
    operator_npub: str,
) -> tuple[dict[str, str] | None, str]:
    """Load credentials from the Secure Courier vault for a given service.

    This is the generic helper that any operator can use.  It accesses
    the courier's credential vault, fetches the encrypted blob, decrypts
    it, and returns the credential dict.

    Returns ``(creds, situation)`` — the same discriminated shape
    ``OperatorRuntime.restore_oauth_session`` uses.  The distinction the whole
    stack rests on is:

    * ``({}, "")`` — **the vault answered and holds nothing for this key.**
      Only this may ever be read as "never onboarded".
    * ``(None, situation)`` — we could not ask.  Reporting this as "nothing
      stored" is what let a cold container tell a connected patron they had
      never authorized, and an out-of-quota database tell a funded patron they
      were broke.

    ``situation`` is an ``ErrorCode`` value the caller can hand straight to a
    situation table: ``secure_courier_unavailable``, ``vault_bootstrapping``,
    ``persistence_quota_exceeded``, or ``persistence_misconfigured``.

    Reserved bookkeeping (``__meta__`` / delivery timestamps) is stripped —
    values only.  Use ``load_vault_blob`` when you need the stamps.

    Args:
        courier_service: The operator's ``SecureCourierService`` instance.
        service: The credential template service name (e.g.,
            ``"tollbooth-sample-operator"``).
        operator_npub: The operator's npub (vault key).
    """
    from tollbooth.credential_meta import strip_meta

    result, situation = await load_vault_blob(courier_service, service, operator_npub)
    if situation or result is None:
        return None, situation
    return strip_meta(result), ""


# ---------------------------------------------------------------------------
# Generic vault-aware config loader
# ---------------------------------------------------------------------------


async def load_config_from_vault(
    courier_service: Any,
    service: str,
    operator_npub: str,
    field_names: list[str],
) -> tuple[dict[str, str], str]:
    """Load specific config fields from the Secure Courier vault.

    Returns ``(fields, situation)``.  ``fields`` maps field_name → value for the
    requested names present in the vault; names absent from the vault are simply
    omitted.  ``situation`` is ``""`` when the vault answered — so an empty dict
    with an empty situation means "asked, and the operator has delivered none of
    these", which is a genuine onboarding state rather than a fault.

    Use this to hydrate operator config at runtime::

        creds, situation = await load_config_from_vault(
            courier, "my-service", npub,
            ["api_key", "api_secret", "host"]
        )
        if situation:
            return warming_up_response(situation)
        client = MyClient(
            api_key=creds.get("api_key"),
            api_secret=creds.get("api_secret"),
            host=creds.get("host"),
        )
    """
    vault_creds, situation = await load_vault_credentials(
        courier_service, service, operator_npub,
    )
    if vault_creds is None:
        return {}, situation
    return {k: v for k, v in vault_creds.items() if k in field_names}, ""
