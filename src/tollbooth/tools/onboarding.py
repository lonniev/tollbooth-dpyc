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


async def load_vault_credentials(
    courier_service: Any,
    service: str,
    operator_npub: str,
) -> dict[str, str] | None:
    """Load credentials from the Secure Courier vault for a given service.

    This is the generic helper that any operator can use.  It accesses
    the courier's credential vault, fetches the encrypted blob, decrypts
    it, and returns the credential dict.

    Args:
        courier_service: The operator's ``SecureCourierService`` instance.
        service: The credential template service name (e.g.,
            ``"tollbooth-sample-operator"``).
        operator_npub: The operator's npub (vault key).

    Returns:
        Dict of credential field names to values, or None if not found.
    """
    if courier_service is None:
        return None
    try:
        vault = courier_service._exchange._credential_vault
        if vault is None:
            return None
        blob = await vault.fetch_credentials(service, operator_npub)
        if blob is None:
            return None
        plaintext = courier_service._exchange._vault_decrypt(blob)
        return json.loads(plaintext)
    except Exception as exc:
        logger.debug("Vault credential load failed for %s: %s", service, exc)
        return None


# ---------------------------------------------------------------------------
# Generic vault-aware config loader
# ---------------------------------------------------------------------------


async def load_config_from_vault(
    courier_service: Any,
    service: str,
    operator_npub: str,
    field_names: list[str],
) -> dict[str, str]:
    """Load specific config fields from the Secure Courier vault.

    Returns a dict of field_name → value for fields that are present
    in the vault.  Missing fields are omitted from the result.

    Use this to hydrate operator config at runtime::

        creds = await load_config_from_vault(
            courier, "my-service", npub,
            ["api_key", "api_secret", "host"]
        )
        client = MyClient(
            api_key=creds.get("api_key"),
            api_secret=creds.get("api_secret"),
            host=creds.get("host"),
        )
    """
    vault_creds = await load_vault_credentials(
        courier_service, service, operator_npub,
    )
    if vault_creds is None:
        return {}
    return {k: v for k, v in vault_creds.items() if k in field_names}
