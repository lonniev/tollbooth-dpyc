"""Onboarding status tool — reports operator configuration readiness.

Every Tollbooth operator has a Pydantic ``BaseSettings`` class that
declares its required configuration.  This tool introspects those
fields to report what's configured, what's missing, and how to
deliver missing values.

Config fields are classified into three provisioning categories:

- **authority**: Provisioned automatically by the Authority during
  registration (e.g., ``neon_database_url``).  The operator's
  ``BootstrapClient`` fetches these on first boot.

- **secret**: Operator-specific credentials that must be delivered
  via Secure Courier DM (e.g., BTCPay keys, API tokens).

- **identity**: The operator's Nostr nsec — set once at deployment.

- **tuning**: Non-secret operational parameters with sensible
  defaults (e.g., ``seed_balance_sats``, ``credit_ttl_seconds``).
  These don't block onboarding.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Fields provisioned by the Authority (auto-filled via BootstrapClient)
AUTHORITY_PROVISIONED = {"neon_database_url"}

# Identity fields — set once at deployment time
IDENTITY_FIELDS = {"tollbooth_nostr_operator_nsec"}

# Fields with sensible defaults that don't block onboarding
TUNING_FIELDS = {
    "seed_balance_sats",
    "credit_ttl_seconds",
    "constraints_enabled",
    "constraints_config",
}

# Everything else with a None default is a secret that needs Secure Courier


def classify_field(name: str) -> str:
    """Return the provisioning category for a settings field."""
    if name in AUTHORITY_PROVISIONED:
        return "authority"
    if name in IDENTITY_FIELDS:
        return "identity"
    if name in TUNING_FIELDS:
        return "tuning"
    return "secret"


def get_onboarding_status_for(
    settings: BaseSettings,
    vault_credentials: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Introspect a Settings instance and return onboarding status.

    Args:
        settings: The operator's Pydantic BaseSettings instance.
        vault_credentials: Optional dict of credentials loaded from
            the Secure Courier vault.  Fields present here are treated
            as configured even if the corresponding env var is absent.

    Returns a dict suitable for JSON serialization with:
    - ``ready``: True if all required fields are configured
    - ``configured``: list of field names that have values
    - ``missing``: list of dicts with field name, category, and hint
    - ``summary``: human-readable status string
    """
    vault = vault_credentials or {}
    configured: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for name, field_info in settings.model_fields.items():
        category = classify_field(name)
        value = getattr(settings, name, None)

        # Check env var first, then vault
        has_value = value is not None
        if has_value and isinstance(value, str):
            has_value = len(value.strip()) > 0
        if not has_value and category == "secret" and name in vault:
            has_value = True

        entry = {
            "field": name,
            "category": category,
        }

        if has_value:
            # Don't leak secret values — just confirm configured
            if category in ("secret", "identity"):
                entry["status"] = "configured"
            else:
                entry["value"] = str(value) if category == "tuning" else "provisioned"
                entry["status"] = "configured"
            configured.append(entry)
        else:
            if category == "tuning":
                # Tuning fields with defaults don't block onboarding
                continue
            entry["status"] = "missing"
            if category == "authority":
                entry["how"] = (
                    "Auto-provisioned by Authority during registration. "
                    "Call get_operator_config or restart the operator to fetch."
                )
            elif category == "secret":
                entry["how"] = (
                    "Deliver via Secure Courier: call request_credential_channel "
                    "then have the operator owner reply with the value via "
                    "encrypted Nostr DM."
                )
            elif category == "identity":
                entry["how"] = (
                    "Set TOLLBOOTH_NOSTR_OPERATOR_NSEC in the deployment "
                    "environment. This is the operator's Nostr private key."
                )
            missing.append(entry)

    # Ready if no non-tuning fields are missing
    ready = len(missing) == 0

    if ready:
        summary = "Operator is fully configured and ready to serve."
    else:
        authority_missing = [m for m in missing if m["category"] == "authority"]
        secret_missing = [m for m in missing if m["category"] == "secret"]
        parts = []
        if authority_missing:
            parts.append(
                f"{len(authority_missing)} authority-provisioned "
                f"{'value' if len(authority_missing) == 1 else 'values'} "
                "pending — restart operator or call get_operator_config"
            )
        if secret_missing:
            names = ", ".join(m["field"] for m in secret_missing)
            parts.append(
                f"{len(secret_missing)} secret{'s' if len(secret_missing) != 1 else ''} "
                f"needed via Secure Courier: {names}"
            )
        summary = "Not ready. " + "; ".join(parts) + "."

    return {
        "ready": ready,
        "configured": configured,
        "missing": missing,
        "summary": summary,
    }


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


async def get_onboarding_status_with_vault(
    settings: BaseSettings,
    courier_service: Any,
    service: str,
    operator_npub: str,
) -> dict[str, Any]:
    """Convenience: run ``get_onboarding_status_for`` with vault check.

    Loads credentials from the Secure Courier vault and passes them
    to ``get_onboarding_status_for`` so that vault-delivered secrets
    show as configured.
    """
    vault_creds = await load_vault_credentials(
        courier_service, service, operator_npub,
    )
    return get_onboarding_status_for(settings, vault_credentials=vault_creds)


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
