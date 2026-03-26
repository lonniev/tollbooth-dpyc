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


def get_onboarding_status_for(settings: BaseSettings) -> dict[str, Any]:
    """Introspect a Settings instance and return onboarding status.

    Returns a dict suitable for JSON serialization with:
    - ``ready``: True if all required fields are configured
    - ``configured``: list of field names that have values
    - ``missing``: list of dicts with field name, category, and hint
    - ``summary``: human-readable status string
    """
    configured: list[dict[str, str]] = []
    missing: list[dict[str, str]] = []

    for name, field_info in settings.model_fields.items():
        category = classify_field(name)
        value = getattr(settings, name, None)

        # Fields with non-None defaults are tuning — skip unless None
        has_value = value is not None
        if has_value and isinstance(value, str):
            has_value = len(value.strip()) > 0

        entry = {
            "field": name,
            "category": category,
        }

        if has_value:
            # Don't leak secret values — just confirm configured
            if category == "secret":
                entry["status"] = "configured"
            elif category == "identity":
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
        missing_names = [m["field"] for m in missing]
        authority_missing = [m for m in missing if m["category"] == "authority"]
        secret_missing = [m for m in missing if m["category"] == "secret"]
        parts = []
        if authority_missing:
            parts.append(
                f"{len(authority_missing)} authority-provisioned "
                f"({'value' if len(authority_missing) == 1 else 'values'}) "
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
# Singleton settings reference — operators register theirs at startup
# ---------------------------------------------------------------------------

_operator_settings: BaseSettings | None = None


def register_operator_settings(settings: BaseSettings) -> None:
    """Register the operator's Settings instance for onboarding introspection."""
    global _operator_settings
    _operator_settings = settings


def get_registered_settings() -> BaseSettings | None:
    """Return the registered operator settings, or None."""
    return _operator_settings
