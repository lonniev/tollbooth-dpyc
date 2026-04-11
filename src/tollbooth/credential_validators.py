"""Credential validation helpers for operator callbacks.

Each operator provides a credential_validator callback to OperatorRuntime.
The callback receives the credential dict and returns a list of error strings
(empty list = valid).

Common validators are provided here for reuse across operators.
"""


def validate_url(value: str, field_name: str) -> str | None:
    """Validate that a field looks like an HTTPS URL."""
    v = (value or "").strip()
    if not v:
        return f"{field_name} is missing."
    if not v.startswith("https://"):
        return f"{field_name} must start with 'https://' (got '{v[:30]}...'). Check for typos."
    return None


def validate_required(value: str, field_name: str) -> str | None:
    """Validate that a field is present and non-empty."""
    if not (value or "").strip():
        return f"{field_name} is missing or empty."
    return None


def validate_btcpay_creds(creds: dict[str, str]) -> list[str]:
    """Validate the standard BTCPay credential trio."""
    errors: list[str] = []
    for check in [
        validate_url(creds.get("btcpay_host", ""), "btcpay_host"),
        validate_required(creds.get("btcpay_api_key", ""), "btcpay_api_key"),
        validate_required(creds.get("btcpay_store_id", ""), "btcpay_store_id"),
    ]:
        if check:
            errors.append(check)
    return errors
