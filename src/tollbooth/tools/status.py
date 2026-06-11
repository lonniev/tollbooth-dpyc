"""service_status response assembly (audit M2.1d).

Extracted from the ``service_status`` closure. The shim keeps the rt/infra
probing (vault + courier lazy-init, wheel version, operator npub, pid) because
those have side effects and runtime coupling; this pure assembler shapes the
result, parses Horizon build-info from the environment, and fingerprints the
operator npub — all testable without a runtime.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

_BUILD_INFO_KEYS = (
    "FASTMCP_CLOUD_URL",
    "FASTMCP_CLOUD_GIT_COMMIT_SHA",
    "FASTMCP_CLOUD_GIT_REPO",
)


def build_service_status(
    *,
    service: str,
    slug: str,
    version: str,
    tollbooth_version: str,
    vault_ok: bool,
    courier_ok: bool,
    operator_npub: str | None,
    process_id: int,
    env: Mapping[str, str],
) -> dict[str, Any]:
    """Assemble the service_status payload.

    ``operator_npub`` is ``None`` only when the runtime could not resolve it
    (the shim caught an exception); a resolved value — even an empty string —
    is fingerprinted, preserving the original closure's behavior.
    """
    build_info: dict[str, str] = {}
    for key in _BUILD_INFO_KEYS:
        val = env.get(key)
        if val:
            build_info[key.lower()] = val

    op_npub_hash = ""
    if operator_npub is not None:
        op_npub_hash = hashlib.sha256(operator_npub.encode()).hexdigest()[:12]

    return {
        "success": True,
        "service": service,
        "slug": slug,
        "version": version,
        "tollbooth_dpyc_version": tollbooth_version,
        "vault_configured": vault_ok,
        "courier_has_vault": courier_ok,
        "operator_npub_hash": op_npub_hash,
        "process_id": process_id,
        "build_info": build_info or None,
    }


def build_upstream_oauth_block(
    creds: Mapping[str, Any] | None,
    *,
    service_name: str,
    refresh_enabled: bool,
    now_ts: float,
) -> dict[str, Any] | None:
    """Build the ``session_status`` ``upstream_oauth`` block from stored creds.

    Returns ``None`` when there is no usable access token (the caller then
    omits the block). Expiry math is runtime-derived: ``expires_at`` is parsed
    defensively (stored as a string), and ``expires_in_seconds`` is clamped at
    zero. Extracted from the session_status closure (audit M2.1e) so the
    token-expiry arithmetic is testable without a runtime.
    """
    if not creds or not creds.get("access_token"):
        return None

    try:
        expires_at = float(creds.get("expires_at", "0") or 0)
    except (TypeError, ValueError):
        expires_at = 0.0

    block: dict[str, Any] = {
        "service": service_name,
        "has_access_token": True,
        "has_refresh_token": bool(creds.get("refresh_token")),
        "refresh_enabled": refresh_enabled,
    }
    if expires_at > 0:
        block["access_token_expires_at"] = expires_at
        block["access_token_expires_in_seconds"] = max(0, int(expires_at - now_ts))
    return block
