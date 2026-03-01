"""LNURL-pay resolution: Lightning address → BOLT11 invoice.

Implements LUD-06 (LNURL-pay) + LUD-16 (Lightning Address) using httpx.
No external dependencies beyond httpx (already a project dep).

Protocol:
  1. Parse ``user@domain`` → ``https://{domain}/.well-known/lnurlp/{user}``
  2. GET that URL → ``{callback, minSendable, maxSendable, ...}``
  3. GET ``{callback}?amount={millisats}`` → ``{pr: "lnbc..."}``
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class LnurlResolutionError(Exception):
    """Failed to resolve a Lightning address to a BOLT11 invoice."""


def _parse_lightning_address(address: str) -> tuple[str, str]:
    """Parse ``user@domain`` into (user, domain). Raises on invalid format."""
    if "@" not in address:
        raise LnurlResolutionError(
            f"Invalid Lightning address (missing @): {address!r}"
        )
    parts = address.split("@")
    if len(parts) != 2:
        raise LnurlResolutionError(
            f"Invalid Lightning address (multiple @): {address!r}"
        )
    user, domain = parts
    if not user:
        raise LnurlResolutionError(
            f"Invalid Lightning address (empty user): {address!r}"
        )
    if not domain:
        raise LnurlResolutionError(
            f"Invalid Lightning address (empty domain): {address!r}"
        )
    return user, domain


async def resolve_lightning_address(
    address: str,
    amount_sats: int,
    *,
    comment: str | None = None,
    timeout: float = 10.0,
) -> str:
    """Resolve a Lightning address to a BOLT11 invoice via LNURL-pay.

    Args:
        address: Lightning address in ``user@domain`` format.
        amount_sats: Amount in satoshis to request.
        comment: Optional comment to attach to the payment.
        timeout: HTTP timeout in seconds.

    Returns:
        BOLT11 invoice string (``lnbc...``).

    Raises:
        LnurlResolutionError: On any failure (parse, network, validation).
    """
    user, domain = _parse_lightning_address(address)
    well_known_url = f"https://{domain}/.well-known/lnurlp/{user}"
    amount_msats = amount_sats * 1000

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Step 1: Fetch LNURL-pay metadata
            resp = await client.get(well_known_url)
            if resp.status_code != 200:
                raise LnurlResolutionError(
                    f"LNURL endpoint returned HTTP {resp.status_code}: "
                    f"{well_known_url}"
                )

            data: dict[str, Any] = resp.json()

            if data.get("status") == "ERROR":
                reason = data.get("reason", "unknown error")
                raise LnurlResolutionError(
                    f"LNURL endpoint error: {reason}"
                )

            min_sendable = data.get("minSendable", 0)
            max_sendable = data.get("maxSendable", 0)
            callback = data.get("callback")

            if not callback:
                raise LnurlResolutionError(
                    "LNURL response missing 'callback' field"
                )

            # Validate amount against endpoint limits
            if amount_msats < min_sendable:
                raise LnurlResolutionError(
                    f"Amount {amount_sats} sats ({amount_msats} msats) below "
                    f"minimum {min_sendable} msats"
                )
            if max_sendable > 0 and amount_msats > max_sendable:
                raise LnurlResolutionError(
                    f"Amount {amount_sats} sats ({amount_msats} msats) above "
                    f"maximum {max_sendable} msats"
                )

            # Step 2: Request invoice from callback
            params: dict[str, Any] = {"amount": amount_msats}
            if comment:
                params["comment"] = comment

            cb_resp = await client.get(callback, params=params)
            if cb_resp.status_code != 200:
                raise LnurlResolutionError(
                    f"LNURL callback returned HTTP {cb_resp.status_code}: "
                    f"{callback}"
                )

            cb_data: dict[str, Any] = cb_resp.json()

            if cb_data.get("status") == "ERROR":
                reason = cb_data.get("reason", "unknown error")
                raise LnurlResolutionError(
                    f"LNURL callback error: {reason}"
                )

            invoice = cb_data.get("pr")
            if not invoice:
                raise LnurlResolutionError(
                    "LNURL callback response missing 'pr' (invoice) field"
                )

            return invoice

    except LnurlResolutionError:
        raise
    except httpx.HTTPError as exc:
        raise LnurlResolutionError(
            f"Network error resolving {address}: {exc}"
        ) from exc
    except Exception as exc:
        raise LnurlResolutionError(
            f"Unexpected error resolving {address}: {exc}"
        ) from exc
