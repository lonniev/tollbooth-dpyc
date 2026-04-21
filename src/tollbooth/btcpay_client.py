"""Async HTTP client for BTCPay Server's Greenfield API."""

from __future__ import annotations

from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class BTCPayError(Exception):
    """Base exception for BTCPay operations."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class BTCPayAuthError(BTCPayError):
    """401/403 — authentication or authorization failure."""


class BTCPayNotFoundError(BTCPayError):
    """404 — resource not found."""


class BTCPayValidationError(BTCPayError):
    """422 — request validation failure."""


class BTCPayServerError(BTCPayError):
    """5xx — server-side error (retryable)."""


class BTCPayConnectionError(BTCPayError):
    """Network/DNS failure (retryable)."""


class BTCPayTimeoutError(BTCPayError):
    """Request timeout (retryable)."""


# ---------------------------------------------------------------------------
# Sats → BTC conversion
# ---------------------------------------------------------------------------

# Default ceiling: 1 BTC.  Any single payout above this is almost certainly
# a unit-mismatch bug (sats confused with BTC → 10^8× overpayment).
_SATS_CONVERSION_MAX_DEFAULT = 100_000_000


def sats_to_btc_string(sats: int, *, max_sats: int = _SATS_CONVERSION_MAX_DEFAULT) -> str:
    """Convert satoshis to an 8-decimal-place BTC string for the BTCPay API.

    Raises ValueError on negative values or values exceeding *max_sats*.
    """
    if sats < 0:
        raise ValueError(f"sats must be non-negative, got {sats}")
    if sats > max_sats:
        raise ValueError(
            f"sats ({sats:,}) exceeds ceiling ({max_sats:,})"
        )
    return f"{sats / 100_000_000:.8f}"


# ---------------------------------------------------------------------------
# Status code → exception mapping
# ---------------------------------------------------------------------------

_STATUS_MAP: dict[int, type[BTCPayError]] = {
    401: BTCPayAuthError,
    403: BTCPayAuthError,
    404: BTCPayNotFoundError,
    422: BTCPayValidationError,
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class BTCPayClient:
    """Async client for BTCPay Server Greenfield API v1.

    Constructor accepts explicit params — no env-var loading.
    Uses ``token`` auth header (not Bearer) per BTCPay convention.
    """

    def __init__(self, host: str, api_key: str, store_id: str) -> None:
        base_url = host.rstrip("/") + "/api/v1"
        self._store_id = store_id
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"token {api_key}"},
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=10.0, pool=5.0),
        )

    # -- internal request dispatcher -----------------------------------------

    async def _request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
    ) -> Any:
        """Send a request and map errors to the BTCPay exception hierarchy."""
        try:
            response = await self._client.request(method, endpoint, json=json_data)
        except httpx.ConnectError as exc:
            raise BTCPayConnectionError(str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise BTCPayTimeoutError(str(exc)) from exc

        if response.status_code >= 400:
            body = response.text
            exc_cls = _STATUS_MAP.get(response.status_code)
            if exc_cls is not None:
                raise exc_cls(body, status_code=response.status_code)
            if response.status_code >= 500:
                raise BTCPayServerError(body, status_code=response.status_code)
            raise BTCPayError(body, status_code=response.status_code)

        return response.json()

    # -- public API methods ---------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        """GET /health — server health status."""
        return await self._request("GET", "/health")

    async def get_store(self) -> dict[str, Any]:
        """GET /stores/{storeId} — store details."""
        return await self._request("GET", f"/stores/{self._store_id}")

    async def create_invoice(
        self,
        amount_sats: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST /stores/{storeId}/invoices — create a Lightning invoice."""
        payload: dict[str, Any] = {
            "amount": str(amount_sats),
            "currency": "SATS",
        }
        if metadata is not None:
            payload["metadata"] = metadata
        return await self._request(
            "POST", f"/stores/{self._store_id}/invoices", json_data=payload
        )

    async def get_invoice(self, invoice_id: str) -> dict[str, Any]:
        """GET /stores/{storeId}/invoices/{invoiceId} — invoice details."""
        return await self._request(
            "GET", f"/stores/{self._store_id}/invoices/{invoice_id}"
        )

    async def get_invoice_payment_methods(self, invoice_id: str) -> list[dict[str, Any]]:
        """GET /stores/{storeId}/invoices/{invoiceId}/payment-methods — payment method details.

        Returns a list of payment method dicts. The Lightning entry contains
        a ``destination`` field with the BOLT11 invoice string.
        """
        return await self._request(
            "GET", f"/stores/{self._store_id}/invoices/{invoice_id}/payment-methods"
        )

    async def get_lightning_invoice(
        self, invoice_id: str, *, retries: int = 3, delay: float = 0.5,
    ) -> str | None:
        """Extract the BOLT11 Lightning invoice string for a BTCPay invoice.

        BTCPay may not populate the Lightning ``destination`` immediately
        after invoice creation.  This method retries up to *retries* times
        with exponential backoff (*delay* doubles each attempt) to give
        the Lightning node time to generate the BOLT11.

        Returns the BOLT11 string (``lnbc...``) or None if no Lightning
        payment method is available after all attempts.
        """
        import asyncio

        for attempt in range(retries):
            try:
                methods = await self.get_invoice_payment_methods(invoice_id)
            except BTCPayError:
                return None
            for method in methods:
                pm_id = method.get("paymentMethodId", "") or method.get("paymentMethod", "")
                if "LN" in pm_id.upper() or "LIGHTNING" in pm_id.upper():
                    dest = method.get("destination", "")
                    if dest and dest.lower().startswith("lnbc"):
                        return dest
            if attempt < retries - 1:
                await asyncio.sleep(delay)
                delay *= 2
        return None

    async def get_api_key_info(self) -> dict[str, Any]:
        """GET /api-keys/current — current API key metadata and permissions."""
        return await self._request("GET", "/api-keys/current")

    async def create_payout(
        self,
        destination: str,
        amount_sats: int,
        payout_method: str = "BTC-LN",
    ) -> dict[str, Any]:
        """POST /stores/{storeId}/payouts — create a store payout.

        Amount is converted from sats to BTC decimal (BTCPay expects BTC).
        """
        amount_btc = sats_to_btc_string(amount_sats)
        payload: dict[str, Any] = {
            "destination": destination,
            "amount": amount_btc,
            "payoutMethodId": payout_method,
        }
        return await self._request(
            "POST", f"/stores/{self._store_id}/payouts", json_data=payload
        )

    async def get_payout_processors(self) -> list[dict[str, Any]]:
        """GET /stores/{storeId}/payout-processors — list configured payout processors."""
        return await self._request(
            "GET", f"/stores/{self._store_id}/payout-processors"
        )

    # -- lifecycle ------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> BTCPayClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
