"""x402 upstream adapter — transparent 402 payment for Operator COGS.

Wraps httpx to intercept HTTP 402 responses from x402-protected upstreams,
sign payment authorizations with the Operator's agentic wallet, and retry.
Patrons never see the 402 handshake.

402 fees are Operator COGS — like server rental. No refunds, no rebates,
no patron visibility. The Operator prices their tools to cover upstream
costs with margin.

Usage::

    from tollbooth.x402_client import X402Client

    creds = await runtime.load_credentials(["wallet_private_key", "wallet_address"])
    x402 = X402Client(
        wallet_private_key=creds["wallet_private_key"],
        wallet_address=creds["wallet_address"],
    )

    # In any tool handler — 402 is invisible to the patron
    resp = await x402.get("https://upstream-x402-api.com/data")

Dependencies are optional — install with ``pip install tollbooth-dpyc[x402]``.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Self

import httpx

logger = logging.getLogger(__name__)

# Header names per x402 HTTP transport spec
_HEADER_PAYMENT_REQUIRED = "payment-required"
_HEADER_PAYMENT_SIGNATURE = "X-PAYMENT"

# Optional imports — graceful degradation
try:
    from x402 import (
        PaymentPayload,
        PaymentRequired,
        PaymentRequirements,
        parse_payment_required,
    )

    _HAS_X402 = True
except ImportError:
    _HAS_X402 = False

try:
    from eth_account import Account
    from eth_account.messages import encode_typed_data

    _HAS_ETH_ACCOUNT = True
except ImportError:
    _HAS_ETH_ACCOUNT = False


def _check_deps() -> None:
    """Raise ImportError with install hint if optional deps are missing."""
    missing: list[str] = []
    if not _HAS_X402:
        missing.append("x402")
    if not _HAS_ETH_ACCOUNT:
        missing.append("eth-account")
    if missing:
        raise ImportError(
            f"x402 adapter requires: {', '.join(missing)}. "
            "Install with: pip install 'tollbooth-dpyc[x402]'"
        )


# EIP-712 typed data for exact-evm scheme (USDC Permit2 authorization)
_EIP712_DOMAIN = {
    "name": "x402",
    "version": "2",
}

_EIP712_TYPES = {
    "TransferWithAuthorization": [
        {"name": "from", "type": "address"},
        {"name": "to", "type": "address"},
        {"name": "value", "type": "uint256"},
        {"name": "validAfter", "type": "uint256"},
        {"name": "validBefore", "type": "uint256"},
        {"name": "nonce", "type": "bytes32"},
    ],
}


class X402Client:
    """HTTP client that transparently handles x402 payment challenges.

    The Operator provides agentic wallet credentials via Secure Courier.
    When an upstream returns 402 with a ``payment-required`` header, this
    client signs the payment authorization and retries automatically.

    402 fees are Operator COGS — no refund, no rebate, no patron visibility.
    """

    def __init__(
        self,
        wallet_private_key: str,
        wallet_address: str,
        facilitator_url: str = "https://x402.org/facilitator",
        *,
        max_retries: int = 1,
        timeout: float = 30.0,
    ) -> None:
        _check_deps()
        self._private_key = (
            wallet_private_key
            if wallet_private_key.startswith("0x")
            else f"0x{wallet_private_key}"
        )
        self._address = wallet_address
        self._facilitator_url = facilitator_url
        self._max_retries = max_retries
        self._http = httpx.AsyncClient(timeout=timeout)

    async def request(
        self, method: str, url: str, **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request, handling 402 transparently.

        Non-402 responses (200, 300, 400, 500) pass through untouched.
        A 402 with a ``payment-required`` header triggers the x402 payment
        dance: parse requirements, sign authorization, retry with payment.
        """
        resp = await self._http.request(method, url, **kwargs)

        for _ in range(self._max_retries):
            if resp.status_code != 402:
                return resp

            payment_header = resp.headers.get(_HEADER_PAYMENT_REQUIRED)
            if not payment_header:
                return resp  # 402 but not x402 — return as-is

            try:
                payload = self._sign_payment(payment_header, url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("x402 payment signing failed: %s", exc)
                return resp  # can't pay — return the 402

            # Retry with payment
            headers = dict(kwargs.pop("headers", {}) or {})
            headers[_HEADER_PAYMENT_SIGNATURE] = payload
            resp = await self._http.request(
                method, url, headers=headers, **kwargs,
            )

        return resp

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        """GET with transparent x402 handling."""
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        """POST with transparent x402 handling."""
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> httpx.Response:
        """PUT with transparent x402 handling."""
        return await self.request("PUT", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        """DELETE with transparent x402 handling."""
        return await self.request("DELETE", url, **kwargs)

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- Private ---------------------------------------------------------------

    def _sign_payment(self, payment_header: str, url: str) -> str:
        """Parse payment requirements, sign, and return base64 payment payload."""
        raw = base64.b64decode(payment_header)
        pr_dict = json.loads(raw)
        payment_required = parse_payment_required(pr_dict)
        # parse_payment_required returns a V1|V2 union; this client implements
        # the current (V2) protocol only. A V1 response is an upstream-version
        # situation, not a crash — narrow explicitly (also satisfies mypy).
        if not isinstance(payment_required, PaymentRequired):
            raise TypeError(
                "Unsupported x402 payment-required version (expected V2). "
                "Upgrade the upstream x402 service."
            )

        if not payment_required.accepts:
            raise ValueError("No payment options in 402 response")

        # Select the first acceptable payment requirement
        selected = payment_required.accepts[0]
        if not isinstance(selected, PaymentRequirements):
            raise TypeError(
                "Unsupported x402 payment-requirements version (expected V2)."
            )

        import os
        import time

        now = int(time.time())
        nonce = "0x" + os.urandom(32).hex()

        # Build EIP-712 authorization
        authorization = {
            "from": self._address,
            "to": selected.pay_to,
            "value": int(selected.amount),
            "validAfter": 0,
            "validBefore": now + selected.max_timeout_seconds,
            "nonce": nonce,
        }

        # Sign via EIP-712 typed data
        chain_id = _parse_chain_id(selected.network)
        domain = {**_EIP712_DOMAIN, "chainId": chain_id}

        signable = encode_typed_data(
            domain_data=domain,
            message_types={"TransferWithAuthorization": _EIP712_TYPES["TransferWithAuthorization"]},
            message_data=authorization,
        )
        signed = Account.sign_message(signable, self._private_key)

        # Build payment payload
        payload = PaymentPayload(
            x402_version=payment_required.x402_version,
            accepted=selected,
            resource=payment_required.resource,
            payload={
                "signature": signed.signature.hex(),
                "authorization": {
                    "from": authorization["from"],
                    "to": authorization["to"],
                    "value": str(authorization["value"]),
                    "validAfter": str(authorization["validAfter"]),
                    "validBefore": str(authorization["validBefore"]),
                    "nonce": nonce,
                },
            },
        )

        payload_json = payload.model_dump_json(by_alias=True)
        return base64.b64encode(payload_json.encode()).decode()


def _parse_chain_id(network: str) -> int:
    """Extract chain ID from CAIP-2 network identifier (e.g. 'eip155:8453' → 8453)."""
    if ":" in network:
        return int(network.split(":")[-1])
    # Common defaults
    _KNOWN = {"base-sepolia": 84532, "base": 8453, "ethereum": 1}
    return _KNOWN.get(network.lower(), 8453)


# Credential template for Secure Courier delivery
def x402_wallet_template() -> Any:
    """Return a CredentialTemplate for x402 agentic wallet credentials.

    Operators deliver these via Secure Courier, same as any other
    service credentials (Schwab API keys, X API creds, etc.).
    """
    from tollbooth.credential_templates import CredentialTemplate, FieldSpec

    return CredentialTemplate(
        service="x402-wallet",
        version=1,
        fields={
            "wallet_private_key": FieldSpec(
                required=True,
                sensitive=True,
                description="Ethereum private key (hex) for the agentic wallet",
            ),
            "wallet_address": FieldSpec(
                required=True,
                sensitive=False,
                description="Ethereum address (0x...) of the agentic wallet",
            ),
            "facilitator_url": FieldSpec(
                required=False,
                sensitive=False,
                description="x402 facilitator URL (default: https://x402.org/facilitator)",
                default="https://x402.org/facilitator",
            ),
        },
        description="Coinbase x402 agentic wallet for upstream API payments",
    )
