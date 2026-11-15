"""Nostr Wallet Connect (NIP-47): ask a person's own wallet to pay an invoice.

A patron's wallet issues a connection string. It names the wallet service, the
relays it listens on, and a secret made for that one connection. Whoever holds
the string can ask that wallet to pay, inside whatever budget the patron set on
it in the wallet. The money never rests with the caller.

    conn = parse_connection_uri(uri)          # uri comes out of the vault
    info = await get_info(conn)
    paid = await pay_invoice(conn, invoice, amount_msat=21_000)
    seen = await lookup_invoice(conn, payment_hash=paid.payment_hash)

Four outcomes, and a caller must tell them apart:

- It returns: the wallet paid, and the preimage it returned matches the invoice.
- :class:`NwcError`: the wallet answered no. ``code`` is the wallet's own.
- :class:`NwcNotSent`: no relay took the request. Nothing was asked of the
  wallet, so asking again is safe.
- :class:`NwcOutcomeUnknown`: the request went out and no usable answer came
  back. The payment may have happened. Call :func:`lookup_invoice` with the
  payment hash before paying anything again.

The connection secret is a spending credential. It lives in the vault, it is
left out of ``repr``, and no message or log line here carries it.

Spec: https://github.com/nostr-protocol/nips/blob/master/47.md
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qs, urlsplit

from tollbooth import bolt11

logger = logging.getLogger(__name__)

__all__ = [
    "InvoiceStatus",
    "NwcAmountMismatch",
    "NwcConnection",
    "NwcError",
    "NwcNotSent",
    "NwcOutcomeUnknown",
    "Payment",
    "WalletInfo",
    "get_info",
    "lookup_invoice",
    "parse_connection_uri",
    "pay_invoice",
]

KIND_INFO = 13194
KIND_REQUEST = 23194
KIND_RESPONSE = 23195

_SCHEMES = ("nostr+walletconnect", "nostrwalletconnect")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RELAYS = 5
_MAX_URI_LENGTH = 2048
# Connecting, and reading the wallet's info event, each get this long.
_SETUP_SECONDS = 8.0
# A wallet's own words are shown to people; keep a hostile one short.
_MAX_WALLET_MESSAGE = 300


class NwcError(Exception):
    """The wallet answered, and the answer was no."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class NwcNotSent(Exception):
    """No relay took the request, so the wallet was never asked."""


class NwcOutcomeUnknown(Exception):
    """The request went out and no usable answer came back.

    A payment may have happened. Look the invoice up before paying again.
    """


class NwcAmountMismatch(ValueError):
    """The invoice does not ask for exactly the amount the caller agreed to."""


@dataclass(frozen=True, slots=True)
class NwcConnection:
    """One wallet connection, as its connection string describes it."""

    wallet_pubkey: str
    relays: tuple[str, ...]
    secret: str = field(repr=False)
    lud16: str | None = None

    @property
    def client_pubkey(self) -> str:
        """The public half of the connection secret; the wallet answers to it."""
        from pynostr.key import PrivateKey  # type: ignore[import-untyped]

        return str(PrivateKey(bytes.fromhex(self.secret)).public_key.hex())


@dataclass(frozen=True, slots=True)
class WalletInfo:
    """What a wallet says about itself and this connection."""

    alias: str | None
    network: str | None
    methods: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Payment:
    """A payment the wallet made, with the preimage that proves it."""

    payment_hash: str
    preimage: str
    fees_paid_msat: int | None


@dataclass(frozen=True, slots=True)
class InvoiceStatus:
    """What the wallet knows about one invoice or payment."""

    payment_hash: str
    state: str | None
    settled: bool
    preimage: str | None
    fees_paid_msat: int | None
    settled_at: int | None


class _RelayRefused(Exception):
    """This relay did not carry the request; another relay may."""


# ---------------------------------------------------------------------------
# Connection string
# ---------------------------------------------------------------------------


def _checked_relay(url: str) -> str:
    """A relay URL this server is willing to dial, or ``ValueError``.

    The connection string is typed in by a patron, so its relay is an address
    an outsider chose for this server to connect to.
    """
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.scheme != "wss" or not host:
        raise ValueError("a wallet relay must be a wss:// address")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("a wallet relay must be a public address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url
    if not address.is_global:
        raise ValueError("a wallet relay must be a public address")
    return url


def parse_connection_uri(uri: str) -> NwcConnection:
    """Read a ``nostr+walletconnect://`` connection string.

    Raises:
        ValueError: the string is not a usable connection. The message never
            quotes the string, which carries the secret.
    """
    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("wallet connection string is empty")
    text = uri.strip()
    if len(text) > _MAX_URI_LENGTH:
        raise ValueError("wallet connection string is too long")

    scheme, sep, rest = text.partition(":")
    if not sep or scheme.lower() not in _SCHEMES:
        raise ValueError("wallet connection string must begin nostr+walletconnect://")
    body, _, query = rest.removeprefix("//").partition("?")
    wallet_pubkey = body.strip("/").lower()
    if not _HEX64.match(wallet_pubkey):
        raise ValueError("wallet connection string names no wallet public key")

    params = parse_qs(query, keep_blank_values=False)
    secrets = params.get("secret", [])
    if len(secrets) != 1 or not _HEX64.match(secrets[0].lower()):
        raise ValueError("wallet connection string carries no usable secret")

    relays = tuple(dict.fromkeys(_checked_relay(r.strip()) for r in params.get("relay", [])))
    if not relays:
        raise ValueError("wallet connection string names no relay")
    if len(relays) > _MAX_RELAYS:
        raise ValueError(f"wallet connection string names more than {_MAX_RELAYS} relays")

    lud16s = params.get("lud16", [])
    lud16 = lud16s[0] if lud16s else None
    return NwcConnection(
        wallet_pubkey=wallet_pubkey,
        relays=relays,
        secret=secrets[0].lower(),
        lud16=lud16,
    )


# ---------------------------------------------------------------------------
# One request, one relay
# ---------------------------------------------------------------------------


def _verified(event: Any, *, kind: int, author: str) -> bool:
    """True when a relay-served event is this kind, by this author, and signed.

    A relay need not honor a filter, so each clause is checked here.
    """
    from pynostr.event import Event  # type: ignore[import-untyped]

    if not isinstance(event, dict) or event.get("kind") != kind or event.get("pubkey") != author:
        return False
    try:
        return bool(Event.from_dict(event).verify())
    except Exception:  # noqa: BLE001 — a malformed event is simply not ours
        return False


def _tag_values(event: dict[str, Any], name: str) -> list[str]:
    return [
        t[1]
        for t in event.get("tags", [])
        if isinstance(t, list) and len(t) > 1 and t[0] == name and isinstance(t[1], str)
    ]


def _recv(ws: Any, deadline: float) -> list[Any] | None:
    """The next relay message, or ``None`` when one cannot be parsed."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("relay did not answer in time")
    ws.settimeout(remaining)
    try:
        msg = json.loads(ws.recv())
    except (json.JSONDecodeError, TypeError):
        return None
    return msg if isinstance(msg, list) and msg else None


def _read_info(ws: Any, conn: NwcConnection) -> dict[str, Any]:
    """The wallet's newest signed info event on this relay."""
    sub = f"nwc-info-{uuid.uuid4().hex[:12]}"
    deadline = time.monotonic() + _SETUP_SECONDS
    ws.send(json.dumps(
        ["REQ", sub, {"kinds": [KIND_INFO], "authors": [conn.wallet_pubkey], "limit": 1}]
    ))
    newest: dict[str, Any] | None = None
    while True:
        msg = _recv(ws, deadline)
        if msg is None:
            continue
        if msg[0] in ("EOSE", "CLOSED") and len(msg) > 1 and msg[1] == sub:
            break
        if msg[0] == "EVENT" and len(msg) >= 3 and msg[1] == sub:
            event = msg[2]
            if _verified(event, kind=KIND_INFO, author=conn.wallet_pubkey) and (
                newest is None or event.get("created_at", 0) > newest.get("created_at", 0)
            ):
                newest = event
    ws.send(json.dumps(["CLOSE", sub]))
    if newest is None:
        raise _RelayRefused("the wallet has published no info event here")
    return newest


def _encryption_for(info: dict[str, Any]) -> str:
    """The scheme to speak, read from the wallet's info event (NIP-47 Encryption)."""
    tags = _tag_values(info, "encryption")
    if not tags:
        return "nip04"
    offered = {scheme for value in tags for scheme in value.split()}
    if "nip44_v2" in offered:
        return "nip44_v2"
    if "nip04" in offered:
        return "nip04"
    raise NwcError("UNSUPPORTED_ENCRYPTION", "the wallet offers no encryption this client speaks")


def _seal(scheme: str, plaintext: str, conn: NwcConnection) -> str:
    if scheme == "nip44_v2":
        from tollbooth.nip44 import encrypt as nip44_encrypt

        return nip44_encrypt(plaintext, conn.secret, conn.wallet_pubkey)
    from tollbooth.nip04 import encrypt as nip04_encrypt

    return nip04_encrypt(plaintext, conn.secret, conn.wallet_pubkey)


def _open(scheme: str, ciphertext: str, conn: NwcConnection) -> str:
    if scheme == "nip44_v2":
        from tollbooth.nip44 import decrypt as nip44_decrypt

        return nip44_decrypt(ciphertext, conn.secret, conn.wallet_pubkey)
    from tollbooth.nip04 import decrypt as nip04_decrypt

    return nip04_decrypt(ciphertext, conn.secret, conn.wallet_pubkey)


def _result_of(content: str, scheme: str, conn: NwcConnection, method: str) -> dict[str, Any]:
    """Open the wallet's signed answer and return its ``result``."""
    try:
        answer = json.loads(_open(scheme, content, conn))
    except Exception as exc:
        raise NwcOutcomeUnknown("the wallet answered and its answer could not be read") from exc
    if not isinstance(answer, dict):
        raise NwcOutcomeUnknown("the wallet's answer was not an object")

    error = answer.get("error")
    if error:
        code = str(error.get("code", "OTHER")) if isinstance(error, dict) else "OTHER"
        text = str(error.get("message", "")) if isinstance(error, dict) else str(error)
        raise NwcError(code[:40], text[:_MAX_WALLET_MESSAGE])

    result = answer.get("result")
    if answer.get("result_type") != method or not isinstance(result, dict):
        raise NwcOutcomeUnknown("the wallet answered a different question than it was asked")
    return result


def _ask_on(
    ws: Any, conn: NwcConnection, method: str, params: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Ask the wallet one thing over one open relay socket.

    Raises :class:`_RelayRefused` only while nothing has been sent to the
    wallet. From the moment the request event is written, any failure is
    :class:`NwcOutcomeUnknown`.
    """
    from pynostr.event import Event  # type: ignore[import-untyped]

    info = _read_info(ws, conn)
    if method not in info.get("content", "").split():
        raise NwcError("NOT_IMPLEMENTED", f"this wallet connection does not allow {method}")
    scheme = _encryption_for(info)

    now = int(time.time())
    request = Event(
        pubkey=conn.client_pubkey,
        kind=KIND_REQUEST,
        content=_seal(scheme, json.dumps({"method": method, "params": params}), conn),
        created_at=now,
        tags=[
            ["p", conn.wallet_pubkey],
            ["encryption", scheme],
            # A request the relay delivers late is one the wallet ignores.
            ["expiration", str(now + int(timeout))],
        ],
    )
    request.sign(conn.secret)
    request_id = str(request.id)

    # Listen before speaking: answers are ephemeral and are never replayed.
    sub = f"nwc-{uuid.uuid4().hex[:12]}"
    ws.send(json.dumps([
        "REQ", sub,
        {"kinds": [KIND_RESPONSE], "authors": [conn.wallet_pubkey], "#e": [request_id]},
    ]))
    setup_deadline = time.monotonic() + _SETUP_SECONDS
    while True:
        msg = _recv(ws, setup_deadline)
        if msg is None or len(msg) < 2 or msg[1] != sub:
            continue
        if msg[0] == "EOSE":
            break
        if msg[0] == "CLOSED":
            raise _RelayRefused("the relay refused the subscription")

    deadline = time.monotonic() + timeout
    try:
        ws.send(json.dumps(["EVENT", request.to_dict()]))
    except Exception as exc:
        raise NwcOutcomeUnknown("the connection dropped while the request was being sent") from exc

    try:
        while True:
            msg = _recv(ws, deadline)
            if msg is None:
                continue
            if msg[0] == "OK" and len(msg) >= 3 and msg[1] == request_id and msg[2] is False:
                # The relay turned the event away, so the wallet never saw it.
                raise _RelayRefused("the relay rejected the request")
            if msg[0] != "EVENT" or len(msg) < 3 or msg[1] != sub:
                continue
            event = msg[2]
            if not _verified(event, kind=KIND_RESPONSE, author=conn.wallet_pubkey):
                continue
            if request_id not in _tag_values(event, "e"):
                continue
            return _result_of(str(event.get("content", "")), scheme, conn, method)
    except (_RelayRefused, NwcError, NwcOutcomeUnknown):
        raise
    except Exception as exc:
        raise NwcOutcomeUnknown(f"no answer from the wallet to {method}") from exc


def _ask(
    conn: NwcConnection, method: str, params: dict[str, Any], timeout: float
) -> dict[str, Any]:
    """Ask the wallet, trying each relay in turn until one carries the request."""
    import websocket  # type: ignore[import-untyped]

    refusals: list[str] = []
    for relay in conn.relays:
        host = urlsplit(relay).hostname
        try:
            ws = websocket.create_connection(relay, timeout=_SETUP_SECONDS)
        except Exception as exc:  # noqa: BLE001
            refusals.append(f"{host}: {type(exc).__name__}")
            continue
        try:
            result = _ask_on(ws, conn, method, params, timeout)
            logger.info("NWC %s answered via %s", method, host)
            return result
        except _RelayRefused as exc:
            refusals.append(f"{host}: {exc}")
        except (NwcError, NwcOutcomeUnknown):
            raise
        except Exception as exc:  # noqa: BLE001 — setup failed before anything was sent
            refusals.append(f"{host}: {type(exc).__name__}")
        finally:
            with contextlib.suppress(Exception):
                ws.close()
    raise NwcNotSent(f"no relay carried {method} to the wallet ({'; '.join(refusals)})")


# ---------------------------------------------------------------------------
# The three calls
# ---------------------------------------------------------------------------


def _msat(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _preimage_for(payment_hash: str, preimage: Any) -> str | None:
    """The preimage, when it is one and it hashes to ``payment_hash``."""
    if not isinstance(preimage, str) or not _HEX64.match(preimage.lower()):
        return None
    if hashlib.sha256(bytes.fromhex(preimage)).hexdigest() != payment_hash:
        return None
    return preimage.lower()


async def get_info(conn: NwcConnection, *, timeout: float = 15.0) -> WalletInfo:
    """Ask the wallet what it is and what this connection may do."""
    result = await asyncio.to_thread(_ask, conn, "get_info", {}, timeout)
    methods = result.get("methods")
    alias, network = result.get("alias"), result.get("network")
    return WalletInfo(
        alias=alias[:80] if isinstance(alias, str) else None,
        network=network[:20] if isinstance(network, str) else None,
        methods=tuple(m for m in methods if isinstance(m, str)) if isinstance(methods, list) else (),
    )


async def pay_invoice(
    conn: NwcConnection, invoice: str, *, amount_msat: int, timeout: float = 60.0
) -> Payment:
    """Ask the wallet to pay ``invoice``, which must ask for exactly ``amount_msat``.

    The amount is checked here, before anything is sent, so a payee cannot be
    paid more than was agreed by returning a larger invoice. Record
    ``bolt11.decode(invoice).payment_hash`` before calling: it is how a payment
    whose outcome is unknown is found again.

    Raises:
        bolt11.Bolt11Error: ``invoice`` is not a readable invoice.
        NwcAmountMismatch: it asks for a different amount, or for none.
        NwcError, NwcNotSent, NwcOutcomeUnknown: see the module docstring.
    """
    asked = bolt11.decode(invoice)
    if asked.amount_msat != amount_msat:
        raise NwcAmountMismatch(
            f"the invoice asks for {asked.amount_msat} msat and {amount_msat} msat was agreed"
        )
    if asked.expires_at <= time.time():
        raise NwcError("OTHER", "the invoice has expired")

    result = await asyncio.to_thread(_ask, conn, "pay_invoice", {"invoice": invoice.strip()}, timeout)
    preimage = _preimage_for(asked.payment_hash, result.get("preimage"))
    if preimage is None:
        raise NwcOutcomeUnknown("the wallet reported a payment without a preimage that proves it")
    return Payment(
        payment_hash=asked.payment_hash,
        preimage=preimage,
        fees_paid_msat=_msat(result.get("fees_paid")),
    )


async def lookup_invoice(
    conn: NwcConnection, *, payment_hash: str, timeout: float = 15.0
) -> InvoiceStatus:
    """Ask the wallet what became of the payment with this hash.

    ``settled`` is true when the wallet returns a preimage that hashes to
    ``payment_hash``, or states the payment settled. A wallet that has never
    seen the hash raises :class:`NwcError` with code ``NOT_FOUND``.
    """
    wanted = payment_hash.lower() if isinstance(payment_hash, str) else ""
    if not _HEX64.match(wanted):
        raise ValueError("payment_hash must be 64 hex characters")

    result = await asyncio.to_thread(
        _ask, conn, "lookup_invoice", {"payment_hash": wanted}, timeout
    )
    state = result.get("state") if isinstance(result.get("state"), str) else None
    preimage = _preimage_for(wanted, result.get("preimage"))
    return InvoiceStatus(
        payment_hash=wanted,
        state=state,
        settled=preimage is not None or state == "settled",
        preimage=preimage,
        fees_paid_msat=_msat(result.get("fees_paid")),
        settled_at=_msat(result.get("settled_at")),
    )
