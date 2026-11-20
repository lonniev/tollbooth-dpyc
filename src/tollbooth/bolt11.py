"""Read what a BOLT11 invoice asks for, before anyone pays it.

A payer needs three facts from an invoice it did not write: how much it asks
for, which payment hash settles it, and when it stops being payable. This
module reads those, and the node that is paid (``payee``: the ``n`` field, or
the key that signed the invoice). It does not pay; the wallet that pays checks
the invoice again.

    asked = decode(invoice)
    asked.amount_msat      # None when the invoice names no amount
    asked.payment_hash     # hex; record it BEFORE paying
    asked.expires_at       # unix seconds

The bech32 checksum is verified, so a mistyped or truncated invoice is refused
here instead of at the wallet.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Bolt11Error", "Invoice", "decode"]

# BOLT11 "Human-Readable Part": ln + network + optional amount + optional multiplier.
_HRP = re.compile(r"^ln(bcrt|bc|tbs|tb)(?:([0-9]+)([munp]?))?$")

# Millisatoshi per unit of each multiplier, as (numerator, denominator).
# One bitcoin is 10^11 msat; `p` is a tenth of a millisatoshi.
_MSAT_PER_UNIT: dict[str, tuple[int, int]] = {
    "": (100_000_000_000, 1),
    "m": (100_000_000, 1),
    "u": (100_000, 1),
    "n": (100, 1),
    "p": (1, 10),
}

_TIMESTAMP_GROUPS = 7      # 35 bits
_SIGNATURE_GROUPS = 104    # 520 bits
_TAG_PAYMENT_HASH = 1      # `p`
_TAG_EXPIRY = 6            # `x`
_TAG_PAYEE = 19            # `n`
_PAYEE_GROUPS = 53         # 265 bits carrying a 33-byte key
_DEFAULT_EXPIRY_SECONDS = 3600
_PAYMENT_HASH_GROUPS = 52  # 260 bits carrying a 256-bit hash
# An invoice is a QR code's worth of text. Anything far longer is not one.
_MAX_LENGTH = 8192


class Bolt11Error(ValueError):
    """The text is not a BOLT11 invoice this module can read."""


@dataclass(frozen=True, slots=True)
class Invoice:
    """What an invoice asks for."""

    network: str
    amount_msat: int | None
    payment_hash: str
    created_at: int
    expiry_seconds: int
    #: The paid node's public key (33 bytes, hex): the ``n`` field, or the key that signed it.
    payee: str = ""

    @property
    def expires_at(self) -> int:
        return self.created_at + self.expiry_seconds


def _to_int(groups: list[int]) -> int:
    value = 0
    for g in groups:
        value = (value << 5) | g
    return value


def _to_bytes(groups: list[int]) -> bytes:
    """5-bit groups as bytes, the last byte padded with zero bits (BOLT11's signed form)."""
    acc, bits, out = 0, 0, bytearray()
    for g in groups:
        acc = (acc << 5) | g
        bits += 5
        while bits >= 8:
            bits -= 8
            out.append((acc >> bits) & 0xFF)
    if bits:
        out.append((acc << (8 - bits)) & 0xFF)
    return bytes(out)


def _signer(hrp: str, data: list[int]) -> str:
    """The key that signed the invoice, recovered from its signature; "" when it cannot be."""
    import hashlib

    from coincurve import PublicKey

    signed = hrp.encode("utf-8") + _to_bytes(data[:-_SIGNATURE_GROUPS])
    sig = _to_bytes(data[-_SIGNATURE_GROUPS:])  # 64 bytes of r||s, then the recovery id
    try:
        key = PublicKey.from_signature_and_message(sig[:65], signed, hasher=lambda m: hashlib.sha256(m).digest())
    except (ValueError, Exception):  # noqa: BLE001 — any failure is "not recoverable"
        return ""
    return str(key.format(compressed=True).hex())


def _amount_msat(digits: str | None, multiplier: str | None) -> int | None:
    if digits is None:
        return None
    if len(digits) > 1 and digits[0] == "0":
        raise Bolt11Error("invoice amount has a leading zero")
    numerator, denominator = _MSAT_PER_UNIT[multiplier or ""]
    scaled = int(digits) * numerator
    if scaled % denominator:
        raise Bolt11Error("invoice amount is finer than a millisatoshi")
    return scaled // denominator


def decode(invoice: str) -> Invoice:
    """Read the amount, payment hash and expiry out of a BOLT11 invoice.

    Raises:
        Bolt11Error: the text is not a well-formed invoice, its checksum does
            not verify, or it carries no single payment hash.
    """
    from pynostr.bech32 import Encoding, bech32_decode  # type: ignore[import-untyped]

    if not isinstance(invoice, str) or not invoice:
        raise Bolt11Error("invoice is empty")
    text = invoice.strip()
    if text[:10].lower() == "lightning:":
        text = text[10:]
    if len(text) > _MAX_LENGTH:
        raise Bolt11Error("invoice is too long")

    hrp, data, spec = bech32_decode(text)
    if hrp is None or data is None or spec != Encoding.BECH32:
        raise Bolt11Error("invoice checksum does not verify")

    match = _HRP.match(hrp)
    if match is None:
        raise Bolt11Error("invoice prefix is not a Lightning invoice")
    network, digits, multiplier = match.groups()

    if len(data) < _TIMESTAMP_GROUPS + _SIGNATURE_GROUPS:
        raise Bolt11Error("invoice is too short")
    created_at = _to_int(data[:_TIMESTAMP_GROUPS])
    fields = data[_TIMESTAMP_GROUPS:-_SIGNATURE_GROUPS]

    payment_hash: str | None = None
    payee = ""
    expiry = _DEFAULT_EXPIRY_SECONDS
    pos = 0
    while pos < len(fields):
        if pos + 3 > len(fields):
            raise Bolt11Error("invoice has a truncated field")
        tag = fields[pos]
        length = (fields[pos + 1] << 5) | fields[pos + 2]
        start = pos + 3
        end = start + length
        if end > len(fields):
            raise Bolt11Error("invoice has a truncated field")
        body = fields[start:end]
        pos = end

        if tag == _TAG_PAYMENT_HASH:
            # BOLT11: a reader MUST skip a `p` field that is not 52 groups long.
            if length != _PAYMENT_HASH_GROUPS:
                continue
            if payment_hash is not None:
                raise Bolt11Error("invoice names more than one payment hash")
            # 260 bits: the 256-bit hash, then 4 bits of padding.
            payment_hash = f"{_to_int(body) >> 4:064x}"
        elif tag == _TAG_EXPIRY:
            expiry = _to_int(body)
        elif tag == _TAG_PAYEE and length == _PAYEE_GROUPS:
            payee = f"{_to_int(body) >> 1:066x}"

    if payment_hash is None:
        raise Bolt11Error("invoice names no payment hash")

    return Invoice(
        network=network,
        amount_msat=_amount_msat(digits, multiplier),
        payment_hash=payment_hash,
        created_at=created_at,
        expiry_seconds=expiry,
        payee=payee or _signer(hrp, data),
    )
