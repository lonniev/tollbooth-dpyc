"""Cashu ecash, the wallet side: blind, unblind, lock and talk to a mint.

An Operator uses this to hold a commission's balance in escrow at a Cashu mint
without ever being able to spend it. The client's Lightning payment is minted
straight into a token locked to two of three Nostr keys (client, worker,
arbiter); a payout is melted straight to a Lightning invoice. The Operator
builds every message, gathers signatures and submits; it holds no key in any
lock, so it can never satisfy one.

What is here, by NUT (https://github.com/cashubtc/nuts):

- NUT-00: :func:`hash_to_curve`, :func:`blind`, :func:`unblind` (blind Diffie-Hellman).
- NUT-11: :func:`p2pk_secret`, :func:`swap_message`, :func:`melt_message`, and
  :func:`schnorr_verify` / :func:`schnorr_sign` for the raw BIP-340 signatures a
  lock needs (over the SHA-256 of the message).
- NUT-12: :func:`verify_dleq`, checked on every promise a mint returns.
- NUT-20: :func:`mint_quote_message`, so a paid invoice mints only with the
  quote key's signature.
- :class:`MintClient`, the HTTP calls, to an https mint on a public address only.

Secrets are hex strings or NUT-10 JSON strings; points and keys travel as
compressed hex. ``mint_sign`` and ``mint_verify`` are the mint's half of
NUT-00, for tests that stand a mint up in-process.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import secrets as _secrets
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from coincurve import PrivateKey, PublicKey, PublicKeyXOnly

#: secp256k1 group order.
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
DOMAIN_SEPARATOR = b"Secp256k1_HashToCurve_Cashu_"
NUT20_TAG = b"Cashu_MintQuoteSig_v1"
SIG_ALL = "SIG_ALL"
SIG_INPUTS = "SIG_INPUTS"
#: NUTs an escrow mint must support: 07 state check, 08 fee change, 11 P2PK, 12 DLEQ, 20 quote signatures.
ESCROW_NUTS = ("7", "8", "11", "12", "20")


class CashuError(Exception):
    """A mint refused, or answered something this wallet will not accept."""

    def __init__(self, message: str, code: int | None = None, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------
# NUT-00: blind Diffie-Hellman
# ---------------------------------------------------------------------------


def _point(hex_key: str) -> PublicKey:
    return PublicKey(bytes.fromhex(hex_key))


def _scalar(value: int) -> bytes:
    return (value % N).to_bytes(32, "big")


def hash_to_curve(message: bytes) -> PublicKey:
    """The point ``Y`` a secret maps to (NUT-00)."""
    msg_hash = hashlib.sha256(DOMAIN_SEPARATOR + message).digest()
    for counter in range(2**16):
        candidate = hashlib.sha256(msg_hash + counter.to_bytes(4, "little")).digest()
        try:
            return PublicKey(b"\x02" + candidate)
        except ValueError:
            continue
    raise CashuError("no curve point found for this secret")


def blind(secret: str, r: bytes | None = None) -> tuple[str, str]:
    """``(B_, r)`` for a secret: ``B_ = Y + rG``. ``r`` is drawn fresh unless given."""
    r = r or PrivateKey().secret
    y = hash_to_curve(secret.encode("utf-8"))
    b_ = PublicKey.combine_keys([y, PrivateKey(r).public_key])
    return b_.format().hex(), r.hex()


def unblind(c_: str, r: str, mint_key: str) -> str:
    """``C = C_ - rK``: the token's signature from the mint's blinded one."""
    minus_r_k = _point(mint_key).multiply(_scalar(N - int(r, 16)))
    return PublicKey.combine_keys([_point(c_), minus_r_k]).format().hex()


def mint_sign(k: bytes, b_: str) -> str:
    """The mint's half: ``C_ = kB_``. For in-process test mints."""
    return _point(b_).multiply(k).format().hex()


def mint_verify(k: bytes, secret: str, c: str) -> bool:
    """The mint's check that ``C == k * hash_to_curve(secret)``. For in-process test mints."""
    return hash_to_curve(secret.encode("utf-8")).multiply(k).format() == bytes.fromhex(c)


# ---------------------------------------------------------------------------
# NUT-12: DLEQ proofs on what the mint returns
# ---------------------------------------------------------------------------


def hash_e(*points: PublicKey) -> bytes:
    joined = "".join(p.format(compressed=False).hex() for p in points)
    return hashlib.sha256(joined.encode("utf-8")).digest()


def verify_dleq(b_: str, c_: str, e: str, s: str, mint_key: str) -> bool:
    """True when the mint proved ``C_`` was signed with the key it published for this amount."""
    a = _point(mint_key)
    e_int, s_int = int(e, 16), int(s, 16)
    if not (0 < s_int < N and 0 < e_int < N):
        return False
    s_g = PrivateKey(_scalar(s_int)).public_key
    minus_e = _scalar(N - e_int)
    r1 = PublicKey.combine_keys([s_g, a.multiply(minus_e)])
    r2 = PublicKey.combine_keys([_point(b_).multiply(_scalar(s_int)), _point(c_).multiply(minus_e)])
    return hash_e(r1, r2, a, _point(c_)) == bytes.fromhex(e)


# ---------------------------------------------------------------------------
# NUT-11: P2PK locks and the messages their signers sign
# ---------------------------------------------------------------------------


def lock_key(xonly_hex: str) -> str:
    """A Nostr public key (32-byte x-only hex) as a lock key. Mints check signatures x-only."""
    if len(xonly_hex) != 64:
        raise ValueError("a lock key is a 32-byte x-only public key in hex")
    PublicKeyXOnly(bytes.fromhex(xonly_hex))  # raises if not on the curve
    return "02" + xonly_hex.lower()


def p2pk_secret(
    data: str,
    *,
    pubkeys: list[str] | None = None,
    n_sigs: int = 1,
    locktime: int | None = None,
    refund: list[str] | None = None,
    sigflag: str = SIG_ALL,
    nonce: str | None = None,
) -> str:
    """A NUT-10 P2PK secret string. Each call draws a fresh nonce unless one is given.

    A lock with a ``locktime`` always names ``refund`` keys: without them the
    token becomes spendable by anyone once the time passes.
    """
    if sigflag not in (SIG_ALL, SIG_INPUTS):
        raise ValueError("sigflag is SIG_ALL or SIG_INPUTS")
    if locktime is not None and not refund:
        raise ValueError("a lock with a locktime must name refund keys")
    keys = pubkeys or []
    if n_sigs < 1 or n_sigs > 1 + len(keys):
        raise ValueError("n_sigs must be between 1 and the number of signing keys")
    tags: list[list[str]] = []
    if keys:
        tags.append(["pubkeys", *keys])
    if n_sigs > 1:
        tags.append(["n_sigs", str(n_sigs)])
    if locktime is not None:
        tags.append(["locktime", str(int(locktime))])
        tags.append(["refund", *(refund or [])])
    tags.append(["sigflag", sigflag])
    body = {"nonce": nonce or _secrets.token_hex(32), "data": data, "tags": tags}
    return json.dumps(["P2PK", body], separators=(",", ":"))


def secret_lock(secret: str) -> dict[str, Any]:
    """``{data, tags}`` of a P2PK secret, nonce aside: SIG_ALL inputs must agree on this."""
    kind, body = json.loads(secret)
    if kind != "P2PK":
        raise ValueError("not a P2PK secret")
    return {"data": body["data"], "tags": body.get("tags", [])}


def swap_message(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> str:
    """SIG_ALL message for a swap: ``secret_i || C_i ... || amount_j || B_j``."""
    return "".join(p["secret"] + p["C"] for p in inputs) + "".join(f"{o['amount']}{o['B_']}" for o in outputs)


def melt_message(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]], quote_id: str) -> str:
    """SIG_ALL message for a melt: the swap message followed by the quote id."""
    return swap_message(inputs, outputs) + quote_id


def schnorr_sign(private_key: bytes, message: str | bytes) -> str:
    """BIP-340 signature over SHA-256 of the message. For agents and tests; the Operator never signs a lock."""
    raw = message.encode("utf-8") if isinstance(message, str) else message
    return PrivateKey(private_key).sign_schnorr(hashlib.sha256(raw).digest()).hex()


def schnorr_verify(pubkey: str, message: str | bytes, signature: str) -> bool:
    """Check a lock signature: ``pubkey`` may be x-only (64 hex) or compressed (66 hex)."""
    raw = message.encode("utf-8") if isinstance(message, str) else message
    try:
        key = bytes.fromhex(pubkey)
        xonly = PublicKeyXOnly(key[1:] if len(key) == 33 else key)
        return bool(xonly.verify(bytes.fromhex(signature), hashlib.sha256(raw).digest()))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# NUT-20: binding a mint quote to a key
# ---------------------------------------------------------------------------


def mint_quote_message(quote: str, outputs: list[dict[str, Any]]) -> bytes:
    """The bytes a quote's key signs to mint it (``Cashu_MintQuoteSig_v1``, length-prefixed)."""

    def len32(b: bytes) -> bytes:
        return len(b).to_bytes(4, "big")

    msg = NUT20_TAG + len32(quote.encode("utf-8")) + quote.encode("utf-8")
    for o in outputs:
        amount = int(o["amount"])
        amt = amount.to_bytes((amount.bit_length() + 7) // 8, "big")
        b_ = bytes.fromhex(o["B_"])
        msg += len32(amt) + amt + len32(b_) + b_
    return msg


# ---------------------------------------------------------------------------
# Amounts, outputs and proofs
# ---------------------------------------------------------------------------


def amount_split(amount: int) -> list[int]:
    """Powers of two that sum to ``amount``, smallest first."""
    if amount < 0:
        raise ValueError("amount must not be negative")
    return [1 << i for i in range(amount.bit_length()) if amount >> i & 1]


def input_fee(proofs: list[dict[str, Any]], fee_ppk: int) -> int:
    """What a mint charges to spend these proofs: ``ceil(n * ppk / 1000)``."""
    return (len(proofs) * fee_ppk + 999) // 1000


@dataclass(frozen=True)
class Pending:
    """Outputs sent to a mint, with what is needed to unblind its answer."""

    outputs: list[dict[str, Any]]
    secrets: list[str]
    rs: list[str]


def outputs_for(amounts: list[int], secrets: list[str], keyset_id: str) -> Pending:
    """Blinded outputs for these amounts and secrets."""
    if len(amounts) != len(secrets):
        raise ValueError("one secret per amount")
    outputs, rs = [], []
    for amount, secret in zip(amounts, secrets, strict=True):
        b_, r = blind(secret)
        outputs.append({"amount": amount, "id": keyset_id, "B_": b_})
        rs.append(r)
    return Pending(outputs=outputs, secrets=list(secrets), rs=rs)


def blank_outputs(count: int, secret: Any, keyset_id: str) -> Pending:
    """NUT-08 blank outputs (amount 0) for a melt's fee change; ``secret()`` draws each secret."""
    pending = outputs_for([1] * count, [secret() for _ in range(count)], keyset_id)
    return Pending(outputs=[{**o, "amount": 0} for o in pending.outputs], secrets=pending.secrets, rs=pending.rs)


def blanks_needed(fee_reserve: int) -> int:
    """How many blank outputs a melt needs to return up to ``fee_reserve`` as change."""
    return max(1, fee_reserve.bit_length()) if fee_reserve > 0 else 0


def proofs_from(promises: list[dict[str, Any]], pending: Pending, keys: dict[int, str]) -> list[dict[str, Any]]:
    """Unblind a mint's promises into proofs, refusing any without a valid DLEQ proof."""
    if len(promises) > len(pending.outputs):
        raise CashuError("the mint returned more promises than outputs")
    proofs = []
    for promise, output, secret, r in zip(promises, pending.outputs, pending.secrets, pending.rs, strict=False):
        amount = int(promise["amount"])
        mint_key = keys.get(amount)
        if mint_key is None:
            raise CashuError(f"the mint has no key for amount {amount}")
        dleq = promise.get("dleq") or {}
        if not dleq or not verify_dleq(output["B_"], promise["C_"], dleq["e"], dleq["s"], mint_key):
            raise CashuError("the mint's promise carries no valid DLEQ proof")
        proofs.append({"amount": amount, "id": promise["id"], "secret": secret, "C": unblind(promise["C_"], r, mint_key)})
    return proofs


def with_witness(proofs: list[dict[str, Any]], signatures: list[str]) -> list[dict[str, Any]]:
    """SIG_ALL inputs: every signature in the first input's witness, none elsewhere."""
    if not proofs:
        raise ValueError("no inputs")
    first = {**proofs[0], "witness": json.dumps({"signatures": signatures}, separators=(",", ":"))}
    return [first, *({k: v for k, v in p.items() if k != "witness"} for p in proofs[1:])]


def y_of(secret: str) -> str:
    """A proof's ``Y`` (NUT-07 state check key)."""
    return hash_to_curve(secret.encode("utf-8")).format().hex()


# ---------------------------------------------------------------------------
# The mint, over HTTP
# ---------------------------------------------------------------------------


def checked_mint_url(url: str) -> str:
    """An https mint on a public address, or ``ValueError``."""
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if parts.scheme != "https" or not host:
        raise ValueError("a mint must be an https:// address")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("a mint must be a public address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return f"https://{parts.netloc}{parts.path.rstrip('/')}"
    if not address.is_global:
        raise ValueError("a mint must be a public address")
    return f"https://{parts.netloc}{parts.path.rstrip('/')}"


def missing_escrow_nuts(info: dict[str, Any]) -> list[str]:
    """The NUTs an escrow needs that this mint's ``/v1/info`` does not claim."""
    nuts = info.get("nuts") or {}
    missing = []
    for nut in ESCROW_NUTS:
        entry = nuts.get(nut)
        if not (isinstance(entry, dict) and entry.get("supported") is True):
            missing.append(nut)
    return missing


class MintClient:
    """The few calls an escrow makes to one mint."""

    def __init__(self, url: str, *, client: httpx.AsyncClient | None = None) -> None:
        self.url = checked_mint_url(url)
        self._client = client

    async def _call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        timeout = httpx.Timeout(20.0, connect=5.0)
        owned = self._client is None
        client = self._client or httpx.AsyncClient(timeout=timeout)
        try:
            resp = await client.request(method, f"{self.url}/v1/{path}", json=body)
        except httpx.TimeoutException as exc:
            raise CashuError("the mint did not answer in time") from exc
        except httpx.HTTPError as exc:
            raise CashuError(f"could not reach the mint: {type(exc).__name__}") from exc
        finally:
            if owned:
                await client.aclose()
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400:
            detail = data.get("detail") if isinstance(data, dict) else None
            code = data.get("code") if isinstance(data, dict) else None
            raise CashuError(f"the mint refused: {str(detail or resp.status_code)[:200]}", code=code, status=resp.status_code)
        return data

    async def info(self) -> dict[str, Any]:
        return dict(await self._call("GET", "info"))

    async def active_keyset(self, unit: str = "sat") -> dict[str, Any]:
        """``{id, input_fee_ppk, keys: {amount: pubkey}}`` of the mint's active keyset for ``unit``."""
        keysets = (await self._call("GET", "keysets")).get("keysets", [])
        active = [k for k in keysets if k.get("unit") == unit and k.get("active")]
        if not active:
            raise CashuError(f"the mint has no active {unit} keyset")
        keyset = active[0]
        keys = (await self._call("GET", f"keys/{keyset['id']}")).get("keysets", [{}])[0].get("keys", {})
        return {"id": keyset["id"], "input_fee_ppk": int(keyset.get("input_fee_ppk") or 0),
                "keys": {int(a): k for a, k in keys.items()}}

    async def mint_quote(self, amount: int, pubkey: str, unit: str = "sat") -> dict[str, Any]:
        """An invoice whose payment mints only with ``pubkey``'s signature (NUT-20)."""
        return dict(await self._call("POST", "mint/quote/bolt11", {"amount": amount, "unit": unit, "pubkey": pubkey}))

    async def mint_quote_state(self, quote: str) -> dict[str, Any]:
        return dict(await self._call("GET", f"mint/quote/bolt11/{quote}"))

    async def mint(self, quote: str, outputs: list[dict[str, Any]], signature: str) -> list[dict[str, Any]]:
        body = {"quote": quote, "outputs": outputs, "signature": signature}
        return list((await self._call("POST", "mint/bolt11", body)).get("signatures", []))

    async def melt_quote(self, bolt11: str, unit: str = "sat") -> dict[str, Any]:
        return dict(await self._call("POST", "melt/quote/bolt11", {"request": bolt11, "unit": unit}))

    async def melt(self, quote: str, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> dict[str, Any]:
        return dict(await self._call("POST", "melt/bolt11", {"quote": quote, "inputs": inputs, "outputs": outputs}))

    async def swap(self, inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list((await self._call("POST", "swap", {"inputs": inputs, "outputs": outputs})).get("signatures", []))

    async def check_state(self, ys: list[str]) -> list[dict[str, Any]]:
        return list((await self._call("POST", "checkstate", {"Ys": ys})).get("states", []))
