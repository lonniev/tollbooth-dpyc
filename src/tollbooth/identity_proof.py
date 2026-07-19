"""Identity proof — single canonical gate for npub ownership.

The gate ``require_proof`` accepts ANY of the following tactics —
caller chooses based on what credentials it has on hand:

1. **Inline Schnorr proof** (kind 27235, NIP-98 style): a freshly-signed
   Nostr event with the tool name in the ``u`` tag and a freshness
   window. Works when the caller holds the nsec.
   ::

       {
           "pubkey": "<hex_pubkey>",
           "kind": 27235,
           "content": "",
           "created_at": <unix_timestamp>,
           "tags": [["u", "<tool_name>"]],
           "sig": "<schnorr_signature>"
       }

2. **Cached dpop_token phrase** (format ``alpha-beta-42``): the
   ``dpop_token`` returned by a prior ``request_npub_proof`` →
   ``receive_npub_proof`` DM round-trip. Works when the caller holds
   only the dpop_token and not the nsec (e.g., a remote AI agent that's
   already proven ownership in this session). The gate hashes it and
   looks up the proven-npub cache supplied by the caller.

The gate is actor-agnostic — Operators, Authorities, and any future
runtime use the same function. Callers pass their own
``proven_cache`` (or omit it to disable tactic 2).

Dependencies: ``pynostr`` (available via ``tollbooth-dpyc[nostr]``).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tollbooth.proven_npub import ProvenNpubCache

logger = logging.getLogger(__name__)

# Dpop_token phrases produced by request_npub_proof have the shape
# ``<word>-<word>-<n>`` — three lowercase letters/digits segments.
_DPOP_TOKEN_RE = re.compile(r"^[a-z]+-[a-z]+-\d+$")

PROOF_EVENT_KIND = 27235
"""NIP-98 HTTP Auth event kind, repurposed for MCP identity proofs."""

DEFAULT_WINDOW_SECONDS = 60
"""Maximum age (in seconds) of a valid proof event."""

MAX_PROOF_JSON_BYTES = 64 * 1024
"""Reject proof payloads larger than this before parsing.

A signed kind-27235 event is well under 2 KB; anything approaching this cap is
an adversarial oversized payload (tool arguments are untrusted AI input). Bound
the input before ``json.loads`` so a 10 MB string can't be fully parsed."""

OWNERSHIP_SENTINEL = "npub_ownership"
"""Sentinel tool name for npub ownership proofs (not tied to a specific tool)."""

ADOPTION_PROOF_TOOL = "operator_adoption_request"
"""Sentinel tool name binding an operator's adoption-request proof.

A canonical, slug-independent ``u``-tag value so an Operator can mint an
inline kind-27235 proof (with its own service nsec) that any chosen
Authority verifies without either side knowing the other's tool-slug. Used
by ``request_adoption`` (operator mints) and ``receive_adoption_request``
(Authority verifies)."""


def _npub_to_hex(npub: str) -> str:
    """Convert a bech32 npub to a hex pubkey string."""
    from pynostr.key import PublicKey  # type: ignore[import-untyped]

    return PublicKey.from_npub(npub).hex()


def create_proof(nsec: str, tool_name: str) -> str:
    """Create a signed kind-27235 identity proof.

    Args:
        nsec: Nostr private key (bech32 nsec1... or hex).
        tool_name: The MCP tool name to embed in the ``u`` tag.

    Returns:
        JSON string of the signed Nostr event.
    """
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]
    from pynostr.event import Event  # type: ignore[import-untyped]

    if nsec.startswith("nsec1"):
        pk = PrivateKey.from_nsec(nsec)
    else:
        pk = PrivateKey(bytes.fromhex(nsec))

    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()),
        # A per-call nonce so two proofs for the same tool within the same wall-clock
        # second are never byte-identical. Without it their event ids collide and the
        # verifier's replay guard rejects the second as already-seen (rapid same-tool
        # callers — a seed loop, an agent keyring — would spuriously fail). The nonce is
        # signed but otherwise inert: verify_proof reads only the ``u`` tag.
        tags=[["u", tool_name], ["nonce", secrets.token_hex(16)]],
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


def create_ownership_proof(nsec: str) -> str:
    """Create a kind-27235 proof for npub ownership (no specific tool).

    Args:
        nsec: Nostr private key (bech32 nsec1... or hex).

    Returns:
        JSON string of the signed Nostr event with ``u=npub_ownership``.
    """
    return create_proof(nsec, OWNERSHIP_SENTINEL)


PROVENANCE_ATTESTATION_TOOL = "npub_proof_request"
"""``u``-tag sentinel marking a kind-27235 event as a proof-request provenance
attestation (not a per-tool caller proof). Distinct from ``OWNERSHIP_SENTINEL``
so a provenance attestation can never be mistaken for a caller's ownership
proof, or vice versa."""


def create_provenance_attestation(
    operator_nsec: str,
    *,
    sender_pubkey_hex: str,
    subject_npub: str,
    service: str,
    challenge: str,
) -> str:
    """Sign an Operator provenance attestation for a Secure-Courier request DM.

    A Secure-Courier DM must be *delivered* from a key other than the
    Operator's own npub in the self-addressed case — Nostr relays silently
    drop self-addressed DMs, so ``open_channel`` sends self-DMs from a
    throwaway ephemeral key (commit #93). That leaves the human staring at an
    unfamiliar sender npub with no verifiable tie to the Operator they believe
    they are dealing with.

    This attestation restores the tie *inside* the (NIP-44 encrypted) DM body:
    the Operator signs — with its **registered** identity key, the asset an
    impostor does not hold — a kind-27235 event that binds:

    - ``sender``: the pubkey the recipient actually sees on the DM (the
      ephemeral delivery key, or the Operator's own key on a patron DM). An
      attestation cannot be lifted onto a DM delivered by a different key.
    - ``subject``: the npub whose ownership/credentials the DM concerns.
    - ``service``: the credential-template / proof service the DM rides on.
    - ``challenge``: the one-time ``dpop_token`` for this exchange, so the
      attestation is bound to this DM and cannot be replayed against another.

    The recipient verifies the signature (``verify_provenance_attestation``)
    and resolves the recovered signer pubkey against the DPYC registry to
    render the trust state. Provenance is thus **Operator-attested**, never
    **requester-asserted** — the design principle this closes.

    Args:
        operator_nsec: The Operator's registered private key (bech32 or hex).
        sender_pubkey_hex: Hex pubkey the DM is delivered from (ephemeral key
            for self-DMs, else the Operator's own pubkey).
        subject_npub: The npub the request concerns (bech32).
        service: The service name the DM rides on.
        challenge: The one-time ``dpop_token`` slug for this exchange.

    Returns:
        JSON string of the signed kind-27235 attestation event.
    """
    from pynostr.key import PrivateKey  # type: ignore[import-untyped]
    from pynostr.event import Event  # type: ignore[import-untyped]

    if operator_nsec.startswith("nsec1"):
        pk = PrivateKey.from_nsec(operator_nsec)
    else:
        pk = PrivateKey(bytes.fromhex(operator_nsec))

    event = Event(
        pubkey=pk.public_key.hex(),
        kind=PROOF_EVENT_KIND,
        content="",
        created_at=int(time.time()),
        tags=[
            ["u", PROVENANCE_ATTESTATION_TOOL],
            ["sender", sender_pubkey_hex],
            ["subject", subject_npub],
            ["service", service],
            ["challenge", challenge],
            ["nonce", secrets.token_hex(16)],
        ],
    )
    event.sign(pk.hex())
    return json.dumps(event.to_dict())


def verify_provenance_attestation(
    attestation_json: str,
    *,
    expected_sender_pubkey_hex: str,
    expected_subject_npub: str,
    expected_challenge: str,
) -> dict[str, Any]:
    """Verify an Operator provenance attestation against what the recipient saw.

    Confirms the attestation is a well-formed, correctly-signed kind-27235
    provenance event whose bound facts (sender, subject, challenge) match the
    DM the recipient actually received — closing the lift-and-replay window.

    This function performs **cryptographic** verification only. It deliberately
    does *not* consult the DPYC registry: the caller resolves the recovered
    ``operator_pubkey_hex`` against the registry to decide the trust state
    (registered + certified → green; registered but novel → amber; unresolvable
    → red). Keeping registry I/O out of this leaf makes it pure and testable,
    and lets the caller apply its own fail-closed policy.

    Unlike ``verify_proof`` there is **no freshness window and no replay
    consumption**: the attestation is an inbound, idempotently-readable fact a
    human may verify minutes or hours after the DM arrived (the same reason the
    proof-reply path dropped its wall-clock gate), and re-reading it is not a
    replay.

    Returns a dict with:
        ``valid``: bool — signature valid and all bound fields match.
        ``operator_pubkey_hex``: str | None — recovered signer, for registry
            resolution (present whenever the signature verified, even if a
            bound field mismatched).
        ``reason``: str — machine-readable failure cause when ``valid`` is
            False (``"pynostr_missing"``, ``"malformed"``, ``"bad_signature"``,
            ``"wrong_kind"``, ``"not_attestation"``, ``"sender_mismatch"``,
            ``"subject_mismatch"``, ``"challenge_mismatch"``), else ``"ok"``.
    """
    result: dict[str, Any] = {
        "valid": False,
        "operator_pubkey_hex": None,
        "reason": "ok",
    }

    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pynostr not installed — cannot verify provenance attestation")
        result["reason"] = "pynostr_missing"
        return result

    if (
        not isinstance(attestation_json, str)
        or len(attestation_json) > MAX_PROOF_JSON_BYTES
    ):
        result["reason"] = "malformed"
        return result

    try:
        event = Event.from_dict(json.loads(attestation_json))
    except Exception:
        result["reason"] = "malformed"
        return result

    try:
        if not event.verify():
            result["reason"] = "bad_signature"
            return result
    except Exception:
        result["reason"] = "bad_signature"
        return result

    # Signature is valid — the signer pubkey is now trustworthy for the caller
    # to resolve against the registry, regardless of any bound-field mismatch.
    result["operator_pubkey_hex"] = event.pubkey

    if event.kind != PROOF_EVENT_KIND:
        result["reason"] = "wrong_kind"
        return result

    def _tag(name: str) -> str | None:
        for tag in event.tags:
            if len(tag) >= 2 and tag[0] == name:
                return tag[1]
        return None

    if _tag("u") != PROVENANCE_ATTESTATION_TOOL:
        result["reason"] = "not_attestation"
        return result
    if _tag("sender") != expected_sender_pubkey_hex:
        result["reason"] = "sender_mismatch"
        return result
    if _tag("subject") != expected_subject_npub:
        result["reason"] = "subject_mismatch"
        return result
    if _tag("challenge") != expected_challenge:
        result["reason"] = "challenge_mismatch"
        return result

    result["valid"] = True
    return result


# Replay protection: track consumed proof event IDs within the window
_consumed_proofs: dict[str, float] = {}  # event_id → expiry time
_CONSUMED_CLEANUP_INTERVAL = 120  # seconds between cleanups
# Hard cap on the consumed-proof set. Entries are only inserted after a proof
# passes full signature + freshness verification, so this set cannot be filled
# with arbitrary junk — but a flood of validly-signed, distinct proofs arriving
# faster than the 120s cleanup could still grow it unbounded. The cap bounds
# memory; 10k live entries is far above any honest operator's concurrent proof
# volume within a single freshness window.
_CONSUMED_MAX_ENTRIES = 10000
_last_cleanup: float = 0.0


def _cleanup_consumed() -> None:
    global _last_cleanup
    now = time.time()
    if now - _last_cleanup < _CONSUMED_CLEANUP_INTERVAL:
        return
    _last_cleanup = now
    expired = [k for k, v in _consumed_proofs.items() if now > v]
    for k in expired:
        del _consumed_proofs[k]


def _record_consumed(event_id: str, expiry: float) -> None:
    """Record a consumed proof id under a hard size cap.

    The interval-based ``_cleanup_consumed`` drains expired ids lazily; this
    cap is the backstop when distinct valid proofs arrive faster than that
    cleanup runs. When full, already-expired ids are purged first; if still
    full, the ids closest to expiry are evicted. Evicting soonest-to-expire is
    safe: an evicted id can only be replayed in the sliver before its proof's
    freshness window lapses, after which the age check rejects it regardless.
    """
    if len(_consumed_proofs) >= _CONSUMED_MAX_ENTRIES:
        now = time.time()
        for k in [k for k, v in _consumed_proofs.items() if now > v]:
            del _consumed_proofs[k]
        if len(_consumed_proofs) >= _CONSUMED_MAX_ENTRIES:
            overflow = len(_consumed_proofs) - _CONSUMED_MAX_ENTRIES + 1
            for k, _ in sorted(
                _consumed_proofs.items(), key=lambda kv: kv[1]
            )[:overflow]:
                del _consumed_proofs[k]
    _consumed_proofs[event_id] = expiry


def verify_proof(
    proof_json: str,
    expected_npub: str,
    tool_name: str,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> bool:
    """Verify a Nostr kind-27235 identity proof.

    Works for any npub holder — operator or patron.

    Args:
        proof_json: JSON string of the signed Nostr event.
        expected_npub: The npub (bech32) the proof must be signed by.
        tool_name: The MCP tool name that must appear in the ``u`` tag.
        window_seconds: Maximum age (in seconds) of the proof event.

    Returns:
        ``True`` if the proof is valid, ``False`` otherwise.
    """
    try:
        from pynostr.event import Event  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pynostr not installed — cannot verify identity proof")
        return False

    if not isinstance(proof_json, str) or len(proof_json) > MAX_PROOF_JSON_BYTES:
        logger.debug(
            "identity_proof: rejecting oversized/invalid payload (%s bytes)",
            len(proof_json) if isinstance(proof_json, str) else "non-str",
        )
        return False

    try:
        event_dict = json.loads(proof_json)
    except (json.JSONDecodeError, TypeError):
        logger.debug("identity_proof: invalid JSON")
        return False

    try:
        event = Event.from_dict(event_dict)
    except Exception:
        logger.debug("identity_proof: invalid Nostr event structure")
        return False

    try:
        if not event.verify():
            logger.debug("identity_proof: signature verification failed")
            return False
    except Exception:
        logger.debug("identity_proof: signature verification error")
        return False

    if event.kind != PROOF_EVENT_KIND:
        logger.debug("identity_proof: wrong kind %d (expected %d)", event.kind, PROOF_EVENT_KIND)
        return False

    try:
        expected_hex = _npub_to_hex(expected_npub)
    except Exception:
        logger.debug("identity_proof: invalid expected npub")
        return False

    if event.pubkey != expected_hex:
        logger.debug("identity_proof: pubkey mismatch")
        return False

    u_values = [tag[1] for tag in event.tags if len(tag) >= 2 and tag[0] == "u"]
    if tool_name not in u_values:
        logger.debug("identity_proof: tool_name %r not in u tags %r", tool_name, u_values)
        return False

    now = time.time()
    age = abs(now - event.created_at)
    if age > window_seconds:
        logger.debug("identity_proof: expired (age=%.1fs, window=%ds)", age, window_seconds)
        return False

    # Replay protection: reject reused proof event IDs
    _cleanup_consumed()
    event_id = getattr(event, "id", None) or event_dict.get("id", "")
    if event_id and event_id in _consumed_proofs:
        logger.debug("identity_proof: replay rejected (event_id=%s)", event_id[:16])
        return False
    if event_id:
        _record_consumed(event_id, now + window_seconds)

    return True


async def require_proof(
    npub: str,
    dpop_token: str,
    tool_name: str,
    *,
    proven_cache: "ProvenNpubCache | None" = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any] | None:
    """Canonical proof-of-ownership gate. Returns ``None`` on success
    (caller proceeds) or a structured error dict to return verbatim.

    Use at the top of every tool whose semantics include "caller is
    acting as ``npub``" — check_balance, purchase_credits,
    account_statement, register_operator, etc. Bootstrap tools
    (request_npub_proof, receive_credentials, register_authority_npub)
    deliberately do NOT gate on this — they're how a candidate proves
    identity in the first place.

    **Tactics accepted, in this order:**

    1. **Cached dpop_token phrase** — when ``proof`` matches the
       ``<word>-<word>-<n>`` shape and ``proven_cache`` is supplied,
       hash it (sha256) and check ``cache.is_proven(hash, npub)``. A
       hit means a prior ``receive_npub_proof`` saw a valid signed-DM
       reply from this npub.
    2. **Inline Schnorr proof** — JSON-encoded kind-27235 event with
       ``tool_name`` in the ``u`` tag, signed by ``npub``, no older
       than ``window_seconds``. Works for any caller that holds the
       nsec; no cache needed.

    Bound to ``tool_name``: a proof issued for one tool will not pass
    for another, preventing replay across the public tool surface.

    Actor-agnostic — Operators, Authorities, and any future runtime
    use the same gate. Pass the cache appropriate to that runtime
    (``await rt.proven_npub_cache()`` for both).
    """
    # Lazy import — constants is leaf-y, but identity_proof is too;
    # keep the dep one-way to be safe.
    from tollbooth.constants import ErrorCode

    if not npub.startswith("npub1") or len(npub) < 60:
        return {
            "success": False,
            "error_code": ErrorCode.NPUB_INVALID,
            "error": (
                "Invalid npub format. Must start with 'npub1' and be at "
                "least 60 characters."
            ),
        }
    if not dpop_token:
        return {
            "success": False,
            "error_code": ErrorCode.PROOF_REQUIRED,
            "error": "dpop_token is required.",
            "next_steps": [
                "Either: sign a kind-27235 Nostr event with your nsec and pass "
                "it as `dpop_token` (one-shot, no relay round-trip).",
                "Or: call request_npub_proof, reply to the DM challenge from "
                "your Nostr client, then call receive_npub_proof — pass the "
                "returned dpop_token as `dpop_token` on every subsequent call.",
            ],
        }

    # Tactic 1: cached dpop_token phrase
    if proven_cache is not None and _DPOP_TOKEN_RE.match(dpop_token):
        # Lazy import to avoid identity_proof ↔ runtime circular dependency.
        from tollbooth.runtime import resolve_npub as _resolve_npub
        try:
            resolved = _resolve_npub(npub)
        except Exception:
            resolved = npub

        import hashlib as _hashlib
        dpop_token_hash = _hashlib.sha256(dpop_token.encode()).hexdigest()
        if await proven_cache.is_proven(dpop_token_hash, resolved):
            return None
        return {
            "success": False,
            "error_code": ErrorCode.PROOF_REFRESH_NEEDED,
            "error": (
                "Your npub-proof cache entry is no longer valid. This is "
                "routine — sign a fresh DM challenge and you're back."
            ),
            "next_steps": [
                "request_npub_proof(patron_npub=<patron_npub>)",
                "Reply to the DM challenge from your Nostr client",
                "receive_npub_proof(patron_npub=<patron_npub>) to cache a "
                "fresh dpop_token",
            ],
        }

    # S4: a dpop_token-shaped token reaching here was NOT accepted as a cached
    # proof (this runtime wired no proven_cache, or the cache-miss branch above
    # already returned). It is definitely not an inline Schnorr event (those are
    # JSON and never match _DPOP_TOKEN_RE), so don't fall through to a confusing
    # "malformed Schnorr" error — tell the caller their token isn't valid here
    # and how to refresh. This changes only the denial message, never what is
    # accepted.
    if _DPOP_TOKEN_RE.match(dpop_token):
        return {
            "success": False,
            "error_code": ErrorCode.PROOF_REFRESH_NEEDED,
            "error": (
                "That looks like a dpop_token, but it isn't a currently-valid "
                "cached proof here. Refresh it, or pass an inline kind-27235 "
                "Schnorr proof instead."
            ),
            "next_steps": [
                "request_npub_proof(patron_npub=<patron_npub>)",
                "Reply to the DM challenge from your Nostr client",
                "receive_npub_proof(patron_npub=<patron_npub>) to cache a "
                "fresh dpop_token",
            ],
        }

    # Tactic 2: inline Schnorr-signed kind-27235 event
    if not verify_proof(dpop_token, npub, tool_name, window_seconds=window_seconds):
        return {
            "success": False,
            "error_code": ErrorCode.PROOF_INVALID,
            "error": "Invalid identity proof.",
        }
    return None
