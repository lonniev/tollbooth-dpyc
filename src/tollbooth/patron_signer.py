"""The single home for patron-side proof signing.

A DPYC patron acts under its own Nostr identity: to call an operator's paid tool it
presents a fresh kind-27235 proof of possession, signed by its nsec and bound to the
tool it is calling. Every server-side Python realization of "a patron calls an operator"
signs *here* — the :class:`~tollbooth.authority_client.AuthorityCertifier`
(operator → Authority) and the agent keyring (an LLM agent → any operator) both hold a
:class:`PatronSigner`. iOS patrons play the same role in native Swift over the Apple
Keychain; this is that role's server-side peer.

Nothing is stored and nothing is *granted* per call: the standing grant is possession of
the nsec plus a funded balance at the operator. The proof is only the per-request
demonstration of possession, minted in memory from the nsec held here.
"""

from __future__ import annotations

from typing import Any

from tollbooth.identity_proof import create_proof


class PatronSigner:
    """Holds a patron's ``(npub, nsec)`` and authenticates its outgoing operator calls.

    ``nsec`` may be empty for a caller that legitimately presents no proof (then the
    proof is ``""``); otherwise every call mints a fresh, tool-bound kind-27235 proof.
    """

    __slots__ = ("_npub", "_nsec")

    def __init__(self, npub: str, nsec: str = "") -> None:
        self._npub = npub
        self._nsec = nsec

    @property
    def npub(self) -> str:
        return self._npub

    def proof(self, tool_name: str) -> str:
        """A fresh in-memory kind-27235 proof bound to ``tool_name`` (``""`` if no nsec)."""
        return create_proof(self._nsec, tool_name) if self._nsec else ""

    def authenticate(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return ``arguments`` with the patron's ``npub`` (if unset) and a fresh
        ``dpop_token`` bound to ``tool_name`` injected — the payload for one operator call."""
        args = dict(arguments or {})
        args.setdefault("npub", self._npub)
        args["dpop_token"] = self.proof(tool_name)
        return args
