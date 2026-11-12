"""Neon persistence for peer grants: which npubs a patron lets watch and pay.

A *peer grant* is the patron's signed approval, minted by this Operator as a
kind-30080 event (``identity_credential.sign_peer_grant``), that another npub
may read the patron's account here and pay for them. A patron may approve
several peers. The envelope is self-verifying; this store only finds it.

Pure persistence: the rules live in ``OperatorRuntime.require_caller_or_peer``.
Envelopes are encrypted with the vault's cipher, like the holder wallet.
"""

from __future__ import annotations

import time
from typing import Any


class PeerGrantStoreUnavailable(Exception):
    """The peer grant store cannot answer; the caller should retry, never assume."""


def schema_statements(t: Any, idx: str) -> list[str]:
    """One row per (patron, peer): the newest grant envelope and its expiry."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t('peer_grants')} ("
            "    patron_npub TEXT NOT NULL,"
            "    peer_npub TEXT NOT NULL,"
            "    envelope_enc TEXT NOT NULL,"
            "    expires_at BIGINT NOT NULL,"
            "    PRIMARY KEY (patron_npub, peer_npub)"
            ")"
        ),
        f"CREATE INDEX IF NOT EXISTS {idx}idx_peer_grants_peer ON {t('peer_grants')}(peer_npub)",
    ]


class PeerGrantStore:
    """Neon-backed store for peer grants. All state lives in Postgres."""

    def __init__(self, vault: Any) -> None:
        if getattr(vault, "_cipher", None) is None:
            raise PeerGrantStoreUnavailable("the vault has no cipher; grants would be stored in the clear")
        self._vault = vault

    def _t(self, table: str) -> str:
        return self._vault._t(table)

    async def _rows(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        result = await self._vault._execute(query, params)
        return list(result.get("rows") or [])

    async def put(self, patron_npub: str, peer_npub: str, envelope: str, expires_at: int) -> None:
        """Record the patron's grant to peer (replacing any older one)."""
        await self._vault._execute(
            f"INSERT INTO {self._t('peer_grants')} (patron_npub, peer_npub, envelope_enc, expires_at) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (patron_npub, peer_npub) DO UPDATE "
            "SET envelope_enc = EXCLUDED.envelope_enc, expires_at = EXCLUDED.expires_at",
            [patron_npub, peer_npub, self._vault._encrypt(envelope), int(expires_at)],
        )

    async def get(self, patron_npub: str, peer_npub: str) -> str | None:
        """The live grant envelope from patron to peer, if any."""
        rows = await self._rows(
            f"SELECT envelope_enc FROM {self._t('peer_grants')} "
            "WHERE patron_npub = $1 AND peer_npub = $2 AND expires_at > $3",
            [patron_npub, peer_npub, int(time.time())],
        )
        return self._vault._decrypt(rows[0]["envelope_enc"]) if rows else None

    async def peers_of(self, patron_npub: str) -> list[dict[str, Any]]:
        """The patron's live peers with their expiry — never the envelope."""
        rows = await self._rows(
            f"SELECT peer_npub, expires_at FROM {self._t('peer_grants')} "
            "WHERE patron_npub = $1 AND expires_at > $2 ORDER BY peer_npub",
            [patron_npub, int(time.time())],
        )
        return [{"peer_npub": r["peer_npub"], "expires_at": int(r["expires_at"])} for r in rows]

    async def patrons_of(self, peer_npub: str) -> list[dict[str, Any]]:
        """The patrons who live-granted this peer, with expiry — never the envelope."""
        rows = await self._rows(
            f"SELECT patron_npub, expires_at FROM {self._t('peer_grants')} "
            "WHERE peer_npub = $1 AND expires_at > $2 ORDER BY patron_npub",
            [peer_npub, int(time.time())],
        )
        return [{"patron_npub": r["patron_npub"], "expires_at": int(r["expires_at"])} for r in rows]

    async def remove(self, patron_npub: str, peer_npub: str) -> bool:
        """Revoke one peer. True if a grant was there."""
        rows = await self._rows(
            f"DELETE FROM {self._t('peer_grants')} WHERE patron_npub = $1 AND peer_npub = $2 "
            "RETURNING peer_npub",
            [patron_npub, peer_npub],
        )
        return bool(rows)

    async def remove_all(self, patron_npub: str) -> None:
        """Revoke every peer of the patron."""
        await self._vault._execute(
            f"DELETE FROM {self._t('peer_grants')} WHERE patron_npub = $1", [patron_npub],
        )
