"""Neon persistence for connection holders: OAuth clients, tokens and the grant wallet.

A *holder* is one connection that carries ``Authorization: Bearer`` — a
Claude.ai connector, a frontend, a script. It holds no identity of its own;
npubs are proven into its *wallet* one Secure Courier exchange at a time, and
each tool call names the npub it acts as.

Pure persistence: the OAuth rules live in ``tollbooth.holder_auth``. Tokens
are stored only as ``sha256(token)``; client records and wallet envelopes are
bearer secrets, so both are encrypted with the vault's cipher.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Literal

TokenKind = Literal["code", "access", "refresh"]


class HolderStoreUnavailable(Exception):
    """The holder store cannot answer; the caller should retry, never assume."""


def token_hash(token: str) -> str:
    """The only form in which a token is ever stored."""
    return hashlib.sha256(token.encode()).hexdigest()


@dataclass(frozen=True)
class TokenRow:
    """A stored token: whose it is, until when, and whether it was spent."""

    holder_id: str
    client_id: str
    expires_at: int
    spent: bool
    meta: dict[str, Any]

    @property
    def live(self) -> bool:
        return not self.spent and self.expires_at > time.time()


def schema_statements(t: Any, idx: str) -> list[str]:
    """Holders: registered OAuth clients, their tokens, and their grant wallets."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t('oauth_clients')} ("
            "    client_id TEXT PRIMARY KEY,"
            "    info_enc TEXT NOT NULL,"
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        ),
        # Codes, access and refresh tokens share one table: a code is just a
        # short-lived token. `spent` marks a redeemed code or a rotated refresh
        # token, so a second presentation is recognised as reuse.
        (
            f"CREATE TABLE IF NOT EXISTS {t('holder_tokens')} ("
            "    token_hash TEXT PRIMARY KEY,"
            "    kind TEXT NOT NULL,"
            "    holder_id TEXT NOT NULL,"
            "    client_id TEXT NOT NULL,"
            "    expires_at BIGINT NOT NULL,"
            "    spent BOOLEAN NOT NULL DEFAULT false,"
            "    meta JSONB NOT NULL DEFAULT '{}'::jsonb"
            ")"
        ),
        f"CREATE INDEX IF NOT EXISTS {idx}idx_holder_tokens_holder ON {t('holder_tokens')}(holder_id)",
        (
            f"CREATE TABLE IF NOT EXISTS {t('holder_wallet')} ("
            "    holder_id TEXT NOT NULL,"
            "    npub TEXT NOT NULL,"
            "    envelope_enc TEXT NOT NULL,"
            "    expires_at BIGINT NOT NULL,"
            "    PRIMARY KEY (holder_id, npub)"
            ")"
        ),
        f"CREATE INDEX IF NOT EXISTS {idx}idx_holder_wallet_npub ON {t('holder_wallet')}(npub)",
    ]


class HolderStore:
    """Neon-backed store for holders. All state lives in Postgres."""

    def __init__(self, vault: Any) -> None:
        # Envelopes and client records are bearer secrets: no cipher, no store.
        if getattr(vault, "_cipher", None) is None:
            raise HolderStoreUnavailable("the vault has no cipher; holder secrets would be stored in the clear")
        self._vault = vault

    def _t(self, table: str) -> str:
        return self._vault._t(table)

    async def _rows(self, query: str, params: list[Any]) -> list[dict[str, Any]]:
        result = await self._vault._execute(query, params)
        return list(result.get("rows") or [])

    # -- OAuth clients ---------------------------------------------------

    async def save_client(self, client_id: str, info_json: str) -> None:
        await self._vault._execute(
            f"INSERT INTO {self._t('oauth_clients')} (client_id, info_enc) VALUES ($1, $2) "
            "ON CONFLICT (client_id) DO UPDATE SET info_enc = EXCLUDED.info_enc",
            [client_id, self._vault._encrypt(info_json)],
        )

    async def client(self, client_id: str) -> str | None:
        rows = await self._rows(
            f"SELECT info_enc FROM {self._t('oauth_clients')} WHERE client_id = $1",
            [client_id],
        )
        return self._vault._decrypt(rows[0]["info_enc"]) if rows else None

    # -- Tokens ------------------------------------------------------------

    async def put_token(
        self,
        token: str,
        kind: TokenKind,
        *,
        holder_id: str,
        client_id: str,
        expires_at: int,
        meta: dict[str, Any],
    ) -> None:
        await self._vault._execute(
            f"INSERT INTO {self._t('holder_tokens')} "
            "(token_hash, kind, holder_id, client_id, expires_at, meta) "
            "VALUES ($1, $2, $3, $4, $5, $6::jsonb)",
            [token_hash(token), kind, holder_id, client_id, int(expires_at), json.dumps(meta)],
        )

    async def token(self, token: str, kind: TokenKind) -> TokenRow | None:
        """Read a token without spending it."""
        rows = await self._rows(
            f"SELECT holder_id, client_id, expires_at, spent, meta "
            f"FROM {self._t('holder_tokens')} WHERE token_hash = $1 AND kind = $2",
            [token_hash(token), kind],
        )
        return _row(rows[0]) if rows else None

    async def spend_token(self, token: str, kind: TokenKind) -> TokenRow | None:
        """Spend a live token atomically; None if it was already spent, expired or unknown.

        Two concurrent redemptions of one code or refresh token: exactly one wins.
        """
        rows = await self._rows(
            f"UPDATE {self._t('holder_tokens')} SET spent = true "
            "WHERE token_hash = $1 AND kind = $2 AND NOT spent AND expires_at > $3 "
            "RETURNING holder_id, client_id, expires_at, spent, meta",
            [token_hash(token), kind, int(time.time())],
        )
        return _row(rows[0]) if rows else None

    async def purge_expired(self) -> None:
        await self._vault._execute(
            f"DELETE FROM {self._t('holder_tokens')} WHERE expires_at < $1",
            [int(time.time())],
        )

    # -- Holders and their wallets ----------------------------------------

    async def hold(self, holder_id: str, npub: str, envelope: str, expires_at: int) -> None:
        """Put a proven npub's grant envelope in the holder's wallet (replacing any older one)."""
        await self._vault._execute(
            f"INSERT INTO {self._t('holder_wallet')} (holder_id, npub, envelope_enc, expires_at) "
            "VALUES ($1, $2, $3, $4) ON CONFLICT (holder_id, npub) DO UPDATE "
            "SET envelope_enc = EXCLUDED.envelope_enc, expires_at = EXCLUDED.expires_at",
            [holder_id, npub, self._vault._encrypt(envelope), int(expires_at)],
        )

    async def envelope(self, holder_id: str, npub: str) -> str | None:
        """The live grant envelope this holder holds for npub, if any."""
        rows = await self._rows(
            f"SELECT envelope_enc FROM {self._t('holder_wallet')} "
            "WHERE holder_id = $1 AND npub = $2 AND expires_at > $3",
            [holder_id, npub, int(time.time())],
        )
        return self._vault._decrypt(rows[0]["envelope_enc"]) if rows else None

    async def holdings(self, holder_id: str) -> list[dict[str, Any]]:
        """Every live npub in the holder's wallet with its expiry — never the envelope."""
        rows = await self._rows(
            f"SELECT npub, expires_at FROM {self._t('holder_wallet')} "
            "WHERE holder_id = $1 AND expires_at > $2 ORDER BY npub",
            [holder_id, int(time.time())],
        )
        return [{"npub": r["npub"], "expires_at": int(r["expires_at"])} for r in rows]

    async def release(self, npub: str) -> None:
        """Drop npub from every wallet — its grants were revoked."""
        await self._vault._execute(
            f"DELETE FROM {self._t('holder_wallet')} WHERE npub = $1", [npub],
        )

    async def forget_holder(self, holder_id: str) -> None:
        """Disconnect a holder: every token it has and its whole wallet."""
        for table in ("holder_tokens", "holder_wallet"):
            await self._vault._execute(
                f"DELETE FROM {self._t(table)} WHERE holder_id = $1", [holder_id],
            )


def _row(r: dict[str, Any]) -> TokenRow:
    meta = r.get("meta") or {}
    return TokenRow(
        holder_id=str(r["holder_id"]),
        client_id=str(r["client_id"]),
        expires_at=int(r["expires_at"]),
        spent=bool(r["spent"]),
        meta=json.loads(meta) if isinstance(meta, str) else dict(meta),
    )
