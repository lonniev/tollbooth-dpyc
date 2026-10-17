"""Holder bearer: the MCP authorization spec as the carrier of DPYC proof grants.

A connection authenticates as a *holder* with ``Authorization: Bearer`` — the
way the MCP authorization spec intends — so a proof grant never has to ride in
a tool argument, where it lands in the model's transcript. The holder carries
no identity of its own. Each npub is proven into the holder's wallet with the
ordinary Secure Courier exchange, and every call names the npub it acts as;
the gate then checks that npub's grant exactly as it checks one passed inline.

Three kinds of bearer reach ``verify_token``:

- a holder access token (``dpyc_ha_…``): the holder, or ``None`` when it is
  unknown or expired, which answers 401 so the client refreshes;
- anything else: an *anonymous* caller, which proves itself with the
  ``dpop_token`` argument as before — a client with no OAuth sends any
  placeholder value;
- no header at all: FastMCP answers 401, which is what starts a client's
  Authorization Code connect.

Connecting asks nothing of the person: a holder holds no npub until one is
proven into it, so ``authorize`` approves at once.
"""

from __future__ import annotations

import secrets
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from fastmcp.server.auth.auth import (
    AccessToken,
    ClientRegistrationOptions,
    OAuthProvider,
    RevocationOptions,
)
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from tollbooth.vaults.holder_store import (
    HolderStore,
    HolderStoreUnavailable,
    TokenKind,
    TokenRow,
    token_hash,
)

ANONYMOUS = "anonymous"

PREFIX: dict[TokenKind, str] = {"code": "dpyc_hc_", "access": "dpyc_ha_", "refresh": "dpyc_hr_"}
TTL_SECONDS: dict[TokenKind, int] = {"code": 5 * 60, "access": 60 * 60, "refresh": 30 * 24 * 3600}

# A verified access token is trusted from memory this long, so a busy
# connection costs one Neon read a minute rather than one per request.
VERIFIED_CACHE_SECONDS = 60
# Client registrations per caller address per hour, per process.
REGISTRATIONS_PER_HOUR = 20


def current_holder() -> str | None:
    """The holder behind the request in flight, or None for an anonymous caller."""
    from fastmcp.server.dependencies import get_access_token

    token = get_access_token()
    if token is None or token.client_id == ANONYMOUS:
        return None
    return token.subject


class HolderAuthProvider(OAuthProvider):
    """OAuth 2.1 authorization server whose subject is a holder, not a person.

    Clients register themselves (RFC 7591); every code and refresh token is
    single-use, and presenting a refresh token a second time disconnects the
    holder, as OAuth 2.1 requires of public clients.
    """

    def __init__(self, store: Callable[[], Awaitable[HolderStore]], *, base_url: str) -> None:
        super().__init__(
            base_url=base_url,
            client_registration_options=ClientRegistrationOptions(enabled=True),
            revocation_options=RevocationOptions(enabled=True),
        )
        self._store = store
        self._verified: dict[str, tuple[AccessToken, float]] = {}

    def get_middleware(self) -> list:
        return [
            Middleware(_AnswerUnavailable),  # type: ignore[arg-type]
            Middleware(_LimitRegistration),  # type: ignore[arg-type]
            *super().get_middleware(),
        ]

    async def _forget_holder(self, holder_id: str) -> None:
        await (await self._store()).forget_holder(holder_id)
        self._verified = {k: v for k, v in self._verified.items() if v[0].subject != holder_id}

    # -- Token mechanics: one way in, one way out --------------------------

    async def _issue(
        self, kind: TokenKind, holder_id: str, client_id: str, meta: dict[str, Any],
    ) -> tuple[str, int]:
        token = PREFIX[kind] + secrets.token_urlsafe(32)
        expires_at = int(time.time()) + TTL_SECONDS[kind]
        await (await self._store()).put_token(
            token, kind, holder_id=holder_id, client_id=client_id,
            expires_at=expires_at, meta=meta,
        )
        return token, expires_at

    async def _redeem(self, kind: TokenKind, token: str, client_id: str) -> TokenRow:
        row = await (await self._store()).spend_token(token, kind)
        if row is None or row.client_id != client_id:
            raise TokenError("invalid_grant", f"{kind} is unknown, expired or already used")
        return row

    async def _token_pair(self, holder_id: str, client_id: str, meta: dict[str, Any]) -> OAuthToken:
        store = await self._store()
        await store.purge_expired()
        access, _ = await self._issue("access", holder_id, client_id, meta)
        refresh, _ = await self._issue("refresh", holder_id, client_id, meta)
        return OAuthToken(
            access_token=access,
            token_type="Bearer",
            expires_in=TTL_SECONDS["access"],
            refresh_token=refresh,
            scope=" ".join(meta.get("scopes", [])) or None,
        )

    # -- Clients ------------------------------------------------------------

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        info = await (await self._store()).client(client_id)
        return OAuthClientInformationFull.model_validate_json(info) if info else None

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        if client_info.client_id is None:
            raise ValueError("client_id is required")
        await (await self._store()).save_client(client_info.client_id, client_info.model_dump_json())

    # -- Authorization Code -------------------------------------------------

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        # Authorizing needs no credential, so it also clears what has lapsed:
        # unredeemed codes cannot pile up between token issues.
        await (await self._store()).purge_expired()
        code, _ = await self._issue(
            "code", str(uuid.uuid4()), str(client.client_id),
            {
                "code_challenge": params.code_challenge,
                "redirect_uri": str(params.redirect_uri),
                "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
                "resource": params.resource,
                "scopes": params.scopes or [],
            },
        )
        return construct_redirect_uri(str(params.redirect_uri), code=code, state=params.state)

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str,
    ) -> AuthorizationCode | None:
        row = await (await self._store()).token(authorization_code, "code")
        if row is None or not row.live or row.client_id != client.client_id:
            return None
        return AuthorizationCode(
            code=authorization_code,
            client_id=row.client_id,
            subject=row.holder_id,
            expires_at=row.expires_at,
            **row.meta,
        )

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        row = await self._redeem("code", authorization_code.code, str(client.client_id))
        return await self._token_pair(
            row.holder_id, row.client_id,
            {"resource": row.meta.get("resource"), "scopes": row.meta.get("scopes", [])},
        )

    # -- Refresh (rotating, with reuse detection) ---------------------------

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str,
    ) -> RefreshToken | None:
        store = await self._store()
        row = await store.token(refresh_token, "refresh")
        if row is None or row.client_id != client.client_id:
            return None
        if row.spent:
            # A rotated refresh token came back: whoever holds it is not the
            # client that rotated it. Disconnect the holder.
            await self._forget_holder(row.holder_id)
            return None
        if not row.live:
            return None
        return RefreshToken(
            token=refresh_token,
            client_id=row.client_id,
            subject=row.holder_id,
            expires_at=row.expires_at,
            scopes=row.meta.get("scopes", []),
        )

    async def exchange_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: RefreshToken, scopes: list[str],
    ) -> OAuthToken:
        row = await self._redeem("refresh", refresh_token.token, str(client.client_id))
        return await self._token_pair(
            row.holder_id, row.client_id,
            {"resource": row.meta.get("resource"), "scopes": scopes or row.meta.get("scopes", [])},
        )

    # -- Access -------------------------------------------------------------

    async def verify_token(self, token: str) -> AccessToken | None:
        """Every bearer that is not ours is an anonymous caller, proven by its arguments."""
        if not token.startswith(PREFIX["access"]):
            return AccessToken(token=token, client_id=ANONYMOUS, scopes=[])
        key, now = token_hash(token), time.time()
        cached = self._verified.get(key)
        if cached is not None and cached[1] > now:
            return cached[0]
        try:
            verified = await self.load_access_token(token)
        except Exception as exc:
            raise HolderStoreUnavailable(str(exc)) from exc
        if verified is not None:
            until = min(now + VERIFIED_CACHE_SECONDS, verified.expires_at or now)
            self._verified[key] = (verified, until)
        return verified

    async def load_access_token(self, token: str) -> AccessToken | None:  # type: ignore[override]
        row = await (await self._store()).token(token, "access")
        resource = row.meta.get("resource") if row else None
        if row is None or not row.live or not self._audience_ok(resource):
            return None
        return AccessToken(
            token=token,
            client_id=row.client_id,
            subject=row.holder_id,
            expires_at=row.expires_at,
            resource=resource,
            scopes=row.meta.get("scopes", []),
        )

    def _audience_ok(self, resource: str | None) -> bool:
        """RFC 8707: a token minted for another resource is not good here."""
        if resource is None or self._resource_url is None:
            return True
        return resource.rstrip("/") == str(self._resource_url).rstrip("/")

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:  # type: ignore[override]
        if token.subject:
            await self._forget_holder(token.subject)


class _AnswerUnavailable:
    """A holder store that cannot answer is a retry, never a 500 or a 401.

    A 401 would send the client to refresh, then to reconnect — asking a
    person to sign in again because the database was waking up.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await self.app(scope, receive, send)
        except HolderStoreUnavailable:
            await JSONResponse(
                {
                    "error": "temporarily_unavailable",
                    "error_description": (
                        "The service is warming up after idle. Retry in 10-15 seconds."
                    ),
                },
                status_code=503,
                headers={"Retry-After": "10"},
            )(scope, receive, send)


class _LimitRegistration:
    """Client registration needs no credential, so it is metered per caller address."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._seen: dict[str, list[float]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"] == "/register" and scope["method"] == "POST":
            now = time.time()
            address = _caller_address(scope)
            recent = [t for t in self._seen.get(address, []) if t > now - 3600]
            if len(recent) >= REGISTRATIONS_PER_HOUR:
                await JSONResponse(
                    {"error": "slow_down", "error_description": "Too many client registrations."},
                    status_code=429,
                    headers={"Retry-After": "3600"},
                )(scope, receive, send)
                return
            self._seen[address] = [*recent, now]
        await self.app(scope, receive, send)


def _caller_address(scope: Scope) -> str:
    """The first hop of X-Forwarded-For, else the socket peer."""
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            return value.decode("latin-1").split(",")[0].strip()
    client = scope.get("client")
    return str(client[0]) if client else ""
