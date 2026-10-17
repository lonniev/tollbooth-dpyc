"""Identity on the connection: who a request may act as, carried in its header.

Every tool call names the npub it acts as; the ``Authorization`` header says
whether the connection may:

- ``Nostr <event>`` — the request is signed by the npub's own key (NIP-98).
  Whoever holds the key needs no grant; the signature is the proof, bound to
  this request's method, path and body.
- ``Bearer dpyc_ha_…`` — a *holder*: a connection with no identity of its own,
  into whose wallet npubs are proven one Secure Courier exchange at a time.
  It acts as any npub its wallet holds, by that npub's grant. Unknown or
  expired, it answers 401 so the client refreshes.
- any other bearer — an anonymous caller, which acts as no npub (a client
  with no OAuth sends any placeholder value);
- no header at all — FastMCP answers 401, which is what starts a client's
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
from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import (
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from starlette.authentication import AuthCredentials, AuthenticationBackend
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tollbooth.vaults.holder_store import (
    HolderStore,
    HolderStoreUnavailable,
    TokenKind,
    TokenRow,
    token_hash,
)

ANONYMOUS = "anonymous"
SIGNED = "nostr-signed"

PREFIX: dict[TokenKind, str] = {"code": "dpyc_hc_", "access": "dpyc_ha_", "refresh": "dpyc_hr_"}
TTL_SECONDS: dict[TokenKind, int] = {"code": 5 * 60, "access": 60 * 60, "refresh": 30 * 24 * 3600}

# A verified access token is trusted from memory this long, so a busy
# connection costs one Neon read a minute rather than one per request.
VERIFIED_CACHE_SECONDS = 60
# Client registrations per caller address per hour, per process.
REGISTRATIONS_PER_HOUR = 20
# The largest request body a signed request may carry: its hash is what the
# signature binds, so the body is read whole before the request is admitted.
MAX_SIGNED_BODY_BYTES = 32 * 1024 * 1024
_BODY_SHA256 = "dpyc.body_sha256"


def _request_token() -> AccessToken | None:
    from fastmcp.server.dependencies import get_access_token

    return get_access_token()


def current_holder() -> str | None:
    """The holder behind the request in flight, or None when it is not a holder."""
    token = _request_token()
    if token is None or token.client_id in (ANONYMOUS, SIGNED):
        return None
    return token.subject


def signed_npub() -> str | None:
    """The npub whose key signed the request in flight, or None."""
    token = _request_token()
    if token is None or token.client_id != SIGNED:
        return None
    return str((token.claims or {}).get("npub") or "") or None


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
            Middleware(_HashSignedBody),  # type: ignore[arg-type]
            Middleware(AuthenticationMiddleware, backend=_ConnectionIdentity(self)),  # type: ignore[arg-type]
            Middleware(AuthContextMiddleware),  # type: ignore[arg-type]
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


class _ConnectionIdentity(AuthenticationBackend):
    """Reads who a request may act as from its ``Authorization`` header."""

    def __init__(self, provider: HolderAuthProvider) -> None:
        self._provider = provider

    async def authenticate(self, conn: HTTPConnection) -> tuple[AuthCredentials, AuthenticatedUser] | None:
        scheme, _, value = (conn.headers.get("authorization") or "").partition(" ")
        if scheme.lower() == "bearer":
            token = await self._provider.verify_token(value.strip())
        elif scheme.lower() == "nostr":
            token = _signed_request_token(conn, value)
        else:
            return None
        if token is None or (token.expires_at and token.expires_at < int(time.time())):
            return None
        return AuthCredentials(token.scopes), AuthenticatedUser(token)


def _signed_request_token(conn: HTTPConnection, value: str) -> AccessToken | None:
    import hashlib

    from tollbooth.identity_proof import verify_request_signature

    npub, _ = verify_request_signature(
        value,
        method=conn.scope["method"],
        path=conn.url.path,
        body_sha256=conn.scope.get(_BODY_SHA256) or hashlib.sha256(b"").hexdigest(),
    )
    if npub is None:
        return None
    return AccessToken(token=value, client_id=SIGNED, scopes=[], claims={"npub": npub})


class _HashSignedBody:
    """Reads a signed request's body ahead of authentication, so the signature's
    ``payload`` hash can be checked, then hands the body on unchanged."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _is_signed(scope):
            await self.app(scope, receive, send)
            return
        import hashlib

        digest, held, size = hashlib.sha256(), [], 0
        while True:
            message = await receive()
            held.append(message)
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > MAX_SIGNED_BODY_BYTES:
                await JSONResponse({"error": "request body too large"}, status_code=413)(scope, receive, send)
                return
            digest.update(chunk)
            if message["type"] != "http.request" or not message.get("more_body"):
                break
        scope[_BODY_SHA256] = digest.hexdigest()

        async def replay() -> Message:
            return held.pop(0) if held else await receive()

        await self.app(scope, replay, send)


def _is_signed(scope: Scope) -> bool:
    for name, value in scope.get("headers", []):
        if name == b"authorization":
            return value[:6].lower() == b"nostr "
    return False

