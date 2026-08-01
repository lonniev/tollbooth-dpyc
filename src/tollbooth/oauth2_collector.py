"""Generic OAuth2 Authorization Code flow helpers for Tollbooth MCP services.

Builds authorization URLs, exchanges codes for tokens, and retrieves
encrypted codes from an external OAuth2 collector. Uses the patron's
npub as the OAuth state parameter — no server-side pending-state storage
needed. The collector encrypts the auth code with SHA-256(state) and this
module decrypts it on retrieval.

Provider-agnostic: callers supply authorize_endpoint, token_endpoint,
scope, etc. MCP services build thin wrappers that bind provider-specific
constants (e.g. Schwab URLs, scopes).

Not exported from ``__init__.py`` — import directly::

    from tollbooth.oauth2_collector import begin_oauth_flow, retrieve_code_from_collector
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import urllib.parse

import httpx

from tollbooth.shortlinks import _ensure_mcp_path

logger = logging.getLogger(__name__)


class OAuthCollectorError(Exception):
    """Base exception for OAuth2 collector operations."""


class OAuthRefreshDenied(OAuthCollectorError):
    """The provider REFUSED the request outright: no retry will ever help.

    The only situation in this module that a human must act on — but *which*
    human depends on ``fault``, and that distinction is the whole point of the
    field. RFC 6749 §5.2 names four refusals and they do not share a culprit:

    * ``fault="grant"`` (``invalid_grant``) — the patron's authorization is
      genuinely dead. This is the ONE case where re-authorizing is the answer.
    * ``fault="client"`` (``invalid_client``, ``unauthorized_client``) — the
      OPERATOR's app credentials are wrong, revoked, or not entitled to this
      grant type. The patron's authorization is untouched, and sending them
      through OAuth again re-authorizes a live account against a broken app.
    * ``fault="request"`` (``invalid_request``) — we sent a malformed request.
      That is our bug, and no human outside the deployment can fix it.

    All four used to raise this exception undifferentiated, so all four reached
    the patron as "your session expired". An operator who rotated an app secret
    watched every patron get told to reconnect.
    """

    def __init__(
        self,
        detail: str,
        *,
        status_code: int = 0,
        oauth_error: str = "",
        fault: str = "grant",
    ):
        self.detail = detail
        self.status_code = status_code
        self.oauth_error = oauth_error
        self.fault = fault
        super().__init__(detail)


class OAuthRefreshUnavailable(OAuthCollectorError):
    """The refresh could not be *completed* — which says nothing about the grant.

    A timeout, a connection failure, a 429, a 5xx, a body that isn't JSON. The
    stored refresh token may be perfectly good; we simply don't know yet, so the
    caller must report "try again", never "re-authorize".

    ``token_may_have_rotated`` is the honest part. Providers that rotate
    single-use refresh tokens (X does) rotate them when the request *arrives*,
    not when we read the answer — so a read timeout can leave the provider
    holding a new token we never saw, and our stored one already spent. When
    that's possible we say so instead of pretending the failure was clean.
    """

    def __init__(
        self,
        detail: str,
        *,
        status_code: int = 0,
        token_may_have_rotated: bool = False,
    ):
        self.detail = detail
        self.status_code = status_code
        self.token_may_have_rotated = token_may_have_rotated
        super().__init__(detail)


# ---------------------------------------------------------------------------
# Token-endpoint transport
# ---------------------------------------------------------------------------

# A token endpoint is not a CDN. httpx's bare default gives 5 seconds to EVERY
# phase, which is a thin margin for an outbound POST to a provider's auth host —
# and the consequence of losing that race is not a slow page, it is a patron
# told to re-authorize a session that was never broken.
TOKEN_ENDPOINT_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)

# Retried ONLY when the connection never opened, which is the one failure that
# provably didn't reach the provider. A read timeout is left to fail: the
# provider may have already rotated a single-use refresh token, and asking again
# with the spent one is how a working grant gets revoked.
_TOKEN_CONNECT_ATTEMPTS = 3
_TOKEN_CONNECT_BACKOFF_S = 1.0

# OAuth2 error codes (RFC 6749 §5.2), sorted by who has to fix them. All three
# sets are terminal — no retry helps — but they are three different errands, and
# only the first belongs to the patron. Everything not listed here is treated as
# "we don't know yet" and reported as unavailable.
_GRANT_IS_DEAD = frozenset({"invalid_grant"})
_CLIENT_IS_AT_FAULT = frozenset({"invalid_client", "unauthorized_client"})
_REQUEST_IS_MALFORMED = frozenset({"invalid_request"})

# oauth_error → the `fault` attributed to OAuthRefreshDenied.
_FAULT_BY_OAUTH_ERROR: dict[str, str] = {
    **{e: "grant" for e in _GRANT_IS_DEAD},
    **{e: "client" for e in _CLIENT_IS_AT_FAULT},
    **{e: "request" for e in _REQUEST_IS_MALFORMED},
}


async def _post_token_endpoint(
    token_endpoint: str, credentials: str, body: dict[str, str],
) -> httpx.Response:
    """POST a form-encoded grant request, retrying only an unopened connection."""
    last: Exception | None = None
    for attempt in range(_TOKEN_CONNECT_ATTEMPTS):
        try:
            async with httpx.AsyncClient(timeout=TOKEN_ENDPOINT_TIMEOUT) as http:
                return await http.post(
                    token_endpoint,
                    headers={
                        "Authorization": f"Basic {credentials}",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    content=urllib.parse.urlencode(body),
                )
        except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
            last = exc
            if attempt + 1 < _TOKEN_CONNECT_ATTEMPTS:
                await asyncio.sleep(_TOKEN_CONNECT_BACKOFF_S * (attempt + 1))
    raise last  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def generate_pkce_pair() -> tuple[str, str]:
    """Generate a PKCE code_verifier and code_challenge (S256).

    Returns:
        (code_verifier, code_challenge) tuple. The verifier is a
        cryptographically random 128-char URL-safe string. The challenge
        is its SHA-256 hash, base64url-encoded without padding.
    """
    import secrets
    verifier = secrets.token_urlsafe(96)[:128]
    challenge = (
        base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode("ascii")).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    return verifier, challenge


# ---------------------------------------------------------------------------
# URL builder
# ---------------------------------------------------------------------------


def build_authorize_url(
    authorize_endpoint: str,
    client_id: str,
    redirect_uri: str,
    state: str,
    *,
    scope: str | None = None,
    extra_params: dict[str, str] | None = None,
) -> str:
    """Construct an OAuth2 authorization URL.

    Args:
        authorize_endpoint: Full URL of the provider's authorize endpoint.
        client_id: OAuth2 client / app key.
        redirect_uri: Registered redirect URI (typically the collector callback).
        state: Opaque state token (the patron's npub in Tollbooth flows).
        scope: Optional OAuth2 scope string (e.g. ``"readonly"``).
        extra_params: Additional query parameters to include.

    Returns:
        Fully-encoded authorization URL string.
    """
    params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if scope is not None:
        params["scope"] = scope
    if extra_params:
        params.update(extra_params)
    return f"{authorize_endpoint}?{urllib.parse.urlencode(params)}"


# ---------------------------------------------------------------------------
# Begin flow
# ---------------------------------------------------------------------------


def begin_oauth_flow(
    patron_npub: str,
    client_id: str,
    redirect_uri: str,
    authorize_endpoint: str,
    *,
    scope: str | None = None,
    provider_name: str = "the provider",
    extra_params: dict[str, str] | None = None,
) -> dict:
    """Start a new OAuth flow. Returns the authorization URL and status dict.

    Uses the patron's npub as the OAuth state parameter — the collector
    encrypts the auth code with SHA-256(npub) and the MCP server decrypts
    it on retrieval.

    Args:
        patron_npub: Patron Nostr public key (``npub1...``).
        client_id: OAuth2 client / app key.
        redirect_uri: Registered redirect URI.
        authorize_endpoint: Provider's authorize URL.
        scope: Optional OAuth2 scope.
        provider_name: Human-readable name for status messages.
        extra_params: Additional query parameters.

    Returns:
        Dict with ``status``, ``authorize_url``, and ``message`` keys.
    """
    url = build_authorize_url(
        authorize_endpoint,
        client_id,
        redirect_uri,
        patron_npub,
        scope=scope,
        extra_params=extra_params,
    )
    return {
        "status": "pending",
        "authorize_url": url,
        "message": (
            f"Open this URL in your browser to authorize with {provider_name}. "
            "After authorizing, call check_oauth_status to confirm."
        ),
    }


# ---------------------------------------------------------------------------
# Token exchange
# ---------------------------------------------------------------------------


async def exchange_code_for_token(
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    token_endpoint: str,
    *,
    code_verifier: str | None = None,
) -> dict:
    """Exchange an authorization code for an access/refresh token pair.

    POST to ``token_endpoint`` with Basic Auth and
    ``grant_type=authorization_code``. Adds a computed ``expires_at``
    field to the returned token dict.

    Args:
        code: The authorization code from the collector.
        client_id: OAuth2 client / app key.
        client_secret: OAuth2 client secret.
        redirect_uri: Redirect URI used during authorization.
        token_endpoint: Provider's token endpoint URL.
        code_verifier: PKCE code_verifier (required for PKCE flows).

    Returns:
        Token dict with ``access_token``, ``refresh_token``, ``expires_at``, etc.

    Raises:
        OAuthRefreshDenied: The provider refused the exchange outright. Read
            ``fault`` / ``oauth_error`` / ``detail`` — the same evidence the
            refresh path surfaces — so a 400 is never a blind
            ``HTTPStatusError``. Only ``fault="grant"`` means the patron must
            start the browser dance again.
        OAuthRefreshUnavailable: The exchange didn't complete (timeout,
            connect failure, 429, 5xx, unparseable body). Retry; the code may
            still be spendable, or the provider may simply have been busy.
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    body: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    if code_verifier:
        body["code_verifier"] = code_verifier

    resp = await _token_endpoint_response(
        token_endpoint, credentials, body, secret_to_redact=code,
    )
    try:
        token = resp.json()
    except Exception as exc:  # a 200 that isn't JSON is not a refusal
        raise OAuthRefreshUnavailable(
            f"token endpoint returned non-JSON on {resp.status_code}: {exc}",
            status_code=resp.status_code,
        ) from exc

    token["expires_at"] = time.time() + token.get("expires_in", 1800)
    return token


async def refresh_access_token(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    token_endpoint: str,
) -> dict:
    """Refresh an expired access token using a refresh token.

    Args:
        client_id: OAuth2 client / app key.
        client_secret: OAuth2 client secret.
        refresh_token: The refresh token from the original authorization.
        token_endpoint: Provider's token endpoint URL.

    Returns:
        New token dict with ``access_token``, ``expires_at``, and
        optionally a rotated ``refresh_token``.

    Raises:
        OAuthRefreshDenied: The provider refused outright; no retry will ever
            help. Read its ``fault`` to learn whose problem it is — only
            ``"grant"`` means the patron must re-authorize.
        OAuthRefreshUnavailable: The refresh didn't complete (timeout, connect
            failure, 429, 5xx, unparseable body). The grant may be perfectly
            fine — retry, and do NOT send the patron back through OAuth.

    Why the two are separated: this function is the only place that sees the
    HTTP status and the provider's error body, so it is the only place that
    *can* tell "your session is gone" from "the network was busy". Collapsing
    both into one exception is what made a five-second blip read to a patron as
    "your X access expired".
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    resp = await _token_endpoint_response(
        token_endpoint,
        credentials,
        {"grant_type": "refresh_token", "refresh_token": refresh_token},
        secret_to_redact=refresh_token,
        # A read/write timeout means the request DID arrive. If this provider
        # rotates single-use refresh tokens, ours may already be spent.
        read_timeout_may_rotate=True,
    )

    try:
        token = resp.json()
    except Exception as exc:  # a 200 that isn't JSON is not a refusal
        raise OAuthRefreshUnavailable(
            f"token endpoint returned non-JSON on {resp.status_code}: {exc}",
            status_code=resp.status_code,
            token_may_have_rotated=True,
        ) from exc

    token["expires_at"] = time.time() + token.get("expires_in", 1800)
    return token


async def _token_endpoint_response(
    token_endpoint: str,
    credentials: str,
    body: dict[str, str],
    *,
    secret_to_redact: str = "",
    read_timeout_may_rotate: bool = False,
) -> httpx.Response:
    """POST to the token endpoint and classify any refusal the way refresh does.

    Shared by :func:`exchange_code_for_token` and :func:`refresh_access_token`
    so a 400 never escapes as a bare ``HTTPStatusError`` that threw away the
    provider's body. On success returns the response; on failure raises
    :class:`OAuthRefreshDenied` or :class:`OAuthRefreshUnavailable` with
    ``fault`` / ``oauth_error`` / ``detail`` filled in from the RFC 6749 §5.2
    body (when the provider sent one).
    """
    try:
        resp = await _post_token_endpoint(token_endpoint, credentials, body)
    except (httpx.ConnectTimeout, httpx.ConnectError) as exc:
        # Every attempt failed to open a connection, so the provider never saw
        # the request and any single-use secret we hold is untouched.
        raise OAuthRefreshUnavailable(
            f"token endpoint unreachable: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        raise OAuthRefreshUnavailable(
            f"token endpoint did not answer: {exc}",
            token_may_have_rotated=read_timeout_may_rotate,
        ) from exc

    if resp.status_code >= 400:
        oauth_error, detail = _token_error_fields(resp)
        # These details are logged and returned. A provider that echoes the
        # rejected grant or auth code back in its error body would otherwise
        # write a live credential into the operator's logs — so the secret
        # never survives the trip out.
        detail = _redact(detail, secret_to_redact)
        if fault := _FAULT_BY_OAUTH_ERROR.get(oauth_error):
            raise OAuthRefreshDenied(
                detail, status_code=resp.status_code, oauth_error=oauth_error,
                fault=fault,
            )
        # A 429 or 5xx is the provider asking for patience, not revoking
        # anything. A 4xx we can't name is likelier a transport or gateway
        # artifact than a considered refusal — treat it as unknown rather than
        # sending the patron to re-authorize on a guess.
        raise OAuthRefreshUnavailable(detail, status_code=resp.status_code)

    return resp


def _redact(text: str, secret: str) -> str:
    """*text* with *secret* masked. A short secret is not worth matching on."""
    if not secret or len(secret) < 8:
        return text
    return text.replace(secret, "<redacted>")


def _token_error_fields(resp: httpx.Response) -> tuple[str, str]:
    """``(oauth_error_code, human_detail)`` from a token-endpoint error response.

    The RFC 6749 §5.2 body is ``{"error": ..., "error_description": ...}``. A
    provider that answers with HTML or an empty body yields an empty code, which
    lands in the "we don't know" branch by design.
    """
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        body = {}
    if not isinstance(body, dict):
        body = {}
    code = str(body.get("error", "")).strip()
    described = str(body.get("error_description", "") or body.get("detail", "")).strip()
    detail = described or (resp.text or "")[:200] or f"HTTP {resp.status_code}"
    return code, f"{resp.status_code} {code or 'unspecified'}: {detail}".strip()


# ---------------------------------------------------------------------------
# Collector retrieval + decryption
# ---------------------------------------------------------------------------


def encrypt_collector_code(code: str, state: str) -> str:
    """Encrypt an authorization code for handoff via the collector.

    The canonical peer of :func:`decrypt_collector_code`: both halves of the
    contract live here so the collector service and the retrieving MCP server
    can't drift. AES-256-GCM with key = SHA-256(state), a random 12-byte IV
    prepended to the ciphertext, URL-safe base64 encoded.

    Args:
        code: The plaintext OAuth2 authorization code.
        state: The state token used during authorization (the npub).

    Returns:
        URL-safe base64 string of (IV + ciphertext + tag).
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = hashlib.sha256(state.encode()).digest()
    iv = os.urandom(12)
    aes = AESGCM(key)
    ct = aes.encrypt(iv, code.encode(), None)
    return base64.urlsafe_b64encode(iv + ct).decode()


def decrypt_collector_code(encrypted_b64: str, state: str) -> str:
    """Decrypt an authorization code encrypted by the collector.

    AES-256-GCM with key = SHA-256(state), IV prepended to ciphertext.

    Args:
        encrypted_b64: URL-safe base64-encoded (IV + ciphertext + tag).
        state: The same state token used during authorization (the npub).

    Returns:
        Plaintext authorization code.

    Raises:
        OAuthCollectorError: If decryption fails.
    """
    key = hashlib.sha256(state.encode()).digest()
    payload = base64.urlsafe_b64decode(encrypted_b64)
    if len(payload) < 28:  # 12 IV + 16 tag minimum
        raise OAuthCollectorError("Encrypted code too short.")
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        iv = payload[:12]
        ct = payload[12:]
        aes = AESGCM(key)
        plaintext = aes.decrypt(iv, ct, None)
        return plaintext.decode()
    except Exception as exc:
        raise OAuthCollectorError(
            "Decryption failed — state token may be wrong or code tampered."
        ) from exc


async def retrieve_code_from_collector(
    collector_url: str,
    state_token: str,
) -> str | None:
    """Fetch an authorization code from the external OAuth2 collector.

    The collector stores codes encrypted with SHA-256(state). This function
    retrieves the encrypted code and decrypts it.

    Args:
        collector_url: Base URL of the collector service.
        state_token: The state token (patron npub) used during authorization.

    Returns:
        Plaintext code string, or ``None`` if not yet available.

    Raises:
        OAuthCollectorError: If decryption fails.
        httpx.HTTPStatusError: If the collector returns an unexpected error.
    """
    url = _ensure_mcp_path(collector_url)
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "retrieve_code", "arguments": {"state": state_token}},
        "id": 1,
    }
    async with httpx.AsyncClient() as http:
        resp = await http.post(
            url,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
        )
        if resp.status_code != 200:
            raise OAuthCollectorError(
                f"Collector returned HTTP {resp.status_code}"
            )
        # Response may be SSE (text/event-stream) or plain JSON — our Accept
        # header allows both, so the collector may answer with either. Handle
        # the SSE framing first, then fall back to a plain-JSON body.
        text = resp.text
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                content = data.get("result", {}).get("structuredContent", {})
                if not content.get("found"):
                    return None
                return decrypt_collector_code(content["code"], state_token)

        # Plain JSON response (Content-Type: application/json, no SSE framing).
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            return None
        content = data.get("result", {}).get("structuredContent", {})
        if not content.get("found"):
            return None
        return decrypt_collector_code(content["code"], state_token)
