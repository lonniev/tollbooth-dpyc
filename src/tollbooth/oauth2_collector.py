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

import base64
import hashlib
import json
import logging
import time
import urllib.parse

import httpx

from tollbooth.shortlinks import _ensure_mcp_path

logger = logging.getLogger(__name__)


class OAuthCollectorError(Exception):
    """Base exception for OAuth2 collector operations."""


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
        httpx.HTTPStatusError: If the token endpoint returns an error status.
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    body: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    if code_verifier:
        body["code_verifier"] = code_verifier

    async with httpx.AsyncClient() as http:
        resp = await http.post(
            token_endpoint,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            content=urllib.parse.urlencode(body),
        )
        resp.raise_for_status()
        token = resp.json()

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
    """
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

    async with httpx.AsyncClient() as http:
        resp = await http.post(
            token_endpoint,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            content=urllib.parse.urlencode({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }),
        )
        resp.raise_for_status()
        token = resp.json()

    token["expires_at"] = time.time() + token.get("expires_in", 1800)
    return token


# ---------------------------------------------------------------------------
# Collector retrieval + decryption
# ---------------------------------------------------------------------------


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
        text = resp.text
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                data = json.loads(line[6:])
                content = data.get("result", {}).get("structuredContent", {})
                if not content.get("found"):
                    return None
                return decrypt_collector_code(content["code"], state_token)
    return None
