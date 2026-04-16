"""OAuth2 provider configuration for standard begin_oauth / check_oauth_status tools.

Operators supply provider-specific constants. The wheel handles tool
registration, PKCE, collector polling, vault persistence, and session
activation.

Usage::

    from tollbooth.oauth_config import OAuthProviderConfig

    runtime = OperatorRuntime(
        oauth_provider=OAuthProviderConfig(
            authorize_url="https://api.schwabapi.com/v1/oauth/authorize",
            token_url="https://api.schwabapi.com/v1/oauth/token",
            scopes="readonly",
        ),
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable


@dataclass(frozen=True)
class OAuthProviderConfig:
    """Provider-specific OAuth2 configuration.

    Args:
        authorize_url: The provider's OAuth2 authorization endpoint.
        token_url: The provider's OAuth2 token endpoint.
        scopes: Space-separated OAuth2 scopes.
        pkce: Whether to use PKCE (required by X/Twitter, optional for Schwab).
        on_token_received: Optional async callback ``(npub, token_dict) -> dict``
            called after successful token exchange. Return dict is merged into
            vault storage (e.g., Schwab returns ``{"account_hash": "..."}``).
        refresh_enabled: Whether to auto-refresh expired tokens on session restore.
        service_name: Vault service name for patron token storage
            (defaults to "oauth").
    """

    authorize_url: str
    token_url: str
    scopes: str = ""
    pkce: bool = False
    on_token_received: Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]] | None = None
    refresh_enabled: bool = True
    service_name: str = "oauth"
    client_id_field: str = "client_id"
    client_secret_field: str = "client_secret"
