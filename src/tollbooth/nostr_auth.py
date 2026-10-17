"""Sign every HTTP request with an nsec (NIP-98), for a client that holds one.

Whoever holds an npub's key needs no grant: each request carries its own
proof, an ``Authorization: Nostr`` event bound to the method, the URL and the
body. The server learns who signed and acts as that npub for that request.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import httpx

from tollbooth.identity_proof import sign_request


def _auth_bases() -> tuple[type, ...]:
    """The ``Auth`` base of every HTTP client library MCP may be running on.

    FastMCP 3 drives httpx; FastMCP 4 drives httpx2, whose client refuses an
    ``httpx.Auth``. Deriving from both lets one signer serve either.
    """
    bases: list[type] = [httpx.Auth]
    try:
        import httpx2  # type: ignore[import-not-found]

        bases.append(httpx2.Auth)
    except ImportError:
        pass
    return tuple(bases)


class NostrRequestAuth(*_auth_bases()):  # type: ignore[misc]
    """Signs each request with one nsec."""

    requires_request_body = True

    def __init__(self, nsec: str) -> None:
        self._nsec = nsec

    def auth_flow(self, request: Any) -> Generator[Any, Any, None]:
        request.headers["Authorization"] = sign_request(
            self._nsec, method=request.method, url=str(request.url), body=request.content,
        )
        yield request
