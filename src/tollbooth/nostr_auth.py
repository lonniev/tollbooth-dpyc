"""Sign every HTTP request with an nsec (NIP-98), for a client that holds one.

Whoever holds an npub's key needs no grant: each request carries its own
proof, an ``Authorization: Nostr`` event bound to the method, the URL and the
body. The server learns who signed and acts as that npub for that request.
"""

from __future__ import annotations

from collections.abc import Generator

import httpx

from tollbooth.identity_proof import sign_request


class NostrRequestAuth(httpx.Auth):
    """httpx auth that signs each request with one nsec."""

    requires_request_body = True

    def __init__(self, nsec: str) -> None:
        self._nsec = nsec

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = sign_request(
            self._nsec, method=request.method, url=str(request.url), body=request.content,
        )
        yield request
