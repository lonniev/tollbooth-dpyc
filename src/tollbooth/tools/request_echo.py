"""request_echo response assembly — what a client sends, seen from the server.

A diagnostic for one question: does a client carry anything per request that
names the conversation it belongs to — a header, a cookie, a query parameter or
a ``_meta`` key that holds steady within a conversation and changes between
them? Calling ``request_echo`` from two conversations and comparing the two
replies answers it.

The reply reaches the calling model, so it lands in a transcript. A value worth
binding a grant to is exactly a value that must stay out of transcripts, so
every value is reported as a short SHA-256 fingerprint: equal fingerprints mean
equal values, and the value itself never leaves the server. Only names that
describe the client software or the protocol (``User-Agent``, the MCP protocol
version, ``clientInfo``) are shown in the clear.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from typing import Any

# Headers whose values describe the client software or the transport, never a
# person or a conversation. Every other header value is fingerprinted.
_CLEAR_HEADERS = frozenset({
    "accept",
    "accept-encoding",
    "content-type",
    "content-length",
    "host",
    "mcp-method",
    "mcp-name",
    "mcp-protocol-version",
    "user-agent",
    "x-forwarded-proto",
})

# ``_meta`` keys whose whole subtree describes the client software or the
# protocol. Everything else is fingerprinted leaf by leaf.
_CLEAR_META_KEYS = frozenset({
    "io.modelcontextprotocol/protocolVersion",
    "io.modelcontextprotocol/clientInfo",
    "io.modelcontextprotocol/clientCapabilities",
})

_MAX_DEPTH = 6


def fingerprint(value: Any) -> str:
    """A 12-hex SHA-256 prefix of ``value`` — comparable, never reversible."""
    return "sha256:" + hashlib.sha256(str(value).encode()).hexdigest()[:12]


def _fingerprint_tree(value: Any, depth: int = 0) -> Any:
    """Keep the shape of a JSON value; replace every leaf with its fingerprint."""
    if depth >= _MAX_DEPTH:
        return fingerprint(value)
    if isinstance(value, Mapping):
        return {str(k): _fingerprint_tree(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_fingerprint_tree(v, depth + 1) for v in value]
    if value is None:
        return None
    return fingerprint(value)


def build_request_echo(
    *,
    method: str,
    path: str,
    query: Iterable[tuple[str, str]],
    headers: Iterable[tuple[str, str]],
    cookies: Mapping[str, str],
    meta: Mapping[str, Any] | None,
    client_info: Mapping[str, Any] | None,
    process_id: int,
) -> dict[str, Any]:
    """Shape the echo of one request. Pure: no I/O, no runtime."""
    echoed_headers: dict[str, str] = {}
    for name, value in headers:
        key = name.lower()
        echoed_headers[key] = value if key in _CLEAR_HEADERS else fingerprint(value)

    echoed_meta: dict[str, Any] | None = None
    if meta is not None:
        echoed_meta = {
            str(k): (v if k in _CLEAR_META_KEYS else _fingerprint_tree(v))
            for k, v in meta.items()
        }

    return {
        "success": True,
        "url": {
            "method": method,
            "path": path,
            "query": {name: fingerprint(value) for name, value in query},
        },
        "headers": echoed_headers,
        "cookies": {name: fingerprint(value) for name, value in cookies.items()},
        "meta": echoed_meta,
        "client_info": dict(client_info) if client_info else None,
        "process_id": process_id,
        "note": (
            "Values are SHA-256 fingerprints: equal fingerprints mean equal "
            "values. Compare replies from two conversations to see what stays "
            "the same within one and changes between them."
        ),
    }
