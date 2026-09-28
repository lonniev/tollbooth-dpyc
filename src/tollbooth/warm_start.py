"""Begin an operator's bootstrap when a client connects.

A cold operator process wakes on its first request, and that request is almost
always the MCP ``initialize`` handshake — not the paid call that needs the vault.
Waiting for the paid call to start the bootstrap makes the patron pay for the
Oracle, the relays, Neon and the pricing model in series with their request.
Starting it on ``initialize`` lets it run while the handshake finishes and the
client (or the model driving it) decides what to call.
"""

from __future__ import annotations

from typing import Any

from fastmcp.server.middleware import Middleware


class WarmOnInitialize(Middleware):
    """On every MCP handshake, ask the runtime to warm up; pass the handshake through."""

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    async def on_initialize(self, context: Any, call_next: Any) -> Any:
        self._runtime.warm()
        return await call_next(context)
