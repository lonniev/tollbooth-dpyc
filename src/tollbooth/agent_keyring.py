"""An authenticated passthrough to a DPYC operator — the agent's own signing hand.

A DPYC agent is a patron that holds its own nsec. This keyring fronts one upstream
operator as a FastMCP proxy whose connection signs every forwarded request with that
nsec (NIP-98), so the operator knows the agent by its npub while the key stays in this
process — out of the agent's own reasoning context. It also names the agent's npub on
each call that does not name one. Nothing is stored; nothing new is *granted* per call —
the standing grant is possession of the nsec and a funded balance.

Reusable by any agent against any DPYC operator; it is not specific to the Software
Factory.

Run (stdio — e.g. as a local MCP server in claude-code-action's ``--mcp-config``):

    DPYC_KEYRING_UPSTREAM=https://cypher-mcp.fastmcp.app/mcp \
    DPYC_KEYRING_NPUB=npub1... \
    DPYC_KEYRING_NSEC=nsec1... \
    python -m tollbooth.agent_keyring

Requires the ``keyring`` extra (FastMCP): ``pip install tollbooth-dpyc[keyring]``.
"""

from __future__ import annotations

import os
from typing import Any


def with_npub(arguments: dict[str, Any] | None, npub: str) -> dict[str, Any]:
    """``arguments`` naming ``npub`` unless the call already names one."""
    args = dict(arguments or {})
    args.setdefault("npub", npub)
    return args


def build_keyring(upstream_url: str, npub: str, nsec: str) -> Any:
    """Build a FastMCP proxy of *upstream_url* whose every request ``nsec`` signs.

    Lazy-imports FastMCP so importing this module never requires the ``keyring`` extra.
    """
    from fastmcp.server import create_proxy
    from fastmcp.server.middleware import Middleware

    from tollbooth.mcp_client import service_client

    class _NameTheAgent(Middleware):
        """Name the agent's npub on every forwarded call that names none."""

        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            params = context.message
            forwarded = context.copy(
                message=params.model_copy(update={"arguments": with_npub(params.arguments, npub)})
            )
            return await call_next(forwarded)

    proxy = create_proxy(service_client(upstream_url, nsec), name="dpyc-keyring")
    proxy.add_middleware(_NameTheAgent())
    return proxy


def main() -> None:
    """Entry point: read config from the environment and serve over stdio.

    Env: ``DPYC_KEYRING_UPSTREAM`` (operator MCP URL), ``DPYC_KEYRING_NPUB`` (the agent's
    npub), ``DPYC_KEYRING_NSEC`` (the agent's nsec — held only here, never emitted).
    """
    upstream = os.environ.get("DPYC_KEYRING_UPSTREAM", "").strip()
    npub = os.environ.get("DPYC_KEYRING_NPUB", "").strip()
    nsec = os.environ.get("DPYC_KEYRING_NSEC", "").strip()
    missing = [
        name
        for name, value in (
            ("DPYC_KEYRING_UPSTREAM", upstream),
            ("DPYC_KEYRING_NPUB", npub),
            ("DPYC_KEYRING_NSEC", nsec),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"agent_keyring: missing required env: {', '.join(missing)}")
    build_keyring(upstream, npub, nsec).run()


if __name__ == "__main__":
    main()
