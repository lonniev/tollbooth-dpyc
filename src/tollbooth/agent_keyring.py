"""An authenticated passthrough to a DPYC operator — the agent's own signing hand.

A DPYC agent is a patron that holds its own nsec. When it calls a paid operator it must
present a proof of possession; this keyring is where that proof is minted so the agent
never handles the key or a token itself. It fronts one upstream operator as a FastMCP
proxy and, on every forwarded call, injects the agent's npub plus a **freshly-signed,
in-memory** kind-27235 proof bound to the tool being called. Nothing is stored; nothing
new is *granted* per call — the standing grant is possession of the nsec and a funded
balance. The proof is just the mechanical demonstration of possession, minted on the spot.

This is the peer of :class:`tollbooth.authority_client.AuthorityClient` (which signs a
proof per ``certify`` call): the same primitive (``create_proof``), exposed as a local MCP
endpoint so the agent calls the operator's verbs plainly while the nsec stays in this
process — out of the agent's own reasoning context. Reusable by any agent against any
DPYC operator; it is not specific to the Software Factory.

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

from tollbooth.patron_signer import PatronSigner


def signed_arguments(
    tool_name: str, arguments: dict[str, Any] | None, npub: str, nsec: str
) -> dict[str, Any]:
    """Convenience free-function form of :meth:`PatronSigner.authenticate`.

    The signing logic lives in :class:`~tollbooth.patron_signer.PatronSigner` (the single
    patron-signing home); this is a thin wrapper for one-shot use and for tests.
    """
    return PatronSigner(npub, nsec).authenticate(tool_name, arguments)


def build_keyring(upstream_url: str, npub: str, nsec: str) -> Any:
    """Build a FastMCP proxy of *upstream_url* that signs each forwarded call.

    Lazy-imports FastMCP so importing this module never requires the ``keyring`` extra.
    """
    from fastmcp.server import create_proxy
    from fastmcp.server.middleware import Middleware

    signer = PatronSigner(npub, nsec)  # one signing hand for this keyring

    class _DpopAuthMiddleware(Middleware):
        """Inject (npub, fresh kind-27235 proof) into every tool call before forwarding."""

        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            params = context.message
            new_args = signer.authenticate(params.name, params.arguments)
            forwarded = context.copy(message=params.model_copy(update={"arguments": new_args}))
            return await call_next(forwarded)

    proxy = create_proxy(upstream_url, name="dpyc-keyring")
    proxy.add_middleware(_DpopAuthMiddleware())
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
