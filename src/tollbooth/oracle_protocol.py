"""Protocol defining the OracleProtocol — free community concierge tools.

The Oracle is a free, unauthenticated concierge that answers questions
about DPYC membership, governance, onboarding, and tax rates. It does
not require payment or credentials.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tollbooth.actor_types import ToolPathInfo


@runtime_checkable
class OracleProtocol(Protocol):
    """Contract for a DPYC Oracle MCP server.

    All tools are free and unauthenticated. Hot-path tools resolve
    locally; cold-path tools fetch from the GitHub registry.
    """

    @property
    def slug(self) -> str:
        """Short identifier for tool-name prefixing."""
        ...

    @classmethod
    def tool_catalog(cls) -> list[ToolPathInfo]:
        """Return metadata for every tool this actor exposes."""
        ...

    # ── Hot-path (local) ─────────────────────────────────────────

    async def get_tax_rate(self) -> dict[str, Any]:
        """(hot) Return the current certification tax rate."""
        ...

    async def how_to_join(self) -> str:
        """(hot) Return onboarding instructions for new members."""
        ...

    async def service_status(self) -> dict[str, Any]:
        """(hot) Return the Oracle's health and version info."""
        ...

    async def request_citizenship(
        self, npub: str, display_name: str
    ) -> dict[str, Any]:
        """(hot) Begin the citizenship application process."""
        ...

    # ── Cold-path (GitHub HTTP) ──────────────────────────────────

    async def about(self) -> str:
        """(cold) Return a description of the DPYC ecosystem."""
        ...

    async def lookup_member(self, npub: str) -> dict[str, Any] | str:
        """(cold) Look up a member by their Nostr npub."""
        ...

    async def get_rulebook(self) -> str:
        """(cold) Return the community governance rulebook."""
        ...

    async def who_is_first_curator(self) -> dict[str, Any] | str:
        """(cold) Identify the Prime Authority / First Curator."""
        ...

    async def network_versions(self) -> dict[str, Any]:
        """(cold) Return version info for all network components."""
        ...

    async def network_advisory(self) -> str:
        """(cold) Return any active network advisories."""
        ...

    async def confirm_citizenship(
        self, npub: str, challenge_id: str, signed_event_json: str
    ) -> dict[str, Any]:
        """(cold) Confirm citizenship with a signed Nostr challenge."""
        ...
