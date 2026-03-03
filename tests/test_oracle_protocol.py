"""Tests for OracleProtocol compliance."""

from typing import Any

from tollbooth.actor_types import ToolPath, ToolPathInfo
from tollbooth.oracle_protocol import OracleProtocol


class StubOracle:
    """Minimal stub satisfying OracleProtocol."""

    @property
    def slug(self) -> str:
        return "oracle"

    @classmethod
    def tool_catalog(cls) -> list[ToolPathInfo]:
        return [
            ToolPathInfo(tool_name="get_tax_rate", path=ToolPath.HOT),
        ]

    async def get_tax_rate(self) -> dict[str, Any]:
        return {"rate": 2.0}

    async def how_to_join(self) -> str:
        return "Visit dpyc.org"

    async def service_status(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def request_citizenship(
        self, npub: str, display_name: str
    ) -> dict[str, Any]:
        return {"npub": npub}

    async def about(self) -> str:
        return "DPYC ecosystem"

    async def lookup_member(self, npub: str) -> dict[str, Any] | str:
        return {"npub": npub}

    async def get_rulebook(self) -> str:
        return "Rules"

    async def who_is_first_curator(self) -> dict[str, Any] | str:
        return {"curator": "npub1..."}

    async def network_versions(self) -> dict[str, Any]:
        return {"version": "1.0"}

    async def network_advisory(self) -> str:
        return "No advisories"

    async def confirm_citizenship(
        self, npub: str, challenge_id: str, signed_event_json: str
    ) -> dict[str, Any]:
        return {"confirmed": True}


class TestOracleProtocol:
    """Protocol compliance tests."""

    def test_stub_satisfies_protocol(self):
        """StubOracle satisfies the OracleProtocol."""
        oracle = StubOracle()
        assert isinstance(oracle, OracleProtocol)

    def test_dict_does_not_satisfy(self):
        """A plain dict does NOT satisfy the protocol."""
        assert not isinstance({}, OracleProtocol)
