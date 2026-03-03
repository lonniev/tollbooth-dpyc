"""Tests for AuthorityProtocol compliance."""

from typing import Any

from tollbooth.actor_types import ToolPath, ToolPathInfo
from tollbooth.authority_protocol import AuthorityProtocol


class StubAuthority:
    """Minimal stub satisfying AuthorityProtocol."""

    @property
    def slug(self) -> str:
        return "authority"

    @classmethod
    def tool_catalog(cls) -> list[ToolPathInfo]:
        return [
            ToolPathInfo(tool_name="certify_credits", path=ToolPath.HOT),
        ]

    async def certify_credits(
        self, operator_id: str, amount_sats: int
    ) -> dict[str, Any]:
        return {"certified": True}

    async def register_operator(self, npub: str) -> dict[str, Any]:
        return {"npub": npub}

    async def operator_status(self) -> dict[str, Any]:
        return {"status": "active"}

    async def check_balance(self) -> dict[str, Any]:
        return {"balance": 1000}

    async def account_statement(self) -> dict[str, Any]:
        return {"transactions": []}

    async def account_statement_infographic(self) -> dict[str, Any]:
        return {"chart": "..."}

    async def service_status(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def report_upstream_purchase(
        self, amount_sats: int
    ) -> dict[str, Any]:
        return {"reported": True}

    async def purchase_credits(self, amount_sats: int) -> dict[str, Any]:
        return {"invoice_id": "inv_123"}

    async def check_payment(self, invoice_id: str) -> dict[str, Any]:
        return {"settled": True}

    async def check_dpyc_membership(self, npub: str) -> dict[str, Any]:
        return {"member": True}


class TestAuthorityProtocol:
    """Protocol compliance tests."""

    def test_stub_satisfies_protocol(self):
        """StubAuthority satisfies the AuthorityProtocol."""
        authority = StubAuthority()
        assert isinstance(authority, AuthorityProtocol)

    def test_dict_does_not_satisfy(self):
        """A plain dict does NOT satisfy the protocol."""
        assert not isinstance({}, AuthorityProtocol)
