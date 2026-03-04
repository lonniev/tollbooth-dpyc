"""Tests for OperatorProtocol compliance."""

from typing import Any

from tollbooth.actor_types import ToolPath, ToolPathInfo
from tollbooth.operator_protocol import OperatorProtocol


class StubOperator:
    """Minimal stub satisfying OperatorProtocol."""

    @property
    def slug(self) -> str:
        return "brain"

    @classmethod
    def tool_catalog(cls) -> list[ToolPathInfo]:
        return [
            ToolPathInfo(tool_name="check_balance", path=ToolPath.HOT),
        ]

    async def check_balance(self, npub: str) -> dict[str, Any]:
        return {"balance": 500}

    async def account_statement(self, npub: str) -> dict[str, Any]:
        return {"transactions": []}

    async def account_statement_infographic(
        self, npub: str
    ) -> dict[str, Any]:
        return {"chart": "..."}

    async def restore_credits(
        self, npub: str, invoice_id: str
    ) -> dict[str, Any]:
        return {"restored": True}

    async def service_status(self) -> dict[str, Any]:
        return {"status": "ok"}

    async def purchase_credits(
        self, npub: str, amount_sats: int, certificate: str
    ) -> dict[str, Any]:
        return {"invoice_id": "inv_456"}

    async def check_payment(
        self, npub: str, invoice_id: str
    ) -> dict[str, Any]:
        return {"settled": True}

    async def certify_credits(
        self, operator_id: str, amount_sats: int
    ) -> dict[str, Any]:
        return {"certified": True}

    async def register_operator(self, npub: str) -> dict[str, Any]:
        return {"npub": npub}

    async def operator_status(self) -> dict[str, Any]:
        return {"status": "active"}

    async def lookup_member(self, npub: str) -> dict[str, Any] | str:
        return {"npub": npub}

    async def how_to_join(self) -> str:
        return "Visit dpyc.org"

    async def get_tax_rate(self) -> dict[str, Any]:
        return {"rate": 2.0}

    async def about(self) -> str:
        return "DPYC ecosystem"

    async def network_advisory(self) -> str:
        return "No advisories"

    async def session_status(self) -> dict[str, Any]:
        return {"hasPersonalSession": False}

    async def request_credential_channel(
        self, service: str, greeting: str, recipient_npub: str | None,
    ) -> dict[str, Any]:
        return {"success": True}

    async def receive_credentials(
        self, sender_npub: str, service: str, credential_card: str,
    ) -> dict[str, Any]:
        return {"success": True}

    async def forget_credentials(
        self, sender_npub: str, service: str,
    ) -> dict[str, Any]:
        return {"success": True}


class TestOperatorProtocol:
    """Protocol compliance tests."""

    def test_stub_satisfies_protocol(self):
        """StubOperator satisfies the OperatorProtocol."""
        operator = StubOperator()
        assert isinstance(operator, OperatorProtocol)

    def test_dict_does_not_satisfy(self):
        """A plain dict does NOT satisfy the protocol."""
        assert not isinstance({}, OperatorProtocol)
