"""Regression guard for audit finding S1 — credentials must NEVER be echoed.

The Secure Courier wrapper (``SecureCourierService.receive``) strips credential
values before returning. The audit found a tool branch that reached past that
wrapper to the exchange, so raw credential values leaked into the tool result
(→ LLM context, logs, transcripts).

These tests exercise the *registered tool closures* and assert no credential
values survive in any receive result.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.secure_courier import SecureCourierService

# Set a test operator nsec so runtime construction never probes the env/relays.
os.environ.setdefault(
    "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
    "nsec1test000000000000000000000000000000000000000000000000000000",
)

# Distinctive secret values — if any of these strings appear anywhere in a
# tool result, a credential leaked.
SECRET_CREDS = {
    "api_key": "sk-LEAK-CANARY-DO-NOT-ECHO",
    "brain_id": "11111111-2222-3333-4444-555555555555",
}

OP_TEMPLATE = CredentialTemplate(
    service="test-operator",
    version=1,
    fields={
        "api_key": FieldSpec(required=True, sensitive=True, description="API key"),
        "brain_id": FieldSpec(required=True, sensitive=False, description="Brain UUID"),
    },
    description="Operator credentials",
)

PATRON_TEMPLATE = CredentialTemplate(
    service="test-patron",
    version=1,
    fields={
        "api_key": FieldSpec(required=True, sensitive=True, description="API key"),
        "brain_id": FieldSpec(required=True, sensitive=False, description="Brain UUID"),
    },
    description="Patron credentials",
)


def _leaky_courier() -> SecureCourierService:
    """A real SecureCourierService whose exchange returns raw credentials.

    Replacing ``_exchange`` with a mock lets the receive path run without any
    relay or vault I/O while still exercising the *real* ``receive``
    stripping logic in the wrapper. The mock's return dict carries the canary
    credentials — so a tool that bypasses the wrapper will echo them.
    """
    svc = SecureCourierService(
        operator_nsec=PrivateKey().nsec,
        relays=["wss://relay.test"],
        templates={"test-operator": OP_TEMPLATE, "test-patron": PATRON_TEMPLATE},
    )
    exchange = MagicMock()
    # Truthy vault so courier()'s late-attach branch is skipped in the runtime.
    exchange._credential_vault = MagicMock()
    exchange.receive = AsyncMock(
        return_value={
            "success": True,
            "service": "test-operator",
            "fields_received": len(SECRET_CREDS),
            "sensitive_fields": 1,
            "encryption": "nip44",
            "credentials": dict(SECRET_CREDS),
            "user_npub": "npub1patrontestxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "persisted": True,
        }
    )
    svc._exchange = exchange
    return svc


def _register(rt: OperatorRuntime) -> dict:
    """Capture the registered standard tool closures by function name."""
    registered: dict = {}

    def fake_slug_tool(_mcp, _slug, _identify):
        def decorator(fn):
            registered[fn.__name__] = fn
            return fn

        return decorator

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return registered


def _assert_no_creds(result: dict) -> None:
    """No raw credential key or canary value may appear anywhere in result."""
    assert isinstance(result, dict)
    assert "credentials" not in result, f"credentials echoed: {result!r}"
    flat = repr(result)
    for value in SECRET_CREDS.values():
        assert value not in flat, f"credential value {value!r} leaked in {result!r}"


def _runtime() -> OperatorRuntime:
    rt = OperatorRuntime(
        tool_registry={},
        operator_credential_template=OP_TEMPLATE,
        patron_credential_template=PATRON_TEMPLATE,
        service_name="Test Operator",
    )
    rt._courier = _leaky_courier()
    return rt


def test_leaky_exchange_actually_carries_credentials() -> None:
    """Positive control: the mocked exchange really does return credentials.

    Guards against the regression test silently passing because the fixture
    stopped producing a leak.
    """
    courier = _leaky_courier()
    raw = courier._exchange.receive  # AsyncMock
    assert "credentials" in raw.return_value
    assert raw.return_value["credentials"] == SECRET_CREDS


@pytest.mark.asyncio
async def test_receive_credentials_strips_credentials() -> None:
    """receive_credentials must not echo credentials."""
    rt = _runtime()
    tools = _register(rt)
    assert "receive_credentials" in tools

    result = await tools["receive_credentials"](
        sender_npub="npub1patrontestxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        service="test-operator",
        nonce="a-nonce",
    )
    _assert_no_creds(result)
    assert result.get("success") is True


@pytest.mark.asyncio
async def test_receive_patron_credentials_strips_credentials() -> None:
    """receive_patron_credentials must not echo credentials."""
    rt = _runtime()
    tools = _register(rt)
    assert "receive_patron_credentials" in tools

    result = await tools["receive_patron_credentials"](
        sender_npub="npub1patrontestxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        nonce="a-nonce",
    )
    _assert_no_creds(result)
    assert result.get("success") is True


# ---------------------------------------------------------------------------
# A failed vault write: the exchange reports success=False and still carries
# the decrypted values (audit F13).
# ---------------------------------------------------------------------------

FAILED_WRITE = {
    "success": False,
    "service": "test-operator",
    "credentials": dict(SECRET_CREDS),
    "persisted": False,
    "error_code": "credential_vault_unavailable",
    "error": "the vault write failed",
}
SENDER = "npub1patrontestxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


def _courier_whose_vault_write_fails() -> SecureCourierService:
    svc = _leaky_courier()
    svc._exchange.receive = AsyncMock(return_value=dict(FAILED_WRITE))
    return svc


@pytest.mark.asyncio
async def test_service_strips_credentials_when_a_receive_write_fails() -> None:
    result = await _courier_whose_vault_write_fails().receive(
        SENDER, service="test-operator", nonce="a-nonce",
    )
    _assert_no_creds(result)
    assert result["success"] is False
    assert result["error_code"] == "credential_vault_unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("receive_credentials", {"service": "test-operator", "nonce": "a-nonce"}),
        ("receive_patron_credentials", {"nonce": "a-nonce"}),
    ],
)
async def test_tools_strip_credentials_when_the_write_fails(tool: str, args: dict) -> None:
    rt = _runtime()
    rt._courier = _courier_whose_vault_write_fails()
    result = await _register(rt)[tool](sender_npub=SENDER, **args)
    _assert_no_creds(result)
    assert result["success"] is False


@pytest.mark.asyncio
async def test_tools_strip_credentials_a_courier_failed_to() -> None:
    """The tools hold the line even if the service ever stops stripping."""
    rt = _runtime()
    courier = MagicMock()
    courier.receive = AsyncMock(return_value=dict(FAILED_WRITE))
    rt.courier = AsyncMock(return_value=courier)
    tools = _register(rt)
    _assert_no_creds(
        await tools["receive_patron_credentials"](sender_npub=SENDER, nonce="a-nonce")
    )
    _assert_no_creds(
        await tools["receive_credentials"](
            sender_npub=SENDER, service="test-operator", nonce="a-nonce",
        )
    )


@pytest.mark.parametrize("tool", ["receive_credentials", "receive_patron_credentials"])
def test_receive_tools_take_no_credential_card(tool: str) -> None:
    """Credential cards are gone: a reply is collected by its nonce only."""
    import inspect

    params = inspect.signature(_register(_runtime())[tool]).parameters
    assert "credential_card" not in params
    assert "nonce" in params


@pytest.mark.asyncio
async def test_receive_without_a_nonce_is_refused_before_the_courier() -> None:
    rt = _runtime()
    courier = MagicMock()
    courier.receive = AsyncMock()
    rt.courier = AsyncMock(return_value=courier)
    tools = _register(rt)
    for result in (
        await tools["receive_credentials"](sender_npub=SENDER, service="test-operator"),
        await tools["receive_patron_credentials"](sender_npub=SENDER),
    ):
        assert result["success"] is False
        assert result["error_code"] == "nonce_missing"
    courier.receive.assert_not_called()
