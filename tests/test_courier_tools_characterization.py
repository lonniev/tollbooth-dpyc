"""Characterization net for the courier tools (audit M2.1g Phase 1).

The courier tools delegate the actual relay drain to courier.receive /
open_channel / forget (methods tested in test_nostr_credentials /
test_secure_courier_service). Their own logic is validation + orchestration +
the operator-credential validation-callback flow in receive_credentials
(validator passes → cashier reset; validator fails → forget bad creds +
rejection DM + structured error). The credential_card branch / no-echo is
already covered by test_credential_no_echo.py.

Pins current behavior at the tool-closure level before the Phase 2 extraction
to tools/courier.py. §2-sensitive — assertions capture existing behavior.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.constants import ErrorCode
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.runtime import OperatorRuntime, register_standard_tools

os.environ.setdefault(
    "TOLLBOOTH_NOSTR_OPERATOR_NSEC",
    "nsec1test000000000000000000000000000000000000000000000000000000",
)

OP = "npub1operatortestxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
SENDER = "npub1sendertestxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

OP_TEMPLATE = CredentialTemplate(
    service="test-operator",
    version=1,
    fields={"api_key": FieldSpec(required=True, sensitive=True)},
    description="Operator creds",
)
PATRON_TEMPLATE = CredentialTemplate(
    service="test-patron",
    version=1,
    fields={"token": FieldSpec(required=True, sensitive=True)},
    description="Patron creds",
)


class FakeCourier:
    def __init__(self, *, receive_result=None, forget_result=None):
        self.receive_result = receive_result or {"success": True, "service": "test-operator"}
        self.forget_result = forget_result if forget_result is not None else {"success": True}
        self.sent: list[str] = []
        self.store_credentials = AsyncMock()
        self._exchange = SimpleNamespace(
            _credential_vault=SimpleNamespace(store_credentials=self.store_credentials),
        )
        self.open_channel = AsyncMock(return_value={"success": True, "dpop_token": "p"})
        self.forget = AsyncMock(return_value=self.forget_result)

    async def receive(self, sender_npub, service=None, dpop_token=None, request_tool=None):
        return dict(self.receive_result)

    async def redeem_card(self, ncred, service=None):
        return {"success": True, "service": service}

    async def send(self, recipient_npub, message):
        self.sent.append(message)


def _runtime(courier, *, validator=None, on_forget=None):
    rt = OperatorRuntime(
        tool_registry={},
        operator_credential_template=OP_TEMPLATE,
        patron_credential_template=PATRON_TEMPLATE,
        service_name="Test Operator",
    )
    rt._operator_npub = OP  # bypass nsec derivation
    rt._courier = courier
    rt._credential_validator = validator
    rt._on_forget = on_forget
    rt.load_credentials = AsyncMock(return_value={"api_key": "k"})
    rt.require_caller_proof = AsyncMock(return_value=None)  # proof passes
    return rt


def _register(rt):
    tools: dict = {}

    def fake_slug_tool(_mcp, _slug):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn
        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


# ── request_credential_channel ────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_channel_opens_with_service_and_greeting():
    courier = FakeCourier()
    rt = _runtime(courier)
    rt._operator_credential_greeting = "welcome"
    tools = _register(rt)

    r = await tools["request_credential_channel"](sender_npub=SENDER, service="test-operator")
    assert r == {"success": True, "dpop_token": "p"}
    assert courier.open_channel.await_args.args[0] == "test-operator"
    assert courier.open_channel.await_args.kwargs["recipient_npub"] == SENDER


@pytest.mark.asyncio
async def test_request_channel_requires_sender_and_service():
    rt = _runtime(FakeCourier())
    tools = _register(rt)
    assert (await tools["request_credential_channel"](service="x"))["error"] == "sender_npub is required."
    r = await tools["request_credential_channel"](sender_npub=SENDER)
    assert "service is required" in r["error"]


# ── receive_credentials validation-callback flow ──────────────────────

@pytest.mark.asyncio
async def test_receive_missing_dpop_token_and_card():
    rt = _runtime(FakeCourier())
    tools = _register(rt)
    r = await tools["receive_credentials"](sender_npub=SENDER, service="test-operator")
    assert r["error_code"] == ErrorCode.DPOP_TOKEN_MISSING


@pytest.mark.asyncio
async def test_receive_no_validator_returns_result():
    courier = FakeCourier(receive_result={"success": True, "service": "test-operator", "ok": 1})
    rt = _runtime(courier, validator=None)
    tools = _register(rt)
    r = await tools["receive_credentials"](sender_npub=SENDER, service="test-operator", dpop_token="p")
    assert r["ok"] == 1


@pytest.mark.asyncio
async def test_receive_validator_passes_resets_cashier():
    courier = FakeCourier()
    rt = _runtime(courier, validator=lambda creds: [])  # no errors
    rt._cashier = object()  # something to clear
    tools = _register(rt)

    r = await tools["receive_credentials"](sender_npub=SENDER, service="test-operator", dpop_token="p")
    assert r["success"] is True
    assert rt._cashier is None  # cashier reset so it reinitializes from new creds


@pytest.mark.asyncio
async def test_receive_validator_fails_rejects_and_dms():
    courier = FakeCourier()
    rt = _runtime(courier, validator=lambda creds: ["api_key looks wrong"])
    tools = _register(rt)

    r = await tools["receive_credentials"](sender_npub=SENDER, service="test-operator", dpop_token="p")
    assert r["success"] is False
    assert r["validation_errors"] == ["api_key looks wrong"]
    assert "failed validation" in r["error"]
    # bad creds forgotten (vault overwritten with empty) and a rejection DM sent
    courier.store_credentials.assert_awaited_once()
    assert courier.store_credentials.await_args.args[2] == ""
    assert courier.sent and "api_key looks wrong" in courier.sent[0]


@pytest.mark.asyncio
async def test_receive_partial_skips_validator_until_complete():
    """A partial delivery whose merged creds still lack a required field does
    NOT run the operator validator and does NOT wipe — completeness is the
    readiness gate's job, so an interim state can't be flagged 'missing' here."""
    courier = FakeCourier()
    rt = _runtime(courier, validator=lambda creds: ["api_key looks wrong"])
    rt.load_credentials = AsyncMock(return_value={})  # nothing required present yet
    tools = _register(rt)

    r = await tools["receive_credentials"](sender_npub=SENDER, service="test-operator", dpop_token="p")
    assert r["success"] is True            # not rejected despite the failing validator
    assert "validation_errors" not in r
    courier.store_credentials.assert_not_awaited()  # nothing forgotten/wiped


# ── forget_credentials ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_forget_fires_on_forget_callback():
    courier = FakeCourier(forget_result={"success": True})
    seen = []
    rt = _runtime(courier, on_forget=lambda svc, npub: seen.append((svc, npub)))
    tools = _register(rt)

    r = await tools["forget_credentials"](service="test-operator", npub=OP, dpop_token="ok")
    assert r["success"] is True
    assert seen == [("test-operator", OP)]


@pytest.mark.asyncio
async def test_forget_requires_service():
    rt = _runtime(FakeCourier())
    tools = _register(rt)
    r = await tools["forget_credentials"](service="", npub=OP, dpop_token="ok")
    assert "service is required" in r["error"]


# ── patron credential variants ────────────────────────────────────────

@pytest.mark.asyncio
async def test_request_patron_opens_on_patron_service():
    courier = FakeCourier()
    rt = _runtime(courier)
    rt._patron_credential_greeting = "hi patron"
    tools = _register(rt)

    r = await tools["request_patron_credentials"](sender_npub=SENDER)
    assert r == {"success": True, "dpop_token": "p"}
    assert courier.open_channel.await_args.args[0] == "test-patron"


@pytest.mark.asyncio
async def test_receive_patron_missing_dpop_token_and_card():
    rt = _runtime(FakeCourier())
    tools = _register(rt)
    r = await tools["receive_patron_credentials"](sender_npub=SENDER)
    assert r["error_code"] == ErrorCode.DPOP_TOKEN_MISSING


@pytest.mark.asyncio
async def test_receive_patron_dpop_token_branch_drains():
    captured = {}

    class _C(FakeCourier):
        async def receive(self, sender_npub, service=None, dpop_token=None, request_tool=None):
            captured.update(service=service, dpop_token=dpop_token, request_tool=request_tool)
            return {"success": True, "service": service}

    rt = _runtime(_C())
    tools = _register(rt)
    r = await tools["receive_patron_credentials"](sender_npub=SENDER, dpop_token="ph")
    assert r["success"] is True
    assert captured == {"service": "test-patron", "dpop_token": "ph",
                        "request_tool": "request_patron_credentials"}
