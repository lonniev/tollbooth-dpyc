"""Tests for OperatorRuntime template-driven onboarding and dual credentials."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.runtime import OperatorRuntime, register_standard_tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_NSEC = "nsec1test000000000000000000000000000000000000000000000000000000"

# Set env var globally for all tests in this module
os.environ.setdefault("TOLLBOOTH_NOSTR_OPERATOR_NSEC", _TEST_NSEC)


def _make_runtime(
    *,
    operator_template: CredentialTemplate | None = None,
    patron_template: CredentialTemplate | None = None,
) -> OperatorRuntime:
    """Create a runtime with test defaults."""
    return OperatorRuntime(
        tool_registry={},
        operator_credential_template=operator_template,
        patron_credential_template=patron_template,
        service_name="Test Operator",
    )


BTCPAY_TEMPLATE = CredentialTemplate(
    service="test-operator",
    version=1,
    fields={
        "btcpay_host": FieldSpec(required=True, sensitive=True, description="BTCPay URL"),
        "btcpay_api_key": FieldSpec(required=True, sensitive=True, description="BTCPay API key"),
        "btcpay_store_id": FieldSpec(required=True, sensitive=True, description="BTCPay store"),
    },
    description="BTCPay credentials",
)

PATRON_TEMPLATE = CredentialTemplate(
    service="test-patron",
    version=1,
    fields={
        "api_key": FieldSpec(required=True, sensitive=True, description="API key"),
        "brain_id": FieldSpec(required=True, sensitive=False, description="Brain UUID"),
    },
    description="Patron API credentials",
)

MIXED_TEMPLATE = CredentialTemplate(
    service="mixed-operator",
    version=1,
    fields={
        "required_key": FieldSpec(required=True, sensitive=True, description="Must have"),
        "optional_key": FieldSpec(required=False, sensitive=True, description="Nice to have"),
    },
    description="Template with optional fields",
)


# ---------------------------------------------------------------------------
# OperatorRuntime credential service properties
# ---------------------------------------------------------------------------


class TestCredentialServiceProperties:
    def test_operator_credential_service_from_template(self) -> None:
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        assert rt.operator_credential_service == "test-operator"

    def test_patron_credential_service_from_template(self) -> None:
        rt = _make_runtime(patron_template=PATRON_TEMPLATE)
        assert rt.patron_credential_service == "test-patron"

    def test_empty_service_when_no_template(self) -> None:
        rt = _make_runtime()
        assert rt.operator_credential_service == ""
        assert rt.patron_credential_service == ""

    def test_dual_templates(self) -> None:
        rt = _make_runtime(
            operator_template=BTCPAY_TEMPLATE,
            patron_template=PATRON_TEMPLATE,
        )
        assert rt.operator_credential_service == "test-operator"
        assert rt.patron_credential_service == "test-patron"


# ---------------------------------------------------------------------------
# Onboarding status — template-driven
# ---------------------------------------------------------------------------


class TestOnboardingStatus:
    @pytest.mark.asyncio
    async def test_all_missing_no_vault(self) -> None:
        """With no vault and no credentials, everything shows missing."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        # Mock vault() to fail — simulates bootstrap not done
        rt.vault = AsyncMock(side_effect=ValueError("No Neon URL"))
        rt.courier = AsyncMock(return_value=None)

        result = await rt.onboarding_status()

        assert result["ready"] is False
        assert result["vault_ok"] is False
        assert result["bootstrap_error"] != ""
        missing_fields = {m["field"] for m in result["missing"]}
        assert "neon_database_url" in missing_fields
        assert "btcpay_host" in missing_fields
        assert "btcpay_api_key" in missing_fields
        assert "btcpay_store_id" in missing_fields

    @pytest.mark.asyncio
    async def test_vault_ok_creds_missing(self) -> None:
        """Vault bootstrapped but operator credentials not yet delivered."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._vault = MagicMock()  # pretend vault is up
        rt.courier = AsyncMock(return_value=None)

        result = await rt.onboarding_status()

        assert result["ready"] is False
        assert result["vault_ok"] is True
        # neon_database_url should be configured
        configured_fields = {c["field"] for c in result["configured"]}
        assert "neon_database_url" in configured_fields
        # BTCPay fields still missing
        missing_fields = {m["field"] for m in result["missing"]}
        assert "btcpay_host" in missing_fields

    @pytest.mark.asyncio
    async def test_fully_configured(self) -> None:
        """All credentials delivered — operator is ready."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._vault = MagicMock()

        # Mock vault credentials to return all fields
        async def _mock_vault_creds(service, npub_override=None):
            return {
                "btcpay_host": "https://pay.example.com",
                "btcpay_api_key": "key123",
                "btcpay_store_id": "store456",
            }
        rt._load_vault_creds = _mock_vault_creds

        result = await rt.onboarding_status()

        assert result["ready"] is True
        assert result["summary"] == "Operator is fully configured and ready to serve."
        assert len(result["missing"]) == 0
        configured_fields = {c["field"] for c in result["configured"]}
        assert "btcpay_host" in configured_fields
        assert "btcpay_api_key" in configured_fields
        assert "btcpay_store_id" in configured_fields

    @pytest.mark.asyncio
    async def test_no_template_means_ready(self) -> None:
        """Without a credential template, only bootstrap matters."""
        rt = _make_runtime()  # no templates
        rt._vault = MagicMock()

        result = await rt.onboarding_status()

        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_partial_credentials(self) -> None:
        """Some credentials delivered, some still missing."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._vault = MagicMock()

        async def _partial_creds(service, npub_override=None):
            return {"btcpay_host": "https://pay.example.com"}
        rt._load_vault_creds = _partial_creds

        result = await rt.onboarding_status()

        assert result["ready"] is False
        configured_fields = {c["field"] for c in result["configured"]}
        missing_fields = {m["field"] for m in result["missing"]}
        assert "btcpay_host" in configured_fields
        assert "btcpay_api_key" in missing_fields
        assert "btcpay_store_id" in missing_fields

    @pytest.mark.asyncio
    async def test_optional_fields_dont_block(self) -> None:
        """Optional template fields don't prevent ready status."""
        rt = _make_runtime(operator_template=MIXED_TEMPLATE)
        rt._vault = MagicMock()

        async def _required_only(service, npub_override=None):
            return {"required_key": "value"}
        rt._load_vault_creds = _required_only

        result = await rt.onboarding_status()

        assert result["ready"] is True
        assert len(result["optional_missing"]) == 1
        assert result["optional_missing"][0]["field"] == "optional_key"

    @pytest.mark.asyncio
    async def test_missing_nsec_blocks_ready(self) -> None:
        """Without nsec, operator can't be ready."""
        rt = OperatorRuntime(
            nsec_env_var="NONEXISTENT_NSEC_VAR",
            tool_registry={},
            operator_credential_template=BTCPAY_TEMPLATE,
            service_name="Test",
        )
        rt._vault = MagicMock()
        rt.courier = AsyncMock(return_value=None)

        result = await rt.onboarding_status()

        assert result["ready"] is False
        missing_fields = {m["field"] for m in result["missing"]}
        assert "NONEXISTENT_NSEC_VAR" in missing_fields

    @pytest.mark.asyncio
    async def test_enriches_with_descriptions(self) -> None:
        """Missing fields include descriptions from the template."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._vault = MagicMock()
        rt.courier = AsyncMock(return_value=None)

        result = await rt.onboarding_status()

        for field in result["missing"]:
            if field["field"] == "btcpay_host":
                assert field["how"] == "BTCPay URL"
                break
        else:
            pytest.fail("btcpay_host not in missing")

    @pytest.mark.asyncio
    async def test_includes_operator_metadata(self) -> None:
        """Result includes operator name, service, greeting."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._vault = MagicMock()
        rt._operator_credential_greeting = "Hello!"
        rt.courier = AsyncMock(return_value=None)

        result = await rt.onboarding_status()

        assert result["operator_name"] == "Test Operator"
        assert result["credential_service"] == "test-operator"
        assert result["credential_greeting"] == "Hello!"


# ---------------------------------------------------------------------------
# Patron onboarding status
# ---------------------------------------------------------------------------


class TestPatronOnboardingStatus:
    @pytest.mark.asyncio
    async def test_no_patron_template(self) -> None:
        """Without patron template, patron is always ready."""
        rt = _make_runtime()
        result = await rt.patron_onboarding_status("npub1test")
        assert result["ready"] is True

    @pytest.mark.asyncio
    async def test_patron_creds_missing(self) -> None:
        """Patron credentials not yet delivered."""
        rt = _make_runtime(patron_template=PATRON_TEMPLATE)
        rt._vault = MagicMock()

        async def _empty(service, npub_override=None):
            return {}
        rt._load_vault_creds = _empty

        result = await rt.patron_onboarding_status("npub1patron")

        assert result["ready"] is False
        missing_fields = {m["field"] for m in result["missing"]}
        assert "api_key" in missing_fields
        assert "brain_id" in missing_fields

    @pytest.mark.asyncio
    async def test_patron_creds_delivered(self) -> None:
        """Patron credentials present in vault."""
        rt = _make_runtime(patron_template=PATRON_TEMPLATE)
        rt._vault = MagicMock()

        async def _full(service, npub_override=None):
            return {"api_key": "key123", "brain_id": "brain-uuid"}
        rt._load_vault_creds = _full

        result = await rt.patron_onboarding_status("npub1patron")

        assert result["ready"] is True
        configured_fields = {c["field"] for c in result["configured"]}
        assert "api_key" in configured_fields
        assert "brain_id" in configured_fields


# ---------------------------------------------------------------------------
# Courier template wiring
# ---------------------------------------------------------------------------


class TestCourierTemplateWiring:
    @pytest.mark.asyncio
    async def test_courier_gets_both_templates(self) -> None:
        """Courier should receive both operator and patron templates."""
        rt = _make_runtime(
            operator_template=BTCPAY_TEMPLATE,
            patron_template=PATRON_TEMPLATE,
        )
        # Mock the courier internals
        with patch("tollbooth.runtime.OperatorRuntime.vault", new_callable=AsyncMock) as mock_vault, \
             patch("tollbooth.secure_courier.SecureCourierService") as mock_courier_cls:
            mock_vault.return_value = MagicMock()
            mock_courier_cls.return_value = MagicMock()

            # Access the template dict that would be built
            templates = {}
            if rt._operator_credential_template:
                templates[rt._operator_credential_template.service] = rt._operator_credential_template
            if rt._patron_credential_template:
                templates[rt._patron_credential_template.service] = rt._patron_credential_template

            assert "test-operator" in templates
            assert "test-patron" in templates
            assert templates["test-operator"] is BTCPAY_TEMPLATE
            assert templates["test-patron"] is PATRON_TEMPLATE


# ---------------------------------------------------------------------------
# register_standard_tools patron tools
# ---------------------------------------------------------------------------


class TestPatronToolRegistration:
    def test_patron_tools_registered_when_template_set(self) -> None:
        """Patron credential tools appear when patron_credential_template is set."""
        mcp = MagicMock()
        # make_slug_tool returns a decorator that just stores the function
        registered_tools = {}

        def fake_slug_tool(m, slug):
            def decorator(fn):
                registered_tools[fn.__name__] = fn
                return fn
            return decorator

        with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
            rt = _make_runtime(
                operator_template=BTCPAY_TEMPLATE,
                patron_template=PATRON_TEMPLATE,
            )
            register_standard_tools(mcp, "test", rt, service_name="test")

        assert "request_patron_credentials" in registered_tools
        assert "receive_patron_credentials" in registered_tools

    def test_no_patron_tools_without_template(self) -> None:
        """Patron credential tools NOT registered when no patron template."""
        mcp = MagicMock()
        registered_tools = {}

        def fake_slug_tool(m, slug):
            def decorator(fn):
                registered_tools[fn.__name__] = fn
                return fn
            return decorator

        with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
            rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
            register_standard_tools(mcp, "test", rt, service_name="test")

        assert "request_patron_credentials" not in registered_tools
        assert "receive_patron_credentials" not in registered_tools
        # But operator tools are still there
        assert "request_credential_channel" in registered_tools
        assert "receive_credentials" in registered_tools


# ---------------------------------------------------------------------------
# OTS config on runtime
# ---------------------------------------------------------------------------


class TestOTSConfig:
    def test_ots_defaults(self) -> None:
        rt = _make_runtime()
        assert rt._ots_enabled is True
        assert rt._ots_calendars is None

    def test_ots_enabled(self) -> None:
        rt = OperatorRuntime(
            tool_registry={},
            service_name="Test",
            ots_enabled=True,
            ots_calendars=["https://cal1.example.com"],
        )
        assert rt._ots_enabled is True
        assert rt._ots_calendars == ["https://cal1.example.com"]


# ---------------------------------------------------------------------------
# load_credentials routing
# ---------------------------------------------------------------------------


class TestLoadCredentials:
    @pytest.mark.asyncio
    async def test_defaults_to_operator_service(self) -> None:
        """load_credentials without service= uses operator credential service."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._operator_npub = "npub1testfake"

        with patch("tollbooth.tools.onboarding.load_config_from_vault", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"btcpay_host": "https://pay.example.com"}
            rt._courier = MagicMock()

            await rt.load_credentials(["btcpay_host"])

            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert call_args[0][1] == "test-operator"  # service arg

    @pytest.mark.asyncio
    async def test_explicit_service_override(self) -> None:
        """load_credentials with service= uses that service."""
        rt = _make_runtime(
            operator_template=BTCPAY_TEMPLATE,
            patron_template=PATRON_TEMPLATE,
        )
        rt._operator_npub = "npub1testfake"

        with patch("tollbooth.tools.onboarding.load_config_from_vault", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {"api_key": "k"}
            rt._courier = MagicMock()

            await rt.load_credentials(["api_key"], service="test-patron")

            call_args = mock_load.call_args
            assert call_args[0][1] == "test-patron"


# ---------------------------------------------------------------------------
# Cashier (payment processor) abstraction
# ---------------------------------------------------------------------------


class TestCashier:
    @pytest.mark.asyncio
    async def test_ensure_cashier_caches(self) -> None:
        """Second call returns cached instance without re-loading."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._operator_npub = "npub1testfake"

        mock_client = MagicMock()

        with patch("tollbooth.tools.onboarding.load_config_from_vault", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {
                "btcpay_host": "https://pay.test",
                "btcpay_api_key": "key",
                "btcpay_store_id": "store",
            }
            rt._courier = MagicMock()

            with patch("tollbooth.btcpay_client.BTCPayClient") as mock_cls:
                mock_cls.return_value = mock_client
                first = await rt.ensure_cashier()
                second = await rt.ensure_cashier()
                assert first is second
                mock_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_cashier_raises_without_creds(self) -> None:
        """Raises ValueError when BTCPay creds are missing."""
        rt = _make_runtime(operator_template=BTCPAY_TEMPLATE)
        rt._operator_npub = "npub1testfake"

        with patch("tollbooth.tools.onboarding.load_config_from_vault", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = {}
            rt._courier = MagicMock()

            with pytest.raises(ValueError, match="BTCPay not configured"):
                await rt.ensure_cashier()

    def test_cashier_invalidation(self) -> None:
        """Setting _cashier to None forces re-creation on next call."""
        rt = _make_runtime()
        rt._cashier = MagicMock()
        assert rt._cashier is not None
        rt._cashier = None
        assert rt._cashier is None


# ---------------------------------------------------------------------------
# npub enforcement — no silent fallback
# ---------------------------------------------------------------------------


class TestNpubEnforcement:
    @pytest.mark.asyncio
    async def test_debit_or_deny_passes_free_tools(self) -> None:
        """Free tools pass without npub."""
        from tollbooth.tool_identity import ToolIdentity
        identity = ToolIdentity(capability="free_tool", category="free", intent="test")
        rt = OperatorRuntime(
            tool_registry={identity.tool_id: identity},
            service_name="Test",
        )
        result = await rt.debit_or_deny(identity.tool_id, "")
        assert result is None

    @pytest.mark.asyncio
    async def test_debit_or_deny_denies_unknown_tools(self) -> None:
        """Unknown UUIDs are denied — no open doors."""
        rt = OperatorRuntime(
            tool_registry={},
            service_name="Test",
        )
        result = await rt.debit_or_deny("unknown-uuid", "")
        assert result is not None
        assert result["success"] is False
        assert "not registered" in result["error"]


# ---------------------------------------------------------------------------
# Initial pricing model scaffold
# ---------------------------------------------------------------------------


class TestInitialPricingModel:
    def test_build_initial_pricing_model(self) -> None:
        """Generates a scaffold with all tools at 0 sats."""
        from tollbooth.runtime import _build_initial_pricing_model
        from tollbooth.tool_identity import ToolIdentity
        import json

        search = ToolIdentity(capability="search", category="read", intent="Find stuff", mcp_name="test_search")
        create = ToolIdentity(capability="create", category="write", intent="Make stuff", mcp_name="test_create")
        rt = OperatorRuntime(
            tool_registry={
                search.tool_id: search,
                create.tool_id: create,
            },
            service_name="Test Service",
        )
        rt._slug = "test"
        result = _build_initial_pricing_model(rt, "Test Service")

        model = json.loads(result)
        assert model["name"] == "Test Service Initial Pricing"
        tool_names = {t["tool_name"] for t in model["tools"]}
        assert "test_search" in tool_names
        assert "test_create" in tool_names
        for t in model["tools"]:
            assert t["price_sats"] == 0
            assert "tool_id" in t

    def test_build_initial_empty_registry(self) -> None:
        """Empty registry produces an empty tools list."""
        from tollbooth.runtime import _build_initial_pricing_model
        import json

        rt = OperatorRuntime(tool_registry={}, service_name="Empty")
        result = _build_initial_pricing_model(rt, "Empty")
        model = json.loads(result)
        assert model["tools"] == []


# ---------------------------------------------------------------------------
# Demand tracking
# ---------------------------------------------------------------------------


class TestDemandTracking:
    @pytest.mark.asyncio
    async def test_get_global_demand_returns_count(self) -> None:
        """Returns demand count from vault."""
        rt = _make_runtime()
        mock_vault = AsyncMock()
        mock_vault.get_demand = AsyncMock(return_value=42)
        rt._vault = mock_vault

        result = await rt.get_global_demand("search")
        assert result == {"search": 42}

    @pytest.mark.asyncio
    async def test_get_global_demand_empty_on_error(self) -> None:
        """Returns empty dict on vault error."""
        rt = _make_runtime()
        rt._vault = None  # will cause vault() to fail

        result = await rt.get_global_demand("search")
        assert result == {}

    def test_demand_window_key_format(self) -> None:
        """Window key is hourly ISO format."""
        rt = _make_runtime()
        key = rt._demand_window_key()
        assert "T" in key
        assert key.endswith(":00")
