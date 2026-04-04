"""Tests for ToolIdentity and STANDARD_IDENTITIES."""

from __future__ import annotations

from tollbooth.tool_identity import (
    STANDARD_IDENTITIES,
    ToolIdentity,
    capability_uuid,
)


class TestCapabilityUUID:
    def test_deterministic(self) -> None:
        """Same capability always produces the same UUID."""
        a = capability_uuid("check_balance")
        b = capability_uuid("check_balance")
        assert a == b

    def test_different_capabilities_differ(self) -> None:
        assert capability_uuid("check_balance") != capability_uuid("post_tweet")

    def test_is_valid_uuid_string(self) -> None:
        uid = capability_uuid("check_balance")
        import uuid
        parsed = uuid.UUID(uid)
        assert parsed.version == 5


class TestToolIdentity:
    def test_tool_id_property(self) -> None:
        ti = ToolIdentity(capability="check_balance", category="free", intent="test")
        assert ti.tool_id == capability_uuid("check_balance")

    def test_frozen(self) -> None:
        ti = ToolIdentity(capability="x", category="free", intent="y")
        try:
            ti.capability = "z"  # type: ignore[misc]
            assert False, "should be frozen"
        except AttributeError:
            pass


class TestStandardIdentities:
    def test_not_empty(self) -> None:
        assert len(STANDARD_IDENTITIES) > 20

    def test_all_have_valid_categories(self) -> None:
        valid = {"free", "read", "write", "heavy", "restricted"}
        for name, identity in STANDARD_IDENTITIES.items():
            assert identity.category in valid, f"{name} has invalid category {identity.category!r}"

    def test_all_have_intent(self) -> None:
        for name, identity in STANDARD_IDENTITIES.items():
            assert identity.intent, f"{name} is missing intent"

    def test_all_uuids_unique(self) -> None:
        ids = [identity.tool_id for identity in STANDARD_IDENTITIES.values()]
        assert len(ids) == len(set(ids)), "Duplicate UUIDs in STANDARD_IDENTITIES"

    def test_key_tools_present(self) -> None:
        """Spot-check that critical tools are registered."""
        for name in (
            "check_balance", "purchase_credits", "check_payment",
            "service_status", "session_status", "get_pricing_model",
            "set_pricing_model", "check_price",
        ):
            assert name in STANDARD_IDENTITIES, f"Missing standard identity: {name}"
