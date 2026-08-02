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
    def test_explicit_tool_id(self) -> None:
        """tool_id is now an explicit field, NOT derived from capability."""
        ti = ToolIdentity(
            tool_id="aaaaaaaa-1111-2222-3333-444444444444",
            capability="check_balance",
            category="free",
            intent="test",
        )
        # Explicit value is preserved verbatim; capability is decorative.
        assert ti.tool_id == "aaaaaaaa-1111-2222-3333-444444444444"

    def test_capability_uuid_helper_still_available(self) -> None:
        """capability_uuid() is retained as a REPL helper for picking
        an initial UUID seed — not called at runtime by the wheel."""
        assert capability_uuid("check_balance") == "f09d5935-e58d-52a6-a5d3-ff82794ec7fb"

    def test_frozen(self) -> None:
        ti = ToolIdentity(
            tool_id="bbbbbbbb-1111-2222-3333-444444444444",
            capability="x",
            category="free",
            intent="y",
        )
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

    def test_keys_are_uuids(self) -> None:
        """All keys in STANDARD_IDENTITIES are UUIDs matching the identity's tool_id."""
        for key, identity in STANDARD_IDENTITIES.items():
            assert key == identity.tool_id, f"Key mismatch: {key} != {identity.tool_id}"

    def test_all_uuids_unique(self) -> None:
        ids = list(STANDARD_IDENTITIES.keys())
        assert len(ids) == len(set(ids)), "Duplicate UUIDs in STANDARD_IDENTITIES"

    def test_all_tool_ids_are_uuid_v5(self) -> None:
        """No hand-written v4 literals in the standard catalog (issue #175)."""
        import uuid

        for key, identity in STANDARD_IDENTITIES.items():
            ver = uuid.UUID(key).version
            assert ver == 5, (
                f"{identity.capability} tool_id is UUID v{ver}, expected v5: {key}"
            )

    def test_key_capabilities_present(self) -> None:
        """Spot-check that critical capabilities are registered."""
        capabilities = {ti.capability for ti in STANDARD_IDENTITIES.values()}
        for cap in (
            "check_balance", "purchase_credits", "check_payment",
            "service_status", "session_status", "get_pricing_model",
            "set_pricing_model", "check_price",
            "list_canonical_identities",
            "restore_neon_schema",
            "get_nostr_profile",
            "publish_nostr_profile",
        ):
            assert cap in capabilities, f"Missing capability: {cap}"
