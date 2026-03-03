"""Tests for actor_types module — enums and frozen dataclass."""

import pytest

from tollbooth.actor_types import ActorRole, ToolPath, ToolPathInfo


class TestActorRole:
    """ActorRole enum tests."""

    def test_values(self):
        assert ActorRole.OPERATOR == "operator"
        assert ActorRole.AUTHORITY == "authority"
        assert ActorRole.ORACLE == "oracle"

    def test_all_members(self):
        assert set(ActorRole) == {
            ActorRole.OPERATOR,
            ActorRole.AUTHORITY,
            ActorRole.ORACLE,
        }


class TestToolPath:
    """ToolPath enum tests."""

    def test_values(self):
        assert ToolPath.HOT == "hot"
        assert ToolPath.COLD == "cold"
        assert ToolPath.DELEGATION == "delegation"

    def test_all_members(self):
        assert set(ToolPath) == {
            ToolPath.HOT,
            ToolPath.COLD,
            ToolPath.DELEGATION,
        }


class TestToolPathInfo:
    """ToolPathInfo frozen dataclass tests."""

    def test_creation_with_defaults(self):
        info = ToolPathInfo(tool_name="check_balance", path=ToolPath.HOT)
        assert info.tool_name == "check_balance"
        assert info.path == ToolPath.HOT
        assert info.delegates_to is None
        assert info.requires_auth is True
        assert info.cost_tier == "FREE"
        assert info.agent_hint == ""
        assert info.supersedes == ""

    def test_creation_with_all_fields(self):
        info = ToolPathInfo(
            tool_name="certify_credits",
            path=ToolPath.DELEGATION,
            delegates_to=ActorRole.AUTHORITY,
            requires_auth=True,
            cost_tier="HEAVY",
            agent_hint="Delegates to Authority — may be slow.",
            supersedes="Previously required separate Authority connection.",
        )
        assert info.delegates_to == ActorRole.AUTHORITY
        assert info.cost_tier == "HEAVY"
        assert "Authority" in info.agent_hint
        assert "Previously" in info.supersedes

    def test_frozen_immutability(self):
        info = ToolPathInfo(tool_name="about", path=ToolPath.COLD)
        with pytest.raises(AttributeError):
            info.tool_name = "something_else"  # type: ignore[misc]

    def test_frozen_immutability_path(self):
        info = ToolPathInfo(tool_name="about", path=ToolPath.COLD)
        with pytest.raises(AttributeError):
            info.path = ToolPath.HOT  # type: ignore[misc]
