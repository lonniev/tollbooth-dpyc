"""session_status reports a clean orphan state while the DPYC registry is
still propagating an approved adoption — not a scary 'Bootstrap failed'.

The public members registry is the source of truth: an operator that can't
yet resolve its OWN entry there is not_registered (orphan), whether it was
never adopted or an Authority just approved it and the GitHub-raw members
file hasn't propagated yet (~5 min CDN). It bootstraps automatically once
discoverable. No operator-side state is tracked.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.asyncio


def _register_tools(rt):
    import tollbooth.runtime as rtmod

    tools: dict = {}

    def fake_slug_tool(_m, _s):
        def deco(fn):
            tools[fn.__name__] = fn
            return fn

        return deco

    with patch("tollbooth.slug_tools.make_slug_tool", side_effect=fake_slug_tool):
        rtmod.register_standard_tools(MagicMock(), "test", rt, service_name="test")
    return tools


def _orphan_rt(bootstrap_error: str):
    rt = MagicMock()
    rt.operator_npub = MagicMock(return_value="npub1op")
    rt._vault = None
    rt.vault = AsyncMock(side_effect=Exception(bootstrap_error))
    return rt


async def test_registry_not_found_reads_as_not_registered_with_propagation_hint():
    err = (
        "Bootstrap failed: Cannot resolve Authority: npub1xdv... not found "
        "in DPYC registry.. Operator may not be registered with an Authority."
    )
    tools = _register_tools(_orphan_rt(err))
    r = await tools["session_status"]()
    assert r["lifecycle"] == "not_registered"
    assert "propagat" in r["message"].lower()


async def test_classic_not_registered_still_not_registered():
    tools = _register_tools(_orphan_rt("Operator is not registered; no Neon URL"))
    r = await tools["session_status"]()
    assert r["lifecycle"] == "not_registered"


async def test_other_bootstrap_error_is_warming_up():
    # A non-registry bootstrap failure (e.g. a relay hiccup) is transient —
    # it must NOT be mislabeled as orphan.
    tools = _register_tools(_orphan_rt("Bootstrap failed: relay connection timeout"))
    r = await tools["session_status"]()
    assert r["lifecycle"] == "warming_up"
