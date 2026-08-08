"""Rotating one operator secret without restating the bundle.

Before this path existed, applying a single reissued key meant re-delivering
the operator's whole credential bundle over Secure Courier — and every field
omitted from that reply is a field destroyed. An operator rotating a BTCPay
key could take out its own ``neon_api_key`` by answering the welcome DM with
only the three fields it thought had changed.

So the assertion that matters most here is the negative one: **a vault that
could not be read is never merged into.** A blind merge would write back only
the fields we happened to see, silently erasing the rest.
"""

from __future__ import annotations

import os

import pytest

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.runtime import OperatorRuntime

_TEST_NSEC = "nsec1test000000000000000000000000000000000000000000000000000000"
os.environ.setdefault("TOLLBOOTH_NOSTR_OPERATOR_NSEC", _TEST_NSEC)

OPERATOR_SERVICE = "test-operator"
OPERATOR_NPUB = "npub1" + "c" * 58


def _operator_runtime() -> OperatorRuntime:
    """A runtime whose operator template mirrors a real Authority's bundle.

    ``operator_npub`` is stubbed rather than derived from a test nsec: the
    identity is incidental to a read-merge-write, and deriving it would tie
    these tests to a shared process-wide env var.
    """
    rt = OperatorRuntime(
        tool_registry={},
        service_name="Test Operator",
        operator_credential_template=CredentialTemplate(
            service=OPERATOR_SERVICE,
            version=1,
            fields={
                "btcpay_host": FieldSpec(description="BTCPay host"),
                "btcpay_api_key": FieldSpec(description="BTCPay API key"),
                "btcpay_store_id": FieldSpec(description="BTCPay store id"),
                "neon_api_key": FieldSpec(description="Neon API key"),
            },
        ),
    )
    rt.operator_npub = lambda: OPERATOR_NPUB  # type: ignore[method-assign]
    return rt


class TestUpdateOperatorCredential:
    @pytest.mark.asyncio
    async def test_merges_one_field_and_preserves_the_rest(self, monkeypatch) -> None:
        rt = _operator_runtime()
        captured: dict = {}

        async def _load(npub, *, service=None):
            return {
                "btcpay_host": "https://btcpay.example",
                "btcpay_api_key": "OLD",
                "btcpay_store_id": "STORE",
                "neon_api_key": "NEON",
            }, ""

        async def _store(npub, data, *, service=None, just_delivered=None):
            captured.update(
                npub=npub, data=data, service=service, just_delivered=just_delivered,
            )
            return True

        monkeypatch.setattr(rt, "load_patron_session", _load)
        monkeypatch.setattr(rt, "store_patron_session", _store)

        assert await rt.update_operator_credential("btcpay_api_key", "NEW") is True

        assert captured["data"]["btcpay_api_key"] == "NEW"
        # The three the operator did not mention must survive untouched — this
        # is the entire reason the tool exists.
        assert captured["data"]["btcpay_host"] == "https://btcpay.example"
        assert captured["data"]["btcpay_store_id"] == "STORE"
        assert captured["data"]["neon_api_key"] == "NEON"
        assert captured["service"] == OPERATOR_SERVICE
        assert captured["npub"] == rt.operator_npub()
        # Stamped as delivered even though only one field moved.
        assert captured["just_delivered"] == ["btcpay_api_key"]

    @pytest.mark.asyncio
    async def test_refuses_to_merge_into_an_unread_blob(self, monkeypatch) -> None:
        """A cold vault must abort the write, not write what little it saw."""
        rt = _operator_runtime()
        stored: list[dict] = []

        async def _cold(npub, *, service=None):
            return None, "vault_bootstrapping"

        async def _store(npub, data, *, service=None, just_delivered=None):
            stored.append(data)
            return True

        monkeypatch.setattr(rt, "load_patron_session", _cold)
        monkeypatch.setattr(rt, "store_patron_session", _store)

        assert await rt.update_operator_credential("btcpay_api_key", "NEW") is False
        assert stored == []

    @pytest.mark.asyncio
    async def test_creates_the_blob_when_the_vault_is_genuinely_empty(
        self, monkeypatch,
    ) -> None:
        """Empty-with-no-situation is a real state, distinct from unreadable."""
        rt = _operator_runtime()
        captured: dict = {}

        async def _empty(npub, *, service=None):
            return None, ""

        async def _store(npub, data, *, service=None, just_delivered=None):
            captured["data"] = data
            return True

        monkeypatch.setattr(rt, "load_patron_session", _empty)
        monkeypatch.setattr(rt, "store_patron_session", _store)

        assert await rt.update_operator_credential("btcpay_host", "H") is True
        assert captured["data"] == {"btcpay_host": "H"}

    @pytest.mark.asyncio
    async def test_returns_false_without_an_operator_template(self) -> None:
        rt = OperatorRuntime(tool_registry={}, service_name="No Template")
        rt.operator_npub = lambda: OPERATOR_NPUB  # type: ignore[method-assign]
        assert await rt.update_operator_credential("btcpay_api_key", "NEW") is False


class TestSharedCredentialMerge:
    """``update_credential_field`` is the one read-merge-write for both actors."""

    @pytest.mark.asyncio
    async def test_honours_the_service_the_caller_names(self, monkeypatch) -> None:
        rt = _operator_runtime()
        captured: dict = {}

        async def _load(npub, *, service=None):
            captured["load_service"] = service
            return {"existing": "kept"}, ""

        async def _store(npub, data, *, service=None, just_delivered=None):
            captured["store_service"] = service
            captured["data"] = data
            return True

        monkeypatch.setattr(rt, "load_patron_session", _load)
        monkeypatch.setattr(rt, "store_patron_session", _store)

        ok = await rt.update_credential_field(
            "npub1" + "a" * 58, "field", "value", service="some-other-service",
        )

        assert ok is True
        assert captured["load_service"] == "some-other-service"
        assert captured["store_service"] == "some-other-service"
        assert captured["data"] == {"existing": "kept", "field": "value"}

    @pytest.mark.asyncio
    async def test_patron_update_delegates_to_the_shared_helper(
        self, monkeypatch,
    ) -> None:
        """The patron entry point keeps its behaviour after the extraction."""
        rt = _operator_runtime()
        seen: dict = {}

        async def _shared(npub, field, value, *, service):
            seen.update(npub=npub, field=field, value=value, service=service)
            return True

        monkeypatch.setattr(rt, "update_credential_field", _shared)
        monkeypatch.setattr(rt, "_patron_storage_service", lambda s: "patronsvc")

        assert await rt.update_patron_credential(
            "npub1" + "b" * 58, "account_hash", "H",
        ) is True
        assert seen["service"] == "patronsvc"
        assert seen["field"] == "account_hash"
