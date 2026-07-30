"""Per-field delivered_at on courier receipt and field listings (issue #166).

Confirms the gap that existed before the fix: vaulting a field left no way to
ask "how old is this secret?", then shows that receive / update stamp
ISO-8601 times and listing / onboarding surfaces expose them.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pynostr.key import PrivateKey

from tollbooth.credential_meta import META_KEY, get_delivered_at, strip_meta
from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_credentials import NostrCredentialExchange
from tollbooth.runtime import OperatorRuntime, classify_operator_secrets

_OPERATOR = PrivateKey()
os.environ.setdefault("TOLLBOOTH_NOSTR_OPERATOR_NSEC", _OPERATOR.nsec)

PATRON = "npub1" + "a" * 58


class MockCredentialVault:
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def _key(self, service: str, npub: str) -> str:
        return f"{service}:{npub}"

    async def store_credentials(self, service: str, npub: str, encrypted_blob: str) -> None:
        self._store[self._key(service, npub)] = encrypted_blob

    async def fetch_credentials(self, service: str, npub: str) -> str | None:
        return self._store.get(self._key(service, npub))

    async def delete_credentials(self, service: str, npub: str) -> bool:
        key = self._key(service, npub)
        if key in self._store:
            del self._store[key]
            return True
        return False


def _patron_template() -> dict[str, CredentialTemplate]:
    return {
        "patron-svc": CredentialTemplate(
            service="patron-svc",
            version=1,
            fields={
                "api_key": FieldSpec(required=True, sensitive=True, description="key"),
                "note": FieldSpec(required=False, sensitive=False, description="note"),
            },
        ),
    }


def _exchange_with_vault(vault: MockCredentialVault) -> NostrCredentialExchange:
    return NostrCredentialExchange(
        nsec=_OPERATOR.nsec,
        relays=["wss://relay.test.invalid"],
        templates=_patron_template(),
        credential_vault=vault,
    )


class TestVaultStoreStampsDelivery:
    @pytest.mark.asyncio
    async def test_first_store_records_delivered_at(self) -> None:
        vault = MockCredentialVault()
        ex = _exchange_with_vault(vault)
        before = datetime.now(UTC)

        ok = await ex._vault_store(
            "patron-svc",
            PATRON,
            {"api_key": "sk-1", "note": "hello"},
            just_delivered=["api_key", "note"],
        )
        assert ok is True

        blob = await ex._vault_fetch("patron-svc", PATRON)
        assert blob is not None
        # Value path never shocks callers with meta
        assert strip_meta(blob) == {"api_key": "sk-1", "note": "hello"}
        assert META_KEY in blob

        api_ts = get_delivered_at(blob, "api_key")
        note_ts = get_delivered_at(blob, "note")
        assert api_ts is not None and note_ts is not None
        # ISO-8601 with timezone; fresher than the moment before store
        parsed = datetime.fromisoformat(api_ts)
        assert parsed.tzinfo is not None
        assert parsed >= before

    @pytest.mark.asyncio
    async def test_partial_redelivery_only_restamps_delivered_field(self) -> None:
        vault = MockCredentialVault()
        ex = _exchange_with_vault(vault)

        await ex._vault_store(
            "patron-svc",
            PATRON,
            {"api_key": "sk-old", "note": "kept"},
            just_delivered=["api_key", "note"],
        )
        first = await ex._vault_fetch("patron-svc", PATRON)
        first_note = get_delivered_at(first, "note")
        first_key = get_delivered_at(first, "api_key")

        await ex._vault_store(
            "patron-svc",
            PATRON,
            {"api_key": "sk-new", "note": "kept"},
            just_delivered=["api_key"],
        )
        second = await ex._vault_fetch("patron-svc", PATRON)
        assert get_delivered_at(second, "api_key") != first_key
        assert get_delivered_at(second, "note") == first_note


class TestListingsSurfaceTimestamps:
    def _runtime_with_exchange(self, exchange) -> OperatorRuntime:
        rt = OperatorRuntime(
            patron_credential_template=CredentialTemplate(
                service="patron-svc",
                version=1,
                fields={
                    "api_key": FieldSpec(required=True, sensitive=True, description="key"),
                    "note": FieldSpec(required=False, sensitive=False, description="note"),
                },
            ),
        )
        courier = MagicMock()
        courier._exchange = exchange
        rt.courier = AsyncMock(return_value=courier)  # type: ignore[method-assign]
        rt.operator_npub = MagicMock(return_value="npub1operatorxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")  # type: ignore[method-assign]
        return rt

    @pytest.mark.asyncio
    async def test_list_fields_includes_delivered_at(self) -> None:
        vault = MockCredentialVault()
        ex = _exchange_with_vault(vault)
        await ex._vault_store(
            "patron-svc",
            PATRON,
            {"api_key": "sk-1"},
            just_delivered=["api_key"],
        )
        rt = self._runtime_with_exchange(ex)

        details = await rt.list_patron_credential_field_details(PATRON)
        assert len(details) == 1
        assert details[0]["field"] == "api_key"
        assert details[0]["delivered_at"] is not None
        # name-only list stays a plain list of strings
        names = await rt.list_patron_credential_fields(PATRON)
        assert names == ["api_key"]
        # meta itself never appears as a field
        assert META_KEY not in names

    @pytest.mark.asyncio
    async def test_legacy_blob_lists_with_null_timestamp(self) -> None:
        """Blobs written before timestamps existed still list, without inventing a stamp."""
        vault = MockCredentialVault()
        exp = _exchange_with_vault(vault)
        # Write a legacy shape directly into the vault (no __meta__)
        plain = json.dumps({"api_key": "legacy"})
        blob = exp._vault_encrypt(plain)
        await vault.store_credentials("patron-svc", PATRON, blob)

        rt = self._runtime_with_exchange(exp)
        details = await rt.list_patron_credential_field_details(PATRON)
        assert details == [{"field": "api_key", "delivered_at": None}]

    @pytest.mark.asyncio
    async def test_update_patron_credential_stamps_field(self) -> None:
        vault = MockCredentialVault()
        ex = _exchange_with_vault(vault)
        rt = self._runtime_with_exchange(ex)

        ok = await rt.update_patron_credential(PATRON, "api_key", "sk-set")
        assert ok is True
        details = await rt.list_patron_credential_field_details(PATRON)
        by_name = {d["field"]: d["delivered_at"] for d in details}
        assert by_name["api_key"] is not None

    @pytest.mark.asyncio
    async def test_patron_onboarding_status_carries_delivered_at(self) -> None:
        vault = MockCredentialVault()
        ex = _exchange_with_vault(vault)
        await ex._vault_store(
            "patron-svc",
            PATRON,
            {"api_key": "sk-1", "note": "n"},
            just_delivered=["api_key", "note"],
        )
        rt = self._runtime_with_exchange(ex)
        status = await rt.patron_onboarding_status(PATRON)
        configured = {c["field"]: c for c in status["configured"]}
        assert "api_key" in configured
        assert configured["api_key"]["delivered_at"] is not None
        assert "delivered_at" in configured["note"]


class TestClassifyOperatorSecretsTimestamps:
    def test_configured_entry_carries_stamp_when_supplied(self) -> None:
        fields = {
            "btcpay_host": FieldSpec(required=True, sensitive=False, description="h", lifecycle="set_once"),
        }
        cfg, miss, opt = classify_operator_secrets(
            fields,
            {"btcpay_host"},
            {"btcpay_host"},
            delivered_at={"btcpay_host": "2026-07-30T00:00:00+00:00"},
        )
        assert miss == [] and opt == []
        assert cfg[0]["field"] == "btcpay_host"
        assert cfg[0]["delivered_at"] == "2026-07-30T00:00:00+00:00"

    def test_missing_stamp_omitted_not_null(self) -> None:
        """Configured entries without a known stamp stay shape-compatible with pre-#166 clients."""
        fields = {
            "btcpay_host": FieldSpec(required=True, sensitive=False, description="h", lifecycle="set_once"),
        }
        cfg, _, _ = classify_operator_secrets(fields, {"btcpay_host"}, {"btcpay_host"})
        assert "delivered_at" not in cfg[0]
