"""Tests for the provisioning GRANT self-heal.

The ALTER OWNER + REVOKE sequence can strand a table with an empty ACL
(``relacl = {}``), which strips even the owner's implicit privileges —
the 2026-06-06 excalibur outage. Provisioning must therefore end with an
unconditional GRANT back to the operator role.
"""

from __future__ import annotations

import pytest

from tollbooth.authority.tenant_provisioner import (
    provision_operator_schema,
    restore_operator_grants,
    schema_name_for_npub,
)

NPUB = "npub19gqrkwsssnz5dl6mj54memy65g4lq7qu7efu532nma3p8c6yzugse3f7a2"
BASE_URL = "postgresql://neondb_owner:pw@ep-test-pooler.aws.neon.tech/neondb"


class _RecordingVault:
    """Records every SQL statement; all statements succeed."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def _execute(self, query: str, params: list | None = None) -> dict:
        self.statements.append(query)
        return {"rows": [], "rowCount": 0, "command": "OK"}


class TestRestoreOperatorGrants:
    @pytest.mark.asyncio
    async def test_grants_all_tables_and_sequences(self) -> None:
        vault = _RecordingVault()
        await restore_operator_grants(vault, "op_test")
        assert (
            'GRANT ALL ON ALL TABLES IN SCHEMA "op_test" TO "op_test"'
            in vault.statements
        )
        assert (
            'GRANT ALL ON ALL SEQUENCES IN SCHEMA "op_test" TO "op_test"'
            in vault.statements
        )


class TestProvisionEndsWithSelfHeal:
    @pytest.mark.asyncio
    async def test_grant_runs_after_revoke(self) -> None:
        vault = _RecordingVault()
        schema = schema_name_for_npub(NPUB)
        await provision_operator_schema(vault, NPUB, base_url=BASE_URL)

        grant = vault.statements.index(
            f'GRANT ALL ON ALL TABLES IN SCHEMA "{schema}" TO "{schema}"'
        )
        revokes = [
            i for i, s in enumerate(vault.statements) if s.startswith("REVOKE")
        ]
        assert revokes, "expected REVOKE statements during provisioning"
        assert grant > max(revokes), (
            "self-heal GRANT must be the final word after every REVOKE"
        )
