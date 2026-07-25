"""Migrate existing operator schemas to per-operator Postgres roles.

Run once (idempotent) to create isolated LOGIN roles for each operator,
transfer schema/table ownership, and revoke Authority DML access.
Delivers operator-scoped credentials via updated bootstrap DM.

Usage::

    python -m tollbooth.authority.role_migration
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tollbooth.authority.tenant_provisioner import (
    create_operator_role,
    extract_authority_role,
    generate_operator_password,
    get_all_operator_config,
    neon_url_for_operator,
    revoke_authority_access,
    store_operator_config,
    transfer_schema_ownership,
)

logger = logging.getLogger(__name__)


async def migrate_single_operator(
    vault: Any,
    npub: str,
    base_url: str,
    authority_nsec_hex: str,
) -> dict[str, Any]:
    """Migrate one operator to a per-role isolated schema.

    1. Create Postgres role matching the schema name.
    2. Transfer schema + table ownership to the new role.
    3. Revoke Authority DML access.
    4. Build operator-scoped connection URL.
    5. Encrypt password and store in bootstrap_config.
    6. Update stored neon_database_url with operator credentials.

    Returns a result dict with success status and details.
    """
    config = await get_all_operator_config(vault, npub)
    schema = config.get("schema", "")
    if not schema:
        return {"success": False, "npub": npub, "error": "No schema in bootstrap_config"}

    password = generate_operator_password()

    try:
        await create_operator_role(vault, schema, password)
        await transfer_schema_ownership(vault, schema)

        authority_role = extract_authority_role(base_url)
        await revoke_authority_access(vault, schema, authority_role)

        new_url = neon_url_for_operator(base_url, schema, password)

        # Encrypt password before storing
        from tollbooth.vault_encryption import VaultCipher
        cipher = VaultCipher(nsec_hex=authority_nsec_hex)
        encrypted_pw = cipher.encrypt(password)
        await store_operator_config(vault, npub, "role_password", encrypted_pw)
        await store_operator_config(vault, npub, "neon_database_url", new_url)

        logger.info(
            "Migrated operator %s schema=%s to isolated role",
            npub[:20], schema,
        )
        return {"success": True, "npub": npub, "schema": schema}

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Migration failed for operator %s schema=%s: %s",
            npub[:20], schema, exc,
        )
        return {"success": False, "npub": npub, "schema": schema, "error": str(exc)}


async def migrate_all_operators(
    vault: Any,
    base_url: str,
    authority_nsec_hex: str,
) -> list[dict[str, Any]]:
    """Migrate all existing operators to per-role isolation.

    Enumerates operators from bootstrap_config and migrates each.
    Failures are logged and returned but do not stop the batch.
    """
    t = vault._t if hasattr(vault, '_t') else lambda x: x
    result = await vault._execute(
        f"SELECT DISTINCT npub FROM {t('bootstrap_config')} WHERE key = 'schema'"
    )
    rows = result.get("rows", [])
    npubs = []
    for row in rows:
        if isinstance(row, list):
            npubs.append(row[0])
        else:
            npubs.append(row.get("npub", ""))

    logger.info("Found %d operators to migrate", len(npubs))

    results = []
    for npub in npubs:
        if not npub:
            continue
        r = await migrate_single_operator(vault, npub, base_url, authority_nsec_hex)
        results.append(r)

    succeeded = sum(1 for r in results if r.get("success"))
    failed = len(results) - succeeded
    logger.info("Migration complete: %d succeeded, %d failed", succeeded, failed)

    return results


async def _main() -> None:
    """CLI entry point for migration.

    Environment variables:
        NEON_DATABASE_URL — Authority's Neon connection string (required)
        TOLLBOOTH_NOSTR_OPERATOR_NSEC — Authority's nsec (required)
        NPUB — If set, migrate only this operator. If empty, migrate all.
    """
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    neon_url = os.environ.get("NEON_DATABASE_URL", "")
    nsec = os.environ.get("TOLLBOOTH_NOSTR_OPERATOR_NSEC", "")
    single_npub = os.environ.get("NPUB", "")

    if not neon_url:
        raise SystemExit("NEON_DATABASE_URL is required")
    if not nsec:
        raise SystemExit("TOLLBOOTH_NOSTR_OPERATOR_NSEC is required")

    # Derive nsec hex
    from pynostr.key import PrivateKey
    if nsec.startswith("nsec1"):
        nsec_hex = PrivateKey.from_nsec(nsec).hex()
    else:
        nsec_hex = nsec

    from tollbooth.vaults.neon import NeonVault
    vault = NeonVault(database_url=neon_url, encryption_nsec_hex=nsec_hex)

    try:
        if single_npub:
            logger.info("Migrating single operator: %s", single_npub[:20])
            r = await migrate_single_operator(vault, single_npub, neon_url, nsec_hex)
            status = "OK" if r.get("success") else "FAILED"
            logger.info("  %s %s %s", status, r.get("npub", "")[:20], r.get("error", ""))
        else:
            results = await migrate_all_operators(vault, neon_url, nsec_hex)
            for r in results:
                status = "OK" if r.get("success") else "FAILED"
                logger.info("  %s %s %s", status, r.get("npub", "")[:20], r.get("error", ""))
    finally:
        await vault.close()


if __name__ == "__main__":
    asyncio.run(_main())
