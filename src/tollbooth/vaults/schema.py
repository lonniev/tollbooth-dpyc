"""Schema preparation: one breadcrumb, one read.

A cold operator process used to run every ``CREATE … IF NOT EXISTS`` its
tables need — 22 to 24 sequential round trips to Neon — before its first tool
could touch the database, almost always to change nothing. Horizon freezes a
process between requests, so that cost could not be moved off the request
path; it had to go.

Now one breadcrumb row, ``tollbooth_schema``, records that the database is
fully prepared for ``SCHEMA_VERSION``. A cold start reads it with one SELECT
and, in the usual case, is done. When it is absent or older, each concern
prepares its own tables in one compound request — a Neon batch, one round
trip and one transaction — and the crumb is written last, so a crumb is only
ever present when everything before it landed.

Each concern keeps its DDL beside the code that uses its tables, as a pure
``schema_statements(t, idx)``: ``t`` names a table in the operator's schema,
``idx`` prefixes index names (index names are schema-scoped in Postgres, and
the fleet's operators share one database). This module only knows the list
and the version.

**Changing any DDL means bumping ``SCHEMA_VERSION``** — and updating
``SCHEMA_DIGEST``; ``tests/test_schema_prep.py`` fails until both move, so a
schema change can never ship behind a crumb that says it is already done.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2
# sha256 of every concern's statements rendered for the test schema "op_test".
SCHEMA_DIGEST = "0d36fb0d481cd85e9508d64c7303d787a432e29ec564808d4dcf5b64d6158c63"

Statements = Callable[[Callable[[str], str], str], list[str]]

# SQLSTATEs that mean "nothing has been prepared here yet": the crumb's table,
# or the operator's schema itself, does not exist.
_ABSENT = {"42P01", "3F000"}


class SchemaPrepError(Exception):
    """A preparation step failed; no breadcrumb was written.

    ``steps`` records every step taken, the failing one last with Neon's own
    error text, so ``restore_neon_schema`` can show exactly where it stopped.
    """

    def __init__(self, steps: list[dict[str, Any]]) -> None:
        failed = steps[-1] if steps else {}
        super().__init__(f"schema step {failed.get('step', '?')!r} failed: {failed.get('error', '')}")
        self.steps = steps


def concerns() -> tuple[tuple[str, Statements], ...]:
    """Every concern that owns tables, in dependency order."""
    from tollbooth.async_jobs import schema_statements as async_jobs
    from tollbooth.coupons.vault import schema_statements as coupons
    from tollbooth.pricing_store import schema_statements as pricing
    from tollbooth.vaults.neon import (
        credential_schema_statements as credentials,
    )
    from tollbooth.vaults.neon import (
        ledger_schema_statements as ledger,
    )
    from tollbooth.vaults.neon import (
        notarization_schema_statements as notarization,
    )

    return (
        ("ledger", ledger),
        ("notarization", notarization),
        ("pricing", pricing),
        ("coupons", coupons),
        ("async_jobs", async_jobs),
        ("credentials", credentials),
    )


def crumb_statements(t: Callable[[str], str], version: int) -> list[str]:
    """The breadcrumb: a one-row table saying which version is prepared."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t('tollbooth_schema')} ("
            "    id BOOLEAN PRIMARY KEY DEFAULT true CHECK (id),"
            "    version INTEGER NOT NULL,"
            "    prepared_at TIMESTAMPTZ NOT NULL DEFAULT now()"
            ")"
        ),
        (
            f"INSERT INTO {t('tollbooth_schema')} (id, version, prepared_at) "
            f"VALUES (true, {int(version)}, now()) "
            "ON CONFLICT (id) DO UPDATE SET version = EXCLUDED.version, prepared_at = now()"
        ),
    ]


def digest(t: Callable[[str], str], idx: str) -> str:
    """A fingerprint of every concern's DDL, for the version guard."""
    h = hashlib.sha256()
    for name, statements in concerns():
        h.update(name.encode())
        for sql in statements(t, idx):
            h.update(sql.encode())
    return h.hexdigest()


async def prepared_version(vault: Any) -> int | None:
    """The version the database says it is prepared for, or None if unprepared.

    Any error other than "nothing here yet" propagates: a read that failed must
    never be mistaken for a database that needs preparing.
    """
    from tollbooth.vaults.neon import NeonQueryError

    try:
        result = await vault._execute(f"SELECT version FROM {vault._t('tollbooth_schema')}")
    except NeonQueryError as exc:
        if exc.code in _ABSENT:
            return None
        raise
    rows = result.get("rows", [])
    return int(rows[0]["version"]) if rows else None


async def prepare_schema(vault: Any, *, force: bool = False) -> list[dict[str, Any]]:
    """Make sure the operator's database is prepared for ``SCHEMA_VERSION``.

    One SELECT when it already is. Otherwise: the operator's schema (a probe,
    then CREATE SCHEMA only if missing — the operator role owns its schema but
    may not CREATE on the database), one compound request per concern, and the
    crumb last. ``force`` skips the read, for ``restore_neon_schema``.

    Returns one ``{"step", "ok"}`` per step taken (empty when nothing was
    needed). A failed step stops the preparation and raises ``SchemaPrepError``
    carrying the steps; no crumb is written.
    """
    if not force:
        have = await prepared_version(vault)
        if have is not None and have >= SCHEMA_VERSION:
            return []

    steps: list[dict[str, Any]] = []
    schema = vault._schema_prefix.rstrip(".")
    idx = f"{schema}_" if schema else ""

    async def step(name: str, action: Any) -> None:
        try:
            await action
        except Exception as exc:
            steps.append({"step": name, "ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})
            raise SchemaPrepError(steps) from exc
        steps.append({"step": name, "ok": True})

    if schema:
        probe = await vault._execute("SELECT 1 FROM pg_namespace WHERE nspname = $1", [schema])
        if not probe.get("rows"):
            await step("schema", vault._execute(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
    for name, statements in concerns():
        await step(name, vault._execute_batch(statements(vault._t, idx)))
    await step("crumb", vault._execute_batch(crumb_statements(vault._t, SCHEMA_VERSION)))
    logger.info("Schema prepared for version %d (%d steps).", SCHEMA_VERSION, len(steps))
    return steps
