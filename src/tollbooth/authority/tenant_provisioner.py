"""Neon tenant provisioning for operator isolation.

Each registered operator gets its own PostgreSQL schema within the
Authority's Neon database. The schema name is derived from the operator's
npub to ensure uniqueness and prevent cross-tenant access.

The Authority's bootstrap_config table stores operator-specific settings
(like the schema-qualified Neon URL) in the Authority's own schema,
gated by Schnorr proof in the get_operator_config tool.
"""

from __future__ import annotations

import hashlib
import logging
import re
import secrets
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qs, urlencode, quote

logger = logging.getLogger(__name__)


def schema_name_for_npub(npub: str) -> str:
    """Derive a Postgres-safe schema name from an npub.

    Uses first 16 chars of SHA-256 hex digest — short, unique, safe.
    Prefixed with 'op_' so it doesn't collide with system schemas.
    """
    digest = hashlib.sha256(npub.encode()).hexdigest()[:16]
    return f"op_{digest}"


def _validate_schema_name(schema: str) -> None:
    """Reject schema names that aren't safe Postgres identifiers."""
    if not re.match(r"^[a-z][a-z0-9_]*$", schema):
        raise ValueError(f"Unsafe schema name: {schema!r}")


def neon_url_with_schema(base_url: str, schema: str) -> str:
    """Append search_path option to a Neon connection URL.

    The operator connects with this URL and all tables resolve within
    the operator's isolated schema.
    """
    _validate_schema_name(schema)
    parsed = urlparse(base_url)
    params = parse_qs(parsed.query)
    params["options"] = [f"-c search_path={schema}"]
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(query=new_query))


def extract_authority_role(database_url: str) -> str:
    """Parse the Authority's Postgres role name from its connection URL."""
    return urlparse(database_url).username or ""


def neon_url_for_operator(base_url: str, schema: str, password: str) -> str:
    """Build a connection URL with operator-scoped role credentials.

    Replaces the Authority's user:pass with the operator role and
    generated password. Keeps ``search_path`` option.
    """
    parsed = urlparse(base_url)
    encoded_pw = quote(password, safe="")
    netloc = f"{schema}:{encoded_pw}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    params = parse_qs(parsed.query)
    params["options"] = [f"-c search_path={schema}"]
    new_query = urlencode(params, doseq=True)
    return urlunparse(parsed._replace(netloc=netloc, query=new_query))


# -- Per-operator Postgres role management ------------------------------------

_PROVISIONER_TABLES = (
    "balances", "transactions", "anchors", "tool_demand",
    "operator_pricing_models", "credentials", "session_bindings",
)


_SAFE_IDENTIFIER = re.compile(r"^[a-z0-9_]+$")
_SAFE_PASSWORD = re.compile(r"^[A-Za-z0-9_\-]+$")


async def create_operator_role(vault: Any, schema: str, password: str) -> None:
    """Create a Postgres LOGIN role for an operator schema.

    Idempotent: if the role already exists, resets its password.
    """
    if not _SAFE_IDENTIFIER.match(schema):
        raise ValueError(f"Unsafe schema name: {schema!r}")
    if not _SAFE_PASSWORD.match(password):
        raise ValueError("Password contains unsafe characters")
    try:
        await vault._execute(
            f'CREATE ROLE "{schema}" WITH LOGIN PASSWORD \'{password}\''
        )
        logger.info("Created Postgres role '%s'", schema)
    except Exception as exc:
        # Neon HTTP API returns 400 with error body — extract the message
        exc_text = str(exc).lower()
        resp_text = ""
        if hasattr(exc, "response"):
            try:
                resp_text = exc.response.text.lower()
            except Exception:
                pass
        if "already exists" in exc_text or "already exists" in resp_text:
            await vault._execute(
                f'ALTER ROLE "{schema}" WITH PASSWORD \'{password}\''
            )
            logger.info("Reset password for existing role '%s'", schema)
        else:
            raise

    # Allow Authority to SET ROLE for ownership transfers
    await vault._execute(f'GRANT "{schema}" TO CURRENT_USER')
    await vault._execute(f'GRANT USAGE ON SCHEMA "{schema}" TO "{schema}"')
    await vault._execute(f'GRANT CREATE ON SCHEMA "{schema}" TO "{schema}"')
    await vault._execute(
        f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" '
        f'GRANT ALL ON TABLES TO "{schema}"'
    )
    await vault._execute(
        f'ALTER DEFAULT PRIVILEGES IN SCHEMA "{schema}" '
        f'GRANT ALL ON SEQUENCES TO "{schema}"'
    )


async def transfer_schema_ownership(vault: Any, schema: str) -> None:
    """Transfer schema and table ownership to the operator role."""
    await vault._execute(f'ALTER SCHEMA "{schema}" OWNER TO "{schema}"')
    for table in _PROVISIONER_TABLES:
        try:
            await vault._execute(
                f'ALTER TABLE "{schema}".{table} OWNER TO "{schema}"'
            )
        except Exception:
            pass  # table may not exist if operator never fully initialized


async def revoke_authority_access(
    vault: Any, schema: str, authority_role: str,
) -> None:
    """Revoke the Authority's DML access to an operator schema."""
    await vault._execute(
        f'REVOKE ALL ON ALL TABLES IN SCHEMA "{schema}" FROM PUBLIC'
    )
    await vault._execute(
        f'REVOKE ALL ON SCHEMA "{schema}" FROM PUBLIC'
    )
    if authority_role:
        await vault._execute(
            f'REVOKE ALL ON ALL TABLES IN SCHEMA "{schema}" FROM "{authority_role}"'
        )


def generate_operator_password() -> str:
    """Generate a secure random password for an operator role."""
    return secrets.token_urlsafe(32)


# -- Bootstrap table ---------------------------------------------------------

def _t(vault: Any, table: str) -> str:
    """Schema-qualified table name, delegated to the vault if available."""
    return vault._t(table) if hasattr(vault, '_t') else table


async def ensure_bootstrap_table(vault: Any) -> None:
    """Create the bootstrap_config table if it doesn't exist.

    Stores operator-specific key-value pairs in the Authority's schema.
    Access is gated by Schnorr proof in the get_operator_config tool.
    """
    await vault._execute(f"""
        CREATE TABLE IF NOT EXISTS {_t(vault, 'bootstrap_config')} (
            npub TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (npub, key)
        )
    """)


async def provision_operator_schema(
    vault: Any,
    npub: str,
    base_url: str = "",
    authority_nsec_hex: str = "",
) -> tuple[str, str]:
    """Create an isolated Postgres schema with a per-operator role.

    Creates the schema, tables, a LOGIN role scoped to the schema,
    transfers ownership, and revokes Authority DML access.

    Returns ``(schema_name, role_password)``.
    """
    schema = schema_name_for_npub(npub)

    # Create schema (idempotent). The SDK's ensure_schema() creates
    # the actual tables (balances, transactions, etc.) on first boot.
    await vault._execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

    # Create operator role, transfer ownership, revoke Authority access
    password = generate_operator_password()
    await create_operator_role(vault, schema, password)
    await transfer_schema_ownership(vault, schema)
    if base_url:
        authority_role = extract_authority_role(base_url)
        await revoke_authority_access(vault, schema, authority_role)

    logger.info("Provisioned schema '%s' with isolated role for operator %s", schema, npub[:16])
    return schema, password


async def store_operator_config(
    vault: Any, npub: str, key: str, value: str
) -> None:
    """Store a config entry in the bootstrap table."""
    await vault._execute(
        f"INSERT INTO {_t(vault, 'bootstrap_config')} (npub, key, value) "
        "VALUES ($1, $2, $3) "
        "ON CONFLICT (npub, key) "
        "DO UPDATE SET value = $3, created_at = now()",
        [npub, key, value],
    )


async def get_operator_config_value(
    vault: Any, npub: str, key: str
) -> str | None:
    """Retrieve a config entry from the bootstrap table."""
    result = await vault._execute(
        f"SELECT value FROM {_t(vault, 'bootstrap_config')} WHERE npub = $1 AND key = $2",
        [npub, key],
    )
    rows = result.get("rows", [])
    if rows:
        return rows[0][0] if isinstance(rows[0], list) else rows[0].get("value")
    return None


async def get_all_operator_config(vault: Any, npub: str) -> dict[str, str]:
    """Retrieve all config entries for an operator."""
    result = await vault._execute(
        f"SELECT key, value FROM {_t(vault, 'bootstrap_config')} WHERE npub = $1",
        [npub],
    )
    config: dict[str, str] = {}
    for row in result.get("rows", []):
        if isinstance(row, list):
            config[row[0]] = row[1]
        else:
            config[row.get("key", "")] = row.get("value", "")
    return config
