-- Optionality's NeonVault recovery DDL
-- Run in Neon's SQL editor (or psql) against Optionality's database.
-- All statements are idempotent (IF NOT EXISTS) — safe to re-run.
--
-- Schema name is derived from sha256(operator_npub)[:16] with `op_` prefix.
-- For Optionality (npub1qchpfnuw76gjmg6y9w4e5yaj9xcq97ft6rd49mj4j0vrsdrkujpqzl7tz0)
-- the schema is op_37b76c01fac616ef. Confirm against service_status's
-- operator_npub_hash field (`37b76c01fac6`).

CREATE SCHEMA IF NOT EXISTS op_37b76c01fac616ef;
SET search_path TO op_37b76c01fac616ef, public;

-- == NeonVault tables (vaults/neon.py::NeonVault.ensure_schema) ==

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.balances (
    npub        TEXT PRIMARY KEY,
    ledger_json TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    last_flush  TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.transactions (
    id              BIGSERIAL PRIMARY KEY,
    npub            TEXT NOT NULL,
    tx_type         TEXT NOT NULL,
    amount_api_sats INTEGER NOT NULL,
    tool_name       TEXT,
    invoice_id      TEXT,
    detail          TEXT,
    balance_after   INTEGER NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS op_37b76c01fac616ef_idx_transactions_npub
    ON op_37b76c01fac616ef.transactions(npub);
CREATE INDEX IF NOT EXISTS op_37b76c01fac616ef_idx_transactions_created
    ON op_37b76c01fac616ef.transactions(created_at);

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.anchors (
    id               BIGSERIAL PRIMARY KEY,
    root_hash        TEXT NOT NULL UNIQUE,
    leaf_count       INTEGER NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    ots_receipts_json TEXT,
    snapshot_json    TEXT NOT NULL,
    leaf_hashes_json TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    confirmed_at     TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS op_37b76c01fac616ef_idx_anchors_created
    ON op_37b76c01fac616ef.anchors(created_at);
CREATE INDEX IF NOT EXISTS op_37b76c01fac616ef_idx_anchors_status
    ON op_37b76c01fac616ef.anchors(status);

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.tool_demand (
    tool_name  TEXT NOT NULL,
    window_key TEXT NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tool_name, window_key)
);

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.authority_config (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.operator_pricing_models (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator   TEXT NOT NULL,
    name       TEXT NOT NULL,
    model_json JSONB NOT NULL,
    is_active  BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS op_37b76c01fac616ef_one_active_per_operator
    ON op_37b76c01fac616ef.operator_pricing_models (operator) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS op_37b76c01fac616ef_idx_pricing_models_operator
    ON op_37b76c01fac616ef.operator_pricing_models (operator);

-- == Credential vault tables (vaults/neon.py::NeonCredentialVault.ensure_schema) ==
-- These also hold the `proven_npub:*` service entries that the Secure Courier
-- proof flow caches across cold-starts.

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.credentials (
    service         TEXT NOT NULL,
    npub            TEXT NOT NULL,
    encrypted_blob  TEXT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (service, npub)
);

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.session_bindings (
    caller_id  TEXT NOT NULL,
    service    TEXT NOT NULL,
    npub       TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (caller_id, service)
);

-- == Bootstrap config (authority/tenant_provisioner.py::provision_operator_schema) ==
-- Used by the wheel's bootstrap to remember its own Neon URL / schema / role password.

CREATE TABLE IF NOT EXISTS op_37b76c01fac616ef.bootstrap_config (
    npub       TEXT NOT NULL,
    key        TEXT NOT NULL,
    value      TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (npub, key)
);

-- == Sanity check ==
-- Verify every table now exists in the schema. All 9 should appear.
SELECT tablename
FROM pg_tables
WHERE schemaname = 'op_37b76c01fac616ef'
ORDER BY tablename;
