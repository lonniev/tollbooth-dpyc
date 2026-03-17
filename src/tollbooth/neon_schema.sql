-- NeonVault schema for tollbooth-dpyc
-- Run once per Neon database. Safe to re-run (IF NOT EXISTS).
-- Also available programmatically via NeonVault.ensure_schema().

-- Core ledger state (one row per npub)
CREATE TABLE IF NOT EXISTS balances (
    npub TEXT PRIMARY KEY,
    ledger_json TEXT NOT NULL,        -- Full UserLedger serialized via to_json()
    version INTEGER NOT NULL DEFAULT 1, -- Optimistic concurrency control
    last_flush TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Append-only transaction journal (the irrefutable record)
CREATE TABLE IF NOT EXISTS transactions (
    id BIGSERIAL PRIMARY KEY,
    npub TEXT NOT NULL,
    tx_type TEXT NOT NULL,            -- 'credit', 'debit', 'rollback', 'expire', 'snapshot'
    amount_api_sats INTEGER NOT NULL,
    tool_name TEXT,                    -- NULL for credits/snapshots, tool name for debits
    invoice_id TEXT,                   -- NULL for debits, invoice for credits
    detail TEXT,                       -- Human-readable context
    balance_after INTEGER NOT NULL,    -- Running balance snapshot
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_transactions_npub ON transactions(npub);
CREATE INDEX IF NOT EXISTS idx_transactions_created ON transactions(created_at);

-- Global demand counters for surge pricing (one row per tool per time window)
CREATE TABLE IF NOT EXISTS tool_demand (
    tool_name TEXT NOT NULL,
    window_key TEXT NOT NULL,         -- Truncated timestamp, e.g. "2026-03-05T14:00"
    count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (tool_name, window_key)
);

-- Operator pricing models (runtime-configurable tool pricing)
CREATE TABLE IF NOT EXISTS operator_pricing_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator TEXT NOT NULL,
    name TEXT NOT NULL,
    model_json JSONB NOT NULL,        -- {name, tools: [...], pipeline: [...]}
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_per_operator
    ON operator_pricing_models (operator) WHERE is_active = true;
CREATE INDEX IF NOT EXISTS idx_pricing_models_operator
    ON operator_pricing_models (operator);

-- Nostr-signed tool ACLs (per-tool authorization for non-operator callers)
CREATE TABLE IF NOT EXISTS tool_acls (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operator TEXT NOT NULL,
    tool_pattern TEXT NOT NULL,
    signed_event_json JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (operator, tool_pattern)
);
CREATE INDEX IF NOT EXISTS idx_tool_acls_operator
    ON tool_acls (operator);
