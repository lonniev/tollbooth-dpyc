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
