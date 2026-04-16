-- Neon cleanup for tollbooth-dpyc v0.10.0
--
-- TWO CONTEXTS:
--   A) Run against the AUTHORITY's Neon (public schema cleanup)
--   B) Run against each OPERATOR's Neon (per-schema cleanup)
--
-- Safe to run multiple times (idempotent).
-- Run the commented-out SELECT queries first to preview what's affected.

-- =====================================================================
-- PART A: AUTHORITY NEON — public schema orphans
--
-- After per-operator role isolation, each operator has its own
-- op_{hash} schema. The public schema retains:
--   LIVE:  authority_config, bootstrap_config,
--          operator_pricing_models (Authority's own pricing),
--          balances (Authority's own ledger),
--          transactions (Authority's own journal),
--          anchors (Authority's own OTS),
--          tool_demand (Authority's own surge counters),
--          credentials, session_bindings
--   DEAD:  tool_acls (never wired into any server)
--
-- Patron data that was once in public.balances / public.transactions
-- before isolation is now duplicated in op_* schemas. Those rows
-- in public are orphaned — operators no longer read/write them.
-- However, they may serve as a historical record. The safest
-- approach is to leave them and let the Authority decide.
-- =====================================================================

-- 1a. DROP tool_acls from public (never used by any server)

-- Preview:
-- SELECT count(*) FROM public.tool_acls;

DROP TABLE IF EXISTS public.tool_acls;
DROP INDEX IF EXISTS public.idx_tool_acls_operator;

-- 1b. Audit orphaned patron rows in public.balances
--
-- These are balances that were created before per-operator schema
-- isolation. The same patrons now have balances in op_* schemas.
-- Listing them for manual review — DO NOT auto-delete.

-- SELECT npub, last_flush, created_at
-- FROM public.balances
-- ORDER BY last_flush DESC;

-- 1c. Audit orphaned transactions in public.transactions

-- SELECT count(*) AS total_public_txns,
--        min(created_at) AS oldest,
--        max(created_at) AS newest
-- FROM public.transactions;

-- =====================================================================
-- PART B: PER-OPERATOR SCHEMA — run in each op_* schema
--
-- Replace {schema} with the actual operator schema name
-- (e.g., op_a1b2c3d4e5f6g7h8).
-- =====================================================================

-- 2a. DROP tool_acls from operator schema (never used)

-- DROP TABLE IF EXISTS {schema}.tool_acls;
-- DROP INDEX IF EXISTS {schema}.idx_tool_acls_operator;

-- 2b. Scrub stale demurrage from model_json
--
-- Prior to v0.10.0, demurrage was a pipeline constraint.
-- Now it is a top-level tranche_lifetime field. All operators
-- should have been re-saved, but this catches stragglers.

-- Preview:
-- SELECT id, operator, name,
--        jsonb_path_query_array(model_json, '$.pipeline[*] ? (@.type == "demurrage")') AS stale
-- FROM {schema}.operator_pricing_models
-- WHERE model_json @> '{"pipeline": [{"type": "demurrage"}]}';

-- UPDATE {schema}.operator_pricing_models
-- SET model_json = jsonb_set(
--     model_json,
--     '{pipeline}',
--     (
--         SELECT coalesce(jsonb_agg(step), '[]'::jsonb)
--         FROM jsonb_array_elements(model_json -> 'pipeline') AS step
--         WHERE step ->> 'type' != 'demurrage'
--     )
-- ),
-- updated_at = now()
-- WHERE model_json @> '{"pipeline": [{"type": "demurrage"}]}';

-- =====================================================================
-- PART C: ALSO RUN ON AUTHORITY for its own pricing model
-- =====================================================================

-- Preview:
-- SELECT id, operator, name
-- FROM public.operator_pricing_models
-- WHERE model_json @> '{"pipeline": [{"type": "demurrage"}]}';

UPDATE public.operator_pricing_models
SET model_json = jsonb_set(
    model_json,
    '{pipeline}',
    (
        SELECT coalesce(jsonb_agg(step), '[]'::jsonb)
        FROM jsonb_array_elements(model_json -> 'pipeline') AS step
        WHERE step ->> 'type' != 'demurrage'
    )
),
updated_at = now()
WHERE model_json @> '{"pipeline": [{"type": "demurrage"}]}';

-- =====================================================================
-- PART D: VERIFY
-- =====================================================================

-- Confirm tool_acls is gone from public
-- SELECT schemaname, tablename FROM pg_tables WHERE tablename = 'tool_acls';

-- Confirm no demurrage steps remain anywhere
-- SELECT schemaname, tablename
-- FROM pg_tables
-- WHERE tablename = 'operator_pricing_models';
-- Then for each schema:
-- SELECT count(*) FROM {schema}.operator_pricing_models
-- WHERE model_json @> '{"pipeline": [{"type": "demurrage"}]}';

-- =====================================================================
-- PART E: CODE CLEANUP (not SQL — wheel changes)
--
-- After running this script, remove dead code from tollbooth-dpyc:
--   - src/tollbooth/acl_store.py (dead — never imported by runtime)
--   - src/tollbooth/tools/acl.py (dead — never registered)
--   - src/tollbooth/acl_verify.py (dead — never imported by runtime)
--   - tests/test_acl_*.py (tests for dead code)
--   - Remove tool_acls CREATE from ensure_schema() in vaults/neon.py
--   - Remove tool_acls from neon_schema.sql
-- =====================================================================
