# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.16.7] — 2026-05-02

### Fixed
- Expired tranches are now collected on ledger deserialization (`from_json`). Previously, `_collect_expired` only ran during `debit()` and `to_json()` — if the process was down during a tranche's expiry window, the expired sats vanished without incrementing `total_expired_api_sats`.

## [0.16.6] — 2026-04-30

### Removed
- Diagnostic print statements and inline error diagnostics from paid_tool decorator and debit_or_deny (added in 0.16.2–0.16.5 to trace excalibur proof bug). Root cause was a duplicate debit_or_deny call in excalibur-mcp's _prepare_x_client, not a wheel issue.

## [0.16.3] — 2026-04-30

### Fixed
- Diagnostic print statements now use stderr with flush=True and avoid f-string `!r` syntax incompatible with Python 3.12. Fixed import ordering for `sys` module.

## [0.16.2] — 2026-04-30

### Added
- Diagnostic logging in `paid_tool` decorator showing proof extraction path, kwargs keys, and bound argument keys at gate time. Will reveal whether FastMCP/Claude.ai is dropping the proof parameter before it reaches debit_or_deny.

## [0.16.1] — 2026-04-30

### Fixed
- `paid_tool` decorator now extracts `proof` from kwargs directly in addition to `inspect.signature().bind()`. FastMCP's Pydantic argument marshaling could drop optional default-valued parameters from the bound arguments, causing proof tokens to arrive empty at `debit_or_deny`.

## [0.16.0] — 2026-04-27

### Added
- **Field-level patron credential CRUD** — `update_patron_credential`, `delete_patron_credential`, `get_patron_credential`, `list_patron_credential_fields` on `OperatorRuntime`. Read-merge-write on the existing encrypted blob; no schema changes.
- Three new standard tools: `update_patron_credential`, `delete_patron_credential`, `get_patron_credential_fields`. Free, registered on all MCPs.

## [0.15.7] — 2026-04-26

### Fixed
- Bootstrap config relay filter `max_age_seconds` raised from 30 days to 1 year. The old 30-day client-side filter was causing bootstrap failures when the Authority's config DM aged past the window, even though relays still held it.

## [0.15.6] — 2026-04-26

### Fixed
- Bootstrap relay discovery now uses the operator's configured relays instead of a hardcoded list. Previously `_read_config_from_relays` ignored `OperatorRuntime._relays`, causing bootstrap failures when the Authority's config DM was on relays not in `BOOTSTRAP_RELAYS`.

## [0.15.5] — 2026-04-26

### Changed
- **Poison-keyed proof** — proof cache key changed from `session_id:npub` to `sha256(poison):npub`. Proof now survives MCP restarts because the calling application holds the raw poison phrase and supplies it on each paid tool call. The MCP stores only the hashed poison in Neon.
- `request_npub_proof` and `receive_npub_proof` now return `proof_token` in the response for the calling application to remember
- `debit_or_deny` accepts poison phrases (e.g. `bold-hawk-42`) as the `proof` parameter; Schnorr signatures still accepted for restricted tools

### Added
- `OperatorRuntime.restore_oauth_session(patron_npub)` — generic restore-refresh-persist cycle for OAuth tokens. Loads from vault, checks expiry, refreshes via `OAuthProviderConfig.token_url`, persists rotated tokens back to vault. Eliminates duplicated refresh logic in schwab-mcp and excalibur-mcp.

## [0.15.4] — 2026-04-24

### Fixed
- Default relay set broadened from `nostr.wine` alone to 5 relays (`relay.primal.net`, `relay.damus.io`, `nos.lol`, `nostr.wine`, `relay.nostr.band`) — patron replies from clients like Oxchat were invisible when their relays didn't include the operator's single default
- `PeriodicRefreshConstraint` now supports `scope` parameter (`global` / `per_patron`) matching `FiniteSupplyConstraint`

## [0.15.3] — 2026-04-23

### Changed
- Proof confirmation DM to patron now carries the same enriched message as the tool response (operator name, session, expiry timestamp) instead of generic "npub ownership confirmed"

## [0.15.2] — 2026-04-22

### Fixed
- Patron-requested proof cache durations (e.g. 36h) were silently capped to 2h by `SessionCache`'s global TTL — container now uses `MAX_PROVEN_TTL` so per-entry expiry is honored

### Changed
- `receive_npub_proof` response message now includes operator name, truncated npub, session ID, and human-readable expiry timestamp (e.g. "valid until 2026-04-24 09:30 UTC (36 hours from now)")
- Added `expires_at` field to proof confirmation response for programmatic use

## [0.15.1] — 2026-04-22

### Fixed
- `record_invoice_settled` fallback path now defaults `created_at` to `settled_at` instead of empty string — invoices recovered after cold start no longer have missing timestamps

## [0.15.0] — 2026-04-21

### Added
- **x402 upstream adapter** (`X402Client`) — transparent HTTP 402 payment for operators consuming Coinbase x402-protected APIs. Per-tool opt-in, optional `[x402]` dependency group.
- `x402_wallet_template()` credential template for Secure Courier delivery of agentic wallet credentials
- BOLT11 retry with exponential backoff in `get_lightning_invoice()` — fixes QR codes showing checkout URL instead of Lightning invoice

### Changed
- `MAX_PROVEN_TTL` raised from 24 hours to 7 days (604,800 seconds) — patrons can request proof cache durations up to "7d"

## [0.14.2] — 2026-04-20

### Added
- FiniteSupplyConstraint global scope now backed by Neon persistence via shared `tool_demand` table (sentinel `window_key="__total__"`)
- `EnvironmentSnapshot.supply_total_for()` helper for lifetime invocation lookups
- `OperatorRuntime.fire_and_forget_supply_increment()` increments lifetime counter after each tool call
- E2E tests for FiniteSupply (3-then-deny loop, independent per-tool counters, constraint removal, re-apply with higher cap)
- E2E tests for SurgePricing (price steps with demand, volume discount with multiplier < 1.0, surge + supply composition)

### Changed
- `get_global_demand()` now returns both hourly demand and lifetime total in a single dict
- `FiniteSupplyConstraint` no longer accepts `current_count` — global scope reads from `EnvironmentSnapshot`

### Fixed
- `test_paid_tool` suite: proof replay cache cleared between tests, missing fake caches injected, stale error assertion updated
- `test_runtime_onboarding` demand tracking test updated for dual-fetch API

## [0.11.6] — 2026-04-16

- fix shortlinks + collector URL: don't double /mcp/ path (v0.11.6)

## [0.11.5] — 2026-04-16

- fix check_oauth_status: resolve collector MCP, not Val Town callback (v0.11.5)

## [0.11.4] — 2026-04-16

- credential lifecycle: not-yet-delivered is a state, not an error (v0.11.4)

## [0.11.3] — 2026-04-16

- fix NameError: oauth_cfg → _opc in begin_oauth (v0.11.3)

## [0.11.2] — 2026-04-16

- OAuth credential field mapping: vendor names → protocol names (v0.11.2)

## [0.11.1] — 2026-04-16

- patron-chosen proof cache duration (v0.11.1)

## [0.11.0] — 2026-04-16

- schema symmetry: NeonCredentialVault uses _t(), remove tool_acls dead code (v0.11.0)

## [0.10.1] — 2026-04-16

- clarify Secure Courier tools as human-in-the-loop (v0.10.1)
- fix lint: remove unused imports (field, time)

## [0.10.0] — 2026-04-16

- promote demurrage from constraint to tranche lifetime (v0.10.0)

## [0.9.3] — 2026-04-15

- opportunistic OTS notarization on paid tool calls (v0.9.3)

## [0.9.2] — 2026-04-15

- vault-backed proof cache survives serverless cold starts (v0.9.2)

## [0.9.1] — 2026-04-15

- bump version to 0.9.1 for proof cache diagnostics release
- Add diagnostic logging to proof cache for session_id investigation

## [0.9.0] — 2026-04-14

- channel-bound npub proof via FastMCP session_id (v0.9.0)

## [0.8.4] — 2026-04-14

- NIP-40 expiration + since-timestamp filter for clean relay exchanges (v0.8.4)

## [0.8.3] — 2026-04-14

- request purges stale DMs, receive retries 4x with 2s pause (v0.8.3)

## [0.8.2] — 2026-04-14

- fix: add missing import time as _time in receive_npub_proof (v0.8.2)

## [0.8.1] — 2026-04-14

- fix: _nsec_hex → _privkey_hex in receive_npub_proof drain loop (v0.8.1)

## [0.8.0] — 2026-04-14

- standard OAuth2 tools via OAuthProviderConfig (v0.8.0)

## [0.7.9] — 2026-04-14

- configurable npub_proof_field, greeting, and on_npub_proven callback (v0.7.9)

## [0.7.8] — 2026-04-14

- receive_npub_proof drains relay — pop all, one summary DM (v0.7.8)

## [0.7.7] — 2026-04-14

- fix: write instructions via _mcp_server (read-only property) (v0.7.7)

## [0.7.6] — 2026-04-14

- DM is the proof — no Schnorr event required from patron (v0.7.6)

## [0.7.5] — 2026-04-14

- use courier.receive() public API, load proof from vault (v0.7.5)

## [0.7.4] — 2026-04-14

- append DPYC agent guidance to all MCP server instructions (v0.7.4)

## [0.7.3] — 2026-04-14

- clarify proof cache lifecycle in tool docs and error messages (v0.7.3)

## [0.7.2] — 2026-04-14

- use Secure Courier for npub proof exchange (v0.7.2)

## [0.7.1] — 2026-04-14

- fix: unpack DM candidates as dicts, not tuples (v0.7.1)

## [0.7.0] — 2026-04-14

- npub ownership proof caching for agentic callers (v0.7.0)
- fix: E402 lint — move pynostr import to top of file

## [0.6.7] — 2026-04-14

- remove session_verifier — proof is always required (v0.6.7)

## [0.6.6] — 2026-04-13

- add session_verifier hook — waive proof for OAuth-authenticated patrons (v0.6.6)

## [0.6.5] — 2026-04-13

- fix: account_statement_infographic passes proof to debit_or_deny (v0.6.5)

## [0.6.4] — 2026-04-13

- purchase_credits returns BOLT11 lightning_invoice from BTCPay payment methods (v0.6.4)

## [0.6.3] — 2026-04-13

- AuthorityCertifier passes proof on certify_credits and check_balance (v0.6.3)
- docs: update README for v0.6.2 — unified proof param, ad valorem pricing, Python 3.12+

## [0.6.2] — 2026-04-13

- delete operator_proof.py, remove legacy fallback in set_pricing_model

## [0.6.1] — 2026-04-13

- set_pricing_model accepts proof as separate param (not embedded in JSON)

## [0.6.0] — 2026-04-13

- security: unified proof parameter across all tool signatures (v0.6.0)
- add proof: str to all standard tool signatures with npub
- security: unified proof parameter — npub always paired with proof

## [0.5.2] — 2026-04-13

- bump v0.5.2
- security: debit_or_deny returns Either(denial_dict, cost_int) (C-3)

## [0.5.1] — 2026-04-13

- bump v0.5.1
- security: reject negative amounts in ToolPricing.compute() (C-2)

## [0.5.0] — 2026-04-12

- remove Horizon OAuth from wheel — npub-only identity everywhere

## [0.4.9] — 2026-04-11

- fix: credential validator reloads creds from vault after courier strips them

## [0.4.8] — 2026-04-11

- fix: pass service as keyword arg to redeem_credential_card

## [0.4.7] — 2026-04-11

- fix: courier rejection logs show expected poison and sender npub

## [0.4.6] — 2026-04-11

- bump v0.4.6 for PyPI publish
- fix: forward force_relay through SecureCourierService.receive()

## [0.4.5] — 2026-04-11

- Bump to v0.4.5
- credential_validator callback: operators validate their own creds
- Validate operator credentials at receive time, reject + DM on failure
- receive_credentials: force_relay option to skip vault cache
- Auto-fix common btcpay_host URL typos: htps:// → https://, http:// → https://

## [0.4.4] — 2026-04-11

- fix: validate btcpay_host starts with https:// at init

## [0.4.3] — 2026-04-11

- fix: active_tranches excludes expired tranches, add expired_tranches field

## [0.4.2] — 2026-04-11

- fix: exclude OTS tools from registry when ots_enabled=False

## [0.4.1] — 2026-04-11

- fix: trust-root vault reads NEON_DATABASE_URL from env

## [0.4.0] — 2026-04-11

- ad valorem paid_tool + purchase_mode for Authority-as-Operator

## [0.3.3] — 2026-04-10

- prune patron credential tools from registry when unused

## [0.3.2] — 2026-04-10

- lazy MCP name resolution via mcp_name_for()

## [0.3.1] — 2026-04-10

- paid_tool records function name for accurate MCP name stamping

## [0.3.0] — 2026-04-10

- single tool identity: UUID for machines, full MCP name for humans

## [0.2.17] — 2026-04-10

- service_status: expose operator slug for namespace filtering

## [0.2.16] — 2026-04-09

- enforce UUID-only tool identity — remove legacy from_dict fallback
- service_status: include operator_npub_hash for patron DM verification

## [0.2.15] — 2026-04-09

- chore: bump to v0.2.15 — closed-door billing gate, CI Node.js 24
- Fix flaky relay probe test: use distinct mock latencies
- chore: bump GitHub Actions to Node.js 24 (checkout v6, setup-python v6)
- fix: remove unused variables flagged by ruff (F841)
- fix: closed-door billing gate — no open-door fallbacks

## [0.2.14] — 2026-04-05

- feat: explicit priced flag — TBD is a lifecycle state, not a price value

## [0.2.13] — 2026-04-05

- fix: initial pricing model uses actual MCP tool names from mcp.list_tools()

## [0.2.12] — 2026-04-05

- feat: UUID-keyed internals — all economic paths use UUID, not short names

## [0.2.11] — 2026-04-05

- fix: remove Oracle tools from STANDARD_IDENTITIES

## [0.2.10] — 2026-04-05

- feat: Oracle-delegated tools use oracle_ namespace

## [0.2.9] — 2026-04-05

- fix: block paid-category tools at 0 sats — TBD is a real gate

## [0.2.8] — 2026-04-05

- fix: reset_pricing_model is a true reset — erase + restore default

## [0.2.7] — 2026-04-05

- fix: always emit category and intent in ToolPrice serialization

## [0.2.6] — 2026-04-05

- feat: reset_pricing_model tool — delete stale models, re-initialize fresh

## [0.2.5] — 2026-04-05

- refactor: remove legacy UUID fallback — clean cut, no backward compat

## [0.2.4] — 2026-04-05

- fix: security — require operator_proof for set_pricing_model + legacy UUID fallback
- fix: lint — remove unused DPYC_NAMESPACE import in test

## [0.2.3] — 2026-04-04

- fix: invalidate PricingResolver cache after set_pricing_model

## [0.2.2] — 2026-04-04

- feat: UUID-based tool identity — decouple pricing from code

## [0.2.1] — 2026-04-03

- chore: bump to v0.2.1 for PyPI publish
- feat: PKCE + refresh_access_token in oauth2_collector
- feat: patron_proof constraint + identity_proof refactor
- feat: check_authority_balance — operator asks Authority for its own tax balance
- fix: clarify check_balance and account_statement are patron-facing
- fix: remove NEON_DATABASE_URL env var fallback — bootstrap only

## [0.2.0] — 2026-04-02

- feat: v0.2.0 — clean Neon schema isolation, no diagnostic hacks

## [0.1.191] — 2026-04-02

- fix: don't schema-qualify credentials/session_bindings — they already work

## [0.1.190] — 2026-04-02

- fix: schema-qualify ALL table names in _execute, not just ensure_schema

## [0.1.189] — 2026-04-02

- fix: parse operator schema from connection URL, not SHOW search_path

## [0.1.188] — 2026-04-02

- fix: revert pooler strip — Neon HTTP SQL API requires the pooler endpoint

## [0.1.187] — 2026-04-02

- diag: expose resolved_schema from _resolve_target_schema

## [0.1.186] — 2026-04-02

- fix: create all vault tables in operator schema, not public

## [0.1.185] — 2026-04-02

- diag: key fingerprint, nsec fingerprint, round-trip test, vault source

## [0.1.184] — 2026-04-02

- diag: fetch_ledger exception type + unqualified SELECT test

## [0.1.183] — 2026-04-02

- diag: encryption state and fetch_ledger result in check_balance

## [0.1.182] — 2026-04-02

- diag: raw Neon query in check_balance bypassing cache and search_path

## [0.1.181] — 2026-04-02

- diag: expose vault_search_path in service_status

## [0.1.180] — 2026-04-02

- fix: append ,public to Neon search_path — root cause of vanishing ledger data

## [0.1.179] — 2026-04-02

- fix: use Neon direct endpoint instead of pooler for read-after-write consistency

## [0.1.178] — 2026-04-02

- diag: add cache entry diagnostics to check_balance

## [0.1.177] — 2026-04-02

- diag: add runtime_id and vault_id to service_status and check_balance

## [0.1.176] — 2026-04-02

- diag: read-back verification after ledger flush to Neon

## [0.1.175] — 2026-04-02

- diag: add vault_endpoint to service_status for Neon routing diagnostics

## [0.1.174] — 2026-04-02

- fix: credit operations report persistence status — no more silent flush failures

## [0.1.173] — 2026-04-02

- fix: onboarding status tools trigger courier late-attach

## [0.1.172] — 2026-04-02

- fix: comprehensive credential vault diagnostics — stop swallowing errors

## [0.1.171] — 2026-04-01

- fix: don't cache empty ledgers from failed vault fetches on cold start

## [0.1.170] — 2026-04-01

- fix: cold start bugs — session_status warming window, auto-reconcile pending invoices

## [0.1.169] — 2026-04-01

- feat: session_status returns operator lifecycle state instead of courier_configured

## [0.1.168] — 2026-04-01

- fix: session_status guidance — tell LLM the operator is ready
- fix: add SessionCache and PatronSessionCache to __all__ (lint F401)

## [0.1.167] — 2026-04-01

- feat: themed infographic — InfographicTheme, MetricDef, InfographicSections

## [0.1.166] — 2026-04-01

- fix: wire demurrage TTL into restore_credits, fix stale tests for v0.1.166

## [0.1.165] — 2026-03-31

- feat: rename tranche_expiration to demurrage, bump v0.1.165

## [0.1.164] — 2026-03-31

- feat: tranche_expiration constraint, remove config-based credit TTL

## [0.1.163] — 2026-03-31

- chore: bump to v0.1.163 for authority_client npub rename
- fix: rename operator_id to npub in authority_client and protocol

## [0.1.162] — 2026-03-31

- feat: add get_patron_onboarding_status tool, rename get_onboarding_status

## [0.1.161] — 2026-03-30

- fix: bump to v0.1.161 — v0.1.160 tag had stale pyproject.toml version

## [0.1.160] — 2026-03-30

- fix: restore ephemeral agent key before relay subscription on cold start

## [0.1.159] — 2026-03-29

- feat: create_shortlink() utility for tollbooth-shortlinks MCP service

## [0.1.158] — 2026-03-29

- feat: OTS on by default, resolve_relays(), operator_settings, relay tests

## [0.1.157] — 2026-03-29

- feat: SessionCache[T] + PatronSessionCache with Neon vault persistence

## [0.1.156] — 2026-03-29

- feat: @runtime.paid_tool() decorator + constraint gate bug fix + OTS standard tools

## [0.1.155] — 2026-03-29

- feat: on_forget callback — forget means forget everywhere

## [0.1.154] — 2026-03-29

- fix: forget_credentials accepts npub for patron credential revocation

## [0.1.153] — 2026-03-29

- fix: require explicit service param on all credential tools
- fix: remove unused 'now' variable in diagnostic logging (F841)
- fix: CI matrix Python 3.12+3.13 (matches requires-python >=3.12)

## [0.1.152] — 2026-03-29

- chore: bump version to 0.1.152 (Python >=3.12)
- chore: require Python >=3.12 (matches Horizon)

## [0.1.151] — 2026-03-29

- fix: use correct FastMCP Cloud env vars for build_info

## [0.1.150] — 2026-03-29

- feat: service_status reports build/deploy info from env

## [0.1.149] — 2026-03-28

- fix: better courier diagnostics — event kinds, candidate counts

## [0.1.148] — 2026-03-28

- fix: late-attach credential vault to courier on retry

## [0.1.147] — 2026-03-28

- fix: use inner DM timestamp for NIP-17 gift wraps

## [0.1.146] — 2026-03-28

- fix: remove time-based pre-filter from DM candidate search

## [0.1.145] — 2026-03-28

- feat: diagnostic logging for receive_credentials relay state

## [0.1.144] — 2026-03-28

- fix: remove hardcoded "Personal Brain" from infographic header

## [0.1.143] — 2026-03-28

- fix: auto-seed pricing model when stored model is missing tools

## [0.1.142] — 2026-03-28

- feat: store_patron_session / load_patron_session on OperatorRuntime
- cleanup: remove defensive int() casts from debit_or_error

## [0.1.141] — 2026-03-28

- fix: swapped args in ledger.debit() call + broken rollback_debit

## [0.1.140] — 2026-03-28

- fix: force int() on both sides of balance comparison in debit_or_error

## [0.1.139] — 2026-03-28

- feat: service_status reports tollbooth_dpyc_version

## [0.1.138] — 2026-03-28

- fix: enforce int types on Tranche and UserLedger numeric fields

## [0.1.137] — 2026-03-28

- feat: thank-you DM on invoice settlement + defensive int cast

## [0.1.136] — 2026-03-28

- fix: no per-field credential rotation — use full operator template
- fix: pricing resolver refresh() uses sentinel, not 0.0
- fix: remove unused json import in bootstrap.py (F401)
- test: cashier abstraction, npub enforcement, auto-seed, demand tracking

## [0.1.135] — 2026-03-28

- docs: improve credential tool docstrings for agent discoverability

## [0.1.134] — 2026-03-28

- refactor: rename _btcpay → _cashier, ensure_btcpay → ensure_cashier

## [0.1.133] — 2026-03-28

- fix: invalidate BTCPay cache on credential receive/forget

## [0.1.132] — 2026-03-28

- fix: pass btcpay + authority_npub to credit tool functions

## [0.1.131] — 2026-03-28

- chore: bump version to 0.1.131
- fix: revert npub fallback — require npub for all patron tools

## [0.1.130] — 2026-03-28

- fix: pass operator_npub to AuthorityCertifier in purchase_credits

## [0.1.129] — 2026-03-27

- fix: debit_or_error falls back to operator npub when patron npub empty

## [0.1.128] — 2026-03-27

- chore: bump version to 0.1.128
- fix: npub fallback, auto-seed pricing, stale vault detection
- fix: remove unused variables in test_runtime_onboarding (F841)

## [0.1.127] — 2026-03-27

- chore: bump version to 0.1.127
- feat: extract core helpers to OperatorRuntime
- refactor: clean credential architecture — dual templates, no tiers

## [0.1.126] — 2026-03-27

- feat: onboarding shows template-only fields (not just Settings fields)

## [0.1.125] — 2026-03-27

- feat: optional credential fields + tuning field expansion

## [0.1.124] — 2026-03-27

- feat: FieldSpec.description + enrich onboarding with template hints

## [0.1.123] — 2026-03-27

- feat: onboarding returns credential_greeting and operator_name

## [0.1.122] — 2026-03-27

- refactor: clean up bootstrap debug scaffolding

## [0.1.121] — 2026-03-27

- fix: convert bech32 nsec to hex for VaultCipher

## [0.1.120] — 2026-03-27

- debug: reset cached bootstrap failure + full traceback in error

## [0.1.119] — 2026-03-27

- debug: full traceback on relay read failure

## [0.1.118] — 2026-03-27

- debug: hex validation + content preview in decrypt errors

## [0.1.117] — 2026-03-27

- fix: NIP-04 decrypt param is ciphertext_with_iv not ciphertext

## [0.1.116] — 2026-03-27

- debug: relay diagnostics in bootstrap_error (events, errors per relay)

## [0.1.115] — 2026-03-27

- debug: relay poll diagnostics — errors, event counts, pubkey hex

## [0.1.114] — 2026-03-27

- debug: always include bootstrap_error and vault_ok in onboarding

## [0.1.113] — 2026-03-27

- feat: relay-based bootstrap — Neon URL via Nostr DM, no OAuth needed

## [0.1.112] — 2026-03-27

- fix: onboarding actively bootstraps vault + reports bootstrap_error
- fix: PricingResolver cache_ts=0.0 appears fresh on new CI runners
- debug: add diagnostic assertion to pricing resolver test
- fix: unpin pytest-asyncio — version 1.3.0 may mishandle async class methods in CI
- ci: use importlib import mode for test isolation
- fix: revert TYPE_CHECKING in pricing — breaks CI, ignore F821 instead
- fix: restore TYPE_CHECKING for pricing_model.py ToolPricing
- fix: TYPE_CHECKING for ToolPricing + noqa for ruff
- fix: revert TYPE_CHECKING guard — ToolPricing used at runtime
- fix: ruff lint cleanup — unused imports + formatting
- ci: add ruff lint step to CI workflow

## [0.1.111] — 2026-03-27

- fix: onboarding marks neon_database_url green when vault is bootstrapped

## [0.1.110] — 2026-03-27

- feat: service_status shows bootstrap_error for diagnostics

## [0.1.109] — 2026-03-27

- fix: receive_credentials credential_card + credential_greeting

## [0.1.108] — 2026-03-26

- feat: infographic renderer + tool restored in standard tools

## [0.1.107] — 2026-03-26

- fix: account_statement gets days param, infographic is operator-specific

## [0.1.106] — 2026-03-26

- feat: register_standard_tools — all DPYC tools in one call

## [0.1.105] — 2026-03-26

- feat: OperatorRuntime — core DPYC protocol engine for all operators

## [0.1.104] — 2026-03-26

- feat: create_operator_proof + bootstrap sends Schnorr proof

## [0.1.103] — 2026-03-26

- chore: bump version to 0.1.103
- feat: generic vault-aware onboarding — any operator, any secrets
- chore: update uv.lock

## [0.1.102] — 2026-03-25

- chore: bump version to 0.1.102
- feat: get_onboarding_status tool — introspects operator config readiness
- feat: ensure_bootstrapped() — lazy singleton for operator startup
- feat: BootstrapClient — discover config from Authority using only nsec

## [0.1.101] — 2026-03-25

- chore: bump to 0.1.101 for vault encryption release
- feat: nsec-derived AES-256-GCM encryption for NeonVault

## [0.1.100] — 2026-03-22

- feat: rename anchoring → Bitcoin notarization, add to operator catalog

## [0.1.99] — 2026-03-22

- fix: belt-and-suspenders migration on cache miss path too

## [0.1.98] — 2026-03-22

- fix: migrate perpetual tranches in LRU cache, not just on vault hydration

## [0.1.97] — 2026-03-22

- chore: bump version to 0.1.97 for release
- fix: all tranches must expire — default 7-day TTL, migrate perpetuals
- chore: sync uv.lock

## [0.1.96] — 2026-03-21

- chore: bump version to 0.1.96
- feat: bridge ToolPrice → ToolPricing for ad valorem pricing support

## [0.1.95] — 2026-03-21

- chore: bump version to 0.1.95
- refactor: rename AuthorityCertifier.certify() to certify_credits()

## [0.1.94] — 2026-03-21

- chore: bump version to 0.1.94
- fix: migrate legacy perpetual rollback tranches to 7-day expiry on load
- fix: rollback adds sats to soonest-expiring tranche instead of creating new one
- fix: rollback tranches expire after 7 days instead of never

## [0.1.93] — 2026-03-21

- chore: sync uv.lock with v0.1.93
- feat: expose pending_invoice_ids in check_balance_tool response

## [0.1.92] — 2026-03-17

- feat: add Nostr-signed tool ACLs and identity proof verification
- feat: add operator catalog conformance validator and bump to 0.1.92

## [0.1.91] — 2026-03-14

- chore: bump version to 0.1.91
- feat: add constraint schemas, ToolPrice fields, and tranche details in check_balance

## [0.1.90] — 2026-03-13

- fix: AuditedVault proxies _execute for PricingModelStore (#100)

## [0.1.89] — 2026-03-13

- Merge pull request #99 from lonniev/feat/pricing-crud-tools
- feat: add pricing CRUD tool functions for operator self-service
- Merge pull request #98 from lonniev/feat/pricing-models
- feat: add runtime-configurable pricing models

## [0.1.88] — 2026-03-10

- feat: retrieve via MCP JSON-RPC, revert response_mode=form_post

## [0.1.87] — 2026-03-10

- fix: POST for Horizon compat + response_mode=form_post

## [0.1.86] — 2026-03-10

- fix: use Horizon /mcp/ prefix for collector retrieve URL

## [0.1.85] — 2026-03-10

- feat: add resolve_service_by_name() for registry discovery (#97)

## [0.1.84] — 2026-03-10

- feat: extract reusable OAuth2 collector client module (#96)

## [0.1.83] — 2026-03-09

- feat: persist pending Courier state in vault for cold-start recovery (#95)

## [0.1.82] — 2026-03-09

- fix: use resolved service for template matching in multi-template servers (#94)

## [0.1.81] — 2026-03-09

- feat: ephemeral agent npub for self-DM avoidance (#93)

## [0.1.80] — 2026-03-08

- chore: bump version to 0.1.80
- Merge pull request #91 from lonniev/refactor/lookup-cache-path
- refactor: update DEFAULT_REGISTRY_URL to lookup cache path

## [0.1.79] — 2026-03-07

- Merge pull request #90 from lonniev/feat/courier-provenance-invoice-dm
- feat: welcome DM provenance block + invoice DM callback

## [0.1.78] — 2026-03-07

- fix: remove legacy royalty payout + fix tax incidence (#89)

## [0.1.77] — 2026-03-06

- Merge pull request #88 from lonniev/feat/authority-config-table
- feat: add authority_config table to NeonVault schema

## [0.1.76] — 2026-03-06

- chore: bump version to 0.1.76
- feat: DPYC identity credential primitive + federated trust cold path (#87)
- chore: clarify citizen vs operator registration metadata (#86)

## [0.1.75] — 2026-03-05

- feat: surge pricing + fire-and-forget flushes (#85)

## [0.1.74] — 2026-03-05

- Merge pull request #84 from lonniev/chore/ecosystem-links-relay-hints
- chore: bump version to 0.1.74
- chore: add ECOSYSTEM_LINKS constant + relay propagation delay hints

## [0.1.73] — 2026-03-04

- Merge pull request #83 from lonniev/feat/obsolete-practices-metadata
- feat: obsolete practices metadata + npub identity guidance for agents

## [0.1.72] — 2026-03-04

- Merge pull request #82 from lonniev/feat/qr-upload-trademark
- feat: upload credential card QR to nostr.build + common-law trademark notices

## [0.1.71] — 2026-03-04

- Merge pull request #81 from lonniev/fix/oracle-text-response
- fix: handle plain-text Oracle responses in OracleClient._parse_result()

## [0.1.70] — 2026-03-04

- feat: auto-restore DPYC session from vault on cold start (#80)
- Merge pull request #79 from lonniev/chore/v0.1.69-readme
- docs: update README for v0.1.69 features

## [0.1.69] — 2026-03-04

- Merge pull request #78 from lonniev/feat/credential-card-dm
- fix: Python 3.10 compat for datetime.fromisoformat Z suffix
- feat: add OPERATOR_BASE_CATALOG and AUTHORITY_BASE_CATALOG to protocols
- feat: send credential card via Nostr DM after first-time credential receipt

## [0.1.68] — 2026-03-03

- Merge pull request #77 from lonniev/feat/oracle-client
- docs: add OracleClient and resolve_oracle_service to README
- feat: add OracleClient and resolve_oracle_service()

## [0.1.67] — 2026-03-03

- fix: unwrap CallToolResult in AuthorityCertifier (#76)

## [0.1.66] — 2026-03-03

- Merge pull request #75 from lonniev/feat/authority-certifier
- feat: add AuthorityCertifier for server-to-server auto-certification

## [0.1.65] — 2026-03-03

- Merge pull request #74 from lonniev/feat/slug-prefixing
- feat: add make_slug_tool() factory for slug-prefixed MCP tool names

## [0.1.64] — 2026-03-03

- fix: harden _npub_to_hex against pynostr TypeError on bad checksum (#73)

## [0.1.63] — 2026-03-03

- Merge pull request #72 from lonniev/fix/ci-extras
- fix: install nostr and qr extras in CI test workflow
- feat: define OperatorProtocol, AuthorityProtocol, OracleProtocol

## [0.1.62] — 2026-03-02

- Merge pull request #70 from lonniev/feat/poison-phrase-instructions
- feat: instruct agent to share poison phrase with human

## [0.1.61] — 2026-03-02

- Merge pull request #69 from lonniev/feat/relay-probe-liveness
- feat: add probe_relay_liveness for dynamic relay negotiation

## [0.1.60] — 2026-03-02

- Merge pull request #68 from lonniev/feat/qr-credential-card
- feat: QR credential card for scan-and-paste credential reuse

## [0.1.59] — 2026-03-02

- Merge pull request #67 from lonniev/fix/tolerant-credential-parser
- fix: silently drop unknown fields in credential validation (v0.1.59)

## [0.1.58] — 2026-03-02

- Merge pull request #66 from lonniev/refactor/remove-tax-coupling
- feat: add direct_purchase_tool as replacement for purchase_tax_credits_tool
- refactor: remove tax-specific coupling from tollbooth-dpyc

## [0.1.57] — 2026-03-01

- Merge pull request #65 from lonniev/feat/pop-and-acknowledge-dms
- feat: pop-and-acknowledge all relay DMs during receive (v0.1.57)

## [0.1.56] — 2026-03-01

- fix: purge stale DMs from local queue after poison match (v0.1.56) (#64)

## [0.1.55] — 2026-03-01

- fix: tempered greedy @@@ regex for robust multi-field parsing (v0.1.55) (#63)

## [0.1.54] — 2026-03-01

- fix: @@@ regex handles multiline values, drop (REQUIRED) labels (v0.1.54) (#62)

## [0.1.53] — 2026-03-01

- feat: add NeonCredentialVault for credential persistence (v0.1.53) (#61)

## [0.1.52] — 2026-03-01

- Merge pull request #60 from lonniev/feat/lnurl-payout-resolution
- feat: resolve Lightning addresses to BOLT11 invoices before payout (v0.1.52)

## [0.1.51] — 2026-03-01

- Merge pull request #59 from lonniev/feat/registry-resolution
- Add DPYCRegistry with resolve_authority_npub for NSEC-only identity (v0.1.51)

## [0.1.50] — 2026-03-01

- Merge pull request #58 from lonniev/feat/parameterize-welcome
- Parameterize welcome greeting + multi-exchange poison keys (v0.1.50)

## [0.1.49] — 2026-02-28

- Fix receive_credentials: scan all DMs instead of short-circuiting on first (#57)

## [0.1.48] — 2026-02-28

- Fix NIP-44v2: use ChaCha20 stream cipher + HMAC-SHA256 (not Poly1305) (#56)

## [0.1.47] — 2026-02-28

- Merge pull request #55 from lonniev/refactor/dry-version
- DRY version: read from importlib.metadata, bump to 0.1.47

## [0.1.46] — 2026-02-28

- Merge pull request #54 from lonniev/fix/nip17-freshness-filter
- Fix NIP-17 inbound receive path + leftover JSON text

## [0.1.45] — 2026-02-28

- @@@ delimiter template + NIP-17 hardening (#53)
- Merge pull request #52 from lonniev/feat/readme-update
- Update README with 12 new modules, 9 feature sections, and current architecture

## [0.1.44] — 2026-02-27

- Repair bare/half-quoted dict keys in lenient JSON parser (#51)

## [0.1.43] — 2026-02-27

- Lenient JSON parsing in Secure Courier — tolerate human-typed dicts (#50)

## [0.1.42] — 2026-02-27

- Sanitize smart quotes before JSON parse in Secure Courier receive (#49)

## [0.1.41] — 2026-02-27

- Fix gift wrap decrypt noise + add anti-replay poison slug (#48)

## [0.1.40] — 2026-02-27

- Fix gift wrap relay rejection — past-only timestamp fuzz (#47)

## [0.1.39] — 2026-02-27

- Fix base64 padding in NIP-44/NIP-04 decrypt — Primal strips trailing '=' (#46)

## [0.1.38] — 2026-02-27

- Migrate outbound DMs from NIP-04 (kind 4) to NIP-17 gift wraps (kind 1059) (#45)

## [0.1.37] — 2026-02-27

- Bump to v0.1.37 — Constraint Engine Phase 3 (ConstraintGate middleware) (#44)
- Add Constraint Engine Phase 3 — opt-in middleware integration (#43)

## [0.1.36] — 2026-02-27

- Add Constraint Engine — Phases 1-2 (additive, no production impact) (#42)

## [0.1.35] — 2026-02-26

- Add SecureCourierService — reusable Secure Courier wrapper (#41)

## [0.1.34] — 2026-02-26

- Bump to v0.1.34 — relay diagnostics + proactive DM notifications
- Merge pull request #40 from lonniev/feat/proactive-dm-notifications
- Merge pull request #39 from lonniev/feat/courier-health
- Add Nostr relay health diagnostic tools (courier_health, courier_ping)
- Add proactive low-balance and expiration Nostr DM notifications

## [0.1.33] — 2026-02-26

- Merge pull request #38 from lonniev/feat/conversational-dm
- Bump version to 0.1.33
- Merge fix/nip17-gift-wrap into feat/conversational-dm
- Add conversational DM flow — welcome, success, and error messages
- Fix NIP-17 gift-wrap subscription to handle metadata-protected DMs

## [0.1.32] — 2026-02-26

- Merge pull request #37 from lonniev/feat/welcome-dm
- Add NostrProfile, welcome DM, and outbound DM support (v0.1.32)

## [0.1.31] — 2026-02-26

- Bump version to 0.1.31 (credential vaulting)
- Add credential vaulting — vault-first lookup for Secure Courier (#36)

## [0.1.30] — 2026-02-26

- Bump version to 0.1.30 (subscription race-condition fix)
- Fix open_channel race condition — run subscription synchronously

## [0.1.29] — 2026-02-26

- Merge pull request #35 from lonniev/feat/secure-courier
- Add Secure Courier Service — Nostr DM credential exchange (v0.1.29)

## [0.1.28] — 2026-02-25

- Merge pull request #34 from lonniev/feat/nip44-encrypted-audit
- NIP-44 encrypted audit events — patron privacy for Nostr publishing

## [0.1.27] — 2026-02-25

- Merge pull request #33 from lonniev/feat/nostr-only
- Remove JWT/Ed25519 certificate path — Nostr-only (Phase 2+3)

## [0.1.26] — 2026-02-25

- Merge pull request #32 from lonniev/feat/nostr-certificate
- Add Nostr certificate verification (Phase 1 dual-mode)

## [0.1.25] — 2026-02-24

- Merge pull request #31 from lonniev/fix/ssl-cert-verification
- Fix SSL certificate verification bypass in Nostr relay connections

## [0.1.24] — 2026-02-23

- Merge pull request #30 from lonniev/feat/tool-pricing
- Add ToolPricing for dynamic tool cost computation

## [0.1.23] — 2026-02-23

- Merge pull request #29 from lonniev/feat/ots-bitcoin-anchoring
- Add OpenTimestamps Bitcoin anchoring (MerkleTree, OTS calendar, anchor tools)

## [0.1.22] — 2026-02-23

- Bump version to 0.1.22
- Merge pull request #28 from lonniev/feat/serverless-flush-strategy
- Upgrade LedgerCache flush strategy for serverless environments

## [0.1.21] — 2026-02-23

- Merge pull request #27 from lonniev/feat/nostr-audit-publisher
- Bump version to 0.1.21
- Add Nostr audit event publisher for vault write diagnostics

## [0.1.20] — 2026-02-23

- Bump version to 0.1.20
- Use Neon JSON row mode instead of arrayMode
- Fix NeonVault row parsing: use arrayMode for Neon HTTP API

## [0.1.19] — 2026-02-23

- Bump version to 0.1.19
- Merge pull request #26 from lonniev/feat/neon-vault
- Pin Python dependencies to exact versions
- Add NeonVault backend for ACID ledger persistence via Neon Postgres

## [0.1.18] — 2026-02-22

- Bump version to 0.1.18
- Merge pull request #25 from lonniev/fix/vault-daily-child-duplicates
- Fix vault daily-child duplication from stale graph cache

## [0.1.17] — 2026-02-22

- Merge pull request #23 from lonniev/feat/account-statement
- Add account_statement_tool for customer-facing purchase and usage reports

## [0.1.16] — 2026-02-22

- Merge pull request #22 from lonniev/feat/tranche-credit-expiration
- Add tranche-based credit expiration with FIFO consumption

## [0.1.15] — 2026-02-21

- Merge pull request #21 from lonniev/feat/soft-delete-vault
- Add soft-delete to TheBrainVault to avoid Azure cache staleness

## [0.1.14] — 2026-02-21

- Merge pull request #20 from lonniev/fix/child-based-vault-discovery
- Switch vault member discovery from link labels to children array

## [0.1.13] — 2026-02-21

- Merge pull request #19 from lonniev/fix/azure-affinity-vault-discovery
- Fix Azure affinity causing stale link data in vault member discovery

## [0.1.12] — 2026-02-21

- Merge pull request #18 from lonniev/refactor/link-based-vault
- Replace JSON-note vault index with link-based member discovery

## [0.1.11] — 2026-02-21

- Merge pull request #17 from lonniev/feat/thebrain-vault
- Add TheBrainVault as canonical VaultBackend implementation

## [0.1.10] — 2026-02-20

- Merge pull request #16 from lonniev/feat/detect-payout-processor
- Detect missing payout processor in btcpay_status and hint in check_payment

## [0.1.9] — 2026-02-20

- Bump version to 0.1.9
- Merge pull request #15 from lonniev/fix/test-protocol-claims
- Fix test certificates to include dpyc_protocol claim

## [0.1.8] — 2026-02-20

- Merge pull request #14 from lonniev/feat/dpyp-protocol-versioning
- Add dpyc_protocol verification and bump to 0.1.8

## [0.1.7] — 2026-02-20

- Merge pull request #13 from lonniev/fix/certificate-sub-claim
- Fix certificate sub claim extraction and bump to v0.1.7
- Merge pull request #12 from lonniev/feat/nostr-keypair-script
- Add Nostr keypair generation script and DPYC identity docs

## [0.1.6] — 2026-02-19

- Merge pull request #11 from lonniev/feat/vault-flush-durability
- Add flush retry, reconcile_pending_invoices, bump to v0.1.6

## [0.1.5] — 2026-02-19

- Bump version to 0.1.5
- Merge pull request #10 from lonniev/feat/split-purchase-tools
- Split purchase_credits_tool into certified and direct variants

## [0.1.4] — 2026-02-19

- Bump version to 0.1.4
- Merge pull request #9 from lonniev/feat/version-provenance
- Add runtime version provenance to btcpay_status diagnostic

## [0.1.3] — 2026-02-19

- Merge pull request #8 from lonniev/fix/bare-base64-keys-and-cleanup
- Accept bare base64 keys, remove authority_url scaffolding
- Merge pull request #7 from lonniev/fix/bump-version-0.1.2
- Bump version to 0.1.2 for PyPI release
- Merge pull request #6 from lonniev/feat/authority-status-diagnostic
- Surface Authority trust chain in btcpay_status diagnostics
- Merge pull request #5 from lonniev/feat/authority-certificate-verification
- Make Authority certificate mandatory — no untrusted operation
- Add Authority certificate verification for purchase flow
- Merge pull request #4 from lonniev/feat/dpyc-readme-and-tool-metadata
- Add configuration table, tool function docs, and development section
- Merge pull request #3 from lonniev/feat/arch-diagram-link
- Update Architecture section with three-party protocol view

## [0.1.1] — 2026-02-18

- Bump version to 0.1.1 for PyPI release
- Merge pull request #2 from lonniev/feat/marketing-readme
- Marketing README + CI/CD for PyPI auto-publish
- Merge pull request #1 from lonniev/feat/initial-extraction
- Extract Tollbooth commerce layer from thebrain-mcp (Task 42 Phase 1)

## [0.1.0-prior-art] — 2026-02-18

- Initial scaffold: Python packaging structure for Tollbooth DPYC

