# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.30.x → 0.33.0 series — TL;DR

Four back-to-back releases triggered by a real-world recovery: a patron's
1,000-sat Lightning payment had cleared at BTCPay but never made it to the
operator's ledger. The root cause turned out to be one bug (`CREATE SCHEMA
IF NOT EXISTS` fails for the per-operator Postgres role even when the
schema already exists) hidden behind layers of silent failure. Fixing each
layer's silence is what these releases collectively do — and 0.33.0
extends the same "don't lie about persistence" treatment to the credential
vault, after the user flagged the symmetric symptom in the ncred path.

- **0.30.0** — `check_payment_tool`, `_create_purchase_invoice`, and
  `restore_credits_tool` now detect `_vault_unavailable` ledgers and
  refuse to credit instead of returning `persisted: true` and dropping the
  data on the floor. Three latent footguns closed.
- **0.31.0** — `NeonVault._execute` reads the Neon error body before
  `raise_for_status`, so the actual SQL message reaches the logs.
  `LedgerCache._load_from_vault` logs the underlying exception. New
  operator-restricted `restore_neon_schema` admin tool re-runs
  `ensure_schema` on demand. Visibility, finally.
- **0.32.0** — `NeonVault.ensure_schema` probes `pg_namespace` before
  attempting `CREATE SCHEMA IF NOT EXISTS`. Postgres evaluates the
  database-level CREATE privilege before the IF NOT EXISTS short-circuit,
  so per-operator roles aborted setup of EVERY downstream table — the
  recovery-blocker. Now skipped when the schema already exists.
- **0.33.0** — `_vault_store` (credential write path) returns `bool`
  instead of swallowing exceptions. Both `redeem_credential_card` and
  `exchange.receive` now propagate `persisted: bool` + `error_code:
  "credential_vault_unavailable"` honestly. The "ncred state lag"
  symptom (creds appear missing for a few refreshes after delivery,
  then suddenly present) was the agent retrying until a write
  actually landed — now the agent sees `success: false` on the
  failed attempt and the human gets one consistent answer.

After 0.32.0 + Authority re-provisioning + Horizon redeploy, the patron's
1,000 sats were restored cleanly via `restore_credits` against BTCPay's
authoritative settled state. No data loss.

## [0.37.2] — 2026-05-24

### Fixed — operator-restricted tools now accept cached proof tokens

`set_pricing_model`, `reset_pricing_model`, `restore_neon_schema`,
and `restore_credits` called the bare `identity_proof.verify_proof`,
which only validates inline kind-27235 events. That meant a caller
who had completed the standard `request_npub_proof` →
`receive_npub_proof` handshake — and held a cached poison-keyed
proof token — couldn't use it for these operator-restricted tools.
Their only path was an inline-signed JSON event, which agents
without a Studio-style proof-signing helper can't easily produce.

Fix: route all four call sites through `rt.require_caller_proof()`
like paid tools already do. That helper wraps
`identity_proof.require_proof`, which checks the proven-npub cache
first and falls back to inline Schnorr — so both tactics work and
the operator can use either path.

## [0.37.1] — 2026-05-24

### Fixed — `check_price` flat branch passes kwargs to `compute()`

The flat-pricing branch of `check_price` called `pricing.compute()`
with no kwargs, which meant the new categorical-multiplier table
(0.37.0) couldn't resolve any param values. The FE preview always
got back the bare base price even when a deal_scenario kwargs
preview was supplied. Now passes `**parsed_kwargs` through, and the
response includes a `multipliers` field exposing the full table so
the FE can render a price-by-selection matrix instead of just the
current point.

Also: `check_price`'s `tool_id` argument now accepts a bare
capability string (e.g. `"deal_scenario"`) in addition to a UUID.
The wheel resolves via `capability_uuid()`. FE callers usually have
the capability name and don't want to derive UUIDs locally.

## [0.37.0] — 2026-05-24

### Added — categorical multipliers in `ToolPricing`

The existing `ToolPricing` only supported fixed costs and
percent-of-numeric-arg costs. Optionality needed a tool whose price
scales by the product of two enum-valued kwargs (``difficulty`` ×
``mode``), which neither shape covered. Added a `multipliers` field
to `ToolPricing` and a parallel `multipliers` dict on `ToolPrice` so
the pricing model JSON round-trips it.

Shape on the JSON side:

```json
{
  "tool_name": "optionality_deal_scenario",
  "price_sats": 1,
  "multipliers": {
    "difficulty": {"apprentice": 1, "journeyman": 2, "adept": 3, "sovereign": 4},
    "mode":       {"fiction": 1, "historical": 5, "live": 10}
  }
}
```

Result: `compute(difficulty="sovereign", mode="live")` returns
`ceil(1 × 4 × 10) = 40` sats. Missing param values resolve to a
multiplier of 1 (no surcharge). Compatible with the existing
fixed/percent shapes — they compose multiplicatively in the order
fixed → percent → multipliers → ceiling.

`ToolIdentity` gains a parallel `pricing_hint_multipliers` field
(frozen-tuple form for hashability) so operators can declare the
seed table in their `_DOMAIN_TOOLS` list; `_build_initial_pricing_model`
includes it in the initial pricing model entry. Operators on existing
models pick up the multipliers after a `reset_pricing_model` call.

`check_price(tool_id, tool_kwargs="...")` now correctly returns the
multiplier-scaled price when the kwargs match a configured tool.

## [0.36.0] — 2026-05-24

### Changed — restore_credits is now operator-restricted (BREAKING)

`restore_credits` was previously `category="free"` and gated on the
patron's own proof — any patron could paste an invoice ID and recover
their own balance. That misaligns with the responsibility model:

- The operator owns the books. Manual credit grants are the operator's
  discretionary action, not a patron self-service.
- Patron-self-serve restore opens replay-after-BTCPay-refund vectors,
  invoice-ID enumeration, and bypasses operator-side support visibility.
- Every other credit-issuing tool in the protocol (e.g.
  `authority_certify_credits`) is gated on the *issuer's* proof, not the
  recipient's. This brings `restore_credits` into line.

API change:

- Parameter renamed: ``npub`` → ``patron_npub`` (recipient ledger).
- ``proof`` must now be signed by the OPERATOR's nsec for the runtime
  tool name (e.g. ``optionality_restore_credits``). Patron proofs are
  rejected with ``operator_proof_invalid``.
- Tool category in ``STANDARD_IDENTITIES`` flipped from ``free`` to
  ``restricted``.

Patron-side recovery flow: patrons who paid but didn't get credits
escalate to the operator's support, who calls restore_credits with the
patron's npub. The check_payment happy-path (with the anti-replay token
from purchase_credits) is unchanged and remains patron-self-serve.

Studio: the patron-side entry point in PR #47's RecoverPaymentSheet
needs to be rewired or removed in a follow-up PR. The operator-side
entry point (in PricingDetailView's overflow menu) needs to sign with
the operator's nsec, not the patron's. See companion Studio PR for the
client-side adjustments.

## [0.35.0] — 2026-05-24

### Fixed — operator-restricted tools now verify the runtime mcp_name, not the bare capability

Three operator-restricted tools — `set_pricing_model`, `reset_pricing_model`,
and `restore_neon_schema` — were calling `verify_proof(proof, npub,
"<bare_capability>")` directly instead of `rt.runtime_name(...)`. Since
wheel 0.24.0 the proof's `u`-tag is checked against the runtime mcp_name
(`<slug>_<func>`, e.g. `optionality_set_pricing_model`), so any caller
signing the wire name — including the Pricing Studio's `signRuntimeProof`
— got `"Invalid proof — only the operator can modify pricing."` back.

Same bug shape as `AuthorityCertifier.certify_credits` in 0.28.0: the
verifier had moved to the namespaced name and a handful of callers
weren't updated. Fix is one line per site: route through
`rt.runtime_name("...")` like the rest of the codebase.

This explains why the Studio got stuck on Set Pricing Model after the
0.34.0 Reconcile diff added the oracle entries — the call was correctly
signed but rejected by the gate.

## [0.34.0] — 2026-05-24

### Fixed — oracle tools now in initial pricing model (no Reconcile false positive)

The 5 `oracle_*` delegation tools (`oracle_about`, `oracle_get_tax_rate`,
`oracle_how_to_join`, `oracle_lookup_member`, `oracle_network_advisory`)
are wire-exposed from `register_standard_tools` on every wheel-built MCP,
but were deliberately excluded from `STANDARD_IDENTITIES` with the
rationale "they're never gated, so they don't need pricing entries." That
made the Studio's Reconcile flow keep offering them as "new tools to
price" on every reset, since `list_tools()` returns them but
`get_pricing_model` doesn't.

The "never gated" claim is still true — they're routed through the free
`oracle_tool` decorator that bypasses `debit_or_deny` entirely. But the
pricing model serving as a complete inventory of what's exposed is more
useful than the pricing model being terse, so they now get
`STANDARD_IDENTITIES` entries with `category="free"` and price 0. The
wheel doesn't consult these entries at oracle-call time (no runtime
change), but the initial model is complete from day one and Reconcile
stops flagging them.

For existing operators: after redeploying on 0.34.0, the first
`reset_pricing_model` will include the oracle entries. Until then, the
Reconcile diff continues to show them as "new" — clicking accept is
harmless (entries are ignored at runtime).

## [0.33.0] — 2026-05-24

### Fixed — credential storage no longer lies about persistence

`SecureCourierService._vault_store()` previously caught all exceptions
from the credential `INSERT` and logged a warning, but returned no
signal to the caller. Both credential receive paths — DM-based
(`exchange.receive`) and ncred-based (`exchange.redeem_credential_card`)
— then unconditionally returned `success: true` with the message
"Credentials stored in vault for future sessions." If the underlying
write had actually failed (cold-start Neon hiccup, schema missing,
permission denied, etc.), the credentials were nowhere in Neon while
the agent and the user both believed they were.

This is the same lie-shape as the 0.30.0 `_vault_unavailable` ledger
bug, just one layer over. It manifests as the "ncred state lag"
symptom: after an ncred delivery, `get_patron_onboarding_status` keeps
returning `missing` for some number of refreshes, then suddenly
returns `configured` once a later attempt's write actually lands.

Fix:

1. `_vault_store` now returns `bool` (True if Neon actually accepted
   the write, False otherwise). The warning log includes exception
   type and message — same diagnostic visibility 0.31.0 added for the
   ledger path.
2. Both `redeem_credential_card` and `exchange.receive` propagate the
   flag honestly: response now includes `persisted: bool`, and on
   failure includes `error_code: "credential_vault_unavailable"` plus
   a clear retry message. `success` is `false` when persistence failed
   and a vault was supposed to receive the write.
3. The success DM to the patron (DM-path only) is sent only when
   `persisted=True` — pre-fix, the patron got "thanks, got it" even
   when nothing was stored.

The agent (Claude.ai, Studio, any MCP client) can now branch on
`success` or `error_code` to retry receive_credentials cleanly, and
the human gets one consistent answer about whether the credentials
made it.

Existing operators with `on_credentials_received` callbacks: this
release does NOT change which path runs them — the ncred shortcut
still bypasses `SecureCourierService.receive`'s post-process block
(callbacks, session_bindings writes, credential-card DM echo). That's
a separate follow-up for the future when operators actually rely on
those callbacks; for now no operator in this monorepo sets one.

## [0.32.0] — 2026-05-24

### Fixed — ensure_schema no longer aborts on role-isolated tenants

`NeonVault.ensure_schema()` opened with `CREATE SCHEMA IF NOT EXISTS
{op_xxx}` to be defensive. But the per-operator Postgres role
provisioned by `tenant_provisioner.provision_operator_schema` owns
the schema (CREATE inside it works) but does NOT have CREATE
privilege on the database itself. Postgres evaluates the database-
level privilege before the `IF NOT EXISTS` short-circuit, so the
statement raises `permission denied for database neondb` even when
the schema already exists.

Since `ensure_schema()` is called as the very first step of
`_get_vault()`, the exception aborted the entire setup. No CREATE
TABLE ran; subsequent `SELECT ... FROM op_xxx.balances` queries got
`relation does not exist` and `_load_from_vault` returned
`_vault_unavailable=True` forever. The 0.30.0 vault_unavailable
guards correctly refused to pretend persistence worked, but the
underlying issue stayed invisible until 0.31.0's surfaced the SQL
error body.

Fix: probe `pg_namespace` for the schema BEFORE attempting
`CREATE SCHEMA`. Skip the create entirely when it already exists.
The CREATE attempt is reserved for the genuine first-time case
where a privileged role is bootstrapping; for the per-operator role
visiting an Authority-provisioned schema (the production path),
ensure_schema now flows straight to the CREATE TABLE statements,
which succeed against the schema the role owns.

Hit by Optionality after schema drop + Authority re-provision: the
schema came back via the privileged Authority role's
`tenant_provisioner.provision_operator_schema`, but the wheel could
never populate it because the very first ensure_schema call failed
for the operator role.

## [0.31.0] — 2026-05-24

### Added — `restore_neon_schema` admin tool + Neon error body surfaced

When Optionality's patron-recovery flow ran into persistent 400s from
Neon's HTTP SQL API, the wheel was blind to *why*: `_execute` called
`resp.raise_for_status()` before reading the body, so the actual SQL
error message ("relation does not exist", "permission denied",
"password authentication failed", etc.) never reached the logs or
the caller. The result: `Failed to load ledger from vault` with no
hint of which root cause to chase.

Three surgical changes:

1. `NeonVault._execute` now reads the response body on 4xx before
   `raise_for_status`. If Neon returned a JSON body with a `message`
   field (its standard SQL-error shape), the wheel raises
   `NeonQueryError` with that message + a query excerpt. Only
   bodyless 4xx and 5xx fall through to the opaque `HTTPStatusError`.

2. `LedgerCache._load_from_vault` no longer swallows the underlying
   exception silently. The warning now includes the exception type
   and message so logs tell you what failed instead of just that
   something did.

3. New operator-restricted `restore_neon_schema` tool re-runs
   `ensure_schema()` on `NeonVault`, `PricingModelStore`, and the
   credential vault (if configured). Idempotent (uses
   `CREATE TABLE IF NOT EXISTS`). Returns per-step success / error
   so the operator can diagnose without redeploying. Requires
   nsec-signed proof — same gate as `reset_pricing_model`.

The fix doesn't change behavior for the happy path; it only converts
silent-and-opaque failure into loud-and-actionable failure.

## [0.30.0] — 2026-05-24

### Fixed — restore_credits and pending-invoice tracking no longer silently fail on cold start

Same root cause as 0.29.0's `check_payment` fix: `LedgerCache.get()`
on a cold serverless instance returns an uncached `UserLedger`
flagged `_vault_unavailable = True`. The other two ledger-write
paths shared the bug:

1. `_create_purchase_invoice` recorded a pending invoice into that
   short-lived ledger; the patron's BTCPay invoice existed at BTCPay
   but the Tollbooth ledger had no record of it. Subsequent
   `check_balance` showed no pending invoices, so the Studio's
   reconcile-pending UX never offered a way to verify settlement.
2. `restore_credits_tool` — the recovery path of last resort — would
   credit an uncached ledger and return `success: true` with the
   credits about to be garbage-collected.

Both now check `_vault_unavailable` and respond appropriately:

- `_create_purchase_invoice` logs `error` (the invoice creation
  itself still succeeds — BTCPay is the source of truth — but the
  log makes the recovery situation obvious) and skips the no-op
  mark_dirty/flush. The patron's path: pay the invoice, then call
  `restore_credits` on the next warm call.
- `restore_credits_tool` refuses outright, returning `success:
  false + error_code: "vault_unavailable"` so the caller retries.

A deeper fix would be in `LedgerCache.flush_user`, which currently
returns `True` for "nothing to flush" — indistinguishable from
"successfully flushed." That's a wider refactor; the three callsite
guards in 0.29.0 + 0.30.0 cover every wheel-internal write path.

## [0.29.0] — 2026-05-24

### Fixed — check_payment no longer lies about persistence on cold start

When `LedgerCache.get()` couldn't reach Neon (serverless cold start),
it returned an *uncached* empty ledger flagged `_vault_unavailable =
True`. `check_payment_tool` then mutated that in-memory ledger,
called `cache.mark_dirty(user_id)` (a silent no-op when the entry
isn't in `_entries`), then `cache.flush_user(user_id)` (returns
`True` for "nothing to flush" — indistinguishable from "successfully
flushed"). The patron saw `Settled + credits_granted: N + persisted:
true`, but the credits lived in a UserLedger that got garbage-
collected the moment the function returned. A subsequent
`account_statement` showed balance 0.

The fix: `check_payment_tool` now checks `_vault_unavailable` before
crediting. If true, it returns `success: false + persisted: false +
error_code: "vault_unavailable"` with a clear retry prompt. The
patron's LN payment is safe at BTCPay; the next `check_payment` call
hits a warmed cache, finds the invoice isn't in `credited_invoices`,
and credits normally (idempotent).

`check_balance_tool` already handled the same flag correctly; only
the write paths needed this guard. `_create_purchase_invoice` and
`restore_credits_tool` have the same latent pattern but their
silent-failure consequences are less severe (the BTCPay invoice
still exists; the pending-invoice tracking just lags). Patching them
is left to a follow-up.

## [0.28.0] — 2026-05-24

### Fixed — AuthorityCertifier signs the runtime mcp_name, not the bare capability

`AuthorityCertifier.certify_credits` was signing its kind-27235 proof
for the bare capability string `"certify_credits"`. Since wheel
0.24.0 the verifier on the Authority side gates against the runtime
mcp_name (`<slug>_<func>`, e.g. `"authority_certify_credits"`). The
mismatch caused every Operator → Authority certify call to fail with
`Invalid identity proof.` — which in turn broke patron-side
`purchase_credits` flows on every Operator, since the wheel auto-
certifies through the Authority on each patron top-up.

The fix is one line in `authority_client.py`: sign the proof using
`self._certify_tool_name` (the same wire name the call uses) instead
of the literal `"certify_credits"`. No public API change.

Tests still pass; the regression slipped past coverage because the
Operator → Authority cert path is exercised only by integration
scenarios, not unit tests.

## [0.27.0] — 2026-05-23

### Fixed — Oracle registry failures no longer silently masked

`_register_operator_via_oracle`, `_update_operator_via_oracle`,
`_deregister_operator_via_oracle`, and `_register_via_oracle`
previously did:

```python
return json.loads(block.text).get("commit_url", block.text)
```

When the Oracle returned `{"success": false, "error": "..."}`, the
missing `commit_url` key caused the *entire failure JSON string* to be
stuffed into the outer `commit_url` field, while the outer tool
returned `success: true`. Studios and dashboards then displayed the
operator as registered with a JSON blob where a GitHub URL should
have been.

The four helpers now share a single parser, `_parse_oracle_commit_url`,
that detects `success: false` and raises `OracleRegistryError`. The
outer `try/except` handlers behave correctly:

- `register_operator`: logs warning, returns `commit_url=""` instead
  of a fake URL. Local ledger still consistent.
- `update_operator` / `deregister_operator`: propagate as
  `{"success": false, "error": "Update failed: ..."}`.
- `register_authority`: logs warning, omits `commit_url` from the
  response.

No public API change. Internal failure semantics only.

## [0.26.0] — 2026-05-23

### Changed — Authority consent now cryptographically witnessed (BREAKING)

`authority_register_operator`, `authority_update_operator`, and
`authority_deregister_operator` now require **two** independent
Schnorr identity proofs:

1. **`proof`** — signed by the *Operator's* npub. Existing behavior;
   proves the caller really controls the operator npub they claim.
2. **`authority_proof`** — signed by the *Authority's own* npub. New.
   Cryptographic witness that the human who controls the Authority's
   nsec has consented to this discretionary action. Apps holding the
   Authority's nsec (e.g. the Pricing Studio) produce this proof
   inline when their user clicks "adopt".

Without the Authority proof, anyone who knew an operator's public
npub and held that operator's own nsec could register / mutate /
remove themselves under any Authority's signature without that
Authority's awareness. Closes the gap where the Operator's
self-signed proof was the only gate.

New `ErrorCode.AUTHORITY_CONSENT_REQUIRED` distinguishes the
Authority-side failure from the Operator-side `PROOF_REQUIRED` /
`PROOF_INVALID` codes, so calling apps can render the right UX
remedy.

**Breaking change.** Clients calling these three tools without an
`authority_proof` argument will receive
`{success: false, error_code: "authority_consent_required", ...}`.
Update calling code to mint a fresh Authority-side proof and pass it.

The Pricing Studio's `argsWithProof` machinery already produces
Operator proofs from Keychain; producing the Authority proof is the
same code path keyed by the Authority's npub.

## [0.25.0] — 2026-05-19

### Changed — DRY pass on the proof gate and operator bootstrap

Three repeated patterns collapsed.

**`OperatorRuntime.require_caller_proof(npub, proof, capability)`** —
the dozen standard-tool sites that wrote
`await require_proof(npub, proof, rt.runtime_name("X"), proven_cache=await rt.proven_npub_cache())`
now write `await rt.require_caller_proof(npub, proof, "X")`. Same
semantics, one short line.

**Module-level imports of `capability_uuid` and `require_proof`** —
four lazy `from tollbooth.tool_identity import capability_uuid` and
fourteen lazy `from tollbooth.identity_proof import require_proof`
imports peppered inside `runtime.py` function bodies are gone. Verified
no circular dependency.

**`register_standard_tools` returns the `@tool` decorator** — operator
servers previously typed the slug twice:

```python
tool = make_slug_tool(mcp, "schwab")        # slug literal once
register_standard_tools(mcp, "schwab", runtime, …)  # slug literal twice
```

Now register the standard tools first and use the returned decorator:

```python
tool = register_standard_tools(mcp, "schwab", runtime, …)
```

All five in-tree operators (schwab, brain, excalibur, taxsort, sample)
updated in lock-step.


## [0.24.1] — 2026-05-19

### Changed — Oracle-delegated tools mount under the operator's slug

`register_standard_tools` previously registered oracle delegations under
a bare `oracle_*` namespace (e.g. `oracle_about`), so an operator like
brain-mcp exposed both `brain_*` (its own paid tools) and `oracle_*`
(delegated free tools) on the same wire. That mixed namespace broke any
downstream client trying to derive the operator's slug from `tools/list`
via a longest-common-prefix scan.

Oracle delegations now mount under `<slug>_oracle_*` — e.g.
`brain_oracle_about`, `schwab_oracle_about`. Every wire-exposed tool on
a given operator now shares a single slug prefix, and downstream slug
detection can fall back to LCP without special-casing oracle. The intent
was always that every operator delegates oracle calls; this change makes
that uniform on the wire.

Breaking for any client calling the bare `oracle_*` names — switch to the
slug-prefixed form. Pricing Studio's `MCPService` already resolves the
slug via the `_service_status` marker so it picks the new names up
automatically.


## [0.24.0] — 2026-05-18

### Changed — identity proof signs the runtime tool name, not the capability seed

`debit_or_deny` and the dozen standard-tool `require_proof` callers passed
`identity.capability` (the Python function identifier — e.g. `check_balance`)
as the tool name the proof's `u` tag had to match. But the capability is
deliberately internal: it exists to derive a stable `tool_id` UUID that
survives FastMCP slug renames, and to decouple the in-process function name
from the on-wire tool name. Every other server boundary — the MCP wire, the
pricing model's `tool_name` field, audit logs — uses the runtime name
(`mcp_name`, e.g. `schwab_check_balance`). The proof verifier was the lone
exception, and that exception forced external callers (the Pricing Studio
App, AI agents) to know about the capability/runtime-name split.

The runtime name (mcp_name) is now the ONE external identifier. A new
`OperatorRuntime.runtime_name(capability)` helper resolves it from a
capability seed (function-local code stays readable):

```python
if err := await require_proof(
    npub, proof,
    rt.runtime_name("check_balance"),
    proven_cache=await rt.proven_npub_cache(),
):
    return err
```

`debit_or_deny` itself passes `name = self.mcp_name_for(tool_id)` to
`require_proof` — same value FastMCP exposes on the wire, same value the
pricing model lists.

**Breaking for external proof signers.** Any caller that was signing the
short capability name (e.g. `["u", "check_balance"]`) will now hit
`proof_invalid`. Sign the runtime tool name instead
(e.g. `["u", "schwab_check_balance"]`). The Pricing Studio App will be
updated to match in lock-step.

`identity.capability` is unchanged — still the seed for the UUID, still
internal. `tool_id` is unchanged — still UUID5(capability), still stable
across slug renames. No Neon migration needed.


## [0.23.1] — 2026-05-17

### Fixed — Authority issues `dpyp-01-base-certificate`, not `tollbooth-cert-v1`

`certify_credits` in `authority/tools.py` stamped the certificate's
`dpyc_protocol` claim with the legacy string `tollbooth-cert-v1`, but
the verifier in `certificate.py` (and every test fixture, and every
Operator) only knows `dpyp-01-base-certificate`. Operators paying a
certified Authority for credits would see:
`Certificate rejected: Unsupported protocol 'tollbooth-cert-v1'.`

Stale string carried over from the pre-v0.22 standalone
`tollbooth-authority` repo when the Authority code moved into the
wheel as `tollbooth.authority`. Tests didn't catch it because the
unit tests for the issuer mocked the protocol field, and integration
tests never crossed an Authority→Operator certified-purchase boundary.

The fix is one-line — replace the legacy string with the canonical
protocol identifier.


## [0.23.0] — 2026-05-17

### Fixed — single canonical proof-of-ownership gate

`require_proof` and `OperatorRuntime.debit_or_deny` were two parallel
implementations of the same concern. `debit_or_deny` accepted both a
cached poison phrase (looked up in the proven-npub cache) and an inline
Schnorr-signed kind-27235 event. `require_proof` only accepted the
Schnorr tactic. Tools added in v0.19 called `require_proof` *before*
`debit_or_deny`, so the cache populated by `receive_npub_proof` was
never readable by paid tools — a successful DM exchange would still
fail the next paid call with `Invalid identity proof.`

`require_proof` is now the single canonical gate. It is async, accepts
an optional `proven_cache: ProvenNpubCache`, and dispatches by proof
shape: a `<word>-<word>-<n>` poison phrase is sha256-hashed and looked
up in the cache; anything else is verified as a Schnorr kind-27235
event with the tool name in the `u` tag. Empty proof returns
`proof is required.` with next-steps describing *both* tactics. All
error responses carry an `error_code` (`PROOF_REQUIRED`,
`PROOF_REFRESH_NEEDED`, `PROOF_INVALID`, `NPUB_INVALID`).

`debit_or_deny` no longer duplicates the proof logic — it computes the
target npub (operator npub for restricted tools, caller npub for
everything else) and delegates to the gate. The 17 inline
`require_proof(...)` call sites across `runtime.py` (12) and
`authority/tools.py` (5) now `await require_proof(..., proven_cache=
await rt.proven_npub_cache())`. Actor-agnostic — Operators and
Authorities pass their own cache.

### Breaking change

`tollbooth.identity_proof.require_proof` is now async and takes the
cache as a keyword argument. External callers (none known) must
update accordingly. Internal call sites already migrated.


## [0.22.1] — 2026-05-16

### Fixed — `service_status.vault_configured` reflects readiness, not prior use

The `vault_configured` field in `service_status` used to check
`rt._vault is not None`, which only flipped true after some other tool
had lazily instantiated the vault. On a fresh process, a properly-
configured Authority would still report `vault_configured: false`
until the first vault-touching tool ran, which contradicted the
field's name.

`service_status` now triggers the lazy init itself (same pattern the
adjacent `courier` check uses) and reports `true` if the vault can be
opened. Failures (missing `NEON_DATABASE_URL`, unreachable backend)
are caught and reported as `false` so `service_status` stays a
non-failing diagnostic.


## [0.22.0] — 2026-05-16

### Added — `register_authority_tools(mcp, runtime)` mixin (Phase B completes the Authority code unification)

The 10 Authority `@tool` definitions that every Authority MCP previously
forked in its own `server.py` are now defined exactly once in
`tollbooth.authority.tools`. Authority repos call
`register_authority_tools(mcp, runtime)` analogous to how operator MCPs
call `register_standard_tools(mcp, slug, runtime)`.

Tools moved into the wheel:

- `register_operator`
- `update_operator`
- `deregister_operator`
- `get_operator_config`
- `operator_status`
- `certify_credits` (the ad-valorem revenue tool)
- `check_dpyc_membership`
- `register_authority_npub`
- `confirm_authority_claim`
- `check_authority_approval`

Also exported:

- `AUTHORITY_DOMAIN_TOOLS` / `AUTHORITY_TOOL_REGISTRY` — the ToolIdentity
  list/dict that Authority MCPs merge into their OperatorRuntime
  `tool_registry`.
- `OPERATOR_CREDENTIAL_TEMPLATE` — the standard BTCPay-store-credentials
  template that every Authority's OperatorRuntime uses for its cashier.

### Net effect across Authority repos (after this lands)

Each Authority repo's `server.py` collapses from ~1000 lines of forked
tool definitions to roughly 30 lines of actor-specific configuration:

```python
from fastmcp import FastMCP
from tollbooth.authority import (
    AUTHORITY_TOOL_REGISTRY,
    OPERATOR_CREDENTIAL_TEMPLATE,
    register_authority_tools,
)
from tollbooth.runtime import OperatorRuntime, register_standard_tools
from tollbooth.tool_identity import STANDARD_IDENTITIES

mcp = FastMCP("tollbooth-authority-mine", instructions="…")
runtime = OperatorRuntime(
    tool_registry={**STANDARD_IDENTITIES, **AUTHORITY_TOOL_REGISTRY},
    purchase_mode="direct",
    ots_enabled=True,
    operator_credential_template=OPERATOR_CREDENTIAL_TEMPLATE,
)
register_standard_tools(mcp, "authority", runtime, ...)
register_authority_tools(mcp, runtime)
```

Eight legacy modules per Authority repo (`actor.py`, `config.py`,
`nostr_signing.py`, `onboarding.py`, `registry.py`, `replay.py`,
`role_migration.py`, `tenant_provisioner.py`) become deletable — their
contents now live in `tollbooth.authority.*`.

### Architectural note

This completes the architectural unification started in v0.21.0 Phase A
(which moved the 6 supporting modules). NA and NE repos were always
intended as *forks-to-get-started, distinct thereafter*, not as ongoing
clones of canonical. With this release, anything that benefits multiple
Authorities lives in the wheel, while each Authority's repo holds only
its actor-specific identity, instructions, and (optionally) overrides.

## [0.21.0] — 2026-05-16

### Added — `tollbooth.authority` package (Phase A of Authority code unification)

The six supporting modules that every Authority MCP previously kept its
own copy of are now defined exactly once in the wheel:

- `tollbooth.authority.onboarding` — `OnboardingState`,
  `OnboardingChallenge`, `AUTHORITY_CLAIM_TEMPLATE`,
  `AUTHORITY_APPROVAL_TEMPLATE`
- `tollbooth.authority.nostr_signing` — `AuthorityNostrSigner`,
  `NOSTR_CERT_KIND`
- `tollbooth.authority.replay` — `ReplayTracker`
- `tollbooth.authority.tenant_provisioner` — `provision_operator_schema`,
  `neon_url_for_operator`, etc.
- `tollbooth.authority.role_migration` — CLI to migrate legacy schemas
- `tollbooth.authority.settings` — `AuthoritySettings` (pydantic-settings)

Authority repos that previously forked these six modules verbatim
across `tollbooth-authority`, `tollbooth-authority-northamerica`, and
`tollbooth-authority-newengland` can now delete their local copies and
import from `tollbooth.authority.*` instead. Each Authority repo's
diff after adoption: 8 files deleted, server.py imports updated.

### Phase B coming next (v0.22.0)

The 10 Authority `@tool` definitions in each Authority's `server.py`
(register_operator, update_operator, deregister_operator,
get_operator_config, operator_status, certify_credits, check_dpyc_membership,
register_authority_npub, confirm_authority_claim, check_authority_approval)
will be promoted into a `register_authority_tools(mcp, runtime)` mixin
function in this package. After v0.22.0 lands, each Authority repo's
`server.py` collapses to roughly:

```python
from fastmcp import FastMCP
from tollbooth.authority import register_authority_tools
from tollbooth.runtime import OperatorRuntime, register_standard_tools

mcp = FastMCP("tollbooth-authority-mine", instructions="…")
runtime = OperatorRuntime(...)
register_standard_tools(mcp, "authority", runtime, ...)
register_authority_tools(mcp, runtime)
```

— roughly 30 actor-specific lines instead of 1000.

## [0.20.0] — 2026-05-16

### Added — generic parent-Authority resolution

New `tollbooth.registry.resolve_my_parent_npub(own_npub)` reads the
caller's own entry from dpyc-community and returns its
`upstream_authority_npub`. Authority MCPs use this to escalate
onboarding claims to their *registered* parent — no longer hardcoding
Prime as the only approver.

For Lonnie-Authority and Tollbooth-Authority-NorthAmerica the parent
is still Prime, so observable behavior is unchanged. For
Tollbooth-Authority-NewEngland, the parent is now NorthAmerica — its
onboarding claim escalates to NA, not Prime. The protocol now cascades
transparently through arbitrary chain depth.

**Architectural note:** this is the first step of promoting Authority-
generic code from the three Authority repos (canonical, NorthAmerica,
NewEngland) into the wheel. The Authority repos previously each held
their own `_resolve_prime_npub()` helper; they'll now import this
wheel-side helper instead. More Authority-class code (onboarding state
machine, the 3 onboarding tools, certify_credits) is a planned future
v0.21+ promotion to a full `register_authority_tools(mcp, runtime)`
mixin analogous to `register_standard_tools`.

## [0.19.0] — 2026-05-16

### Security — Breaking API: every tool that names a proof now verifies it

Closed a class of bug where standard tools accepted `proof: str = ""` in
their signature but never actually called `verify_proof`. The pattern
looked security-aware (proof is a parameter!) without enforcing anything —
any caller with someone else's npub could enumerate balances, scrape
account statements, and read OAuth state. The Authority service had
silently patched its own tools with a local `_verify_operator_proof`
helper; that fix is now hoisted into the wheel as `identity_proof.require_proof`
and applied uniformly.

**Affected standard tools** (now require non-empty `npub` and `proof`,
with proof verified via Schnorr against npub bound to the tool name):

- `check_balance`, `purchase_credits`, `check_payment`, `restore_credits`
- `account_statement`, `account_statement_infographic`, `get_patron_onboarding_status`
- `update_patron_credential`, `delete_patron_credential`, `get_patron_credential_fields`
- `forget_credentials`, `begin_oauth`, `check_oauth_status`

**Bootstrap tools intentionally still allow empty params** — they're how
a candidate proves identity in the first place: `request_credential_channel`,
`receive_credentials`, `request_patron_credentials`, `receive_patron_credentials`,
`request_npub_proof`, `receive_npub_proof`, `check_proof_status`.

**Caller impact**: any MCP client that called these tools without proof
will receive a clear error response with `next_steps` pointing at the
request/receive_npub_proof handshake. Pricing Studio already drives that
handshake (per `feedback_human_in_loop_proof`); other clients (Claude
Desktop, Claude Code, ad-hoc curls) need to wire it up.

**Deferred to follow-up**: `session_status` (two-mode anonymous-or-per-patron
design), `check_price` (per-npub pricing nuance), `service_status` (system-level).

## [0.18.0] — 2026-05-13

### Security
- **Vault AAD is now enforced — no silent fallback.** `VaultCipher.decrypt()` previously retried-without-AAD on tag failure, a back-compat shim added in v0.14.0 for ciphertext written before AAD support landed. That shim made AAD purely advisory: a ciphertext written with `aad="oauth/access_token"` could be swapped into the `aad="oauth/refresh_token"` slot, the AAD-aware decrypt would fail, the shim's retry-without-AAD path would succeed, and the application got the wrong-slot plaintext silently. AAD now does what its name says — a tag mismatch raises `InvalidTag` and decrypt fails. **Mildly breaking**: any pre-AAD-era ciphertext written before v0.14.0 (2026-04-19) that has not been rotated through a write since then is now un-decryptable. After ~24 days of normal traffic (OAuth refresh cycles, ledger updates, Secure Courier redelivery), the realistic survivor population is set-once preferences and untouched operator credentials; both recover through normal operator/patron flows (`update_patron_credential`, `receive_credentials`).

### Added
- **`cryptography>=46.0.5` is now an explicit dependency.** The wheel uses `AESGCM` from `cryptography` in `vault_encryption.py` but had been relying on transitive resolution. Fresh installs without other pinning could pull older versions with known CVEs. 46.0.5 is the floor that includes the 2024-2025 fix cohort consumers of this wheel already run.

### Fixed
- Three pre-existing ruff lint findings on `main` that had been silently failing the Tests workflow's Lint step — an unused `f""` prefix in `proven_npub.py`, an unused `except as e` binding in `runtime.py`, and an unused `PaymentRequired` import in `x402_client.py`. No behavior change.

## [0.17.4] — 2026-05-12

### Fixed
- **BTCPay invoices now carry an `orderId` so Lightning wallets render a real description.** BTCPay's per-store Lightning Description Template — typically `"{StoreName} (Order ID: {OrderId})"` — gets substituted into the BOLT11 description that paying wallets read. Without an `orderId`, wallets like Wallet of Satoshi showed `"Paid to <StoreName> (Order ID:)"` with an empty field, which looked like a bug to the patron. Every credit-purchase invoice now sets `orderId` to `"dpyc-{purpose}-{user_id[:16]}-{utc_timestamp}"` — enough to disambiguate purchases in the patron's wallet history without exposing the full npub. Any pre-set `orderId` in `extra_metadata` is honored.

## [0.17.3] — 2026-05-05

### Fixed
- **`update_patron_credential` (and friends) now work on OAuth-only operators.** Previously, `update_patron_credential(npub, field='account_hash', value=<hash>)` returned `"No credential service configured"` for schwab-mcp because no `patron_credential_template` was wired. The new `OperatorRuntime._patron_storage_service` helper falls back to the OAuth provider's `service_name` when no patron template is set, so per-patron preferences (e.g. `account_hash`, `default_brain_id`) land in the OAuth vault entry alongside the tokens — surviving refresh cycles automatically. Closes the contradiction where `account_hash_required` errors directed callers at a tool that refused to write.
- Same fallback for `delete_patron_credential`, `get_patron_credential`, `list_patron_credential_fields`. Explicit `service=` argument and the `patron_credential_service` (set-once template) both still take precedence; the OAuth fallback is the third option.

## [0.17.2] — 2026-05-05

### Changed (breaking — internal codes)
- **`oauth_situation_response` now maps situations 1:1 to ErrorCodes.** Previously two distinct situations (token_expired vs no_credentials, operator_not_configured vs no_oauth_config) collapsed to the same code, losing the diagnostic specificity calling agents need. Each situation now keeps its own code — shared recovery flows are expressed via shared `next_steps`, not shared codes.
- Removed `ErrorCode.OAUTH_REFRESH_NEEDED` (split into `OAUTH_TOKEN_EXPIRED` for returning patrons and `OAUTH_NOT_YET_AUTHORIZED` for first-time patrons).
- Removed `ErrorCode.OPERATOR_NOT_CONFIGURED` (split into `OAUTH_NOT_WIRED` for deployment-side issues and `OPERATOR_CREDENTIALS_MISSING` for credential-delivery issues).
- Unknown situations now return `OAUTH_SITUATION_UNKNOWN` with the raw situation echoed, instead of silently masquerading as a routine refresh.

### Added
- **`OperatorRuntime.npub_validation_error(npub, *, param)`** — DRY validation helper returning `None` if valid or a structured `{success, error_code, error}` dict otherwise. Distinguishes `NPUB_MISSING` (param absent) from `NPUB_INVALID` (bad format).
- **`OperatorRuntime.proof_validation_error(proof, *, param)`** — same shape for proof_token presence checks. Returns slug-qualified `next_steps` recipe.
- New `ErrorCode` entries: `NPUB_MISSING`, `PROOF_MISSING`, `OAUTH_TOKEN_EXPIRED`, `OAUTH_NOT_YET_AUTHORIZED`, `OAUTH_NOT_WIRED`, `OPERATOR_CREDENTIALS_MISSING`, `OAUTH_SITUATION_UNKNOWN`.
- Standard tools (`begin_oauth`, `check_oauth_status`, `request_npub_proof`, `receive_npub_proof`, `check_proof_status`, `update/delete_patron_credential`, `get_patron_credential_fields`, `get_patron_onboarding_status`) now use the new helpers instead of inline `if not npub: return …; try: resolve_npub(…) except: return …` duplication. All free-tool denial paths now carry `error_code`.

## [0.17.1] — 2026-05-04

### Added
- **`OperatorRuntime.oauth_situation_response(situation)`** — generic helper that maps the situation strings returned by `restore_oauth_session` to canonical `{success, error_code, error, next_steps}` dicts. Tool names in `next_steps` are qualified with the runtime's slug so the response is directly invocable. Eliminates duplication of situation→error_code tables across consumer MCPs (schwab, excalibur, future OAuth2 operators).
- **`ErrorCode.OAUTH_REFRESH_NEEDED`** and **`ErrorCode.OPERATOR_NOT_CONFIGURED`** for the standard OAuth-session-restoration outcomes.

## [0.17.0] — 2026-05-04

### Added
- **Structured `error_code` field** on every denial path from `debit_or_deny` and the `paid_tool` decorator's `catch_errors` fallback. Calling agents can branch on stable strings (e.g. `proof_refresh_needed`, `insufficient_balance`, `upstream_auth_refresh_needed`) without parsing prose. New `tollbooth.constants.ErrorCode` enumerates the codes.
- **Patron-actionable `next_steps` lists** on routine refresh situations (`proof_required`, `proof_refresh_needed`, `insufficient_balance`, `upstream_auth_refresh_needed`) so calling LLMs can route directly to the recovery flow.
- **`check_proof_status` standard tool** — free, no side effects. Mirrors `check_oauth_status` for the npub-proof flow: a calling agent can ask "will my next paid call accept this proof_token?" before burning credits on a guaranteed failure. Returns `status` (valid|expired|unknown) and runtime-derived `expires_in_seconds`.
- **`ProvenNpubCache.proof_status(poison_hash, npub)`** — read-only sibling of `is_proven`; never mutates cache state on expiry.
- **Optional `patron_npub` argument on `session_status`** — when supplied, the response includes an `upstream_oauth` block with the patron's stored OAuth token expiry (runtime-derived from vault state) so clients can refresh proactively rather than reactively.
- **Generic upstream-auth detection in `paid_tool`** — exception messages mentioning `401`, `unauthorized`, `invalid_grant`, or `token expired` are remapped to `error_code: "upstream_auth_refresh_needed"` with a `begin_oauth → check_oauth_status` next-steps recipe.

### Changed
- Removed magic-number TTL claims from user-facing error strings. Where a number was previously printed (e.g. "10-15 seconds after a cold start"), the wording now describes the behavior ("typically resolves shortly") so docstring "examples" don't ossify into false contracts.

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

