# Deferred Operator Adoption — Implementation Plan (wheel-wide)

Status: design approved (deferred-courtship model), implementation pending.
Scope: `tollbooth-dpyc` SDK feature; every operator + Authority inherits it on a pin bump.

## Decisions baked in (override if wrong)

1. **Request delivery = MCP-to-MCP.** `request_adoption` (operator-side) calls the Authority's `receive_adoption_request` over FastMCP `Client` (request/response, Horizon-friendly — no Authority-side inbox polling). The secure-courier DM is used only for the *owner notification*.
2. **Operator proof = inline self-sign.** `request_adoption` runs on the operator, which holds its own nsec on Horizon, so it mints an inline Schnorr (kind-27235) adoption proof binding `operator_npub → authority_npub` and forwards it. The Authority verifies it inline with the existing `identity_proof` verify primitive — no DM round-trip. (Symmetric to how `authority_proof` is produced by the nsec-holder.) The human-in-the-loop moment is the `request_adoption` invocation itself.
3. **Approval phasing.** Phase 1 = route (A) Studio approve via `authority_proof` + durable queue + owner-notification DM. Phase 2 (additive) = route (B) `check_adoption_replies` async-DM approval.
4. **Owner-notify target = the Authority's own npub** (`runtime.operator_npub()`); the Studio, holding that nsec, surfaces it.
5. **Registry timing.** Oracle/dpyc-community registration moves from request-time to **approval-time**, so pending (un-adopted) operators never appear in the public roster.

## State — Authority-side Neon table (durable, replaces the in-memory `_onboarding` singleton weakness)

`operator_adoption_requests`:
```
operator_npub  TEXT PRIMARY KEY
service_url    TEXT NOT NULL DEFAULT ''
status         TEXT NOT NULL DEFAULT 'pending'   -- pending|approved|rejected|provisioned
poison_hash    TEXT NOT NULL DEFAULT ''          -- for Phase-2 async-DM reply match
note           TEXT NOT NULL DEFAULT ''
requested_at   TIMESTAMPTZ DEFAULT now()
decided_at     TIMESTAMPTZ
expires_at     TIMESTAMPTZ                        -- requested_at + 7d
```
Keyed per operator → concurrent requests are fine and survive cold starts (the precedent's singleton was one-at-a-time and process-local).

---

## Work units (each a releasable checkpoint; SDK-first)

### Unit 0 — Characterize `register_operator` (safety net, no behavior change)
Add `tests/test_authority/test_register_operator_characterization.py` pinning today's contract: (a) denies without operator `proof`, (b) denies without `authority_proof` (`AUTHORITY_CONSENT_REQUIRED`), (c) on both valid → ledger row + `provision_operator_schema` called + oracle register + bootstrap DM + `neon_database_url` returned. This makes the Unit 1 extraction provably faithful (characterize-then-extract).

### Unit 1 — Extract `_provision_operator(npub, service_url)` (pure refactor)
In `authority/tools.py`, lift the body of `register_operator` (lines ~685–742: ledger → `provision_operator_schema` → `store_operator_config` → `_register_operator_via_oracle` → `_resend_bootstrap_dm` → result dict) into a module-private `async def _provision_operator(runtime, npub, service_url) -> dict`. `register_operator` becomes:
```python
err = await require_proof(npub, proof, runtime.runtime_name("register_operator"), …);            if err: return err
err = await _require_authority_consent(runtime, authority_proof, runtime.runtime_name("register_operator")); if err: return err
return await _provision_operator(runtime, npub, service_url)
```
Unit 0 tests stay green = faithful. **Wire API unchanged.** Releasable alone.

### Unit 2 — Adoption store (`authority/adoption_store.py`)
CRUD over `operator_adoption_requests`, mirroring `pricing_store.py` conventions (`vault._t(...)`, `vault._execute(...)`, `rowCount` camelCase):
- `ensure_schema(vault)`
- `upsert_pending(vault, operator_npub, service_url, poison_hash, note) -> None` (idempotent; refreshes `expires_at`)
- `get(vault, operator_npub) -> dict|None`
- `list_pending(vault) -> list[dict]`
- `mark(vault, operator_npub, status) -> bool` (sets `decided_at`)
- `prune_expired(vault) -> int`
Unit tests against a fake vault (assert SQL params, status transitions, rowCount handling).

### Unit 3 — Authority tools + identities + errors
`authority/tools.py`:
- `receive_adoption_request(operator_npub, proof, service_url)` — category `free`, **gated by inline operator-npub proof** (verify the kind-27235 adoption proof signed by `operator_npub`, bound to this Authority's npub). Effect: `ensure_schema` + `upsert_pending`; fire-and-forget `open_channel("operator_adoption", greeting="Operator <npub> (<url>) requests adoption — approve in Studio or reply yes + <poison>", recipient_npub=runtime.operator_npub())`; store `poison_hash`. Returns `{success, status:"pending"}`. **Never provisions.**
- `list_adoption_requests(authority_proof)` — `restricted` (`_require_authority_consent`). Returns `list_pending`.
- `approve_adoption(operator_npub, authority_proof)` — `restricted`. → `_provision_operator(runtime, operator_npub, row.service_url)`; `mark(..., "provisioned")`. Returns the provisioning result.
- `reject_adoption(operator_npub, authority_proof, reason="")` — `restricted`. `mark(..., "rejected")`; optional reject DM to the operator.
- `get_adoption_status(operator_npub)` — `free`. Read-only status for the operator's poll (no proof; status isn't sensitive, it's the operator's own request).

`tool_identity.py`: frozen UUIDs + `ToolIdentity` entries (categories above). `constants.py`: `ADOPTION_PENDING`, `ADOPTION_NOT_FOUND`, `ADOPTION_ALREADY_PROVISIONED`.

Tests: inbound proof gate, owner-consent gate on approve/reject/list, approve→`_provision_operator` called once, reject path, idempotent re-request.

### Unit 4 — Operator-side tools (`register_standard_tools` in `runtime.py`)
- `request_adoption(authority_npub, proof, note="")` — `restricted` (operator-proof gated on the operator's own npub). Steps: resolve the Authority's MCP URL from the registry (`resolve_authority_service` / Oracle `lookup_member(authority_npub)`); mint inline adoption proof with the operator nsec bound to `authority_npub`; `Client(authority_url).call_tool("<authority_slug>_receive_adoption_request", {...})`; return `{status:"pending", authority_npub}`. Defensive (timeout, structured error if the Authority is unreachable).
- `adoption_status(authority_npub, proof)` — `free`. MCP-to-MCP `get_adoption_status` poll; returns pending/approved/provisioned/rejected.

FastMCP `Client` MCP-to-MCP is the same pattern proven in dpyc-oracle `list_services`. Tests mock the Authority `Client`.

### Unit 5 — Phase 2: async-DM approval (additive)
`check_adoption_replies(authority_proof)` — `restricted`. Drains `exchange.receive(sender_npub=runtime.operator_npub(), service="operator_adoption")`; for each reply whose poison matches a pending row's `poison_hash`, finalize via `_provision_operator` + `mark provisioned`. Lets the owner approve by replying to the Nostr DM without the Studio. Ship after Phase 1 is proven.

### Unit 6 — Release + propagation + client
- SDK release (`/release`): minor bump (additive feature), CHANGELOG.
- Fleet pin bump → every operator gains `request_adoption`/`adoption_status`; every Authority gains the adoption tools. (cypher-mcp inherits it for free.)
- **Studio (Swift, separate repo):** a "Pending Adoptions" queue (`list_adoption_requests`) with Approve/Reject actions that mint `authority_proof` from the stored Authority nsec → `approve_adoption`/`reject_adoption`. Plus surface the `operator_adoption` notification DM.

---

## Lifecycle (unchanged — confirmed)
Operator stays `not_registered` (session_status, runtime.py:2641) until `approve_adoption` provisions the Neon tenant; then bootstrap DM → `warming_up` → `ready`. No lifecycle code changes.

## Sequencing / dependencies
Unit 0 → 1 (refactor) is the foundation and ships first. Unit 2 (store) precedes Unit 3 (Authority tools). Unit 3 precedes Unit 4 (operator side needs the Authority endpoint to call). Unit 5 is optional/after. Unit 6 last.

## Open sub-details to finalize during Unit 3
- Exact canonical message for the inline adoption proof (bind operator_npub + authority_npub + service_url; reuse `identity_proof` verify). Keep it a self-contained kind-27235 event, no relay round-trip.
- Whether `reject_adoption` notifies the operator by DM or only sets status (operator's `adoption_status` poll would see `rejected` either way).
