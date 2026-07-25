# Changelog

All notable changes to this project will be documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## 0.71.2 — 2026-07-25

### Fixed — proactive Neon compute usage reads real numbers (and never blanks)

The consumption-history probe queried Neon at **monthly** granularity, but a
monthly bucket for an *in-progress* month can come back empty — so every
project's compute read `null` / `status: "unknown"` even with a valid key and
`configured: true`. Query at **daily** granularity now (buckets exist from the
1st onward, summed to month-to-date).

And the probe no longer fails mutely: when compute is still unavailable — an HTTP
error, a scope/plan limitation, or simply no rows in range — the reason rides
back on `neon_api.usage_note` (status + Neon's own message, credential-free), so
`unknown` always says *why* instead of showing a blank. Tests cover the daily
query, the HTTP-failure note, and the empty-rows note.

## 0.71.1 — 2026-07-25

### Fixed — renamed-with-frozen-UUID tools resolve their real wire name in proofs

`network_persistence_health` (0.71.0 rename of `network_books_health`) kept its
original frozen `tool_id` so pricing and proofs stay stable — but `runtime_name`
re-hashed the *new* capability string via `capability_uuid()`, producing a UUID
that no longer matched the frozen id. `mcp_name_for` missed and returned the raw
UUID as the "wire name", so the owner-consent gate expected a `u` tag of
`df1368fa-…` instead of `authority_network_persistence_health`. Every Authority
proof mismatched — masked by `_require_authority_consent` as a blanket
`authority_consent_required`, so the Pricing Studio Persistence Status pane could
never satisfy consent.

`runtime_name` now resolves a capability through its **registered identity**
(the authoritative `tool_id`), falling back to the name hash only for a
brand-new, unregistered capability. This fixes the rename and any future one —
and also repairs `runtime_name` for dynamic tools (whose ids are seeded from a
`dyn:`-prefixed string, which the old hash path likewise missed). `capability_uuid`
stays a seed-only helper, as its docstring already promised. Regression test added.

## 0.71.0 — 2026-07-25

### Changed — "Books" is now "Persistence" throughout the SDK surface

The "Books" metaphor is retired. It always read as *more* than the ledgers, and
the persistence layer is not forever Neon — so the symbols now name the concern,
not the implementation. Clean break, no compat alias (per doctrine):

- `network_books_health` → **`network_persistence_health`** (the Authority tool;
  UUID frozen). Its `own_books` field → **`own_store`**; `OwnBooksStatus` →
  `OwnStoreStatus`; the `neon_books_alert` channel → `neon_persistence_alert`.
- Pricing Studio ships the matching FE rename (`network_persistence_health`
  lookup + `own_store` decode) — **Authorities must redeploy on 0.71.0** for the
  live call to resolve against the new build.

### Fixed — proactive Neon watch surfaces real compute usage, and real errors

- `NeonAdminClient.project_usage` now reads per-project `compute_time_seconds`
  from Neon's `consumption_history/projects` endpoint (the `/projects` list omits
  it), so `used_pct`/`status` are real instead of always `"unknown"`. Best-effort
  and tolerant — a consumption-history failure degrades usage to `None`, never an
  outage.
- A failing `/projects` call now surfaces Neon's own message, and front-loads the
  most common cause — an org-scoped key missing `neon_org_id` — as an actionable
  hint instead of a bare status line.

### Changed — ruff 0.16.0 clean, fleet-wide

Cleaned every ruff 0.16.0 violation across the SDK: targeted inline `# noqa`
where a lint is intentional (best-effort probes, deliberate broad catches),
genuine fixes everywhere else. No global ignore, no pinned-back linter — all code
is our responsibility.

## 0.70.0 — 2026-07-24

### Added — `NEON_API_KEY` is a Secure-Courier–delivered Authority secret, not env

The Neon control-plane key is not needed to bootstrap an Authority, so it is no
longer an environment secret. `neon_api_key` (sensitive) and `neon_org_id` are
now optional fields on the Authority's operator credential template — delivered
via Secure Courier and read from the vault by the proactive compute-quota watch
(`network_books_health`). They surface in onboarding `optionalMissing`, so the
courier DM requests them and `receive_credentials` vaults them; the FE hint now
reads "Deliver … via Secure Courier."

### Removed

`neon_api_key` / `neon_org_id` env settings on the Authority — deliver them via
Secure Courier instead (Pricing Studio → Authority → Persistence Status → Deliver).

## 0.69.1 — 2026-07-22

### Fixed — restricted tools deny non-operators with `restricted`, not a misleading proof error

`debit_or_deny` now checks `caller == operator` **before** verifying the proof
for `restricted` (operator-only) tools. Previously the proof was verified first,
against the *operator's* npub — so a non-operator caller (whose own proof was
perfectly valid) failed that check and got `proof_refresh_needed` ("your
npub-proof cache is no longer valid, re-sign-in"). That was misleading, and on a
non-best-effort call it read as an auth bounce that logged the caller out. The
access check is an npub comparison and the operator npub is public, so ordering
it first leaks nothing; a non-operator now gets a clear `restricted` ("This tool
is restricted to the operator.") and the operator path is unchanged.

## 0.69.0 — 2026-07-22

### Changed — courier freshness window 15 min → 1 hour (human-paced replies)

The Secure Courier's default freshness window (`_DEFAULT_FRESHNESS`) — how long
the operator remembers a pending channel after sending the DM — goes from 15
minutes to 1 hour. It governs both npub-proof challenges and credential handoffs,
which are human-in-the-loop: a person has to notice the DM, match a code, and
sign a reply, often not at a keyboard the instant it arrives. 15 minutes forced
a scramble, and against a background worker on a ~30-minute tick it structurally
missed the window (the tick landed after the challenge had already lapsed),
leaving the proof perpetually pending. One hour lets the parties wait for
ordinary human responsiveness. The reply itself remains un-timestamp-gated
(`tools/proof.py`) — this only bounds how long a challenge stays claimable.
Consumers that pass an explicit `freshness_window` are unaffected.

## 0.68.0 — 2026-07-21

### Added — proof requests carry a Device-Grant `verify_at` (where the code was shown)

`request_npub_proof` gains an optional `verify_at`: a free-form statement of
*where the initiating agent already showed the recipient this one-time code* —
a URL, or "your Claude.ai conversation", "the Grok session". It is OAuth 2.0
Device Authorization Grant (RFC 8628) applied to the Nostr proof flow: the
`dpop_token` is the `user_code`; `verify_at` is the (generalized)
`verification_uri`. The recipient approves only if the code in the DM matches
the one displayed there — so an unsolicited request, whose code they've never
seen anywhere, is refused (no anchor → no trust). Signed into the attestation
as a `verify_at` tag (tamper-evident) and written as an IMPORTANT line in the DM
next to the code. Omitted when not given.

## 0.67.1 — 2026-07-21

### Fixed — request `origin` is emitted only when something is actually observed

0.67.0 could stamp a misleading origin: on a platform that hides the client IP
(Horizon hands the app localhost), the harvest fell back to `127.0.0.1` and
emitted `127.0.0.0/24 · <user-agent>` — a loopback address that "could be
anywhere", plus a self-reported UA. Now: private/loopback/link-local addresses
are discarded (they are the internal proxy, not the client); the real-client-IP
header set is broadened (`true-client-ip`, `fly-client-ip`, `fastly-client-ip`,
… + more geo headers); and the `origin` tag is emitted ONLY when an *observed*
signal survives (a public IP or an edge geo). A self-reported User-Agent alone
yields no tag — we omit rather than assert a "trust me" origin the operator
never observed.

## 0.67.0 — 2026-07-21

### Added — proof requests carry an operator-observed `origin` so the human can judge an unsolicited ask

`request_npub_proof` is free and unauthenticated — anyone can trigger the operator
to send a genuinely operator-signed proof-request DM to any npub. The attestation
proves *who signed it*, but said nothing about *who triggered it, from where*.

The operator now best-effort harvests the triggering client's provenance
server-side — geo + a coarsened source IP (last octet dropped) + the claimed
client agent — from the transport headers (`CF-IPCountry`, `CF-Connecting-IP` /
`X-Forwarded-For`, `User-Agent`), and signs it into the attestation as an
`origin` tag (tamper-evident; observed, never client self-report). It is also
shown as a `Requested from:` line in the DM. Best-effort: when the transport
exposes nothing (no HTTP context / no edge geo), the tag is simply omitted.

This is display-only by design — it gives a human the missing datum to *decide*
whether to accept a request, not a gate that blocks it.

## 0.66.1 — 2026-07-21

### Changed — the proof-request `reason` is carried once (signed tag only), keeping the payload lean

0.66.0 spliced the `reason` into three places — the greeting, the signed
attestation tag, and a `Reason:` provenance line. That bloats the DM and
duplicates free-text into the parse-sensitive body. The reason now rides in
exactly one place: the tamper-evident `reason` tag on the attestation (where a
recipient reads it trustworthily). The greeting and provenance block are back to
their lean, fixed form. No behaviour change for a recipient that reads the tag;
the wire payload is smaller and single-sourced.

## 0.66.0 — 2026-07-21

### Added — proof / credential requests can carry a signed, human-readable purpose

`request_npub_proof` (and the underlying `SecureCourier.open_channel` →
`create_provenance_attestation`) gain an optional `reason`: a plain-language purpose the
Operator states for the request — *"I'm working on your request XYZ and need the Operator
to do ABC for you."* It is signed into the `kind:27235` provenance attestation as a `reason`
tag (tamper-evident, bound to the signer) **and** shown in the DM body's Message Provenance
block. This gives a recipient the one fact that makes a stranger's ask judgeable — especially
in the unknown-signer case, where a claimed identity alone is not enough to decide. Omitting
`reason` leaves the attestation and DM exactly as before (no tag, no line).

## 0.65.1 — 2026-07-19

### Fixed — a rejected inline npub proof now says *why*, and the dpop_token shape is documented

A self-minted `kind:27235` proof that was refused returned one opaque `proof_invalid`
"Invalid identity proof." — the real reason was computed and discarded at DEBUG, forcing
key-holding agents into guess-and-check. `require_proof` now surfaces a machine-readable
`reason` on the denial (`malformed_json` for a base64/NIP-98-wrapped token, `tool_mismatch`
— with `expected_u` naming the tool — when the `u` tag held the endpoint URL instead of the
tool name, plus `signature_invalid` / `npub_mismatch` / `wrong_kind` / `expired` / `replayed`).
The accept/reject logic is **byte-for-byte unchanged** — `verify_proof(...) -> bool` is now a
thin wrapper over the reason-returning `_verify_proof_reason`; only the explanation is added,
never what is accepted. Every `dpop_token` docstring now states the exact shape: raw JSON (not
base64, not NIP-98), a `u` tag holding this tool's exact name from `tools/list` (not the
endpoint URL), `content:""`, `created_at` within the freshness window, and a recommended
`nonce`. Closes tollbooth-dpyc#137 (reported by Scout).

## 0.65.0 — 2026-07-19

### Added — a proof-request DM now proves who is asking, not just who is answering

The npub-proof handshake authenticated the patron to the Operator but never the Operator to the
patron. In the self-addressed case (an Operator proving its own npub) relays drop
self-addressed DMs, so the request is *delivered* from a throwaway ephemeral key — and the human
was shown that unfamiliar key as the "Operator", with no way to tell a legitimate front-end
worker from an impostor who guessed their npub and timed a request well. The only safe move was
to let it expire, so the honest flow failed closed while a well-timed attack would have
succeeded.

The seal cannot carry the fix — relays require the distinct delivery key — so provenance now
rides *inside* the encrypted DM body, attested by the Operator rather than asserted by the
requester.

- **Operator provenance attestation.** `identity_proof.create_provenance_attestation` signs a
  kind-27235 event with the Operator's **registered** identity key (the asset an impostor does
  not hold) binding the **delivery key**, the **subject** npub, the **service**, and the
  one-time **challenge** (`dpop_token`). `open_channel` embeds it in every request DM.
  `verify_provenance_attestation` checks the signature and that each bound fact matches the DM
  the recipient actually saw — so an attestation cannot be lifted onto a DM sent by a different
  key, nor replayed against another exchange. It performs no registry I/O: it surfaces the
  recovered signer pubkey for the recipient to resolve, keeping the fail-closed trust decision
  (registered+certified → green; registered but novel → amber; unresolvable → red) at the edge.
- **Honest attribution in the DM body.** The `Operator:` line now always shows the Operator's
  **registered** npub, never the ephemeral delivery key; the delivery key is shown on its own
  labeled line that says, in plain words, that its authority comes from the attestation and not
  from the key itself.
- **Additive and backward compatible.** If signing is unavailable the DM omits the attestation
  block and renders amber (never green) at the client — envelope-absent is never trusted. No
  change to the NIP-59 gift-wrap, the reply path, or any consumer; `thebrain-mcp`'s live DM flow
  is unaffected.

Client-side three-state rendering (Pricing Studio) and steering agent-held identities to inline
self-attestation instead of the DM round-trip are tracked as follow-ups.

## 0.64.3 — 2026-07-19

### Added — Neon books health: a 402 is no longer a "warming up" lie, and the Authority learns first

The DPYC economy's accounting books are Neon, and the books are the Authority's charge. When
an operator's Neon project exhausted its compute quota, Neon answered HTTP 402 and the operator
went dark — but the runtime misclassified that 402 as `warming_up` and told patrons to *"retry
shortly"*, so a real outage masqueraded as a cold start and the Authority learned last, from a
patron complaint. This release fixes the classification and routes the signal to the party
responsible for the books.

- **Distinct classification (A).** `NeonQueryError` now carries the HTTP `status`; a 402 is
  classified as the new `PERSISTENCE_QUOTA_EXCEEDED` error code and `quota_exceeded` lifecycle
  state — separate from `warming_up` (transient cold start) and `persistence_misconfigured`
  (SQL/permission). The paid-tool gate and `session_status` now return an honest, non-transient
  message ("retrying will NOT help") instead of inviting a retry, and the pricing resolver
  stops burning its cold-start retry budget on a condition only a human can clear
  (`PricingResolver.last_error_quota`).
- **Operator → Authority alert (B).** On catching a Neon 402, the operator's runtime
  fire-and-forgets a rate-limited alert to its Authority via the new
  `authority_receive_neon_402_alert` tool (`AuthorityCertifier.report_neon_quota_exceeded`).
  It works while the operator's own books are dark because it uses only the operator nsec and
  the Authority's registry-resolved endpoint — neither touches Neon. The Authority records a
  durable latest-state row (`neon_alert_store`) and DMs its owner.
- **Proactive watch (C).** New restricted `network_books_health` Authority tool reports, from
  most to least proactive: per-Neon-project compute-quota posture (hours used, %, reset date,
  ok/warning/critical/exhausted) via a new `NeonAdminClient` (needs an org-scoped
  `NEON_API_KEY`; `configured=false` otherwise); reactive self-detection of the Authority's own
  books; and the operator-reported alerts. `list_neon_alerts` exposes the reactive queue.
  Pricing Studio renders these as a Network Books Health panel.

## 0.64.2 — 2026-07-18

### Fixed — delivered operator secrets are now visible in onboarding status

- `onboarding_status` enumerated only the operator's *declared* credential template, so
  operator secrets delivered into the auto-included optional slots (the field-report
  `github_repo` / `github_token`, added to the courier template in 0.64.1) vaulted correctly
  but were invisible everywhere — including Pricing Studio, which reads this surface. It now
  enumerates the *effective* template (declared + auto-included) via the new pure
  `classify_operator_secrets`, so a delivered operator secret appears under `configured`.
  Auto-included fields that were never delivered are omitted entirely (not listed as
  `optional_missing`), so operators who don't use field reports aren't nagged, and onboarding
  readiness is unchanged. Resolves the first live Scout field report (tollbooth-sample#64 →
  tollbooth-dpyc#132): an operator can now confirm which field-report secrets are set and
  which repo reports route to. (Rotate = re-courier the field; revoke = `forget_credentials`;
  patron-scope listing already exists as `get_patron_credential_fields`. Per-field delivery
  timestamps remain a future enhancement.)

## 0.64.1 — 2026-07-18

### Changed — field reports need no per-operator template edit

- The runtime now auto-includes the optional `github_repo` / `github_token` field-report
  secrets in every operator's courier-facing credential template
  (`OperatorRuntime._courier_operator_template`). Because the Secure Courier silently drops
  fields not present in the template, `report_issue` previously required each operator to
  spread `ISSUE_REPORTING_CREDENTIAL_FIELDS` into its own template before the secrets could be
  delivered. Now an operator enables field reports fleet-wide by simply couriering the two
  secrets — no code change. The fields stay optional, so onboarding readiness (which reads the
  declared template) and operators that never use field reports are unaffected. Provisioning
  across the fleet is therefore just the Renovate SDK pin bump plus the per-operator Secure
  Courier delivery.

## 0.64.0 — 2026-07-18

### Added — `report_issue`: patron-filed field reports as GitHub issues

- Every operator now exposes a standard `<slug>_report_issue(npub, dpop_token, title, body,
  tool_name)` tool. A proven patron who finds a tool's metadata or response wrong or confusing
  files a field report as a GitHub issue **on that operator's own repo** — the defect lands where
  the tool lives. The **author of record is the caller's npub** (for an assistant reporting under
  a "Scout" identity, that npub is Scout), stamped into the issue body with an authoritative
  `<!-- dpyc-field-report reporter="npub1..." tool="..." -->` marker that the Service Desk keys on.
  Any marker token smuggled into the report text is neutralized so provenance cannot be spoofed.
- **Proof-gated and metered**: no npub / no proof → no issue, and the tool is priced (seeds at a
  1-sat floor via its `ToolIdentity` pricing hint) so a free write to an issue tracker cannot be
  abused — "balance is the cap". A not-configured operator, a validation failure, or a GitHub
  rejection all return a clean situation and **refund** the fee (nothing was filed).
- Operators opt in by spreading `ISSUE_REPORTING_CREDENTIAL_FIELDS` (`github_repo` +
  `github_token`, both optional) into their `operator_credential_template` and delivering the two
  via Secure Courier — same path as every other operator secret. The token needs only
  `issues:write` on the repo, never code scope. New `github_issues_client.py` (httpx, WASI-safe via
  the wasmcp seam) and `tools/report_issue.py` carry the implementation.

## 0.63.4 — 2026-07-18

### Fixed — operator display name now reaches the community roster

- `register_operator` never carried the operator's chosen name: `_provision_operator`
  hardcoded `display_name=npub[:16] + "..."`, so every Authority-mediated registration
  landed in `members/operators/*.json` named by a truncated npub (both `display_name` and
  `services[].name`). The `register_operator` tool now accepts an optional `display_name`
  and threads it through `_provision_operator` → `_register_operator_via_oracle`, falling
  back to the truncated npub only when no name is supplied. The deferred `approve_adoption`
  path forwards `display_name` too (empty until the adoption store captures one).

## 0.63.3 — 2026-07-16

### Added — npub-proof challenge DM now stamps the request time

- The npub-ownership proof-request DM omitted *when* the challenge was raised. `request_npub_proof_tool`
  now prepends a single terse `Requested: <YYYY-MM-DD HH:MM UTC>` line to the greeting the courier
  places directly above the `@@@` fields, so the timestamp lands in the preamble without touching the
  shared `open_channel` path used by every other credential exchange. Proof DMs stay succinct
  notifications, not documents. Closes #120.

## 0.63.2 — 2026-07-15

### Fixed — OAuth2 collector retrieval no longer drops plain-JSON responses

- `retrieve_code_from_collector` only parsed Server-Sent-Events framing (lines beginning
  `data: `). When the collector answered with a plain `Content-Type: application/json`
  body — which the request's own `Accept: application/json, text/event-stream` header
  permits — no line matched and the function fell through to `return None`.
  `check_oauth_status` reads that `None` as "code not yet available" and reports `pending`,
  so a completed OAuth flow appeared stuck forever.
- The retrieval now accepts **either** framing: it decodes SSE `data:` frames as before and
  falls back to parsing the whole body as a single JSON object. This removes the dependency
  on the collector host's transport-framing default, so a future FastMCP/host change that
  flips SSE↔JSON cannot re-break the path. Reuses the SDK's `decrypt_collector_code`
  (no new crypto). Closes the OAuth `pending` regression seen by collector consumers
  (e.g. schwab-mcp) after the 2026-07-09 collector redeploy. (#116, #118)

## 0.63.1 — 2026-07-14

### Fixed — `create_proof` mints a unique event per call (no more replay-collision)

- `create_proof` produced whole-second, fixed-body kind-27235 events, so two proofs for
  the **same tool** minted within the **same wall-clock second** were byte-identical →
  same event id → the verifier's replay guard rejected the second as already-seen. Rapid
  same-tool callers (a seed loop, an agent keyring) hit spurious "Invalid identity proof".
- Each proof now carries a per-call `nonce` tag (`secrets.token_hex(16)`), signed but
  otherwise inert — `verify_proof` reads only the `u` tag — so distinct mints can never
  collide by construction. Replay protection is unaffected (re-presenting the same token
  is still rejected).

## 0.63.0 — 2026-07-14

### Added — `PatronSigner`: the single home for patron-side proof signing

- `tollbooth.patron_signer.PatronSigner` holds a patron's `(npub, nsec)` and authenticates
  its outgoing operator calls — `proof(tool)` mints a fresh, tool-bound kind-27235 proof;
  `authenticate(tool, args)` returns the ready payload (npub + fresh `dpop_token`). It is
  now the one place server-side Python signs on a patron's behalf (the peer of iOS's
  Keychain-backed native signer). Empty `nsec` yields an empty proof (parity with prior behaviour).
- **`AuthorityCertifier` refactored onto `PatronSigner`** — its hand-rolled `_make_proof`
  and inline payload assembly are gone; it holds a `PatronSigner` and calls `authenticate()`.
  Behaviour preserved exactly (including the historical `check_balance` proof-name, flagged
  for a separate follow-up). The agent keyring uses the same signer.

### Added — `tollbooth.agent_keyring`: an authenticated passthrough to a DPYC operator

- A reusable **agent keyring** — the peer of `AuthorityClient`. A DPYC agent (a patron
  holding its own nsec) fronts an upstream paid operator through this FastMCP proxy; on
  every forwarded call it injects the agent's npub and a **freshly-signed, in-memory**
  kind-27235 proof bound to the tool being called (`create_proof`). Nothing is stored,
  and nothing new is *granted* per call — the standing grant is possession of the nsec
  plus a funded balance; the proof is only the mechanical demonstration of possession.
- Run as a local stdio MCP server (e.g. in a CI agent's `--mcp-config`) so the agent
  calls the operator's verbs plainly while the nsec stays in the keyring process, out of
  the agent's own reasoning context:
  `DPYC_KEYRING_UPSTREAM=... DPYC_KEYRING_NPUB=... DPYC_KEYRING_NSEC=... python -m tollbooth.agent_keyring`.
- New optional extra `keyring` (pulls FastMCP; imported lazily). `signed_arguments()` is a
  pure, FastMCP-free helper (the injection logic) and is unit-tested independently.

## 0.62.4 — 2026-07-12

### Fixed — a missing `[prefect]` extra degrades gracefully instead of poisoning the container

- **`_ensure_async_executor` no longer crashes the first drill when the long-runner creds are vaulted but the `prefect` runtime is absent.** Constructing `PrefectClosureExecutor` imports `prefect` (the optional `[prefect]` extra); an operator who couriered `prefect_api_url`/`prefect_api_key` but pinned `tollbooth-dpyc[nostr]` (no `prefect`) hit an `ImportError` raised **after** `_async_executor_resolved` was already set — so the first `start_async_job` on each container errored and every later job short-circuited to in-process, silently. Observed as optionality's `deal_scenario` timing out `job_timed_out` with `recovered:true` while **no** flow run ever reached Prefect. The construction is now wrapped: a missing extra logs a loud, actionable warning (`add the [prefect] extra`), records the reason, and falls back to in-process — never propagates.
- **`service_status.durable_jobs` now reports `detached_executor_resolved` and `detached_executor_error`.** `detached_executor_active: false` was ambiguous — lazily-unprobed vs. probed-and-failed look identical. The new fields make a misconfigured operator (creds present, extra missing, empty creds) diagnosable without reading container logs.

## 0.62.3 — 2026-07-11

### Fixed — a cold-vault hiccup no longer pins a container to in-process execution

- **`_ensure_async_executor` no longer caches a *transient* creds-load failure.** It set `_async_executor_resolved = True` **before** loading the long-runner creds, so if the vault threw on a container's first job — a cold Neon on warm-up, exactly when the first request lands — the probe bailed and that container was pinned to **in-process execution for its whole life**, even though the creds were present and would load a moment later. Every deal/judge/tip on it then ran in-process and risked the `max_runtime` hard-cap (observed as a live `deal_scenario` timing out `job_timed_out` despite the detached executor being active elsewhere in the fleet). The resolution is now cached only on a **definitive** answer (creds loaded, present or absent); a load exception leaves the probe unresolved so the next job retries. A genuine "no creds" answer is still cached (no wasteful re-probing).

## 0.62.2 — 2026-07-11

### Changed — durable async jobs can now carry per-job state

- **`register_job_spec` shape callback now receives the job's params: `shape_result(raw, params)`.** The detached (closure) path runs only the sealed `http_request` in Prefect, so `shape_result` — which settles the completed run back in the MCP — previously had no access to the job's identifying arguments (npub, entry_id, …). A stateful operator job (open a journal entry, record an evaluation) could not perform its param-dependent side effects on the detached path. `params` is the same kwargs `build_closure` received, threaded through so the settle step is symmetric with the in-process runner. **Breaking:** existing `shape_result(raw)` callbacks must accept a second `params` argument (ignore it if stateless).
- **A `build_closure` that raises `AsyncJobSituation` now settles terminally.** `start_async_job` treats a curated situation from the build step (a pre-flight rejection — a not-found entry, an unfunded-provider probe) as a terminal, refundable outcome — persist the structured situation, refund the fare, return the situation response — instead of routing it through the generic "dispatch failed" in-process fallback. Symmetric with a runner raising one; lets a closure job reject cheaply before dispatch.

## 0.62.1 — 2026-07-09

### Security — hardening batch (audit-driven)

- **`check_payment` / `restore_credits` now verify invoice ownership before crediting.** A settled `invoice_id` is not a bearer token: crediting confirms the invoice's `metadata.user_id` matches the account being credited (refuses `invoice_owner_mismatch`). Previously any caller who learned another patron's settled invoice_id — they surface in tool results, DMs, and logs — could claim it, and the per-ledger `credited_invoices` idempotency guarded only the victim's ledger, so it minted free credits for the claimer (cross-account double-issuance). The `check_payment` settlement path is also rewritten onto the 0.62.0 CAS `LedgerCache.mutate()` write-through, so the credit is idempotent and conflict-safe against fresh state rather than a `get_fresh`+`flush_user` dance.
- **Credential vault upgraded from unauthenticated NIP-04 AES-256-CBC to authenticated AES-256-GCM.** API keys, OAuth tokens, and the ephemeral agent nsec (`agent_nsec_hex`) were self-encrypted with NIP-04 CBC — no MAC, so ciphertext was malleable and not tamper-evident. They now use the same `VaultCipher` (GCM) the ledger uses. Legacy blobs (identified by the NIP-04 `?iv=` marker) still decrypt and are re-encrypted with GCM on their next write, aging the unauthenticated population out.
- **Self-provisioning actors (Authorities) now encrypt their ledger at rest.** The `vault_source="env"` path constructed `NeonVault` with no encryption key, storing financial balances in a plaintext Postgres column readable by Neon/DB admins. It now encrypts with the actor's own nsec. A keyless `NeonVault` also logs a loud warning (previously silent), and a legacy-plaintext read under an encrypting cipher warns so the migration is observable.
- **Audit publisher never broadcasts cleartext financials.** Ledger-update events to non-npub targets previously fell through to a plaintext publish (balance/deposited/consumed) on public relays; any target that can't be NIP-44-encrypted is now skipped rather than leaked.
- **Bounded untrusted string arguments before parsing.** `verify_proof` rejects `proof_json` over 64 KiB and the Secure Courier `receive` tool bounds `dpop_token`/`credential_card` — a 10 MB payload can no longer be fully parsed (mild DoS from adversarial AI tool input).
- **Defense-in-depth SQL identifier guard** on `transfer_schema_ownership` (schema names are already SHA-256-derived and validated on role creation).
- Removed orphaned `acl_verify` / `acl_store` / `tools.acl` bytecode with no source (import-shadowing hazard); corrected the NIP-04 shared-secret docstring (raw x-coordinate, no hash step).

## 0.62.0 — 2026-07-09

### Fixed — balance ledger is now write-through and conflict-safe across replicas

- **`NeonVault.store_ledger` no longer clobbers on a version conflict.** The `balances` table's optimistic-concurrency guard used to *detect* a conflict (guarded UPDATE returns 0 rows) and then **fall through to an unconditional UPSERT that overwrote the newer row** — silently losing a balance mutation whenever a horizontally-scaled fleet wrote the same ledger (e.g. a +1000 top-up on one replica vanishing when another replica flushed a stale copy). It now raises `LedgerVersionConflict`; the caller re-fetches and re-applies. The no-cached-version path inserts with `ON CONFLICT DO NOTHING` and likewise refuses to blind-overwrite.
- **Ledger mutations are write-through with read-modify-write retry.** New `LedgerCache.mutate(user_id, fn)` fetches the CURRENT ledger from the definitive store, applies `fn`, and CAS-writes it; on `LedgerVersionConflict` it re-fetches and re-applies (bounded retries) so concurrent replicas can't clobber or double-spend. `debit()` and the new `credit()` route through it; the runtime billing / chain-credit / rollback paths and the credit tools (`check_payment`, `restore_credits`, `reconcile_pending_invoices`, `purchase`) now read fresh before writing. No balance change is deferred to a lossy write-behind flush.
- **Never zero a balance on a cold read.** `mutate` raises `LedgerUnavailableError` instead of applying a mutation to an empty fallback ledger when the store is unreadable; a debit then surfaces "service warming up — retry, no fare charged" rather than a false "insufficient balance".
- **`check_balance` / `account_statement` read fresh**, so the displayed balance is authoritative (fixes the chip-vs-paid-path disagreement).
- New exceptions in `tollbooth.vault_backend`: `LedgerVersionConflict`, `LedgerUnavailableError`, `LedgerWriteError`.

Tradeoff: ~1 extra Neon round-trip per paid call — the deliberate cost of making Neon the definitive store so the MCP business logic can scale horizontally.

## 0.61.0 — 2026-07-09

### Fixed — `max_runtime_seconds` is now a hard cap on a claim-check attempt

- **`OperatorRuntime._run_job` wraps the runner in `asyncio.wait_for(..., timeout=max_runtime_seconds)`.** Previously the runner was `await`ed unbounded, so `max_runtime_seconds` was only the stale-reclaim threshold — a runner whose own I/O timeout was missing or too generous (e.g. an LLM SDK's 10-minute default) left the job row `running` well past its declared budget, and a polling frontend's ceiling expired before any terminal state was written. Now an attempt that outruns its budget is cancelled at its next `await` point.
- **A budget timeout is terminal and refundable, not retried.** On timeout the job is failed with a curated `AsyncJobSituation` (`error_code="job_timed_out"`, transient, "This request took too long… No fare was charged.") and the debit is rolled back — retrying would just burn a second full budget the frontend has already stopped waiting for. Writing a terminal `error` also forecloses the stale-reclaim race (the row never lingers `running` for a watchdog to re-kick).
- **Cancellation is cooperative** — a runner parked on I/O (every DPYC runner: an LLM/HTTP round-trip on `await`) is interrupted cleanly; a runner stuck in non-awaiting CPU work would not be. Operators should still bound their own upstream calls (a shorter per-call timeout yields a faster, more specific situation); this cap is the backstop that guarantees no runner outlives its declared budget.

## 0.60.0 — 2026-07-08

### Changed — BREAKING: Nostr relays come from one source of truth (`dpyc-community/relays.json`)

- **The wheel no longer carries any hardcoded relay list.** Four drifting sets — `nostr_diagnostics.DEFAULT_RELAY` / `FALLBACK_RELAY_POOL`, `bootstrap_relay.BOOTSTRAP_RELAYS`, `nostr_profile.PROFILE_RELAYS` (and the Pricing Studio app's own list) — are replaced by a single curated set fetched over GitHub raw HTTPS from `https://raw.githubusercontent.com/lonniev/dpyc-community/main/relays.json`. Edit that file and the whole federation follows on its next cache refresh; no code release needed to tune the relay set.
- **New `relay_registry.RelayRegistry` / `get_relays()`** — synchronous `httpx.Client` fetch, 3-day in-process TTL cache, primary-first ordering. **Fail-closed** on a cold cache with an unreachable registry (`RelayRegistryError`); **stale-if-error** otherwise (serves the last-known-good set with a short backoff so an outage can't cause a fetch on every call). Sync by design because every relay consumer in the wheel (courier, bootstrap, profile, audit) is synchronous.
- **`resolve_relays()` no longer takes a `configured` argument.** The set is governed solely by the registry; it probes liveness and returns the live relays in registry (primary-first) order, or the full set unprobed if none respond.
- **Removed the `relays=` constructor parameter from `OperatorRuntime`** and the Authority-only **`TOLLBOOTH_NOSTR_RELAYS`** env override. Relay choice is no longer per-server configuration. (No consumer passed either — verified across all repos.)
- **Curated set:** `relay.primal.net` (primary), `relay.damus.io`, `nos.lol`, `relay.nostr.band` — chosen for being both reliable **and** open for writes (paywalled/metadata-only relays like `nostr.wine` are excluded because they would reject arbitrary patron courier DMs). See `dpyc-community/RELAYS.md` for the curation criteria and usage protocol.
- **Spin/WASI note:** Spin operators (e.g. tollbooth-fermyon) fetch `relays.json` through the `wasi:http` seam — verify `raw.githubusercontent.com` is in `allowed_outbound_hosts` when bumping the pin.

## 0.59.1 — 2026-07-03

### Fixed — restore operator→Authority M2M calls broken by the 0.57.0 `dpop_token` rename

- **`authority_client` and `request_adoption` now send the identity token under `dpop_token`, not the pre-0.57.0 `proof` kwarg.** The 0.57.0 rename (`proof` / `poison` / `proof_token` → one `dpop_token`) renamed every tool *parameter* but missed three outbound *calls*: `certify_credits` and `check_balance` (`authority_client.py`) and `receive_adoption_request` (`runtime.py`). Each still passed `proof=`, which the Authority's pydantic-typed tools reject with `unexpected keyword argument: proof` — silently breaking **every patron credit purchase** (which drives `certify_credits` at the operator's upstream Authority) and cross-Authority adoption. Latent since 0.57.0 because these fire only on real M2M certification / adoption, not on ordinary tool calls. The signed-token *value* is unchanged; only the kwarg name was wrong.
- **Regression guard:** `test_authority_client` now asserts the certify call sends `dpop_token` and never the old `proof` key.

## 0.59.0 — 2026-06-30

### Added — optional author-declared time budget for async jobs (`expected_seconds`)

- **`start_async_job` now accepts an optional `expected_seconds`** — a caller-declared *prediction* of how long the work takes (distinct from `max_runtime_seconds`, which is a safety ceiling). When a consumer passes it (e.g. a dynamic-block author who sized and pays ad valorem for their block), the advised poll cadence trusts it: the **first wait is ~75% of the budget**, then each subsequent poll waits ~75% of what remains (geometric tightening), capped at a 300s single-hop ceiling and floored at 5s near/after the budget. Leave it unset (`0`) and the existing steady-ceiling countdown is unchanged — so other claim-check consumers are unaffected.
- **Schema:** new `expected_seconds INTEGER NOT NULL DEFAULT 0` column on `async_jobs`, with an idempotent `ADD COLUMN IF NOT EXISTS` retrofit for operators provisioned earlier. Persisted on create, read back on the row, and consumed by `poll_backoff_seconds(elapsed, max_runtime, expected_seconds)`.
- **Why two regimes:** for a job whose duration is genuinely known, polling through the middle is wasted round-trips; sleeping most of the budget up front and tightening at the end is both cheaper and lower-latency. For a job where the number is only a ceiling (the default), a long first wait would be wrong — so the budget curve is strictly opt-in.

## 0.58.0 — 2026-06-30

### Changed — claim-check polling cadence counts down to the deadline instead of up from the start

- **`fetch_async_job` / `start_async_job` now advise an adaptive `poll_after_seconds`** instead of a constant `3`. A fixed 3s tick told an agent to poll a 400s job ~130 times; naive exponential backoff is worse — it advises the *longest* wait right when the result is most imminent, since the longer a job has already run the closer it is to done. The new cadence counts DOWN toward the deadline by which the job is guaranteed resolved (the calling tool's `max_runtime_seconds`): it holds at a steady ceiling through the bulk of the run (bounding how long a finished result waits to be noticed) and TIGHTENS toward a floor in the home stretch. For `resolve_dynamic_block` (`max_runtime=210`) this is a steady ~21s through the middle, then `21 → 16 → 11 → 6 → 5` as the finish nears — flat-then-decreasing, never increasing. (The web frontend has its own client-side backoff and ignores this field; this is the advice an AI agent honors.)
- **New pure helper `poll_backoff_seconds(elapsed, max_runtime)`** in `tollbooth/async_jobs.py`, plus an `elapsed_seconds` column on the job row. No tool-signature change; every claim-check consumer picks up the better cadence on pin bump.

## 0.57.0 — 2026-06-29

### Changed — BREAKING: one name for the possession token — `dpop_token` (retires `proof` / `proof_token` / `poison`)

- **The Secure Courier possession token is now `dpop_token` everywhere it is caller- or code-visible.** It was spelled three ways for one value: `proof_token` (returned by `request_npub_proof`), `poison` (the receive param + the credential-DM wire field + internal symbols), and `proof` (the parameter on every paid/free tool call). Calling it `proof` was wrong — the *proof* is the cached hash the wheel derives; the token is the **D**emonstrated **P**r**o**of-**o**f-**P**ossession credential a caller presents to retrieve/assert it. One general name now spans both the npub-ownership-proof flow and the credential-delivery flow.
- **Tool-signature change (this is the breaking part):** every paid/free tool that took `proof` now takes `dpop_token`. The `paid_tool` decorator extracts `kwargs["dpop_token"]`, so a consumer server must rename its paid-tool params **in lockstep** with this wheel — a server still declaring `proof` will fail every paid call with `proof_required`. No compat shim (clean cut).
- **Wire + error surface:** the credential-DM field is now `dpop_token = @@@…@@@` (a reply drafted against an old `poison = …` template won't parse — re-request a fresh channel). `ErrorCode.POISON_MISSING`→`DPOP_TOKEN_MISSING`, `COURIER_POISON_MISMATCH`→`COURIER_DPOP_TOKEN_MISMATCH`. The `proof_required` denial text now names `dpop_token`. The drain/retry/relay-pinning behavior is byte-for-byte unchanged — this was a rename, never a protocol change.
- **Removed vestigial `proof` params** from the bootstrap courier tools (`request_credential_channel`, `receive_credentials`, `request_patron_credentials`, `receive_patron_credentials`) — they never used them; a *request* tool issues the token, it does not receive one.

### Added — frictionless cold start (proof-vs-credential disambiguation, free "do I need credentials?" probe)

- **`service_status` now returns a free, unauthenticated `patron_auth` block** — `{"mode": "oauth"|"secure_courier"|"none", "patron_credentials_required": bool, ...}` — so an agent can learn whether an operator needs OAuth, a couriered secret, or nothing *before* proving anything. The same block is attached to every `proof_required` denial, killing the chicken-and-egg where the answer hid behind a proof-gated tool.
- **Proof and credential flows now cross-reference each other** in their tool descriptions (`request_npub_proof`/`receive_npub_proof` ↔ `request_credential_channel`/`receive_credentials`), so "run the Secure Courier" is no longer ambiguous between proving npub ownership and delivering a secret.
- **OAuth tool descriptions encode the "try live first" heuristic:** don't pre-emptively `begin_oauth`; a `pending` `check_oauth_status` is not proof of a lapsed session — attempt the live call and fall back only on an explicit `upstream_auth_refresh_needed`.

## 0.56.0 — 2026-06-29

### Added — detached-job failure reasons reach the MCP as curated, frontend-facing situations

- **New `AsyncJobSituation`** (exported from `tollbooth`) — a job runner or a spec's `shape_result` raises it to report a failure the calling FRONTEND should render as informative UX: a machine `error_code`, a safe human `message`, optional `next_steps`, and a `transient` flag. The raw upstream error stays operator-side; only these curated fields cross the tool boundary. Follows the "situations, not failures" convention.
- **`fetch_async_job` and `_run_job` now surface situations.** When `shape_result`/the runner raises `AsyncJobSituation`, the job refunds and the structured fields are returned to the caller AND persisted on the job row (serialized), so a later poll returns the same structured situation (via `situation_response_from_row`). Any *other* (unclassified) exception still refunds with a GENERIC message.
- **Fixed a latent leak:** the previous code stored `str(exc)` from a failed `shape_result` in the job row's `error`, which the already-error fetch branch returned to the patron on a subsequent poll. Now only a generic string (or a curated situation) is ever stored — a raw exception never reaches the patron.
- **The generic `dpyc-job-flow` is now a faithful messenger:** `http_request` returns the response for *every* status (including non-2xx) instead of raising, so the upstream status + body reach the MCP's `shape_result` (which decides success vs failure — domain policy belongs in the operator, not the generic flow). A non-2xx is logged to the Prefect run logs (operator-only) for debugging. Genuine transport errors still fail the run. (Flow-repo change; re-deploy to pick it up.)

## 0.55.3 — 2026-06-29

### Changed — long-runner secrets are normal operator secrets (no separate credential service)

- **Removed the wheel-injected `dpyc-longrunner` Secure Courier service.** It was a third credential-management path — invisible to `onboarding_status`/Pricing Studio and misrouted by the Studio's single-service courier card — to save a one-line per-server template edit, and it pushed Prefect infra secrets onto operators that never run long jobs. `prefect_api_key` is operator infrastructure exactly like `btcpay_api_key`; there was no principled reason to separate it.
- **The three fields are now exported as `tollbooth.credential_templates.LONGRUNNER_CREDENTIAL_FIELDS`** (optional `FieldSpec`s — the wheel still owns the canonical names, which are coupled to the executor wiring and the `dpyc-closure-key-<key_id>` block). Operators that register a long-running job spec spread them into their **own** `operator_credential_template`: `fields={**mine, **LONGRUNNER_CREDENTIAL_FIELDS}`. They are then ordinary optional operator secrets — same service, same `onboarding_status`/Studio surface, same Secure Courier path, delivered from the Studio's courier card without special routing.
- `_closure_key_hex` and the auto-wiring probe now load these from the **default operator credential service** (dropped the `service="dpyc-longrunner"` override and the `_LONGRUNNER_SERVICE` constant). No behavior change to dispatch/poll/refund.
- **Migration:** operators that delivered these under the old `dpyc-longrunner` service must re-courier them under their operator service (clean cutover, no compat shim).

## 0.55.2 — 2026-06-29

### Fixed — detached dispatch actually reaches the standalone Prefect account

- **`PrefectClosureExecutor` never authenticated to the operator's standalone Prefect account**, so on a host platform that sets its own `PREFECT_*` env (e.g. Prefect Horizon), `run_deployment`/poll targeted the *wrong* account (401 / deployment-not-found). `submit` raised, and `start_async_job`'s dispatch-failure handler silently fell back to the **in-process runner** — so detached execution never actually ran from the MCP front, and the very long jobs it exists to protect still died on serverless recycle. Quick jobs masked it by completing in-process before recycling.
- Root cause: `poll` used `os.environ.setdefault(...)`, a **no-op** when the host already set those vars; `submit` set nothing at all. Replaced both with `temporary_settings({PREFECT_API_URL, PREFECT_API_KEY})` — which *re-derives* settings (not defaults them) and is contextvar-scoped so concurrent operators don't clobber each other — forcing the vaulted standalone-account creds for the duration of each `run_deployment` / client call. Verified against a deliberately-wrong ambient env (401 without the override; correct account inside it).

## 0.55.1 — 2026-06-29

### Fixed — closure-path failure semantics symmetric with in-process

- `fetch_async_job` now wraps `shape_result` on the completed branch: if a detached run finishes but its raw result can't be shaped (an upstream non-2xx surfaced as the result, or the op produced nothing usable), the job is failed and the fare **refunded** — symmetric with the in-process runner, which raises+refunds on the same conditions. Previously a shaping exception escaped unhandled, leaving the job stuck and the fee uncredited. The error detail is never surfaced to the caller.
- The generic `dpyc-job-flow` `http_request` op now calls `raise_for_status()`: a non-2xx upstream response marks the flow run FAILED (so the MCP refunds) instead of returning as a "completed" job carrying an error status. Matches an in-process runner's `raise_for_status`. (Flow-repo change; re-deploy `dpyc-job-flow/dpyc-jobs` to pick it up.)

## 0.55.0 — 2026-06-29

### Changed — durable long-runner is a generic, DRY operator capability (not eXcalibur-specific)

- **Wheel-owned `dpyc-longrunner` credential service.** The runtime now auto-injects a built-in Secure Courier credential template (`prefect_api_url`, `prefect_api_key`, `closure_seal_key`) for *every* operator — the same mechanism as the built-in `npub_ownership` template. Any operator unlocks detached execution by couriering these three secrets; no per-server credential-template wiring. The upstream API secret (e.g. the Anthropic key) is **not** here — it stays in the operator's own template and is sealed into the closure locally.
- **Automatic executor wiring.** `start_async_job` opportunistically calls a one-shot `_ensure_async_executor()`: if a job spec is registered and the `dpyc-longrunner` creds are present in the vault, it installs a `PrefectClosureExecutor` bound to this operator's `key_id` — with no `set_async_executor` call in the server. An explicit `set_async_executor(...)` still wins and disables the probe.
- **Per-operator closure keys.** Each operator holds its own `closure_seal_key`; the shared `dpyc-job-flow` selects the right one via a new **non-secret `key_id`** in the cleartext run envelope (names the operator's `dpyc-closure-key-<key_id>` Prefect Secret block). `key_id` = `OperatorRuntime.durable_key_id()`, a public SHA-256 prefix of the operator npub — operators can't open each other's closures. `PrefectClosureExecutor` takes `key_id`; the flow's `dpyc_job_flow(closure_b64, key_id)` loads the keyed block. `_closure_key_hex` now loads from the `dpyc-longrunner` service.

## 0.54.0 — 2026-06-28

### Added — pluggable async-job executor (detached, durable execution off the recycling front)

- New module `tollbooth.async_executor` with a `JobExecutor` Protocol and two implementations. `InProcessExecutor` (the default) preserves today's behavior — the registered runner runs as a concurrent `asyncio` task in the operator's process. `PrefectClosureExecutor` dispatches the work to detached Prefect-managed compute via `run_deployment(..., timeout=0)` (fire-and-return), so a long job (an LLM round-trip, a web-augmented generation) survives a serverless front that freezes/recycles mid-run. The `service_status` Docket diagnostic added in 0.53.2 is what proved Horizon offers no durable in-process backend, motivating this.
- **Spec-driven "closure" path.** `OperatorRuntime.register_job_spec(kind, build_closure, shape_result)` registers a job kind that, under a detached executor, is dispatched as a self-describing **job spec** rather than an in-process call. `build_closure(**params)` runs in the MCP with full vault access (it loads operator secrets locally and bakes them into the spec, e.g. a fully-formed HTTP request); the spec is **AES-256-GCM sealed** (reusing `vault_encryption.VaultCipher`, AAD-bound) before it becomes a run parameter, so secrets reach Prefect only as ciphertext. `shape_result(raw)` turns the flow's raw return into the stored result dict. **No executable code travels to the flow** — only declarative data; op primitives live in the flow's own git-versioned repo.
- `OperatorRuntime.set_async_executor(executor)` installs the executor (default `InProcessExecutor`). The public `start_async_job` / `fetch_async_job` / `register_job_runner` API is unchanged; existing operators are unaffected. Selection is automatic: a kind uses the closure path only when it has a registered spec **and** a non-in-process executor is installed, else it falls back to the in-process runner.
- `start_async_job` now seals+submits the closure and persists the executor handle; if dispatch fails after the row is persisted (e.g. Prefect unreachable) it falls back to an in-process runner when one exists, else refunds — a fee-charged job is never stranded.
- `fetch_async_job` **settles** the closure path: it polls the executor for the detached run's terminal state and writes the result (or refunds on failure) into the Neon row. The old unconditional watchdog re-kick is gone for the closure path (the executor owns durability); the in-process path keeps its atomic re-kick as the only recovery for hosts without a detached executor.
- The detached flow returns its result via a **Prefect Artifact** (auto-associated with the flow run, read back with the MCP's existing Prefect API key), not via Prefect result storage — whose default is the worker's local disk and therefore unreadable from the MCP host. `PrefectClosureExecutor.poll` reads the artifact by flow-run id. The generic flow (`flows/dpyc_job_flow.py`, op primitive `http_request`) ships in this repo for Prefect Managed to clone; it touches no Neon and receives only declarative data.
- New nullable `async_jobs.run_handle TEXT` column (added to the CREATE and retrofitted via `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`; `restore_neon_schema` carries it). New `AsyncJobStore.set_run_handle`.
- New optional `[prefect]` extra (`prefect>=3.0`), imported lazily inside `PrefectClosureExecutor` — operators who never opt in do not need it installed.

## 0.53.2 — 2026-06-28

### Added — `service_status` reports the Docket (async-job) backend

- New `async_jobs` block in `service_status`: `docket_url_set`, `backend` (the URL **scheme only** — `redis`/`rediss`/`memory`/…, never the URL, which can carry credentials), and `durable_across_recycles` (true only for redis/valkey). Lets an operator confirm whether their FastMCP/Horizon deployment has a durable background-task backend, since the `start_async_job` claim-check path runs work via in-process `asyncio.create_task`, which does not survive a serverless container recycle without a redis-backed Docket. A test asserts the credential-bearing URL never appears in the diagnostic.

## 0.53.1 — 2026-06-28

### Fixed — `update_coupon` no longer fails saving a coupon with unchanged use caps

- **Bug:** editing a coupon and saving without touching its use caps (e.g. renaming `EVALUATOR100` in Pricing Studio, or any save that left `uses_per_patron`/`total_uses` alone) failed with `update failed: Object of type ellipsis is not JSON serializable`. `update_coupon_tool` used `...` (Python `Ellipsis`) as its private "leave this cap alone" sentinel and passed it into `CouponVault.update()`, but the vault recognises only its own `_UNSET = object()` sentinel. The unrecognised `Ellipsis` was appended to the SQL params list, where Neon's HTTP JSON encoder rejected it. The bug fired on every save that didn't set both caps.
- **Fix:** the tool no longer invents a second sentinel. It builds the cap kwargs conditionally and **omits** them when unchanged, letting the vault's own `_UNSET` default own the leave-alone semantics — one sentinel, one owner. Added a regression test pinning the unchanged-caps path (existing tests missed it because the test double recorded kwargs without JSON-serializing).

## 0.53.0 — 2026-06-25

### Added — generic upstream HTTP 402 ("renew your subscription") handler

- New module `tollbooth.upstream_payment` turns a **bare HTTP 402** from an upstream API into a structured, human-facing situation instead of an opaque error. Some upstreams (e.g. an X/Twitter developer plan whose billing lapsed, or a metered API past its quota) answer `402 Payment Required` to mean "the paid subscription/access tier tied to these credentials no longer covers this request." No payment this server can make settles it — a human must renew the plan at the provider.
- `upstream_payment_situation(service=, renew_url=, audience=, detail=, status_code=)` builds the situation: `error_code` `upstream_subscription_required`, clear renewal advice woven with the provider's portal URL, `audience` phrasing for whose plan it is (`"operator"` vs `"patron"`), and `transient: False` so schedulers and retry loops stop hammering an endpoint that can't self-recover.
- `classify_upstream_payment(response, ...)` duck-types an httpx-style response and returns the situation only for a bare 402, returning `None` for any other status.
- `is_x402_payment_challenge(headers)` distinguishes this from the **x402 micropayment protocol**: a machine-payable challenge advertises on-chain terms in a `payment-required` header and belongs to `X402Client` (transparent Operator COGS); its absence on a 402 is the human-subscription case. `classify_upstream_payment` routes accordingly.
- New `ErrorCode.UPSTREAM_SUBSCRIPTION_REQUIRED`. No heavy deps — the module is always importable (it does not require the optional `[x402]` extra). Exported from the package root.

## 0.52.3 — 2026-06-23

### Fixed — low-cert-balance reminder now self-notifies the actor who must refill (not the parent Authority)

- **Bug:** when a purchase order was refused with `authority_insufficient_balance`, the SDK dunned the **parent Authority's** owner ("your Authority's own credit balance is empty… `purchase_credits` on your Authority") and the patron-facing situation claimed the Authority was out of credits "for resale." Both were wrong. `certify_credits` is a `paid_tool`, so its fee debits the **purchasing actor's** own ledger held at the Authority (`debit_or_deny(tool_id, npub=caller)`) — never the Authority's own funds. The party that must act is the Operator or sub-Authority itself, which refills by calling its Authority's `purchase_credits`. The parent Authority owner cannot fix a downstream balance and got spurious dunning, while the actor who needed to act was never told.
- **Most visible in multi-level chains:** a secondary Authority (e.g. NewEngland) topping up / certifying up to a penultimate one (NorthAmerica) is itself the "operator" purchasing — so NorthAmerica's owner was dunned for NewEngland's depleted balance, and NewEngland's admin stayed in the dark.
- **Fix:** `_dun_authority_low_certs` → `_dun_self_low_cert_balance`. The relay-deduped marker DM is now a **self-notice** to the purchasing actor's own npub (`rt.operator_npub()`), which Pricing Studio surfaces. Wording corrected on both the DM and the patron-facing situation; misleading docstrings in `authority_client.AuthorityCertifyError` updated. `error_code` (`authority_insufficient_balance`) is unchanged — it's a wire contract across consumers. No change to fee computation, certification, or the refill mechanism (which already worked); only who is notified and what they're told.

## 0.52.2 — 2026-06-22

### Added — `resolve_service_version()` (one fleet-wide version resolver)

- New `tollbooth.version.resolve_service_version(dist_name, source_hint=None)` (also exported as `tollbooth.resolve_service_version`). Resolves a service's own version from `pyproject [project].version`: installed distribution metadata first, with a from-source `pyproject.toml` fallback (walks up from `source_hint`, typically the caller's `__file__`) for deploys that run a checkout without installing it — e.g. Horizon running a flat `py-modules` app, where `importlib.metadata` raises `PackageNotFoundError`. Returns `"0.0.0"` if unresolvable.
- Every operator and Authority can now call this instead of carrying its own copy, so version reporting in `service_status` is identical across the fleet and there is no hand-maintained version constant to drift — `/release` bumping pyproject is the only lever.
- The wheel dogfoods it for its own `__version__` (was a bare `importlib.metadata.version` with no fallback).

## 0.52.1 — 2026-06-22

### Fixed — Nostr profile read/publish no longer crawls relays sequentially

- **`fetch_profile` and `publish_profile_event` now fan out to relays in parallel** (one thread per relay) instead of querying them one-at-a-time. Wall-clock is now bounded by the single slowest relay, not the sum across the whole set. A `get_nostr_profile`/`publish_nostr_profile` call that previously took 10–40s+ (and paid for every relay even after finding the profile) now returns in roughly one relay round-trip.
- **Per-relay socket timeout cut from 10s to 5s** (`_TIMEOUT`), with `settimeout` applied to the read loop so a relay that connects but never sends EOSE can't hang the call. A dead or slow relay no longer holds the whole request hostage.
- Extracted `_fetch_one` / `_publish_one` per-relay helpers (each swallows its own failures → a dead relay yields no result rather than breaking the fan-out). New unit tests pin newest-wins selection, dead-relay tolerance, and field whitelisting.

## 0.52.0 — 2026-06-22

### Changed — BREAKING: decouple `vault_source` and `purchase_mode`; add registry-derived certify-up

- **Two orthogonal axes, no longer conflated:**
  - **`vault_source`** — where the Neon URL comes from: `"env"` (NEON_DATABASE_URL, self-provisioned) or `"authority"` (default, bootstrap DM).
  - **`purchase_mode`** — whether a purchase order certifies up to a parent Authority: `"direct"` (trust-root, no upstream cert), `"certified"` (certify up), or `"auto"` (derive from registry chain).
- **Background:** Sub-Authorities (e.g., NewEngland under NorthAmerica) need `vault_source="env"` (self-provisioned Neon) **and** `purchase_mode="certified"` (certify-up). The old single-axis flag couldn't express this combination — bumping to direct would reroute vault bootstrap, breaking the sub-Authority's own Neon.
- **New `resolve_purchase_mode(own_npub, registry_url)` in `tollbooth/registry.py`:**
  - Reads `upstream_authority_npub` from the dpyc-community registry and applies the certify-up rule:
    - No upstream → `"direct"` (Prime / trust-root)
    - Parent is Prime → `"direct"` (no upstream cert)
    - Non-Prime parent → `"certified"` (certify purchases to the parent Authority)
  - Single source of truth; called lazily by `_effective_purchase_mode()` if `purchase_mode="auto"`.
- **`OperatorRuntime` constructor changes:**
  - Added `vault_source: str = "authority"` parameter (default unchanged).
  - `purchase_mode` now accepts `"auto"` (new); existing `"direct"` and `"certified"` callers unchanged.
  - `vault()` method now keys off `vault_source`, independent of `purchase_mode`.
  - `purchase_credits()` calls `await rt._effective_purchase_mode()` for the certify-up decision.
- **Authority examples updated** (`server.py` in all three authorities):
  - `vault_source="env"` (self-provisions Neon from NEON_DATABASE_URL).
  - `purchase_mode="auto"` (derives direct/certified from registry chain).
  - Canonical Authority sits under Prime → resolves to `"direct"`.
  - NorthAmerica sits under Prime → resolves to `"direct"`.
  - NewEngland sits under NorthAmerica → resolves to `"certified"`.
- **Test suite:** 12 tests in `tests/test_purchase_mode_decoupling.py` cover the rule, auto-caching, failsafe on registry unreachable, and vault source independence.

### ⚠️ Migration

- **If you hardcode `purchase_mode="direct"`**, update to `vault_source="env", purchase_mode="auto"` (or keep explicit `"direct"` if you want to opt out of future registry sync).
- **If you hardcode `purchase_mode="certified"`**, keep it; or switch to `purchase_mode="auto"` to auto-derive.
- **No changes required for operators** — the wheel handles it internally. Authorities and MCP servers using the SDK must update their `OperatorRuntime` constructor calls to include the new split params.

## 0.51.0 — 2026-06-20

### Added — Nostr kind-0 profile tools (self-sovereign patron profiles, no key custody)

- **Two new free standard tools so every operator can serve patron profiles**
  (and every frontend/agent reads/writes them via the backend instead of doing
  relay I/O itself):
  - **`get_nostr_profile(npub)`** — reads an npub's latest public kind-0
    metadata (name, display_name, about, picture, banner, nip05, website,
    lud16) across a broad relay set. Free, no proof — the data is already
    public on relays.
  - **`publish_nostr_profile(npub, signed_event)`** — relays a **client-signed**
    kind-0 to relays after verifying `kind == 0`, `pubkey == npub`, and a valid
    Schnorr signature. **The wheel never holds a patron nsec** — the frontend
    signs (session key or NIP-07) and the wheel only fans the signed event out.
    The signature is the authorization; no proof token, no escrow.
- New module `tollbooth/nostr_profile.py` (`fetch_profile` / `publish_profile_event`),
  self-contained raw-websocket relay I/O mirroring `bootstrap_relay.py`.
- Rationale: kind-0 is public + self-signed, so it rightly lives on relays, not
  the operator vault. Operator secrets stay in the vault via Secure Courier;
  profiles are public and key-owned. We do NOT ask patrons for nsecs.

## 0.50.0 — 2026-06-20

### Changed — operator credential delivery is now merge-on-receive (single-secret deliveries)

- **`receive_credentials` merges the delivered fields into the existing vault
  blob instead of replacing it**, and accepts a *partial* delivery. An operator
  can now open a Secure Courier channel and reply with just **one** secret
  (e.g. a later `anthropic_api_key`) without re-sending — or clobbering — the
  rest (`nostr_credentials.py` `receive()`: load existing → overlay delivered →
  store merged).
- **`validate_payload` gained `partial=True`** — skips the "missing required
  fields" rejection while still enforcing field shape (type, non-empty if
  present) and dropping unknown fields. Used by the merge path.
- **Completeness is enforced by the readiness gate, not per-delivery.**
  `session_status` / `get_operator_onboarding_status` already check the merged
  vault has every required field; an incomplete interim state simply stays
  `warming_up`. The operator-credential validator now runs **only when the
  merged set is complete**, so a partial delivery is never flagged "missing X"
  and wiped.
- `receive_credentials` response now reports `stored_fields`,
  `still_missing_required`, and `optional_missing` so callers (and Studio) can
  show what's left. Re-sending the full batch behaves exactly as before.

## 0.49.0 — 2026-06-19

### Changed — operator bootstrap config is now a NIP-33 replaceable event (durability)

- **The Authority publishes each operator's bootstrap config as a NIP-33
  parameterized-replaceable event (kind 30078, NIP-78 app data) instead of a
  kind-4 DM.** Relays keep only the latest replaceable per (Authority, kind,
  per-operator `d` tag), so the config no longer ages off the way a stream of
  kind-4 DMs does — which is what left operators unable to cold-start bootstrap
  ("No bootstrap config on relays"). Content stays NIP-04-encrypted; discovery
  still needs only the operator's nsec (resolve Authority from the registry →
  read the Authority's replaceable event scoped by the operator's `d` tag). The
  age (`since`) window is gone — a stable replaceable is the current config no
  matter how old. `receive_bootstrap_config` dropped its unused `max_age_seconds`
  parameter.
- ⚠️ **Cold switchover — no transition code.** Operators on this wheel read only
  kind-30078; existing configs were published as kind-4. After every Authority
  is on ≥0.49.0, the Prime Authority must re-run `get_operator_config` (or
  `register_operator`) once per operator to re-publish each config as the
  replaceable event. Until then those operators bootstrap-fail by design. No
  data is at risk — relays only ever carried the config pointer, never the Neon
  tables.

## 0.48.1 — 2026-06-19

### Fixed — check_price no longer falsely previews unpriced tools as flat/0

- **`check_price` now agrees with `debit_or_deny`.** A non-free tool that is
  absent from (or unpriced/TBD in) the active pricing model used to preview as
  `pricing_type: flat, effective_cost_api_sats: 0, success: true` — while the
  real paid call denied it with `tool_not_priced`. `check_price` resolved the
  UUID from the tool registry and returned a 0 default without consulting the
  model. It now mirrors `_resolve_pricing`'s `has_tool`/`is_priced` gate (when
  Neon is available) and returns `error_code: tool_not_priced` for such tools.
  Surfaced while verifying newly-deployed (registry-present, model-absent) tools.

## 0.48.0 — 2026-06-19

### Changed — npub proof delegation cap raised 7 → 30 days

- **`MAX_PROVEN_TTL` is now 30 days (was 7).** Patrons already choose their own
  proof-delegation duration via `request_npub_proof` (`parse_duration` accepts
  e.g. `"30 days"`, `"4 weeks"`); only the hard cap moved. Durations above the cap
  still clamp down rather than erroring. This supports long-lived editorial
  sessions and unattended automation (e.g. a scheduler that holds an operator
  proof_token and cannot re-prove interactively). No API or call-site change.

## 0.47.0 — 2026-06-18

### Changed — kind treatment when an Authority is out of certification credits

- **`purchase_credits` no longer leaks the raw `Authority certification failed: … Insufficient balance: 0 sats available, …` string to patrons.** When the Operator's own balance at its certifying Authority is exhausted, certification is refused — that is the Operator's supply problem, not the patron's. The patron now gets a kind situation response (`error_code: authority_insufficient_balance`) naming the Authority and asking them to be patient and retry shortly, with `next_steps` and the `authority_npub`. Any other certification failure is unchanged.
- **`AuthorityCertifier` propagates the Authority's structured `error_code`.** `AuthorityCertifyError` now carries the Authority's `error_code` (e.g. `insufficient_balance`) so callers branch on the code rather than parsing prose; `purchase_credits` prefers it and falls back to a message-text check for older Authorities. Empty for connection/transport failures.

### Added — relay-deduped "Authority is out of credits" dunning

- **The wheel now reminds the Authority to top up, automatically and at most once per ~10-minute window.** On an Authority-insufficient-balance refusal, the Operator sends a single marker-tagged DM (`["t", "dpyc-dunning"]`) to the Authority's npub asking its human to `purchase_credits`. Dispatch runs on a daemon thread so the patron's response is never delayed, and failures are swallowed (courtesy DM).
- **Relay-as-cache dedup ("send-if-not-sent"), no new persistent state.** Before sending, `NostrCredentialExchange.has_recent_tagged_dm(...)` queries the relays for a recent kind-4 DM we authored to the Authority bearing the marker tag (`{kinds:[4], authors:[me], #p:[authority], #t:[…], since: now-600}`). The dunning DM self-expires via NIP-40 (~10 min), so the relay's own event store *is* the dedup cache — repeated patron attempts in the window send no duplicate, and a fresh reminder is allowed once the marker expires. Best-effort: if the relays dropped the event early, the caller is free to re-send.
- **`send_dm` gained an opt-in `extra_tags` parameter** (appended to the NIP-04 kind-4 leg only — the leg signed by the operator's identity key, hence author-queryable; the ephemeral gift-wrap leg stays untagged). New `ErrorCode.AUTHORITY_INSUFFICIENT_BALANCE`.

## 0.46.1 — 2026-06-18

### Fixed

- **`unregister_dynamic_tool` uses the current FastMCP tool-removal API.** Prefers `mcp.local_provider.remove_tool` (FastMCP 3.x) and falls back to the top-level `remove_tool` on older versions, so retiring a synthesized tool no longer trips a deprecation warning.

## 0.46.0 — 2026-06-18

### Added — runtime tool synthesis (operator-defined dynamic tools)

- **New `tollbooth.dynamic_tools` module + `OperatorRuntime.register_dynamic_tool` / `unregister_dynamic_tool`.** Operators can now synthesize first-class, typed MCP tools at runtime from a declarative parameter schema plus a `runner` callback — e.g. a named, parameter-bound stored query becomes `slug_find_airline_flights(from_city, to_city)`. The machinery is domain-agnostic (no graph/REST/SQL assumptions baked in); the `runner` (`async (params, npub, proof) -> dict`) supplies the behavior. `cypher-mcp` is the first consumer (named Cypher queries); any operator can back a synthesized tool with REST, SQL, a stored prompt, etc., and reuse the identical primitive.
- **Correctly-typed schemas through the billing decorator.** `build_dynamic_handler` builds a handler carrying both `__signature__` and real `__annotations__` (FastMCP derives a tool's input schema from annotations, not `__signature__` alone), so the synthesized tool exposes flat, typed params. `paid_tool`'s `functools.wraps` preserves both, so synthesized tools get debit / refund-on-raise like any paid tool.
- **Register-only; price in the App.** Synthesized tools are inserted into the tool registry (so `check_price`, `list_canonical_identities`, and the pricing model all see them) but carry **no pricing hint** — they stay unpriced ("not priced yet (TBD)") until the operator prices them in the pricing model. No price flows through the MCP; Pricing Studio stays the source of truth.
- **Bootstrap wiring.** `register_standard_tools` now stashes the FastMCP app + slug decorator on the runtime so dynamic (de)registration can happen after bootstrap. Spec persistence and cold-start re-materialization stay with the consumer — the wheel registers/deregisters; it does not dictate where specs live.
- The dynamic-tool param-schema language (`validate_param_schema` / `validate_params`; types `string`/`int`/`float`/`bool`/`list`) lives in `dynamic_tools` as the canonical implementation.

## 0.45.4 — 2026-06-17

### Fixed — tenant schema ownership no longer lags new tables

- **`transfer_schema_ownership` now reassigns *every* table in an operator's schema, not a hand-maintained list.** The static `_PROVISIONER_TABLES` never gained `coupons` (added 0.41.0), so on tenants where that table was created by the provisioning role it stayed admin-owned — the operator role then couldn't `CREATE INDEX` on it ("must be owner of table coupons"), which aborts the entire vault bootstrap and strands the operator in `warming_up` with no paid tools. Ownership transfer now enumerates `pg_tables`, covering every table now and as the schema grows (unsafe identifiers are skipped). Removed the stale `_PROVISIONER_TABLES` constant.

### Added — owner-side tenant repair

- **New restricted Authority tool `repair_operator_schema(operator_npub, authority_proof)`** — reassigns all table ownership in an operator's tenant schema to its own role and re-grants DML, in place. Unlike `register_operator` it does not rotate the operator's DB password or re-send the bootstrap DM, so it repairs a mis-owned tenant (the failure above) without disrupting a working one. Idempotent.

## 0.45.3 — 2026-06-17

### Changed — caller-facing errors survive refund-on-raise

- **A paid tool that raises `ValueError` now surfaces that message to the caller** (under new error code `tool_input_invalid`) instead of the blanket "Tool execution failed. Check operator logs." A `ValueError` is the operator's deliberate caller-facing signal — unknown key, invalid params, a lifecycle situation — so the caller can self-correct rather than being misdirected to operator logs for their own mistake. Surfaced live by cypher-mcp's first-light test: calling `execute_query_by_key` with an unknown key correctly refunded (refund-on-raise intact) but reported a generic failure instead of "No published query named '…'". The debit is still rolled back before the message is built, so this stays a no-charge outcome; non-`ValueError` exceptions remain sanitized (no internal leakage).

## 0.45.2 — 2026-06-17

### Changed — orphan stays orphan until it's publicly discoverable

- **An operator that can't yet resolve its own entry in the public DPYC registry now reads as `not_registered` (orphan), not a scary `Bootstrap failed`.** After an Authority approves an adoption, the public members file (`read-only-lookup-cache.json`, served via GitHub raw CDN) takes a few minutes to propagate. During that window the operator was surfacing *"Bootstrap failed: Cannot resolve Authority … Operator may not be registered"* — alarming, when it's simply not yet discoverable. `session_status` now classifies the registry-not-found / cannot-resolve-authority case as the `not_registered` lifecycle with calm propagation guidance ("the public registry is still propagating … the operator bootstraps automatically once its entry appears"), and `get_operator_onboarding_status` reports "awaiting registry propagation" instead of "bootstrap pending" for that case. The public registry — not the Authority's say-so — is the source of truth for "adopted"; no operator-side state is tracked. The operator auto-heals the moment it's discoverable.

## 0.45.1 — 2026-06-17

### Fixed — request_adoption now works for the orphan it's meant for

- **`request_adoption` no longer requires the operator's vault to verify the caller's proof.** It gated through `require_caller_proof`, which builds the vault-backed proven-npub cache and therefore forced an operator bootstrap — but an un-adopted orphan has no vault yet (bootstrapping is exactly what adoption provisions). The result was a chicken-and-egg failure (`Bootstrap failed: Cannot resolve Authority … Operator may not be registered with an Authority`) on the precise case the tool exists for. It now verifies the caller's **inline kind-27235 proof directly** (`require_proof(..., proven_cache=None)`) — no vault touched. The gate is unchanged (the caller must still hold the operator nsec and sign); only the cached-poison tactic (impossible without a vault) is dropped for this tool. Regression test asserts the vault is never touched.

## 0.45.0 — 2026-06-16

### Added — deferred operator adoption (the courtship)

- **Operators can ask a chosen Authority to adopt them, and the Authority owner approves on their own time** — instead of adoption only happening by the Authority calling `register_operator` out of band. New operator tools `request_adoption(authority_npub, …)` and `adoption_status(authority_npub)` reach the chosen Authority MCP-to-MCP (FastMCP `Client`, matched by tool-name suffix so neither side needs the other's slug). New Authority tools: `receive_adoption_request` (inbound, gated by an inline operator-ownership proof), `list_adoption_requests` / `approve_adoption` / `reject_adoption` (Authority-owner consent), and `get_adoption_status` (free).
- Pending requests are **durable** in a per-Authority Neon table (`authority/adoption_store.py`), keyed by operator npub, so concurrent requests survive Horizon cold starts — unlike the in-memory Authority-onboarding singleton.
- `approve_adoption` and `register_operator` now share one provisioning core (`_provision_operator`), so the deferred-courtship and inline-consent paths produce a byte-identical effect (ledger row + isolated Neon tenant + community-registry registration + bootstrap DM).
- Operator-ownership is proven inline against a canonical sentinel (`identity_proof.ADOPTION_PROOF_TOOL`) — no relay round-trip on the request leg.
- A best-effort owner-notification DM ("review your queue") accompanies each request; the durable queue remains the source of truth.
- New error codes: `adoption_pending`, `adoption_not_found`, `adoption_already_provisioned`. Lifecycle is unchanged — an operator stays `not_registered` until approval provisions its Neon tenant, then the existing bootstrap path takes it to `ready`.

### Deferred (follow-ups — flagged, not dropped)

- Phase 2 — approve-by-Nostr-reply (`check_adoption_replies`): reply "yes + poison" to the notification DM to approve without the Studio. Needs poison plumbing through the courier; today the Studio approve action (cryptographic consent) is the approval path.
- Pricing Studio "Pending Adoptions" queue UI (separate Swift work).

## 0.44.15 — 2026-06-11

### Added (audit M3 — quality & polish)

- **`SessionCache`** gains an optional `max_size` bound (evicts
  least-recently-written entries) and an **opportunistic expiry sweep on
  `set()`** so a write-heavy cache can't accumulate stale entries unbounded.
  Default remains unbounded — TTL-only — so existing callers are unchanged.
- **Developer tooling**: a `Makefile` (`dev`/`test`/`lint`/`type`/`cov`/`all`)
  and a refreshed `CONTRIBUTING.md` documenting the mypy + coverage gates and a
  dependency-pinning policy (exact pins for behavior-critical deps, security
  floor for `cryptography`).
- Tests for previously-uncovered modules: `infographic.py` and
  `constraints/patron_proof.py`.

### Fixed

- **Account infographic double-escaped the service name** — it was passed
  through `escape()` twice (once at the call site, once inside `_text`),
  rendering `<x>` as `&amp;lt;x&amp;gt;`. XSS-safe either way, but the name now
  displays correctly. Surfaced by the new infographic tests.
- **Poison-format proof feedback (S4)**: a proof token of the
  `<word>-<word>-<n>` shape that can't be validated as a cached proof (no cache
  wired, or a miss) now returns a clear "refresh your token / pass an inline
  Schnorr proof" message instead of falling through to a confusing "Invalid
  identity proof". Changes only the denial message — never what is accepted (a
  real Schnorr proof is JSON and never matches the token shape).

### Docs

- `vault_encryption.encrypt` documents the AES-GCM random-nonce volume ceiling
  (~2**32 encryptions/key per NIST SP 800-38D) and that nsec rotation is the
  mitigation — far beyond any per-operator vault's write cadence.
- Test flakiness: the background-flush test now waits on a bounded
  wait-until-condition helper instead of a fixed `asyncio.sleep(0.3)`.

## 0.44.14 — 2026-06-11

### Fixed

- **`restore_neon_schema` never re-created the credential-vault schema.** The
  step guarded on `getattr(rt, "_credential_vault", None)`, but
  `OperatorRuntime` has no such attribute (the live vault lives on
  `rt._courier._exchange._credential_vault`), so it was always None and the
  branch was dead — a restore left the credential tables uncreated. Now builds a
  `NeonCredentialVault` directly from the operator's `NeonVault` (idempotent
  `CREATE TABLE IF NOT EXISTS`), mirroring the `PricingModelStore` step, so a
  restore re-creates the credential schema even on a cold runtime whose courier
  hasn't materialized. Per-step failures are surfaced inline like the other
  steps. Regression-tested. (Resolves the second M2.5/M1.4 §2 follow-up.)

## 0.44.13 — 2026-06-11

### Fixed

- **Authority registration no longer reports phantom success on a failed npub
  persist.** `_set_authority_npub` swallowed a vault **write** failure and then
  cached the authority npub in memory anyway — so registration returned success
  while the cert-critical `authority_npub` was never written, vanishing on the
  next restart and silently breaking certificate verification. The write now
  propagates on failure, the in-memory cache is updated only **after** a
  successful write, and `check_authority_approval` aborts activation with a
  clear "retry once the vault is reachable" error instead of marking the
  Authority `activated`. Onboarding state is preserved on failure so the
  operator can retry. Regression-tested. (Resolves the M1.4 §2 follow-up.)

## 0.44.12 — 2026-06-11

### Changed (audit M1.4)

- **Error-handling pass: no more silent `except Exception: pass`.** All 35 bare
  `except Exception` swallows now log with context at an appropriate level —
  the exact pattern that hid the 0.44.9 tranche bug and 0.44.10 proof bug.
  Levels chosen by risk: `logger.debug` for best-effort cleanup / cold-start /
  diagnostics; `logger.warning` for surge-demand increments, the authority-npub
  vault ops, and operator callbacks. **Money path:** the credit-rollback failure
  handler (`runtime.py`) now logs `ERROR` — a swallowed rollback means a patron
  may have been charged without delivery, which now surfaces for reconciliation
  instead of vanishing. Behavior-preserving (logging only); no control-flow
  change. The 9 narrow specific-exception handlers (`asyncio.CancelledError`,
  ISO-date / NIP / JSON parse-fallbacks, `add_signal_handler` platform fallback,
  `parse_duration` default) are intentionally left quiet — a typed `except` is a
  reasoned swallow, not the anti-pattern.

### Flagged for follow-up

- `authority/tools.py::_set_authority_npub` swallows a vault **write** failure
  then caches the npub in memory — false durability on a certification-critical
  key. Now logs `warning`; a re-raise is likely correct but is a §2
  authority-flow change, left for an explicit decision.

## 0.44.11 — 2026-06-11

### Added (audit M2.5)

- **mypy is now a blocking CI gate.** The package type-checks clean at default
  strictness (`[tool.mypy]`, run in the `Type check` job on every matrix
  Python). Do not silence new errors with per-file overrides — fix the type.
  Adopting it cleared a 27-error backlog and surfaced the bugs below.

### Fixed

- **Cold-start courier rehydration could crash on a JSON expiry.**
  `_resolve_pinned_record` compared `time.time() > p_expiry` where `p_expiry`
  came from a JSON blob and can be a string → `TypeError: '>' not supported
  between 'float' and 'str'`. Now coerced to float before comparison.
- **Dead `_patron_npubs` handling removed from constraint config.**
  `build_constraint` read a key (`_patron_npubs`) nothing serialized and set an
  attribute nothing read; live patron-group scoping is `PricingStep.patron_npubs`
  (`constraints/gate.py`), which enforces its own max-10 rule. Removed.
- **x402 client now narrows the V1|V2 union explicitly.** `_sign_payment`
  implements the current (V2) protocol; a V1 payment-required response now
  raises a clear "upgrade upstream" error instead of an attribute crash.

### Internal

- Type-narrowing None-guards across `proven_npub`, `ledger`, `pricing_resolver`,
  `nostr_credentials`, `runtime`, `coupons/vault`; `ToolConstraint.schema()` is
  now a declared abstract classmethod. The restore-path `_credential_vault`
  branch in `runtime.py` is documented as known-dead (always None — the live
  vault is on `_courier._exchange`); flagged for a separate verified fix.

## 0.44.10 — 2026-06-11

### Fixed

- **`receive_npub_proof` silently dropped valid proof replies.** The drain loop
  carried a pre-challenge timestamp gate (`created_at < challenge_ts - 5s`) that
  popped any reply older than the stored challenge timestamp **without a NACK** —
  so a correctly-signed reply, carrying the correct one-time poison, on the
  correct pinned relay, was discarded as "too old" whenever the patron's Nostr
  client clock skewed behind the server or the human replied at human pace. The
  one-time poison phrase is already the sole anti-replay scoping mechanism (a
  stale reply carries a stale poison → caught as `wrong token`), so the timestamp
  gate added zero security while introducing a hard failure mode. **Removed the
  gate** (and the now-dead `challenge_ts` store/load on the
  `_proof_pending_npub_ownership` session). The relay purge-on-request stays — it
  keeps the rendezvous relay clean between attempts. Regression-tested: a reply
  with `created_at` before the challenge but the correct poison now matches.
  Surfaced live while exercising the proof flow; same shape as the 0.44.9 tranche
  bug — a guard clause swallowing the happy path.

## 0.44.9 — 2026-06-11

### Fixed

- **Credit tranche expiration was silently disabled fleet-wide.**
  `resolve_tranche_lifetime()` called a non-existent `ensure_pricing_store()`
  method since the 2026-03-31 tranche-expiration refactor; the `AttributeError`
  was swallowed by a bare `except`, so it always returned `None`. As the only
  reader of the pricing model's `tranche_lifetime` (used on the billing and
  purchase/restore/reconcile paths), this meant operators who configured a
  tranche lifetime had patron credits that **never expired**. Fixed to build the
  `PricingModelStore` inline (as everywhere else) and to log rather than swallow
  failures. **Behavior change:** pricing models with `tranche_lifetime` set will
  now expire credits on schedule, as originally intended. Regression-tested.

### CI (audit M2.5)

- Ruff now enforces `F821` (undefined names); forward-ref imports resolved via
  `TYPE_CHECKING`.
- Coverage ratchet gate: `--cov-fail-under=67` (a floor, not a target; measured
  ~70%). CI now fails on a coverage regression.

## 0.44.8 — 2026-06-11

### Changed (internal)

- The money gate `debit_or_deny` is decomposed (audit M2.2). Its 306-line
  pipeline is split into three self-contained, independently-testable async
  methods on `OperatorRuntime` — `_resolve_pricing`, `_evaluate_constraints`,
  and `_apply_billing` — with `debit_or_deny` keeping only the tightly-coupled
  header stages (identity / proof / restricted-access / caller-resolve) and
  orchestrating the three. Strictly behavior-preserving (stage methods, not a
  separate state object; no reordering). 306 → 116 lines. No wire-API change.

### Tests

- Characterization net for `debit_or_deny` (`tests/test_credit_gate.py`, 15
  tests) pinning every stage, including the previously-unexercised constraint
  denial / discount / credit and coupon-burn paths — the safety net under the
  decomposition.
- The `authority/` subpackage gains broad coverage (audit M2.4): 23% → 71%,
  with the Schnorr cert signing round-trip, anti-replay tracker, onboarding
  state machine, role-isolation provisioning, and the `certify_credits`
  cert/fee engine now tested.

## 0.44.7 — 2026-06-11

Completes the `register_standard_tools` decomposition (audit M2.1). No wire-API
changes.

### Changed (internal)

- The OAuth2 tools (`begin_oauth`, `check_oauth_status`) moved to
  `tools/oauth.py` (functions over `rt`), the last section of the extraction.
  `check_oauth_status` performs the code→token exchange and persists OAuth
  tokens, so the move is behavior-preserving and characterize-then-extract:
  `tests/test_oauth_tools_characterization.py` pins both tools (PKCE +
  authorize-URL build + verifier persist; token exchange + vault persist +
  `on_token_received` merge) and stays green against the extracted code.
- With this, every standard tool body lives in a tested `tollbooth.tools.*`
  module (coupons, pricing/`check_price`, identities, status, proof, courier,
  oauth) behind a thin proof-gate shim. `register_standard_tools` is now a
  registration layer rather than a 2,600-line god function; `runtime.py` is
  3741 lines (4757 at the start of this audit pass). Logic that was trapped in
  closures — including the §2-sensitive proof drain loop, courier validation
  flow, and OAuth token exchange — is now unit-tested.

## 0.44.6 — 2026-06-11

### Fixed

- Proof tools no longer block the async event loop. `receive_npub_proof` and
  `request_npub_proof` called `exchange._fetch_dms_from_relays()` synchronously
  (the pinned drain and the stale-DM purge) — the same latent block fixed for
  the courier in 0.44.3 (P1). Both now run via `asyncio.to_thread`, verified by
  a teeth-checked non-blocking test.

### Changed (internal)

- Courier tools join the `register_standard_tools` decomposition (audit M2.1).
  `request_credential_channel`, `receive_credentials`, `forget_credentials`,
  and the patron variants moved to `tools/courier.py` (functions over `rt`),
  leaving thin shims. §2-sensitive credential flow, so behavior-preserving:
  `receive_credentials`' operator-credential validation-callback flow (validator
  passes → cashier reset; fails → forget bad creds + rejection DM + structured
  error) is pinned by `tests/test_courier_tools_characterization.py`, which
  stays green against the extracted code. `runtime.py` is now 3938 lines
  (4757 at the start of this audit pass). No wire-API changes.

## 0.44.5 — 2026-06-11

Internal maintenance: the proof tools join the `register_standard_tools`
decomposition (audit M2.1). No wire-API changes.

### Changed (internal)

- `request_npub_proof`, `receive_npub_proof`, and `check_proof_status` moved
  from inline closures into `tools/proof.py` (functions over `rt`). These are
  identity-proof orchestrators (the deterministic, poison-scoped Secure Courier
  drain loop + proven-npub cache), so the move is strictly behavior-preserving.
  It was done characterize-then-extract: the `receive_npub_proof` drain loop —
  previously ~0% covered — is now pinned by
  `tests/test_proof_tools_characterization.py`, which stays green against the
  extracted code (the faithfulness proof). `runtime.py` drops 4461 → 4112 lines;
  `tools/proof.py` is covered to 81%.

### Known issue (tracked)

- `receive_npub_proof` / `request_npub_proof` still call
  `exchange._fetch_dms_from_relays()` synchronously — a latent event-loop block
  of the same class fixed for the courier in 0.44.3 (P1). Deliberately left out
  of this pure move; a separate `asyncio.to_thread` follow-up will address it.

## 0.44.4 — 2026-06-11

Internal maintenance from the 2026-06-10 audit: coverage measurement in CI and
the first tranche of the `register_standard_tools` decomposition (M0.2 + M2.1).
No wire-API changes — tool names, signatures, and behavior are identical.

### CI

- Coverage is now measured on every CI run (pytest-cov, branch + thread),
  printed as a `term-missing` report. **Report-only** — no `--cov-fail-under`
  gate yet (that ratchet lands once the extraction raises the floor). Baseline
  recorded in `CONTRIBUTING.md`: 64%.

### Changed (internal)

- Standard tool bodies are moving out of the ~2,600-line
  `register_standard_tools` closure into testable `tools/` functions, leaving
  thin proof-gate + delegate shims. This release extracts the coupon CRUD
  (`tools/coupons.py`), the `check_price` pure core
  (`build_pricing_preview` / `apply_constraint_preview` in `tools/pricing.py`),
  `list_canonical_identities` (`tools/identities.py`), and the `service_status`
  / `session_status` assembly (`tools/status.py`). `runtime.py` drops from 4757
  to 4461 lines; the extracted logic — previously untested in closures — is now
  covered 80–100% by direct unit tests. Function names are preserved so
  `mcp_name_for` keeps producing identical runtime tool names.

## 0.44.3 — 2026-06-10

Event-loop hardening from the 2026-06-10 SDK audit (P1).

### Changed

- Relay drains no longer block the async event loop (audit P1). `open_channel`
  and the `receive` pinned-relay drain do synchronous websocket I/O (connect +
  recv-until-EOSE, bounded by a per-relay timeout); they now run on a worker
  thread via `asyncio.to_thread` instead of inline, so a slow/timing-out relay
  no longer freezes every other coroutine on the serverless event loop. The
  "buffer populated on return" contract is preserved (the thread is awaited).
  `_ephemeral_agents` access is now lock-guarded to stay safe under the
  concurrency this unlocks. Regression test in
  `tests/test_relay_io_nonblocking.py`.

## 0.44.2 — 2026-06-10

Security and hardening release from the 2026-06-10 SDK audit.

### Security

- **Credential leak on the credential-card redemption path (audit S1).**
  `receive_credentials` and `receive_patron_credentials` redeemed an `ncred`
  card by calling `courier._exchange.redeem_credential_card()` directly and
  returning its result, which still carried the raw `credentials` values —
  bypassing the `SecureCourierService.redeem_card()` wrapper that strips them.
  Raw credential values could surface in the tool result (→ agent context,
  logs). Both tools now route through `courier.redeem_card()`, which vaults and
  then strips. Vaulting is unchanged (it happens inside the exchange), so this
  is a pure removal of the echoed values. Regression guard added in
  `tests/test_credential_no_echo.py`.

### Changed

- Replay-protection set in `identity_proof` is now bounded by a hard cap
  (`_CONSUMED_MAX_ENTRIES = 10000`) with eviction of expired-then-soonest-to-
  expire ids (audit S2). Entries are only inserted after full signature +
  freshness verification, so this is a memory backstop against a flood of
  distinct valid proofs outrunning the 120s lazy cleanup, not a new gate.
- PyPI classifier bumped from `Development Status :: 2 - Pre-Alpha` to
  `4 - Beta` to reflect production use across the operator fleet.

### CI

- The `[x402]` optional extra is now installed in the test workflow, so the 15
  `x402_client` tests run instead of being silently skipped. `pytest` now runs
  with `-ra` so any future skipped module surfaces its reason in the log.

## 0.44.1 — 2026-06-07

### Fixed

- Courier resolve-error prose pointed at the wrong recovery tool in the proof
  and patron-credential flows — `COURIER_TOKEN_EXPIRED` (and siblings) said
  "Call request_credential_channel again" even from `receive_npub_proof`. The
  messages now carry a `{request_tool}` placeholder filled per flow
  (request_npub_proof / request_patron_credentials / request_credential_channel),
  so the hint always names the tool that opens *that* channel. Same fix applied
  to the `poison_missing` and `courier_not_found` strings.
- `OperatorProtocol.receive_credentials` and the obsolete-practice prose still
  described the pre-0.44.0 `receive_credentials(sender_npub, service)` shape —
  updated to include the now-required `poison` argument.

## 0.44.0 — 2026-06-07

Deterministic, poison-scoped Secure Courier retrieval. Agentic clients kept
failing the "pick up your reply" step because the MCP guessed which channel to
drain and swept every configured relay. The retrieve tools now take an explicit
`(sender_npub, service, poison)` and drain exactly the one rendezvous relay the
request pinned — no guessing, one unambiguous answer.

### Changed (breaking)

- `receive_credentials`, `receive_patron_credentials`, and `receive_npub_proof`
  now **require** `poison` (the session phrase / `proof_token` returned by the
  matching `request_*`). Calls without it return `ErrorCode.POISON_MISSING`.
  `receive_credentials` drops the `force_relay` argument.
- `Exchange.receive(sender_npub, *, service, poison)` is now strict: it resolves
  the pending channel for `(sender_npub, service)`, verifies the poison, and
  drains **only** the pinned rendezvous relay. Wrong-poison / undecryptable /
  malformed DMs are NIP-09 deleted and the sender is NACK'd; the first matching
  DM is ACK'd and the scan **stops** (stop-at-match). No vault-first fallback.
- Not-found, mismatch, expiry, and no-pin paths now return structured results
  (`success: False` + `error_code` + `popped`) instead of raising
  `CourierTimeout` / `CourierValidationError`.
- NACK and not-found copy no longer reveal the expected poison phrase
  (previously echoed `got X, expected Y`).
- `receive_npub_proof` drops its 4×/2s retry loop for a single pinned-relay
  drain — the call is human-gated, so one fetch is correct.

### Added

- `Exchange.receive_from_vault(sender_npub, *, service)` — vault-only,
  poison-free credential read. `SecureCourier.restore_session` /
  `ensure_identity` now use it for serverless cold-start session restoration,
  so removing vault-first from the agent path does not break automatic restore.
- `Exchange._resolve_pinned_record(...)` — verifies poison + resolves the pinned
  relay, with cold-start rehydration of the poison, pin, and ephemeral agent key
  from the `__pending__{service}` vault blob.
- `ErrorCode` additions: `POISON_MISSING`, `COURIER_NO_PENDING_RECORD`,
  `COURIER_POISON_MISMATCH`, `COURIER_TOKEN_EXPIRED`, `COURIER_NO_PINNED_RELAY`,
  `COURIER_NOT_FOUND`.
- Per-drain NACK cap (5) so a flood of junk DMs can't be amplified into a flood
  of outbound replies; excess mismatches are popped + deleted silently.

### Migration

Consumers calling these tools must pass the `poison` they received from the
matching `request_*` call (Studio and the React frontends are updated in lockstep).
Credential-card redemption (`ncred1...`) is unchanged and needs no poison.

## 0.43.0 — 2026-06-06

Hardening from the 2026-06-06 excalibur-mcp double outage: a relay-purged
bootstrap DM masked as a cold start, then a re-provisioning ACL wipe
masked as the same cold start.

### Added

- `ErrorCode.PERSISTENCE_MISCONFIGURED` — paid-tool gate now distinguishes
  permanent SQL errors (SQLSTATE classes 28/3D/42: permission denied,
  missing relation, auth failure) from transient warm-up. Permanent
  errors say so, carry the SQL detail, and skip the pointless retry
  backoff. Transient errors keep the existing `warming_up` recipe.
- `session_status` pricing-layer probe — "ready means ready." The
  lifecycle no longer reports `ready` on vault health alone; the pricing
  model must load too. New `misconfigured` lifecycle state for permanent
  persistence errors.
- `restore_operator_grants` — tenant provisioning now ends with an
  idempotent `GRANT ALL ON ALL TABLES/SEQUENCES IN SCHEMA` to the
  operator role. The ALTER OWNER + REVOKE sequence could strand tables
  with an empty ACL (`relacl = {}`), stripping even the owner's implicit
  privileges.
- Opportunistic bootstrap DM re-publication (the OTS pattern) — the
  Authority re-sends an operator's bootstrap config DM on the back of
  `certify_credits` / `operator_status` traffic when the last send is
  older than 7 days (stamped in `bootstrap_config`, in-process throttle,
  fire-and-forget). Free relays purge kind-4 events (< 69 days observed),
  so a one-shot publication guarantees an eventual cold-start outage.
- `authority` optional extra (`pydantic-settings`) — the
  `tollbooth.authority` subpackage's settings dependency is now declared
  instead of assumed from the deployment.

### Changed

- `NeonQueryError` carries the Postgres SQLSTATE in `.code` when Neon
  supplies one.
- `send_bootstrap_config` parses relay replies strictly per NIP-20 —
  `["OK", id, false, "rate-limited"]` no longer counts as published
  (substring matching on "ok" did).
- `receive_bootstrap_config` polls every relay and lets the newest
  config win, instead of stopping at the first relay that answers — a
  stale DM carries a rotated-away role password, which fails worse than
  no DM at all.
- Authority's `_resend_bootstrap_dm` runs the blocking relay publish in
  a thread and stamps `bootstrap_dm_sent_at` on success.

## 0.42.0 — 2026-06-04

### Added — Claim-check async jobs

Library support for slow Operator tools (LLM round-trips, web-search
generations) that would otherwise hold the MCP connection open past
client timeouts.  Such a tool now returns like any normal MCP tool — it
just returns a *claim check* instead of the end item.  The work runs as
a concurrent asyncio task in the Operator's process; the result (an
opaque JSON blob — the output itself, or a pointer such as
`{"entry_id": ...}`) persists in the Operator's Neon `async_jobs`
table.  The Operator defines a companion tool that redeems the claim.

New `tollbooth/async_jobs.py` (`AsyncJobStore`) plus three
`OperatorRuntime` helpers:

- `register_job_runner(kind, runner)` — map a job kind to the async
  callable that performs the work; registration by name is what lets a
  fresh serverless container resume an orphaned job.
- `start_async_job(kind, npub, params, *, tool_id,
  max_runtime_seconds, result_ttl_seconds)` — persist, spawn, return
  `{claim_check, status: "pending"}`.  Call from inside a `@paid_tool`
  body: the fee is assessed for *requesting* the work.  The durations
  are coded by the tool itself — no operator-level settings.
- `fetch_async_job(claim, npub)` — the companion tool's body.  Free to
  call; every lifecycle state returns guidance
  (`running`/`done`/`error`/`expired`).  Doubles as the watchdog:
  pending or stalled jobs found while polling are re-kicked on the
  current container, so serverless recycles can't strand work.  No
  cron — the patron's own polling drives recovery.

Claims are npub-bound (a claim check alone never unlocks another
patron's job).  Terminal failures (3 attempts) refund the fee via the
existing `rollback_debit` and surface a generic error — raw exception
text never reaches the patron.  Expired results are purged
opportunistically, rate-limited like the OTS check.

Nothing new is exposed on the wire by the wheel: no standard tools, no
discovery mechanism.  In the end there are simply some Operator tools
that return a claim check rather than the end item.

## 0.41.1 — 2026-06-03

### Fixed — Coupon tools' runtime-name resolution

The 7 new coupon tools (`mint_coupon`, `list_coupons`, `update_coupon`,
`delete_coupon`, `redeem_coupon`, `list_my_coupons`, `forget_coupon`)
were registered via `@tool` inside `register_standard_tools` but their
`ToolIdentity` entries were missing from `STANDARD_IDENTITIES`.  That
broke proof verification: `OperatorRuntime.runtime_name(capability)`
calls `mcp_name_for(capability_uuid(capability))`, which returned the
bare UUID when the identity wasn't in the registry.  The Studio signed
the proof for `<slug>_list_coupons`; the wheel checked it against a
UUID string.  Mismatch surfaced as `[proof_invalid] Invalid identity
proof.`

The 7 identities are now in `STANDARD_IDENTITIES` (4 restricted +
3 free).  Operators picking up 0.41.1 see proof verification succeed
without any source changes on their side.

## 0.41.0 — 2026-06-03

### Added — Operator-owned discount coupons

Coupons are now first-class operator objects with their own CRUD
surface, not inline params on a constraint.  Each coupon has a
catchy name (the redemption code), a discount %, a calendar window,
per-patron and aggregate usage caps, and a redemption counter that
the wheel maintains atomically as patrons consume tools.

Patrons redeem a code once via `redeem_coupon`; the wheel auto-applies
the discount on subsequent paid tool calls (no per-call code entry).
The constraint chain references a coupon by id; the runtime pre-loads
the caller's redemption rows before walking the chain and burns one
use per applied coupon when the debit commits.

#### New persistence

- `coupons` and `patron_coupons` tables, both per-operator schema,
  added to `NeonVault.ensure_schema`.  Idempotent — upgrading
  operators pick them up on the first paid tool call.
- New `tollbooth/coupons/` module: `Coupon`, `PatronCoupon`,
  `CouponRedemption`, `CouponRedemptionMap`, `CouponsVault`.

#### New tools (registered by `register_standard_tools`)

Operator-restricted (require proof; resolved against `operator_npub()`):

- `mint_coupon(name, discount_percent, valid_from, valid_until,
  uses_per_patron=1, total_uses=None)`
- `list_coupons()`
- `update_coupon(coupon_id, **patch)` — supports
  `clear_uses_per_patron` / `clear_total_uses` to set caps to NULL
- `delete_coupon(coupon_id)` — cascades to patron redemptions

Patron-facing (free; require proof against caller's npub):

- `redeem_coupon(npub, code)` — idempotent
- `list_my_coupons(npub)` — joined view with status per row
- `forget_coupon(npub, coupon_id)` — cosmetic removal

#### Breaking — `CouponConstraint` shape

`CouponConstraint(coupon_id=...)` replaces the old inline-params
form.  The new constraint is a thin reference; the operator owns the
coupon row, the patron owns the redemption.  Per-tool chains carrying
the old shape fail `from_dict` (missing `coupon_id`) and the step is
skipped — operators rebuild via mint + chain-attach.

#### Gate + runtime

- `ConstraintGate.evaluate_chain[_async]` now returns a 3-tuple
  `(denial, effective_cost, consumed_coupon_ids)`.  The last element
  is the deduped list of coupon ids whose discount applied.
- `ConstraintContext.coupon_redemptions: CouponRedemptionMap | None`
  carries pre-loaded redemption snapshots so the constraint stays
  synchronous.
- `OperatorRuntime.debit_or_deny` burns one use per applied coupon
  after a successful debit (and on free / credit success paths).
  `check_price` pre-loads for accurate preview but never burns.

#### Graceful degradation

Orphaned `coupon_id` references (deleted coupons, unredeemed
patrons) return neutral — chain continues at base price, no denial.

## 0.40.0 — 2026-06-02

### Changed (breaking) — Per-tool constraint chains replace operator-wide pipeline

Pricing rows were already per-tool but constraints were a single
operator-wide `PricingModel.pipeline` with optional `tool_ids` filters
as a secondary scope.  Mixed-grain authoring made pedagogical pricing
ugly: a coupon meant to apply to cheap exploration tools but never to
the heavy judge tool required a global discount narrowed by filter.

This release flattens the model: every constraint is owned by exactly
one tool's `ToolPrice.chain` (ordered list of `PipelineStep`).  The
wheel ships a fixed registry of constraint *types*; operators compose
chains of constraint *instances* per tool.  At debit and preview time,
`ConstraintGate.evaluate_chain[_async]` walks the chain
sequentially: each step's `price_modifier` applies to the running
price; a denial short-circuits the walk.

- `PricingModel.pipeline` field — **removed**.
- `PricingModel.to_constraint_config()` — **removed**.
- `PipelineStep.tool_ids` — **removed** (owning tool is implicit).
- `PipelineStep.patron_npubs` — kept (audience filter within a tool).
- `ToolPrice.chain: list[PipelineStep]` — **added**.
- `PricingModel.chain_for(tool_id)` — **added**.
- `ConstraintEngine` and its `ALL_MUST_PASS` / `ANY_MUST_PASS` /
  `FIRST_MATCH` modes — **deleted**.  There is one mode now: walk the
  chain, apply each step, deny short-circuits.
- `ConstraintGate(config)` static-config constructor — **removed**.
  The gate is constructed with no args and `attach_resolver(resolver)`
  wires it to the runtime's `PricingResolver`.
- `PricingResolver.get_constraint_engine()` — **replaced** by
  `get_chain(tool_id) -> list[PipelineStep]`.
- `load_constraints()` / `validate_config()` — **deleted**.
  `load_constraint(step_dict)` and `validate_step(step_dict)` remain
  as per-step utilities the gate and Studio use.
- `OperatorRuntime.debit_or_deny`: chain-walks via the gate; when the
  chain drives the price below zero, the patron is credited (via
  `UserLedger.credit_deposit(abs(price), "chain_credit:<tool>")`)
  instead of debited.  Skips the insufficient-balance gate for the
  credit case.
- `check_price` preview: same chain-walk for both pricing and effect
  reporting; reports a `credit` effect when the chain drives the
  price negative.
- Existing pre-0.40 models with a top-level `pipeline` key
  deserialize cleanly — the key is silently ignored.  Operators
  re-author chains per tool via the Pricing Studio.



### Changed (breaking protocol) — Secure Courier rendezvous-relay pinning

Every persistent failure of `request_credential_channel` /
`request_npub_proof` had the same root cause: sender and receiver
disagreed on which Nostr relay to use, made worse by individual relay
outages. The symptom — `receive_*` pops N stale DMs and never matches
the actual reply — meant the responder published on a relay the
courier's listener wasn't watching.

`NostrCredentialExchange.open_channel()` now pins the rendezvous to
the specific relay it successfully published the challenge on:

- Iterates the configured relay list in order; the first relay that
  accepts the publish becomes the per-conversation rendezvous.
- The committed relay URL is embedded in the DM body as
  `rendezvous_relay = @@@<wss-url>@@@` so the responder knows where
  to publish their reply.
- On failure the DM body is rebuilt with the next candidate URL and
  resigned (the chicken-and-egg of "embed before publish" — the
  embedded URL is always the one publish actually succeeded on).
- The pinned relay is persisted in a new sibling dict
  `_pinned_relays` and in the `__pending__{service}` vault blob for
  cold-start recovery.
- When every relay rejects the publish the courier raises a new
  `CourierUnreachableError` — a lifecycle state, not a stack trace.
  Callers must re-issue the request after checking relay connectivity.

`request_npub_proof` surfaces the committed `rendezvous_relay` in its
response so MCPs and frontends can display it to the human-in-the-loop
responder. `request_credential_channel` already returns the full
`open_channel` result dict so the field propagates naturally.

The receive-side pin enforcement (subscribe-only-to-pin and
mismatch-rejection) is deliberately not part of this release —
responder cooperation via the embedded URL already eliminates the
asymmetry; full receive-side enforcement is a follow-up once every
deployed responder honors the pin.

### Migration

- **Old responders (clients that ignore `rendezvous_relay`):** still
  work as before — receive subscribes to the full relay list, so a
  reply that happens to land on any configured relay is found. They
  just don't get the routing hint.
- **Cooperating responders (Pricing Studio v1.x with this wheel pin):**
  parse `rendezvous_relay` from the courier DM and display it to the
  user, who configures their Nostr client to publish there. End of
  asymmetry; reply success rate jumps.
- **MCP operators:** no code change required; the new
  `rendezvous_relay` field appears in `request_npub_proof` /
  `request_credential_channel` responses. Pass through to your UI.
- **`CourierUnreachableError` is new:** if your wrapper code catches
  `CourierError` it inherits the new subclass automatically. If you
  catch `Exception` specifically and care about distinguishing
  "unreachable" from other failures, branch on
  `isinstance(exc, CourierUnreachableError)`.

## 0.38.0 — 2026-05-27

### Changed (breaking for tool declarations) — `tool_id` is now an explicit, opaque, frozen field

Prior versions derived each tool's `tool_id` (UUID) from a *string*
inside `@runtime.paid_tool(capability_uuid("X"))` and the matching
`ToolIdentity(capability="X", …)`. Renaming `X` in code — exactly what
TheBrain MCP did in `14892cb` ("UUID-keyed internals" / v1.9.20) — caused
the derived UUID to change, which orphaned every pricing-model row in
Neon that was keyed by the OLD UUID. The wheel's gate would look up
the NEW UUID, miss, and refuse the call. Studio's "repair" added in
pricing-studio 1.9.4 also could not see this class of orphan and was
ultimately found to make things worse for operators where the function
name and capability arg in the decorator had diverged.

The fix removes the derivation step entirely. `tool_id` is now a
required, opaque `str` field on `ToolIdentity` (no longer a `@property`
that computes from `capability`). Operators paste an explicit UUID
constant — generated once via `capability_uuid("X")` or `uuid.uuid4()`
at tool birth — and never touch it again. Renaming any field of the
ToolIdentity, the Python function, or the operator slug leaves the
UUID intact and the pricing-model row keyed correctly forever.

#### Migration for operators

For each tool, declare its current canonical UUID as a module-level
constant and pass it to `ToolIdentity(tool_id=...)`. The UUID values
are exactly what `capability_uuid("<capability>")` produced in 0.37.x,
so existing pricing-model rows continue to match without any data
migration. The wheel itself ships frozen constants for every
`STANDARD_IDENTITIES` entry and for every Authority domain tool. Each
operator (`tollbooth-sample`, Optionality, TheBrain, Schwab, Excalibur,
Taxsort, Shortlinks, dpyc-oracle, OAuth2 Collector) updates their own
`_DOMAIN_TOOLS` the same way and bumps their wheel pin.

A separate Studio rewrite (pricing-studio 1.10) replaces the local
UUID-derivation in `ReconciliationViewModel` with a call to the new
`list_canonical_identities()` tool below.

### Added — `list_canonical_identities()` free tool

Returns `{tools: [{tool_id, mcp_name, category, intent, capability}, …]}`
for every tool in the running wheel's `_tool_registry`. Source of truth
for any client (Studio, agents, FE) that needs to know how this MCP
identifies its tools.

Reserved UUID `e7a9c2f6-1d4b-4c3e-8f7a-5b9d2c1e8f3a` (will not collide
with any historical `capability_uuid("...")` derivation; chosen as a
fresh constant).

### Tests

- `tests/test_tool_identity.py` updated: `tool_id` is checked as an
  explicit value not a derivation; `capability_uuid()` is exercised
  separately as the REPL helper it now is.
- `tests/test_paid_tool.py`, `tests/test_runtime_onboarding.py`: every
  `ToolIdentity(...)` constructor call site provides an explicit
  `tool_id` (using `capability_uuid(name)` as the value, preserving
  test semantics).
- Full suite: 1354 passed.


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

## [0.37.3] — 2026-05-24

### Fixed — `receive_npub_proof` decrypts self-DM replies with ephemeral agent nsec

For self-DM npub proofs (an operator or patron proving ownership of
THEIR OWN npub against an MCP that also runs on that npub),
`request_npub_proof`'s `open_channel` correctly generates an
ephemeral agent nsec and sends the challenge from THAT ephemeral.
The patron's reply is therefore encrypted to the ephemeral's pubkey
using the ephemeral ↔ patron ECDH pair.

But `receive_npub_proof` hardcoded `decrypt_key = exchange._privkey_hex`
(the operator's own nsec), so the ECDH was degenerate (patron and
self with the same key) and every popped candidate returned garbage
or empty plaintext. The user's reply was on the relay, signed by the
correct npub, but the wheel never saw the right plaintext — so the
drain reported "popped N DMs, none matched."

This is the same ephemeral-agent lookup the credential-flow
`receive` already does (nostr_credentials.py:1008-1012). Mirroring
it: look up `exchange._ephemeral_agents.get(poison_key)` and use
that nsec for decryption when present; fall back to the operator's
nsec for non-self-DM flows. No protocol change — just the right
key for the right session.

This explains the long sequence of "Scanned and cleaned N DMs but
none matched" failures during the Optionality recovery dance and
the pricing-model-reset attempts. The wheel was sending from one
npub (ephemeral) and reading with another (operator), exactly as
the user diagnosed.

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

- fix: use correct Horizon env vars for build_info

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

