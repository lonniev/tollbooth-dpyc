# Operator Requests — Adoption, Updates and Leaving

Status: as built in tollbooth-dpyc 0.103. Every operator and Authority inherits it with the SDK pin.

An operator and its Authority are two parties, and each request proves exactly one npub. So
anything that needs both of them is a request by one party and a decision by the other,
recorded in between. The operator asks; the Authority's owner decides, on their own time.

| What | Who acts | How |
|---|---|---|
| Be adopted | Operator asks, Authority owner approves | `request_adoption` → `approve_operator_request(kind="adoption")` |
| Change service URL or display name | Operator asks, Authority owner approves | `update_operator` → `approve_operator_request(kind="update")` |
| Leave, or be removed | Either party alone | `deregister_operator(npub)` |

There is no one-step registration. An Authority joining its own parent takes the same path:
its runtime carries `request_adoption` like any operator's.

## Proof

Each call proves its one npub on the connection (see the README's *Identity & Proofs*):

- **The operator** holds its own nsec, so `request_adoption` and `adoption_status` run on the
  operator's service and reach the Authority over `service_client(authority_url, nsec)`,
  whose every request is signed with the operator's key (NIP-98). The Authority's
  `receive_adoption_request` admits a request only when it is signed by `operator_npub`.
  `request_adoption` itself answers only a request signed with the operator's key.
- **The Authority's owner** acts with requests that prove the Authority's own npub (signed
  with its nsec, as the Pricing Studio does). `list_operator_requests`,
  `approve_operator_request` and `reject_operator_request` refuse anyone else with
  `authority_consent_required`.
- **Registry writes** go from the Authority to the dpyc-oracle on a connection signed with
  the Authority's key; the Oracle (behind `SignedRequestProvider`) accepts
  `register_operator`, `update_operator` and `deregister_operator` only when the request is
  signed by the Authority it names.

## The flow

1. **Ask.** On the operator: `request_adoption(authority_npub, service_url, display_name, note)`.
   The operator resolves the Authority's MCP endpoint from the community registry and calls
   its `receive_adoption_request`, which records a pending row and sends the owner a
   best-effort "check your queue" DM. Nothing is provisioned yet.
2. **Wait.** The operator polls `adoption_status(authority_npub)`; it reads the Authority's
   `get_adoption_status` for this operator. The operator's lifecycle stays `not_registered`.
3. **Decide.** The owner reads `list_operator_requests` (the Pricing Studio's *Pending
   Requests*) and calls `approve_operator_request(operator_npub, kind)` or
   `reject_operator_request(operator_npub, kind, reason)`.
   - An approved **adoption** provisions the operator: ledger row, isolated Neon tenant
     (`op_{hash}` schema with its own LOGIN role) and the registry entry via the Oracle, then
     the bootstrap DM. The operator moves `warming_up` → `ready`.
   - An approved **update** applies the requested service URL and/or display name to the
     registry entry.
4. **Leave.** `deregister_operator(npub)` passes when the request proves `npub` (the operator
   leaving) or the Authority's npub (the owner removing it). It removes the registry entry
   and closes the adoption row. The ledger and the Neon tenant stay, so a later adoption
   resumes where it left off.

A rejected operator may ask again; asking again refreshes the row to pending.

## State — the Authority's Neon vault

`tollbooth.authority.operator_requests` keeps one table per kind of request,
`operator_adoption_requests` and `operator_update_requests`, each keyed by `operator_npub`:

```
operator_npub  TEXT PRIMARY KEY
service_url    TEXT NOT NULL DEFAULT ''
display_name   TEXT NOT NULL DEFAULT ''
status         TEXT NOT NULL DEFAULT 'pending'   -- pending|rejected|provisioned|applied|closed
note           TEXT NOT NULL DEFAULT ''
requested_at   TIMESTAMPTZ DEFAULT now()
decided_at     TIMESTAMPTZ
expires_at     TIMESTAMPTZ                        -- requested_at + 7 days
```

`provisioned` closes an approved adoption, `applied` an approved update, and `closed` a
deregistered operator. Rows are durable, so concurrent requests and serverless cold starts
are both fine. The owner DM is a convenience; the table is the source of truth.
