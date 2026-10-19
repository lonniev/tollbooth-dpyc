# Identity Proofs

How a DPYC service knows which npub a request acts for: what the design is for, how it
works, and how to use it as an agent, a web page, a keyed program or an operator.

Current as of `tollbooth-dpyc` 0.104.1. The code is the authority:
`holder_auth.py`, `vaults/holder_store.py`, `tools/proof.py`, `identity_proof.py`,
`identity_credential.py`, `proven_npub.py`.

---

## 1. Intention

- **The patron's key stays with the patron.** An npub's nsec never reaches a service, a
  website or an AI assistant. A service learns that a request may act for an npub from a
  signature that key made.
- **A keyless caller can still act for a patron.** An AI assistant or a web page holds no
  nsec. The patron lets it act for their npub by answering one Nostr DM from their own
  Nostr client, and chooses how long that permission lasts.
- **Proof travels on the connection.** Tool arguments are untrusted input, so they carry
  only the npub a call acts for. The `Authorization` header carries the proof, and nothing
  that proves an npub ever enters a model's transcript.
- **One npub per request.** Every request acts for exactly one npub. An action that needs
  two parties is a request by one and a recorded approval by the other.
- **An npub is a key.** The protocol proves possession of a key and says nothing about who
  holds it. One person may hold several npubs, and one connection may act for several,
  one per request.
- **The patron can take it back.** `forget_credentials` revokes every grant the patron ever
  issued, on every connection.

## 2. Two ways a connection proves an npub

Every tool takes `npub` and its own arguments. The HTTP `Authorization` header decides
whether the request may act for that npub:

| Header | Who uses it | What proves the npub |
|---|---|---|
| `Nostr <base64 kind-27235 event>` | a caller holding the nsec: an operator calling its Authority, an agent with a keyring, Pricing Studio | the signature on this one request |
| `Bearer dpyc_ha_…` | a caller holding no key: an AI assistant, a web page, a headless worker | a grant the patron's DM reply put in this connection's wallet |
| anything else, or none | anyone | nothing: the request is anonymous; free tools answer, paid tools return `proof_required` |

### 2.1 Signed request (NIP-98)

The caller signs each HTTP request with the npub's key: a kind-27235 event whose tags name
the URL (`u`), the `method`, the SHA-256 of the body (`payload`) and a random `nonce`. The
service checks the Schnorr signature, the method, the path and the body hash, and accepts
the event within 60 seconds, once. The signature authorizes that request and its exact
arguments. Nothing is stored and nothing expires later.

### 2.2 Holder connection (OAuth 2.1)

A **holder** is a connection with no npub of its own. It carries a **wallet**: for each npub
proven into it, the grant that proves it.

**Connecting.** The service is its own authorization server:

1. `POST /register` registers the client (dynamic client registration).
2. `GET /authorize` is a conventional OAuth 2.1 authorization request (Authorization Code
   with PKCE). There is no consent screen: a new holder holds nothing, so the service
   approves at once and responds with a brief authorization code as a parameter of the
   `redirect_uri`; the client exchanges it for the bearer token in the subsequent
   `POST /token` request (step 3). The client's `Accept` header determines how the
   MCP authorization server returns the `redirect_uri`:
   - **Default:** a standard `302` redirect. A program reads the code from the `Location`
     header and does not follow it.
   - **`Accept: application/json`:** the same redirect, delivered as a `200` whose body is
     `{"redirect_uri": "<that URL>"}`. A browser's `fetch` follows a `302` on its own and
     cannot read where a cross-origin one points, so the JSON form lets a web page take the
     code without navigating away: no page flash, and no browser needed at all.

   The code is good only with the PKCE verifier the client kept.
3. `POST /token` exchanges the code for an access token and a refresh token.

The service mints a **holder id** at step 2: a random identifier kept only on the service,
never shown to the client. Both tokens resolve to it.

**Proving an npub into the wallet** (the Secure Courier exchange):

1. `request_npub_proof(patron_npub, reason?, verify_at?)` sends the patron a DM and returns
   a one-time `nonce`. The challenge is bound to this holder.
2. The patron reads the DM in their own Nostr client and replies. The reply carries the
   nonce and may say how long the permission lasts, e.g. `cache_duration = @@@7 days@@@`
   (`2h`, `1d`, `two weeks`, `unlimited` all parse). The patron's key signs the reply.
3. `receive_npub_proof(patron_npub, nonce)` finds that reply on the pinned relay, verifies
   the patron's signature, mints a grant and puts it in this holder's wallet. The reply to
   the caller says the npub is held and until when. It carries no grant.

From then on the caller passes only `npub`.

## 3. The nonce

The nonce is a short phrase, such as `keen-eagle-58`. It is not a secret. Its job is to tie
one patron reply to one request:

- `request_npub_proof` issues it and the challenge DM shows it.
- The patron's reply must carry it, so `receive_npub_proof` can pick that reply out of
  everything on the relay, and replies carrying a different nonce are refused.
- After the match, the pending exchange is discarded and the nonce's job is done. The
  grant records `sha256(nonce)` in its `challenge` tag, naming the exchange the patron
  answered.
- Nobody stores, checks or presents it again. Someone who learns a nonce and the npub
  gains nothing: on any other connection `receive_npub_proof` answers
  `courier_nonce_mismatch`, and paid calls stay `proof_required`.

The Secure Courier credential flow (`request_credential_channel` /
`receive_credentials`) uses a nonce the same way, to pick the patron's credential reply.

## 4. The grant

An Operator-signed Nostr event of kind 30080. It binds:

- the patron's npub;
- the patron's verified reply signature and event id, the proof of possession it attests;
- `sha256(nonce)`, the exchange answered;
- an expiry, from the patron's `cache_duration`.

It is stored encrypted in the holder's wallet as `{"grant": <event>}` and never leaves the
service. On every call that names the npub, the service checks:

1. the grant is signed by this service's own key;
2. it names the npub the call acts for;
3. it has not expired;
4. it was issued after the patron's last `forget_credentials` (the revocation watermark).

A connection reaches a wallet only through its own holder id, so one holder can never use
another's grant.

## 5. Lifetimes

Two clocks run independently:

| What | Lifetime | When it ends |
|---|---|---|
| Authorization code | 5 minutes | exchanged once at `/token` |
| Access token | 1 hour | the client refreshes; FastMCP answers 401 to prompt it |
| Refresh token | 30 days, single use | replaced by a new one on every refresh |
| Holder id | as long as the client keeps refreshing | the client revokes it (`/revoke`), or a spent refresh token is presented: either forgets the holder, its tokens and its wallet |
| Grant for an npub | the patron's choice: default 2 hours, at most 30 days | calls for that npub return a proof error; the connection stays live |

So the access token is a short-lived, renewable handle on one persistent connection, while
each npub's permission lasts exactly as long as its patron said. A lapsed grant needs only a
new DM exchange on the same connection.

## 6. Actions with more than one party

A request proves one npub. A two-party action is a workflow:

- **Adoption:** the operator calls `request_adoption` on its own runtime, signed with its
  nsec; the Authority's owner sees it in `list_operator_requests` and calls
  `approve_operator_request` or `reject_operator_request`.
- **Registry changes:** `update_operator` records a request the Authority approves the same
  way.
- **Leaving:** `deregister_operator` is a single-party act by either side.

Each approval is one npub's own request. A tool needing three npubs takes three approvals,
each proven by that npub. One connection holding all three grants may make all three
requests.

## 7. Usage

### An AI agent (keyless)

1. Ask the patron for their npub. Never ask for an nsec.
2. `check_proof_status(patron_npub)`. If `valid`, go to step 5.
3. `request_npub_proof(patron_npub)`. Tell the patron to reply to the DM from their Nostr
   client, and name a duration if they want one. Then **wait** for them to say they replied.
4. `receive_npub_proof(patron_npub, nonce)`, once. Each call drains the relay; calling before
   the patron replies loses the reply.
5. Call paid tools with `npub` and their arguments.

Statuses and errors:

| Seen | Meaning | Do |
|---|---|---|
| `check_proof_status` → `valid` | the connection proves the npub; `expires_in_seconds` says for how long | call tools |
| `not_held` | this connection has never proven it | steps 3–4 |
| `expired`, `revoked`, `invalid` | the grant lapsed, the patron revoked it, or it does not verify | steps 3–4 |
| `proof_invalid` on a paid call | the same three, met mid-session; `reason` says which | steps 3–4 |
| `proof_required` on a paid call | the connection does not hold the npub | steps 3–4 |
| `courier_not_found` | no reply carrying the nonce yet | confirm the patron replied; never poll |
| `courier_nonce_mismatch` | the nonce is not this holder's open challenge | `request_npub_proof` again |

`held_identities()` lists every npub the connection holds, with expiry. Claude.ai and other
MCP clients do the OAuth connect themselves; the agent only runs steps 1–5.

### A web page

`@tollbooth-dpyc/web`: `connectAsHolder()` registers, authorizes by fetch and exchanges the
code in place; `holderBearer()` supplies the access token, refreshing as needed;
`forgetHolder()` revokes the connection, and the service forgets it and every npub it held.

### A program holding the nsec

```python
from tollbooth.identity_proof import sign_request
from tollbooth.mcp_client import service_client

client = service_client(url, nsec)                              # signs every request
header = sign_request(nsec, method="POST", url=url, body=body)  # one request, by hand
```

An agent whose runner holds an nsec can use `python -m tollbooth.agent_keyring`: a local
MCP proxy that signs every forwarded request, with the nsec out of the agent's reach.

### An operator

`OperatorRuntime(holder_auth_url="https://<your-host>")` turns on both proofs, the OAuth
endpoints and the holder wallet. Restricted and paid tools then check the caller with
`debit_or_deny` as usual. A service that keeps no holders, such as the dpyc-oracle,
installs `SignedRequestProvider`: signed requests prove their npub and every other request
is anonymous.

## 8. What a leak gives away

| Leaked | Gives |
|---|---|
| a nonce, with the npub | nothing |
| a signed request | that identical request, once, within 60 s |
| an access token | the holder's npubs until the token expires (≤ 1 hour) |
| a refresh token | the holder, until the real client next refreshes: that reuse forgets the holder for both |
| an nsec | that npub, everywhere: rotate the key |

The patron's `forget_credentials` revokes every grant they issued, wherever it is held.

## 9. Further reading

- [README: Identity & Proofs](../README.md#identity--proofs), the short version.
- [Threat model](https://github.com/lonniev/dpyc-community/blob/main/docs/THREAT-MODEL.md),
  rows S-1 to S-9 and E-3 to E-5.
- [Patent revision log](https://github.com/lonniev/dpyc-community/blob/main/docs/patent/REVISIONS.md),
  Revisions 2–4: how the protocol evolved and why.
