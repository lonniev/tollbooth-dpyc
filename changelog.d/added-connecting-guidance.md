- A "Connecting" section in the server instructions of every operator that
  runs holder auth. In one read it tells an agent that speaks raw HTTP how to
  connect:
  - Free tools and `tools/list` need only `Authorization: Bearer anonymous`.
  - Paid tools take either a NIP-98 signed request (`Authorization: Nostr …`
    or `Bearer nostr.…`) or an OAuth holder connection. That connection is
    made through `/register`, then `/authorize` with PKCE S256 (approved at
    once, and answered as JSON with `Accept: application/json`), then
    `/token`. The npub is then proven into it with `<slug>_request_npub_proof`
    and `<slug>_receive_npub_proof`, both of which refuse a connection that
    is not a holder.
