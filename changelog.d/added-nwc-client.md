- `tollbooth.nwc`: a Nostr Wallet Connect (NIP-47) client, so an Operator can ask a
  patron's own wallet to pay an invoice. Three calls: `get_info`, `pay_invoice` and
  `lookup_invoice`. `pay_invoice` takes the agreed `amount_msat` and refuses, before
  anything is sent, an invoice that asks for any other amount. A request that went out
  and got no answer raises `NwcOutcomeUnknown` and is never sent again on another relay.
- `tollbooth.bolt11`: `decode` reads an invoice's amount, payment hash and expiry, and
  verifies its checksum. Checked against the BOLT11 specification's own examples.
- `scripts/nwc_spike.py`: a hand-run check that pays a Lightning address from a real
  wallet. The connection string is typed at a hidden prompt.
