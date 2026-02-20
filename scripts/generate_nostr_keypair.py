#!/usr/bin/env python3
"""Generate a Nostr keypair for DPYC identity.

Outputs an npub (public key) and nsec (private key) in bech32 format.
The npub is used as your identity in the DPYC Honor Chain:

  - Operators set DPYC_OPERATOR_NPUB in their .env
  - Authorities set DPYC_AUTHORITY_NPUB in their .env
  - Users provide their npub at session time via activate_dpyc()

Requires: pip install nostr-sdk
"""

from __future__ import annotations

import sys

try:
    from nostr_sdk import Keys
except ImportError:
    print("Error: nostr-sdk not installed. Run: pip install nostr-sdk", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    keys = Keys.generate()

    npub = keys.public_key().to_bech32()
    nsec = keys.secret_key().to_bech32()
    hex_pubkey = keys.public_key().to_hex()

    print("=== Nostr Keypair (DPYC Identity) ===")
    print()
    print("npub (public key — share freely, set in .env):")
    print(f"  {npub}")
    print()
    print("nsec (PRIVATE key — back up securely, never commit to git):")
    print(f"  {nsec}")
    print()
    print(f"hex pubkey: {hex_pubkey}")
    print()
    print("--- Environment variable usage ---")
    print()
    print("For an Authority (.env):")
    print(f"  DPYC_AUTHORITY_NPUB={npub}")
    print()
    print("For an Operator (.env):")
    print(f"  DPYC_OPERATOR_NPUB={npub}")


if __name__ == "__main__":
    main()
