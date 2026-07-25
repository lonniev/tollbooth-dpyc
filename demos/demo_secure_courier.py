#!/usr/bin/env python3
"""Live demo of the Secure Courier Service.

Generates a fresh operator keypair, prints the npub for you to DM,
then listens on real Nostr relays for your encrypted message.

Usage:
    python demos/demo_secure_courier.py

Then send a DM from Primal to the printed npub containing:
    {"api_key": "test-key-123", "api_secret": "test-secret-456"}
"""

import asyncio

from pynostr.key import PrivateKey

from tollbooth.credential_templates import CredentialTemplate, FieldSpec
from tollbooth.nostr_credentials import (
    CourierTimeout,
    CourierValidationError,
    NostrCredentialExchange,
)

RELAYS = [
    "wss://relay.primal.net",
    "wss://relay.damus.io",
    "wss://nos.lol",
]

TEMPLATES = {
    "test": CredentialTemplate(
        service="test",
        version=1,
        fields={
            "api_key": FieldSpec(required=True, sensitive=True),
            "api_secret": FieldSpec(required=True, sensitive=True),
        },
        description="Test credentials for Secure Courier demo",
    ),
}


async def main():
    # Generate fresh operator keypair
    operator_key = PrivateKey()
    print("=" * 60)
    print("  SECURE COURIER SERVICE — Live Demo")
    print("=" * 60)
    print()
    print(f"  Operator npub: {operator_key.public_key.bech32()}")
    print()
    print("  Send a DM from Primal to the npub above containing:")
    print()
    print('    {"api_key": "test-key-123", "api_secret": "test-secret-456"}')
    print()
    print(f"  Listening on {len(RELAYS)} relays...")
    print("  Freshness window: 600 seconds")
    print()

    exchange = NostrCredentialExchange(
        nsec=operator_key.nsec,
        relays=RELAYS,
        templates=TEMPLATES,
        freshness_window=600,
    )

    # Open channel (starts subscription)
    _channel = await exchange.open_channel(
        "test", greeting="Hi — this is the Secure Courier demo.",
    )
    print("  Channel open. Waiting for your DM...")
    print()

    # Poll for the DM
    sender_npub = input("  Enter your npub (or press Enter for default): ").strip()
    if not sender_npub:
        sender_npub = "npub1l94pd4qu4eszrl6ek032ftcnsu3tt9a7xvq2zp7eaxeklp6mrpzssmq8pf"
    print()
    print(f"  Looking for DM from: {sender_npub}")
    print()

    max_attempts = 12
    for attempt in range(1, max_attempts + 1):
        try:
            result = await exchange.receive(sender_npub)
            print("  " + "=" * 56)
            print("  POUCH RECEIVED!")
            print("  " + "=" * 56)
            print(f"  Service:    {result['service']}")
            print(f"  Fields:     {result['fields_received']}")
            print(f"  Sensitive:  {result['sensitive_fields']}")
            print(f"  Encryption: {result['encryption']}")
            print()
            # Show credentials (redacted for sensitive)
            for key, value in result["credentials"].items():
                display = value[:4] + "..." if len(value) > 4 else value
                print(f"  {key}: {display}")
            print()
            print(f"  {result['message']}")
            print()
            return

        except CourierTimeout:
            print(f"  Attempt {attempt}/{max_attempts} — no DM yet, retrying in 5s...")
            await asyncio.sleep(5)

        except CourierValidationError as e:
            print(f"  Validation error: {e}")
            return

    print("  Timed out waiting for DM. Make sure you sent it to the right npub.")


if __name__ == "__main__":
    asyncio.run(main())
