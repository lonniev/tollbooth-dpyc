#!/usr/bin/env python3
"""Live test: generate ephemeral keypair, send DM, wait for reply.

1. Generate a fresh ephemeral Nostr keypair
2. Send a NIP-04 DM to the target npub with a @@@ payload
3. Subscribe for replies addressed to the ephemeral npub
4. Print any replies received
"""

import json
import sys
import time

from pynostr.key import PrivateKey

from tollbooth.nostr_credentials import NostrCredentialExchange
from tollbooth.credential_templates import CredentialTemplate, FieldSpec

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    stream=sys.stderr)
logger = logging.getLogger(__name__)

def p(msg):
    """Print with flush."""
    print(msg, flush=True)

TARGET_NPUB = "npub15e3ver3pskqsuqzszrp9mxpks7tqg7kedzyhmxfwlmu2jf8enaas86ll75"
RELAYS = ["wss://relay.damus.io", "wss://nos.lol", "wss://relay.primal.net"]

def main():
    # Step 1: Generate ephemeral keypair
    ephemeral = PrivateKey()
    eph_npub = ephemeral.public_key.bech32()
    eph_hex = ephemeral.public_key.hex()

    print("=" * 60)
    print(f"Ephemeral npub:  {eph_npub}")
    print(f"Ephemeral hex:   {eph_hex}")
    print(f"Target npub:     {TARGET_NPUB}")
    print("=" * 60)

    # Create a minimal exchange using the ephemeral key
    templates = {
        "test": CredentialTemplate(
            service="test",
            version=1,
            fields={
                "answer": FieldSpec(required=True, sensitive=False),
            },
            description="Live ephemeral DM test",
        ),
    }

    ex = NostrCredentialExchange(
        nsec=ephemeral.nsec,
        relays=RELAYS,
        templates=templates,
    )

    # Step 2: Send DM with @@@ payload
    payload_text = (
        "Hello from ephemeral agent!\n\n"
        "Please reply with:\n"
        "  answer = @@@YOUR_REPLY_HERE@@@\n\n"
        f"Session phrase: bright-owl-42\n"
        f"Reply to this npub: {eph_npub}"
    )

    print(f"\nSending DM to {TARGET_NPUB}...")
    try:
        ex.send_dm(TARGET_NPUB, payload_text)
        print("DM sent successfully!")
    except Exception as exc:
        print(f"ERROR sending DM: {exc}")
        sys.exit(1)

    # Step 3: Wait for reply
    print(f"\nWaiting for reply addressed to {eph_npub}...")
    print("(Will check every 15 seconds for up to 5 minutes)")
    print("=" * 60)

    # Decode target npub to hex via pynostr
    from pynostr.key import PublicKey
    target_pub = PublicKey.from_npub(TARGET_NPUB)
    target_hex = target_pub.hex()

    print(f"Target hex: {target_hex}")
    print(f"Ephemeral hex: {eph_hex}")

    max_wait = 300  # 5 minutes
    interval = 15
    start = time.time()

    while time.time() - start < max_wait:
        elapsed = int(time.time() - start)
        print(f"\n--- Check at {elapsed}s ---")

        # Fetch from relays
        try:
            ex._fetch_dms_from_relays()
        except Exception as exc:
            print(f"  Relay fetch error: {exc}")

        # Check buffer
        with ex._lock:
            event_count = len(ex._received_events)
            print(f"  Events in buffer: {event_count}")

            for i, event in enumerate(ex._received_events):
                eid = event.get("id", "?")[:16]
                kind = event.get("kind", "?")
                pubkey = event.get("pubkey", "?")[:16]
                ptags = [t[1][:16] for t in event.get("tags", []) if t[0] == "p"]
                created = event.get("created_at", 0)
                age = int(time.time()) - created if created else "?"
                print(f"  [{i}] kind={kind} from={pubkey}... p={ptags} age={age}s id={eid}...")

                # Try to decrypt if it's from our target
                if event.get("id") not in ex._consumed_ids:
                    try:
                        pt = ex._decrypt_dm(event, target_hex)
                        print(f"  >>> DECRYPTED: {pt[:200]}")
                    except Exception as exc:
                        print(f"  >>> Decrypt failed: {exc}")

        if event_count > 0:
            # Check if any have valid content
            candidates = ex._find_dm_candidates(target_hex)
            if candidates:
                print(f"\n  Found {len(candidates)} candidate(s) from target!")
                # Don't consume — just report
                break

        remaining = max_wait - (time.time() - start)
        if remaining > interval:
            print(f"  Waiting {interval}s... ({int(remaining)}s remaining)")
            time.sleep(interval)
        else:
            break

    print("\n" + "=" * 60)
    print("Done. Buffer contents:")
    with ex._lock:
        for i, event in enumerate(ex._received_events):
            print(f"  [{i}] {json.dumps(event, indent=2)[:300]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
