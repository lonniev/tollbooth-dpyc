"""Secure Courier credential tool bodies (audit M2.1g Phase 2).

Extracted verbatim from the courier closures in register_standard_tools. These
orchestrate the Secure Courier (rt.courier()) — the relay drain itself lives in
courier.receive / open_channel / forget. §2-sensitive credential flow, so the
move is behavior-preserving: the only closure variables that became parameters
are ``service_name`` / ``slug`` (used in receive_credentials' rejection DM).
Behavior is pinned by tests/test_courier_tools_characterization.py and the
credential-no-echo (S1) regression test.
"""

from __future__ import annotations

from typing import Any


async def request_credential_channel_tool(
    rt: Any, sender_npub: str, service: str,
) -> dict[str, Any]:
    """Open a Secure Courier channel and send the operator credential template."""
    if not sender_npub:
        return {
            "success": False,
            "error": "sender_npub is required.",
        }
    if not service:
        return {
            "success": False,
            "error": (
                "service is required. Use the credential_service "
                "from get_operator_onboarding_status or get_patron_onboarding_status. Available: "
                + ", ".join(
                    s for s in [
                        rt.operator_credential_service,
                        rt.patron_credential_service,
                    ] if s
                )
            ),
        }
    courier = await rt.courier()
    if courier is None:
        return {"success": False, "error": "Secure Courier not configured."}
    try:
        return await courier.open_channel(
            service,
            greeting=rt._operator_credential_greeting,
            recipient_npub=sender_npub,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def receive_credentials_tool(
    rt: Any,
    sender_npub: str,
    service: str,
    poison: str,
    credential_card: str,
    *,
    service_name: str,
    slug: str,
) -> dict[str, Any]:
    """Pick up operator credentials (poison drain or ncred card) + validate."""
    if not sender_npub:
        return {
            "success": False,
            "error": "sender_npub is required.",
        }
    if not service:
        return {
            "success": False,
            "error": (
                "service is required. Use the same service from "
                "request_credential_channel. Available: "
                + ", ".join(
                    s for s in [
                        rt.operator_credential_service,
                        rt.patron_credential_service,
                    ] if s
                )
            ),
        }
    if not poison and not credential_card:
        from tollbooth.constants import ErrorCode as _EC
        return {
            "success": False,
            "error_code": _EC.POISON_MISSING,
            "error": (
                "poison is required — pass the session phrase returned by "
                "request_credential_channel (or provide a credential_card)."
            ),
        }
    courier = await rt.courier()
    if courier is None:
        return {"success": False, "error": "Secure Courier not configured."}
    try:
        if credential_card:
            # Route through the wrapper, not courier._exchange directly,
            # so credential values are stripped before returning (audit S1).
            result = await courier.redeem_card(
                credential_card, service=service,
            )
        else:
            result = await courier.receive(
                sender_npub, service=service, poison=poison,
            )

        # Validate credentials via operator callback before accepting.
        # The courier strips credentials from the result for security,
        # so reload them from the vault where they were just stored.
        if result.get("success") and service == rt.operator_credential_service and rt._credential_validator:
            try:
                creds = await rt.load_credentials(list(rt._operator_credential_template.fields.keys()))
            except Exception:
                creds = {}
            errors = rt._credential_validator(creds)
            if errors:
                # Reject: forget the bad creds
                try:
                    vault = courier._exchange._credential_vault
                    if vault:
                        await vault.store_credentials(service, rt.operator_npub(), "")
                except Exception:
                    pass
                # DM the sender about the problem
                rejection_msg = (
                    "Credential rejection from " + (service_name or slug) + ":\n\n"
                    + "\n".join(f"  - {e}" for e in errors)
                    + "\n\nPlease correct and resend."
                )
                try:
                    await courier.send(
                        recipient_npub=sender_npub,
                        message=rejection_msg,
                    )
                    result["rejection_dm_sent"] = True
                except Exception:
                    pass
                return {
                    "success": False,
                    "validation_errors": errors,
                    "error": "Credentials received but failed validation: " + "; ".join(errors),
                    "message": "A rejection DM has been sent. Please correct and resend.",
                }
            rt._cashier = None
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def forget_credentials_tool(
    rt: Any, service: str, npub: str, proof: str,
) -> dict[str, Any]:
    """Delete vaulted credentials for (service, npub); proof of ownership required."""
    if not service:
        return {
            "success": False,
            "error": (
                "service is required. Available: "
                + ", ".join(
                    s for s in [
                        rt.operator_credential_service,
                        rt.patron_credential_service,
                    ] if s
                )
            ),
        }
    if err := await rt.require_caller_proof(npub, proof, "forget_credentials"):
        return err
    target_npub = npub
    courier = await rt.courier()
    if courier is None:
        return {"success": False, "error": "Secure Courier not configured."}
    try:
        result = await courier.forget(target_npub, service)
        if service == rt.operator_credential_service:
            rt._cashier = None
        # Fire on_forget callback so operators can clear caches
        if rt._on_forget and result.get("success"):
            try:
                rt._on_forget(service, target_npub)
            except Exception:
                pass
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}


async def request_patron_credentials_tool(
    rt: Any, sender_npub: str,
) -> dict[str, Any]:
    """Open a Secure Courier channel for patron credential delivery."""
    if not sender_npub:
        return {"success": False, "error": "sender_npub is required."}
    courier = await rt.courier()
    if courier is None:
        return {"success": False, "error": "Secure Courier not configured."}
    try:
        return await courier.open_channel(
            rt.patron_credential_service,
            greeting=rt._patron_credential_greeting,
            recipient_npub=sender_npub,
        )
    except Exception as e:
        return {"success": False, "error": str(e)}


async def receive_patron_credentials_tool(
    rt: Any, sender_npub: str, poison: str, credential_card: str,
) -> dict[str, Any]:
    """Pick up patron credentials (poison drain or ncred card)."""
    if not sender_npub:
        return {"success": False, "error": "sender_npub is required."}
    if not poison and not credential_card:
        from tollbooth.constants import ErrorCode as _EC
        return {
            "success": False,
            "error_code": _EC.POISON_MISSING,
            "error": (
                "poison is required — pass the session phrase returned "
                "by request_patron_credentials (or a credential_card)."
            ),
        }
    courier = await rt.courier()
    if courier is None:
        return {"success": False, "error": "Secure Courier not configured."}
    try:
        service = rt.patron_credential_service
        if credential_card:
            # Route through the wrapper, not courier._exchange directly,
            # so credential values are stripped before returning (audit S1).
            return await courier.redeem_card(
                credential_card, service=service,
            )
        return await courier.receive(
            sender_npub, service, poison=poison,
            request_tool="request_patron_credentials",
        )
    except Exception as e:
        return {"success": False, "error": str(e)}
