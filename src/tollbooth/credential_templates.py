"""Credential template schema definition and validation.

Operators define strict credential templates — no freeform JSON accepted.
Each template declares the expected fields (required/optional, sensitive/not)
and a service name for matching incoming payloads.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FieldSpec:
    """Specification for a single credential field."""

    required: bool = True
    sensitive: bool = True


@dataclass(frozen=True)
class CredentialTemplate:
    """Schema for a service's credential payload.

    Operators declare expected fields; incoming payloads are validated
    against this schema.  Unknown fields are silently ignored, required
    fields enforced, and the ``description`` is rendered for the end user.
    """

    service: str
    version: int
    fields: dict[str, FieldSpec]
    description: str = ""


class TemplateValidationError(ValueError):
    """Raised when a credential payload fails template validation."""


def validate_payload(
    payload: dict,
    template: CredentialTemplate,
) -> dict[str, str]:
    """Validate a credential payload against a template.

    Args:
        payload: Decoded JSON dict from the Nostr DM.
        template: The operator-defined credential template.

    Returns:
        Validated payload dict (only the declared fields, stripped of any
        ``service`` or ``version`` metadata keys).

    Raises:
        TemplateValidationError: On missing required fields, unknown fields,
            or type mismatches.
    """
    # Allow optional service/version metadata in the payload
    metadata_keys = {"service", "version"}
    payload_fields = {k: v for k, v in payload.items() if k not in metadata_keys}

    # Silently drop fields not declared in the template
    known = set(template.fields.keys())
    payload_fields = {k: v for k, v in payload_fields.items() if k in known}

    # Check required fields
    missing = []
    for name, spec in template.fields.items():
        if spec.required and name not in payload_fields:
            missing.append(name)
    if missing:
        raise TemplateValidationError(
            f"Missing required fields: {', '.join(sorted(missing))}"
        )

    # Validate types (all values must be strings)
    for name, value in payload_fields.items():
        if not isinstance(value, str):
            raise TemplateValidationError(
                f"Field '{name}' must be a string, got {type(value).__name__}"
            )
        if template.fields[name].required and not value.strip():
            raise TemplateValidationError(
                f"Required field '{name}' must not be empty"
            )

    return payload_fields


def render_credential_payload_lines(template: CredentialTemplate) -> list[str]:
    """Return only the ``key = @@@placeholder@@@`` lines for a template.

    Used by the welcome DM builder to embed the payload section inside a
    larger structured message without duplicating the Service/Description
    header that :func:`render_delimited_instructions` includes.
    """
    lines: list[str] = []
    for name in template.fields:
        placeholder = f"PASTE_YOUR_{name.upper()}_HERE"
        lines.append(f"  {name} = @@@{placeholder}@@@")
    return lines


def render_delimited_instructions(template: CredentialTemplate) -> str:
    """Render a credential template using @@@ delimiters (mobile-friendly).

    Produces a text block where each field is a simple ``key = @@@value@@@``
    line.  Much easier to fill in on a phone than composing JSON.
    """
    lines = [
        f"Service: {template.service} (v{template.version})",
    ]
    if template.description:
        lines.append(f"Description: {template.description}")
    lines.append("")
    lines.append("Reply with your credentials — paste each value between the @@@ markers:")
    lines.append("")

    for name, spec in template.fields.items():
        placeholder = f"PASTE_YOUR_{name.upper()}_HERE"
        lines.append(f"  {name} = @@@{placeholder}@@@")

    return "\n".join(lines)


def render_template_instructions(template: CredentialTemplate) -> str:
    """Render a credential template as human-readable DM instructions.

    Produces a text block the user can reference when composing their
    Nostr DM payload.
    """
    lines = [
        f"Service: {template.service} (v{template.version})",
    ]
    if template.description:
        lines.append(f"Description: {template.description}")
    lines.append("")
    lines.append("Send a JSON object as a Nostr DM with these fields:")
    lines.append("")

    for name, spec in template.fields.items():
        req = "REQUIRED" if spec.required else "optional"
        lines.append(f'  "{name}": "<your value>"  ({req})')

    lines.append("")
    lines.append("Example:")
    example = {}
    for name, spec in template.fields.items():
        example[name] = f"your_{name}_here"
    import json

    lines.append(f"  {json.dumps(example, indent=2)}")

    return "\n".join(lines)
