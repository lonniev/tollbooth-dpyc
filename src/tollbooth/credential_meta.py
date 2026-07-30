"""Per-field delivery metadata for vaulted credentials.

Credential blobs remain a JSON object keyed by field name → value.  A single
reserved key, ``__meta__``, holds non-secret bookkeeping that must never be
treated as a credential field:

    {
      "api_key": "…",
      "api_secret": "…",
      "__meta__": {
        "delivered_at": {
          "api_key": "2026-07-30T12:34:56.789012+00:00",
          "api_secret": "2026-07-30T12:34:56.789012+00:00"
        }
      }
    }

``delivered_at`` answers "how old is this secret?" after a Secure Courier
receipt (or any other field write that counts as delivery).  Legacy blobs
written before this existed simply have no timestamps — callers see ``None``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any, TypedDict

# Reserved top-level key.  Must never surface as a field name in listings,
# onboarding classification, or credential-card payloads.
class CredentialFieldDetail(TypedDict):
    """One row of a credential-field listing: the name, and when it arrived.

    A plain ``dict[str, str | None]`` cannot say that only ONE of these is nullable, so
    every read of ``["field"]`` widened to ``str | None`` and infected its callers. The
    field name always exists; ``delivered_at`` is None for blobs vaulted before timestamps
    were recorded.
    """

    field: str
    delivered_at: str | None


META_KEY = "__meta__"
_DELIVERED_AT = "delivered_at"


def now_iso() -> str:
    """UTC timestamp in ISO-8601 with offset (the handoff's contract)."""
    return datetime.now(UTC).isoformat()


def is_meta_key(name: str) -> bool:
    """True for the reserved metadata slot (and any future ``__…__`` bits)."""
    return name == META_KEY or (name.startswith("__") and name.endswith("__"))


def credential_field_names(blob: Mapping[str, Any] | None) -> list[str]:
    """Field names only — never exposes ``__meta__``."""
    if not blob:
        return []
    return [k for k in blob if not is_meta_key(k)]


def strip_meta(blob: Mapping[str, Any] | None) -> dict[str, str]:
    """Return plain credential values (stringified), dropping reserved keys.

    Non-string values are coerced with ``str()`` so callers that expect
    ``dict[str, str]`` keep working even if a legacy blob grew odd types.
    """
    if not blob:
        return {}
    out: dict[str, str] = {}
    for key, value in blob.items():
        if is_meta_key(key):
            continue
        out[key] = value if isinstance(value, str) else str(value)
    return out


def get_delivered_at(blob: Mapping[str, Any] | None, field: str) -> str | None:
    """ISO-8601 delivery time for *field*, or ``None`` if never recorded."""
    if not blob or is_meta_key(field):
        return None
    meta = blob.get(META_KEY)
    if not isinstance(meta, dict):
        return None
    stamps = meta.get(_DELIVERED_AT)
    if not isinstance(stamps, dict):
        return None
    value = stamps.get(field)
    return value if isinstance(value, str) and value else None


def delivered_at_map(blob: Mapping[str, Any] | None) -> dict[str, str]:
    """``{field: iso}`` for every field that has a recorded delivery time.

    Only includes fields still present in the blob (stale stamps for deleted
    fields are ignored).
    """
    names = set(credential_field_names(blob))
    if not blob or not names:
        return {}
    meta = blob.get(META_KEY)
    if not isinstance(meta, dict):
        return {}
    stamps = meta.get(_DELIVERED_AT)
    if not isinstance(stamps, dict):
        return {}
    return {
        name: ts
        for name, ts in stamps.items()
        if name in names and isinstance(ts, str) and ts
    }


def apply_delivery_timestamps(
    field_values: Mapping[str, Any],
    *,
    previous: Mapping[str, Any] | None = None,
    just_delivered: Iterable[str] | None = None,
    when: str | None = None,
) -> dict[str, Any]:
    """Build a store blob: plain values + ``__meta__.delivered_at`` stamps.

    Args:
        field_values: The credential fields to persist.  May still contain a
            leftover ``__meta__`` key (which is ignored — we rebuild it).
        previous: Prior blob, used only to carry forward timestamps for fields
            not in *just_delivered*.
        just_delivered: Field names whose values arrived on this write.  Each
            receives a fresh ``delivered_at``.  When ``None``, every field
            whose value is new or changed relative to *previous* is stamped;
            fields whose value is unchanged keep their prior timestamp.
        when: ISO-8601 timestamp to record.  Defaults to ``now_iso()``.

    Returns:
        A blob ready for ``json.dumps`` / vault store.  ``__meta__`` is omitted
        entirely when no timestamps did remain (keeps legacy empty blobs tidy).
    """
    plain = strip_meta(field_values)
    stamp_at = when if when is not None else now_iso()

    prior_stamps: dict[str, str] = {}
    if previous:
        prior_stamps = delivered_at_map(previous)

    if just_delivered is None:
        prior_plain = strip_meta(previous)
        to_stamp = {
            name
            for name, value in plain.items()
            if prior_plain.get(name) != value
        }
    else:
        to_stamp = {name for name in just_delivered if name in plain and not is_meta_key(name)}

    new_stamps: dict[str, str] = {}
    for name in plain:
        if name in to_stamp:
            new_stamps[name] = stamp_at
        elif name in prior_stamps:
            new_stamps[name] = prior_stamps[name]
        # else: present but never stamped (legacy carry) — leave absent

    blob: dict[str, Any] = dict(plain)
    if new_stamps:
        blob[META_KEY] = {_DELIVERED_AT: new_stamps}
    return blob
