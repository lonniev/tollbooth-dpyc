"""Properties: named, typed, non-secret values an npub owns and tunes.

An operator chooses which LLM model each of its uses runs on; a patron chooses
units, a theme, whether the tour runs. Neither is a secret, neither is
required, and each should follow its owner from device to device. Credentials
were the only place to keep such a thing, and they are a secret store: sealed,
listed only by name, declared for delivery by Secure Courier. Properties are
the plain counterpart.

**Declared in code, valued by the owner.** The operator declares what may be
tuned (`PropertySpec`: its kind, its default, its words); the owner sets one
row. Unset means the default, so declaring a property changes nothing until
somebody tunes it.

**Two owners.** ``operator_properties`` belong to the operator npub: anyone may
read them (which model a fare buys is fair to know), only the operator writes
them. ``patron_properties`` belong to each patron npub: only that patron reads
or writes theirs — a timezone says where someone lives.

**Account or device.** A property ``follows`` the account (kept here, on every
device) or the device (kept in that browser alone, never sent here). The
frontend reads both kinds the same way; this module stores only the first.

Stored in the clear in the operator's own schema, as business rows are.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any

from tollbooth.llm_route import TIER_READER, TIER_WRITER

logger = logging.getLogger(__name__)

KINDS = ("llm_model", "choice", "list", "number", "flag", "text")
FOLLOWS = ("account", "device")

TABLE = "actor_properties"

#: How long a process trusts an operator property it read, as the pricing model.
CACHE_TTL_S = 60.0

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9._-]*/[a-z0-9][a-z0-9._:-]*$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_ITEM = re.compile(r"^[a-z0-9_:-]{1,40}$")
_MAX_TEXT = 200
_MAX_SLUG = 120
_MAX_LIST = 50


class PropertyError(ValueError):
    """A property name or value that cannot be kept, said plainly."""


@dataclass(frozen=True)
class PropertySpec:
    """What may be tuned, and how.

    ``kind``:
      - ``llm_model`` — a tier (``writer``, ``reader``) or a provider slug
        (``vendor/model``); ``suggestions`` are offered, any valid slug is kept.
      - ``choice`` — one of ``choices``.
      - ``list`` — some of ``choices``, in the owner's order; with no
        ``choices``, short plain names (lowercase words, digits, ``_:-``).
      - ``number`` — finite, within ``minimum``..``maximum``.
      - ``flag`` — true or false.
      - ``text`` — at most 200 characters, no control characters.
    """

    kind: str
    #: ``None`` means none set: the owner has not chosen, and the reader decides.
    default: Any
    label: str
    description: str = ""
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    suggestions: tuple[str, ...] = ()
    follows: str = "account"

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown property kind {self.kind!r}; use one of {KINDS}")
        if self.follows not in FOLLOWS:
            raise ValueError(f"follows must be one of {FOLLOWS}")
        if self.kind == "choice" and not self.choices:
            raise ValueError("a choice property needs its choices")
        if self.default is not None:
            validate(self, self.default)  # a default that would be refused is a bug

    def describe(self) -> dict[str, Any]:
        """The spec as a page renders it."""
        out: dict[str, Any] = {"kind": self.kind, "label": self.label,
                               "description": self.description, "default": self.default,
                               "follows": self.follows}
        if self.choices:
            out["choices"] = list(self.choices)
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        if self.suggestions:
            out["suggestions"] = list(self.suggestions)
        return out


def llm_model_property(label: str, description: str = "", *, default: str = TIER_WRITER,
                       suggestions: tuple[str, ...] = ()) -> PropertySpec:
    """Which model one use of the LLM runs on. ``default`` is a tier, so an
    unset property runs on whatever the deployment's tier names, as before."""
    return PropertySpec(kind="llm_model", default=default, label=label,
                        description=description, suggestions=suggestions)


def validate(spec: PropertySpec, raw: Any) -> Any:
    """``raw`` as the value ``spec`` keeps, or ``PropertyError``.

    Treats ``raw`` as adversarial: it arrives from a tool argument. A string
    is accepted for every kind, since that is what a form sends.
    """
    k = spec.kind
    if k == "llm_model":
        v = raw.strip() if isinstance(raw, str) else ""
        if v.lower() in (TIER_WRITER, TIER_READER):
            return v.lower()
        if len(v) > _MAX_SLUG or not _SLUG.match(v):
            raise PropertyError(f"{spec.label}: give a tier ({TIER_WRITER}, {TIER_READER}) "
                                "or a model slug like vendor/model")
        return v
    if k == "choice":
        if raw not in spec.choices:
            raise PropertyError(f"{spec.label}: choose one of {', '.join(spec.choices)}")
        return raw
    if k == "list":
        items = _json_list(raw)
        ok = (lambda i: i in spec.choices) if spec.choices else (lambda i: bool(_ITEM.match(i)))
        if items is None or len(items) > _MAX_LIST or not all(ok(i) for i in items):
            raise PropertyError(f"{spec.label}: a list drawn from {', '.join(spec.choices)}"
                                if spec.choices else f"{spec.label}: a list of short plain names")
        return list(dict.fromkeys(items))
    if k == "number":
        n = _number(raw)
        if n is None or (spec.minimum is not None and n < spec.minimum) or \
                (spec.maximum is not None and n > spec.maximum):
            raise PropertyError(f"{spec.label}: a number"
                                + (f" from {spec.minimum:g}" if spec.minimum is not None else "")
                                + (f" to {spec.maximum:g}" if spec.maximum is not None else ""))
        return n
    if k == "flag":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.lower() in ("true", "false"):
            return raw.lower() == "true"
        raise PropertyError(f"{spec.label}: true or false")
    # text
    if not isinstance(raw, str) or len(raw) > _MAX_TEXT or _CONTROL.search(raw):
        raise PropertyError(f"{spec.label}: text of at most {_MAX_TEXT} characters")
    return raw.strip()


def _number(raw: Any) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        try:
            raw = float(raw)
        except ValueError:
            return None
    if not isinstance(raw, (int, float)) or not math.isfinite(raw):
        return None
    return int(raw) if float(raw).is_integer() else float(raw)


def _json_list(raw: Any) -> list[str] | None:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    if not isinstance(raw, list) or not all(isinstance(i, str) for i in raw):
        return None
    return raw


def check_name(name: str, specs: dict[str, PropertySpec]) -> PropertySpec:
    """The spec for ``name``, or a refusal naming the ones there are."""
    spec = specs.get(name) if isinstance(name, str) else None
    if spec is None:
        known = ", ".join(sorted(specs)) or "none"
        raise PropertyError(f"no property {name!r}; these are tunable: {known}")
    return spec


def check_declarations(specs: dict[str, PropertySpec]) -> dict[str, PropertySpec]:
    """Declared names are lowercase words; a typo fails at start-up, not on a write."""
    for name in specs:
        if not _NAME.match(name):
            raise ValueError(f"property name {name!r}: lowercase letters, digits and _")
    return dict(specs)


# ── Standard patron properties ───────────────────────────────────────────

#: Every site's patrons have these; the frontend reads them through one hook.
STANDARD_PATRON_PROPERTIES: dict[str, PropertySpec] = {
    "theme": PropertySpec(kind="choice", default="system", label="Theme",
                          choices=("system", "light", "dark"), follows="device"),
    "timezone": PropertySpec(kind="text", default="auto", label="Time zone",
                             description="An IANA zone, e.g. America/New_York, or auto for the browser's."),
    # Whether the guided tour runs. Which pages have had theirs stays per device.
    "tour": PropertySpec(kind="flag", default=True, label="Tour"),
}


# ── Storage ──────────────────────────────────────────────────────────────


def schema_statements(t: Any, idx: str) -> list[str]:
    """One row per (owner npub, property)."""
    return [
        (
            f"CREATE TABLE IF NOT EXISTS {t(TABLE)} ("
            "    npub TEXT NOT NULL,"
            "    name TEXT NOT NULL,"
            "    value JSONB NOT NULL,"
            "    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
            "    PRIMARY KEY (npub, name)"
            ")"
        ),
    ]


@dataclass
class PropertyStore:
    """Reads and writes the rows; knows nothing of who may."""

    vault: Any
    _cache: dict[str, tuple[float, dict[str, tuple[Any, str]]]] = field(default_factory=dict)

    async def values(self, npub: str, *, cached: bool = False) -> dict[str, tuple[Any, str]]:
        """``{name: (value, updated_at)}`` as stored for ``npub``."""
        if cached and (hit := self._cache.get(npub)) and time.monotonic() - hit[0] < CACHE_TTL_S:
            return hit[1]
        # As text: Neon hands JSONB back already parsed, so a value that is a
        # string arrives as a bare str, indistinguishable from JSON text.
        # 2026-10-29: every text and model property read back as its default.
        r = await self.vault._execute(
            f"SELECT name, value::text AS value, updated_at FROM {self.vault._t(TABLE)} "
            "WHERE npub = $1", [npub])
        out: dict[str, tuple[Any, str]] = {}
        for row in r.get("rows") or []:
            try:
                value = json.loads(row.get("value") or "")
            except (TypeError, ValueError):
                continue
            out[str(row.get("name"))] = (value, str(row.get("updated_at") or ""))
        if cached:
            self._cache[npub] = (time.monotonic(), out)
        return out

    async def set(self, npub: str, name: str, value: Any) -> None:
        await self.vault._execute(
            f"INSERT INTO {self.vault._t(TABLE)} (npub, name, value, updated_at) "
            "VALUES ($1, $2, $3::jsonb, now()) "
            "ON CONFLICT (npub, name) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            [npub, name, json.dumps(value)],
        )
        self._cache.pop(npub, None)

    async def clear(self, npub: str, name: str | None = None) -> int:
        """Return one property, or every one ``npub`` set, to its default."""
        if name is None:
            r = await self.vault._execute(
                f"DELETE FROM {self.vault._t(TABLE)} WHERE npub = $1", [npub])
        else:
            r = await self.vault._execute(
                f"DELETE FROM {self.vault._t(TABLE)} WHERE npub = $1 AND name = $2", [npub, name])
        self._cache.pop(npub, None)
        return int(r.get("rowCount") or r.get("rowcount") or 0)


def listing(specs: dict[str, PropertySpec], stored: dict[str, tuple[Any, str]]) -> list[dict[str, Any]]:
    """Each property, its spec, its value and whether that is the default.

    A stored value its spec would now refuse — the spec changed under it —
    reads as the default rather than as something nobody can set any more.
    """
    out = []
    for name, spec in specs.items():
        value, updated_at, is_default = spec.default, "", True
        if name in stored:
            try:
                value, updated_at, is_default = validate(spec, stored[name][0]), stored[name][1], False
            except PropertyError:
                pass
        out.append({"name": name, **spec.describe(), "value": value,
                    "is_default": is_default, "updated_at": updated_at})
    return out
