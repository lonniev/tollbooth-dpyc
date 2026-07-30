"""Per-field delivered_at bookkeeping for vaulted credentials (issue #166)."""

from __future__ import annotations

from tollbooth.credential_meta import (
    META_KEY,
    apply_delivery_timestamps,
    credential_field_names,
    delivered_at_map,
    get_delivered_at,
    strip_meta,
)


class TestCredentialMetaHelpers:
    def test_strip_and_names_hide_meta(self) -> None:
        blob = {
            "api_key": "k",
            META_KEY: {"delivered_at": {"api_key": "2026-01-01T00:00:00+00:00"}},
        }
        assert strip_meta(blob) == {"api_key": "k"}
        assert credential_field_names(blob) == ["api_key"]
        assert META_KEY not in credential_field_names(blob)

    def test_stamp_on_first_delivery(self) -> None:
        blob = apply_delivery_timestamps(
            {"api_key": "k", "secret": "s"},
            just_delivered=["api_key", "secret"],
            when="2026-07-30T12:00:00+00:00",
        )
        assert strip_meta(blob) == {"api_key": "k", "secret": "s"}
        assert get_delivered_at(blob, "api_key") == "2026-07-30T12:00:00+00:00"
        assert get_delivered_at(blob, "secret") == "2026-07-30T12:00:00+00:00"
        assert delivered_at_map(blob) == {
            "api_key": "2026-07-30T12:00:00+00:00",
            "secret": "2026-07-30T12:00:00+00:00",
        }

    def test_partial_delivery_preserves_prior_stamp(self) -> None:
        prior = apply_delivery_timestamps(
            {"api_key": "old", "secret": "kept"},
            just_delivered=["api_key", "secret"],
            when="2026-07-01T00:00:00+00:00",
        )
        merged = apply_delivery_timestamps(
            {"api_key": "new", "secret": "kept"},
            previous=prior,
            just_delivered=["api_key"],
            when="2026-07-30T12:00:00+00:00",
        )
        assert strip_meta(merged) == {"api_key": "new", "secret": "kept"}
        assert get_delivered_at(merged, "api_key") == "2026-07-30T12:00:00+00:00"
        # secret was not in this delivery → keeps original stamp
        assert get_delivered_at(merged, "secret") == "2026-07-01T00:00:00+00:00"

    def test_deleted_field_drops_stale_stamp(self) -> None:
        prior = apply_delivery_timestamps(
            {"api_key": "k", "extra": "x"},
            just_delivered=["api_key", "extra"],
            when="2026-07-01T00:00:00+00:00",
        )
        remaining = apply_delivery_timestamps(
            {"api_key": "k"},
            previous=prior,
            just_delivered=[],  # pure delete — no fresh delivery
            when="2026-07-30T12:00:00+00:00",
        )
        assert strip_meta(remaining) == {"api_key": "k"}
        assert get_delivered_at(remaining, "api_key") == "2026-07-01T00:00:00+00:00"
        assert get_delivered_at(remaining, "extra") is None
        assert "extra" not in delivered_at_map(remaining)

    def test_legacy_blob_has_no_timestamps(self) -> None:
        legacy = {"api_key": "k"}
        assert get_delivered_at(legacy, "api_key") is None
        assert delivered_at_map(legacy) == {}
        assert credential_field_names(legacy) == ["api_key"]

    def test_auto_stamp_only_changed_values(self) -> None:
        prior = apply_delivery_timestamps(
            {"a": "1", "b": "2"},
            just_delivered=["a", "b"],
            when="2026-01-01T00:00:00+00:00",
        )
        nxt = apply_delivery_timestamps(
            {"a": "1", "b": "changed"},
            previous=prior,
            just_delivered=None,  # auto-detect changes
            when="2026-07-30T00:00:00+00:00",
        )
        assert get_delivered_at(nxt, "a") == "2026-01-01T00:00:00+00:00"
        assert get_delivered_at(nxt, "b") == "2026-07-30T00:00:00+00:00"
