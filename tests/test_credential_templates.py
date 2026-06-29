"""Tests for credential template validation."""

import pytest

from tollbooth.credential_templates import (
    LONGRUNNER_CREDENTIAL_FIELDS,
    CredentialTemplate,
    FieldSpec,
    TemplateValidationError,
    render_delimited_instructions,
    render_template_instructions,
    validate_payload,
)


def test_longrunner_fields_are_optional_operator_secrets():
    """The durable long-runner fields are normal optional operator secrets that
    an operator spreads into its OWN credential template (no separate service)."""
    assert set(LONGRUNNER_CREDENTIAL_FIELDS) == {
        "prefect_api_url", "prefect_api_key", "closure_seal_key"
    }
    # all optional — without them the job falls back to in-process execution
    assert all(not f.required for f in LONGRUNNER_CREDENTIAL_FIELDS.values())
    # the two keys are sensitive; the URL is not
    assert LONGRUNNER_CREDENTIAL_FIELDS["prefect_api_key"].sensitive
    assert LONGRUNNER_CREDENTIAL_FIELDS["closure_seal_key"].sensitive
    assert not LONGRUNNER_CREDENTIAL_FIELDS["prefect_api_url"].sensitive


def test_longrunner_fields_merge_into_an_operator_template():
    """Spreading them into a template yields one service with all fields, and
    they validate + render like any other operator secret (single mgmt path)."""
    tmpl = CredentialTemplate(
        service="excalibur-operator",
        version=1,
        fields={"anthropic_api_key": FieldSpec(required=True), **LONGRUNNER_CREDENTIAL_FIELDS},
    )
    # partial delivery of just the long-runner fields validates (merge-on-receive)
    cleaned = validate_payload(
        {"closure_seal_key": "ab" * 32, "prefect_api_url": "u", "prefect_api_key": "k"},
        tmpl,
        partial=True,
    )
    assert set(cleaned) == {"closure_seal_key", "prefect_api_url", "prefect_api_key"}
    # they render in the welcome DM alongside the operator's own secrets
    text = render_delimited_instructions(tmpl)
    assert "closure_seal_key = @@@" in text and "anthropic_api_key = @@@" in text


def _x_api_template() -> CredentialTemplate:
    """Sample X/Twitter API credential template."""
    return CredentialTemplate(
        service="x",
        version=1,
        fields={
            "api_key": FieldSpec(required=True, sensitive=True),
            "api_secret": FieldSpec(required=True, sensitive=True),
            "access_token": FieldSpec(required=True, sensitive=True),
            "access_secret": FieldSpec(required=True, sensitive=True),
            "display_name": FieldSpec(required=False, sensitive=False),
        },
        description="X/Twitter API v2 credentials (OAuth 1.0a User Context)",
    )


class TestValidatePayload:
    """Tests for validate_payload()."""

    def test_valid_full_payload(self):
        """All required + optional fields accepted."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
            "display_name": "testuser",
        }
        result = validate_payload(payload, tmpl)
        assert result == payload

    def test_valid_required_only(self):
        """Optional fields can be omitted."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
        }
        result = validate_payload(payload, tmpl)
        assert result == payload

    def test_metadata_keys_stripped(self):
        """service and version keys are ignored in validation."""
        tmpl = _x_api_template()
        payload = {
            "service": "x",
            "version": 1,
            "api_key": "key1",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
        }
        result = validate_payload(payload, tmpl)
        assert "service" not in result
        assert "version" not in result
        assert len(result) == 4

    def test_unknown_fields_silently_dropped(self):
        """Fields not in template are silently stripped, known fields pass through."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
            "rogue_field": "evil",
            "notes": "sent from my phone",
        }
        result = validate_payload(payload, tmpl)
        assert "rogue_field" not in result
        assert "notes" not in result
        assert result["api_key"] == "key1"
        assert len(result) == 4

    def test_unknown_fields_do_not_affect_required_check(self):
        """Extra fields don't mask missing required fields."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
            "rogue_field": "evil",
            "notes": "extra stuff",
        }
        with pytest.raises(TemplateValidationError, match="Missing required fields"):
            validate_payload(payload, tmpl)

    def test_missing_required_fields(self):
        """Missing required fields are reported."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
        }
        with pytest.raises(TemplateValidationError, match="Missing required fields"):
            validate_payload(payload, tmpl)

    def test_partial_allows_missing_required(self):
        """partial=True (merge-on-receive) accepts a subset of required fields."""
        tmpl = _x_api_template()
        payload = {"api_key": "key1"}
        result = validate_payload(payload, tmpl, partial=True)
        assert result == {"api_key": "key1"}

    def test_partial_single_optional_field(self):
        """A lone optional field is accepted in partial mode."""
        tmpl = _x_api_template()
        result = validate_payload({"display_name": "solo"}, tmpl, partial=True)
        assert result == {"display_name": "solo"}

    def test_partial_still_rejects_empty_present_required(self):
        """A present-but-empty required field is rejected even in partial mode."""
        tmpl = _x_api_template()
        with pytest.raises(TemplateValidationError, match="must not be empty"):
            validate_payload({"api_key": "   "}, tmpl, partial=True)

    def test_partial_still_drops_unknown(self):
        """Unknown fields are dropped in partial mode too."""
        tmpl = _x_api_template()
        result = validate_payload({"api_key": "k", "bogus": "x"}, tmpl, partial=True)
        assert result == {"api_key": "k"}

    def test_empty_required_field_rejected(self):
        """Required field with empty/whitespace value is rejected."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "   ",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
        }
        with pytest.raises(TemplateValidationError, match="must not be empty"):
            validate_payload(payload, tmpl)

    def test_non_string_value_rejected(self):
        """Non-string values are rejected."""
        tmpl = _x_api_template()
        payload = {
            "api_key": 12345,
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
        }
        with pytest.raises(TemplateValidationError, match="must be a string"):
            validate_payload(payload, tmpl)

    def test_empty_optional_field_allowed(self):
        """Optional field can be empty string."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
            "display_name": "",
        }
        result = validate_payload(payload, tmpl)
        assert result["display_name"] == ""


class TestRenderInstructions:
    """Tests for render_template_instructions()."""

    def test_includes_service_name(self):
        """Instructions mention the service name."""
        tmpl = _x_api_template()
        text = render_template_instructions(tmpl)
        assert "x" in text
        assert "v1" in text

    def test_includes_description(self):
        """Instructions include the template description."""
        tmpl = _x_api_template()
        text = render_template_instructions(tmpl)
        assert "OAuth 1.0a" in text

    def test_marks_required_fields(self):
        """Required fields are labeled REQUIRED."""
        tmpl = _x_api_template()
        text = render_template_instructions(tmpl)
        assert "REQUIRED" in text

    def test_marks_optional_fields(self):
        """Optional fields are labeled optional."""
        tmpl = _x_api_template()
        text = render_template_instructions(tmpl)
        assert "optional" in text

    def test_includes_example_json(self):
        """Instructions include example JSON."""
        tmpl = _x_api_template()
        text = render_template_instructions(tmpl)
        assert "your_api_key_here" in text

    def test_minimal_template(self):
        """Template with no description still renders."""
        tmpl = CredentialTemplate(
            service="test",
            version=1,
            fields={"token": FieldSpec(required=True)},
        )
        text = render_template_instructions(tmpl)
        assert "test" in text
        assert "token" in text


class TestRenderDelimitedInstructions:
    """Tests for render_delimited_instructions()."""

    def test_uses_aaa_markers(self):
        """Output uses @@@ delimiters for each field."""
        tmpl = _x_api_template()
        text = render_delimited_instructions(tmpl)
        assert "@@@PASTE_YOUR_API_KEY_HERE@@@" in text
        assert "@@@PASTE_YOUR_API_SECRET_HERE@@@" in text

    def test_includes_service_and_version(self):
        """Header includes service name and version."""
        tmpl = _x_api_template()
        text = render_delimited_instructions(tmpl)
        assert "x" in text
        assert "v1" in text

    def test_no_required_or_optional_labels(self):
        """Field lines should not include (REQUIRED) or (optional) labels."""
        tmpl = _x_api_template()
        text = render_delimited_instructions(tmpl)
        assert "REQUIRED" not in text
        assert "(optional)" not in text

    def test_includes_description(self):
        """Template description is rendered."""
        tmpl = _x_api_template()
        text = render_delimited_instructions(tmpl)
        assert "OAuth 1.0a" in text

    def test_no_json_in_output(self):
        """Delimited instructions contain no JSON syntax."""
        tmpl = _x_api_template()
        text = render_delimited_instructions(tmpl)
        assert "{" not in text
        assert "}" not in text

    def test_minimal_template(self):
        """Single-field template renders correctly."""
        tmpl = CredentialTemplate(
            service="test",
            version=1,
            fields={"token": FieldSpec(required=True)},
        )
        text = render_delimited_instructions(tmpl)
        assert "token = @@@PASTE_YOUR_TOKEN_HERE@@@" in text
