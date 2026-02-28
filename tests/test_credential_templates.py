"""Tests for credential template validation."""

import pytest

from tollbooth.credential_templates import (
    CredentialTemplate,
    FieldSpec,
    TemplateValidationError,
    render_delimited_instructions,
    render_template_instructions,
    validate_payload,
)


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

    def test_unknown_fields_rejected(self):
        """Fields not in template are rejected."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
            "api_secret": "secret1",
            "access_token": "token1",
            "access_secret": "asecret1",
            "rogue_field": "evil",
        }
        with pytest.raises(TemplateValidationError, match="Unknown fields.*rogue_field"):
            validate_payload(payload, tmpl)

    def test_missing_required_fields(self):
        """Missing required fields are reported."""
        tmpl = _x_api_template()
        payload = {
            "api_key": "key1",
        }
        with pytest.raises(TemplateValidationError, match="Missing required fields"):
            validate_payload(payload, tmpl)

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

    def test_marks_required_and_optional(self):
        """Required and optional labels present."""
        tmpl = _x_api_template()
        text = render_delimited_instructions(tmpl)
        assert "REQUIRED" in text
        assert "optional" in text

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
