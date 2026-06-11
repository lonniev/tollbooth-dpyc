# Developer shortcuts for tollbooth-dpyc. Every target uses uv.
#
# mypy and the test suite must run with ALL extras installed — without the
# x402 extra, --ignore-missing-imports hides real type errors in x402_client
# that CI (which installs every extra) will catch. The EXTRAS list below
# mirrors the CI install exactly.

EXTRAS := --extra dev --extra nostr --extra qr --extra authority --extra x402

.PHONY: dev test lint type cov all help

help:  ## Show this help
	@grep -E '^[a-z]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-6s %s\n", $$1, $$2}'

dev:  ## Install the package with every extra the suite exercises
	uv sync $(EXTRAS)

test:  ## Run the full test suite
	uv run $(EXTRAS) pytest tests/ -q

lint:  ## Ruff lint
	uvx ruff check .

type:  ## Mypy type-check (blocking gate in CI; needs all extras)
	uv run $(EXTRAS) mypy

cov:  ## Run tests with the coverage ratchet gate (matches CI)
	uv run $(EXTRAS) pytest tests/ -q \
		--cov=tollbooth --cov-report=term-missing:skip-covered --cov-fail-under=67

all: lint type cov  ## The full local gate: lint + type + cov
