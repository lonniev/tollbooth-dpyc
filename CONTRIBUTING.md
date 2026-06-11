# Contributing to tollbooth-dpyc

## Development setup

This project uses [`uv`](https://github.com/astral-sh/uv). Install all the
extras the test suite exercises:

```bash
pip install -e ".[dev,nostr,qr,authority,x402]"
```

(`dev` pulls in `pytest`, `pytest-asyncio`, and `pytest-cov`.)

## Running tests

```bash
pytest tests/ -q            # full suite
pytest tests/test_paid_tool.py -v   # a single module
```

CI runs the same suite on Python 3.12 and 3.13 with `ruff check .` first.

### Coverage

Coverage is measured on every CI run and printed as a `term-missing` report.
It is **report-only** today — there is no `--cov-fail-under` gate yet. That
ratchet lands once the `tools/` extraction (audit M2.1) raises the measured
floor.

```bash
pytest tests/ -q --cov=tollbooth --cov-report=term-missing
```

**Baseline: 64%** (line + branch, measured 2026-06-10 at v0.44.3).

The coverage map mirrors the known structure work ahead — the largest gaps are
the ones the audit already flagged:

| Module | Coverage | Tracking |
|---|---|---|
| `runtime.py` | 33% | M2.1 — extract the inline `@tool` closures into `tools/` so they become unit-testable |
| `authority/` (`tools.py` 12%, `role_migration.py` 0%, `replay.py` 31%) | low | M2.4 — add authority tests |
| `infographic.py`, `credential_validators.py` | 0% | M3.6 |

When you add or move code, check the coverage delta — a drop usually means a
behavior path lost its test, not just fewer lines.

## Lint

```bash
ruff check .
```

## Releases

See `CLAUDE.md` §8. In short: bump `version` in `pyproject.toml`, add a
`CHANGELOG.md` entry, commit as `release: vX.Y.Z — …`, then push an annotated
`vX.Y.Z` tag — the tag triggers the PyPI publish workflow.
