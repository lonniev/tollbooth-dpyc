# Contributing to tollbooth-dpyc

## Development setup

This project uses [`uv`](https://github.com/astral-sh/uv). Install all the
extras the test suite exercises:

```bash
make dev          # uv sync with every extra
# or directly:
uv sync --extra dev --extra nostr --extra qr --extra authority --extra x402
```

(`dev` pulls in `pytest`, `pytest-asyncio`, `pytest-cov`, and `mypy`.)

## Make targets

The `Makefile` wraps the common commands, each with the full set of extras CI
installs:

| Target | What it runs |
|---|---|
| `make dev`  | install with all extras |
| `make test` | the full suite |
| `make lint` | `ruff check .` |
| `make type` | `mypy` (the blocking type gate) |
| `make cov`  | the suite with the coverage ratchet gate |
| `make all`  | lint + type + cov — the full local gate |

> **Always run `type`/`cov` with all extras.** Without the `x402` extra,
> mypy's `--ignore-missing-imports` hides real type errors in `x402_client`
> that CI will catch.

## Running tests

```bash
make test
pytest tests/test_paid_tool.py -v   # a single module
```

CI runs the same suite on Python 3.12 and 3.13, with `ruff check .` and `mypy`
as blocking gates before the tests.

### Type checking

`mypy` is a **blocking CI gate** (config in `pyproject.toml` `[tool.mypy]`).
The package type-checks clean at default strictness; fix new errors rather than
adding per-file overrides — adopting the gate is what surfaced the 0.44.9
tranche bug and the 0.44.10 proof bug.

### Coverage

Coverage is measured on every CI run with a **ratchet gate**:
`--cov-fail-under=67` (a floor, not a target — measured ~70%). CI fails on a
regression. Raise the floor as coverage climbs; never lower it.

```bash
make cov
```

When you add or move code, check the coverage delta — a drop usually means a
behavior path lost its test, not just fewer lines.

## Lint

```bash
make lint    # ruff check .
```

## Dependency pinning policy

The fleet deploys serverless on Prefect Horizon, which **builds from
`uv.lock`** — so reproducible, byte-stable resolutions matter more than
floating to the newest release.

- **Runtime deps the wheel's behavior depends on are pinned exactly**
  (`httpx==0.28.1`, `pynostr==0.7.0`). Pin to the version that actually has
  what we import; a surprise minor bump should be a deliberate PR, not a
  transitive accident.
- **`cryptography` uses a security floor** (`>=46.0.5`) rather than an exact
  pin — we want CVE fixes to flow in, and its API surface we use is stable.
  The floor is the cohort that includes the 2024–2025 fix set.
- Bumping a pin is a normal change: edit `pyproject.toml`, regenerate
  `uv.lock` (`uv lock --upgrade-package <name>`), and let CI verify. Editing
  `pyproject.toml` alone is **not** enough — Horizon builds the lockfile.

## Releases

See `CLAUDE.md` §8. In short: bump `version` in `pyproject.toml`, add a
`CHANGELOG.md` entry, commit as `release: vX.Y.Z — …`, then push an annotated
`vX.Y.Z` tag — the tag triggers the PyPI publish workflow.
