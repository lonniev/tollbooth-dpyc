"""One-shot recovery: re-run ensure_schema() on Optionality's NeonVault.

Why this script exists: in early Optionality runs (wheel <0.31.0), the
NeonVault's startup `ensure_schema()` could silently fail (Neon SQL
error body wasn't read) and leave the `balances` table absent while
the runtime continued reporting `vault_configured: true`. Restoring
the schema requires an operator-restricted call, so we sign a
kind-27235 proof here with the operator's nsec and invoke the
`restore_neon_schema` admin tool.

Usage:
    export OPTIONALITY_OPERATOR_NSEC=nsec1...
    uvx --from "tollbooth-dpyc[nostr]==0.31.0" --with fastmcp \\
        python scripts/restore_optionality_schema.py

The nsec is read from the env var, used once to produce the proof,
and never written to disk or logged. Set the env var on the same
line as the command if your shell history is sensitive.
"""

import asyncio
import json
import os
import sys

from fastmcp import Client
from tollbooth.identity_proof import create_proof  # type: ignore


OPTIONALITY_URL = "https://optionality-mcp.fastmcp.app/mcp"


async def main() -> int:
    nsec = os.environ.get("OPTIONALITY_OPERATOR_NSEC", "").strip()
    if not nsec.startswith("nsec1"):
        print(
            "ERROR: set OPTIONALITY_OPERATOR_NSEC env var to the operator's "
            "nsec1... value. Found:",
            "missing" if not nsec else "non-nsec value",
            file=sys.stderr,
        )
        return 2

    proof = create_proof(nsec, "restore_neon_schema")

    async with Client(OPTIONALITY_URL) as c:
        tools = await c.list_tools()
        tool = next(
            (t for t in tools if t.name.endswith("_restore_neon_schema")),
            None,
        )
        if tool is None:
            print("ERROR: restore_neon_schema not exposed on Optionality.", file=sys.stderr)
            return 3

        result = await c.call_tool(tool.name, {"proof": proof})
        for block in result.content:
            if hasattr(block, "text"):
                print(block.text)
                # Try to pretty-print for readability
                try:
                    parsed = json.loads(block.text)
                    print("\n---")
                    print(json.dumps(parsed, indent=2))
                except Exception:
                    pass
                return 0 if (
                    isinstance(parsed, dict) and parsed.get("success")
                ) else 1

    print("ERROR: no response from restore_neon_schema.", file=sys.stderr)
    return 4


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
