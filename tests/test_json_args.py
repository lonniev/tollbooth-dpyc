"""Arguments a client serialised one time too many.

The bug: a client sends `{"patch": "{\"doc\": …}"}` — the right value, encoded
once more than the schema says — and pydantic reports `dict_type`, which reads
as "your payload is malformed" when it is not. Clients do this unevenly, so it
presents as a size limit or as two sibling tools disagreeing, and an
investigator goes looking for a server bug that does not exist.

The danger in fixing it is over-reach: parsing a string a caller MEANT to be a
string would silently change their data. Most of what follows guards that.
"""

from __future__ import annotations

import json

from tollbooth.json_args import coerce_json_arguments

DOC = {"doc": {"blocks": [{"kind": "text", "text": "x" * 1100} for _ in range(9)]}}
OBJ = {"type": "object", "additionalProperties": True}
ARR = {"type": "array"}
STR = {"type": "string"}


class TestItParsesWhatWasOverEncoded:
    def test_a_stringified_object_is_parsed(self):
        args = {"patch": json.dumps(DOC)}
        assert coerce_json_arguments(args, {"patch": OBJ}) == ["patch"]
        assert args["patch"] == DOC

    def test_size_is_irrelevant(self):
        """The report read this as a size limit. It never was."""
        big = {"doc": {"blocks": [{"text": "x" * 5000} for _ in range(50)]}}
        assert len(json.dumps(big)) > 250_000
        args = {"patch": json.dumps(big)}
        coerce_json_arguments(args, {"patch": OBJ})
        assert args["patch"] == big

    def test_a_stringified_array_is_parsed(self):
        args = {"items": '[1, 2, 3]'}
        assert coerce_json_arguments(args, {"items": ARR}) == ["items"]
        assert args["items"] == [1, 2, 3]

    def test_a_real_object_passes_through_untouched(self):
        args = {"patch": DOC}
        assert coerce_json_arguments(args, {"patch": OBJ}) == []
        assert args["patch"] is DOC


class TestItRefusesToOverReach:
    def test_a_declared_string_is_never_parsed(self):
        """Even when its text is valid JSON. This is the dangerous case."""
        args = {"note": '{"still": "a string"}'}
        assert coerce_json_arguments(args, {"note": STR}) == []
        assert args["note"] == '{"still": "a string"}'

    def test_a_union_with_string_is_never_parsed(self):
        """`str | dict` declares under anyOf — a string is legitimate there."""
        union = {"anyOf": [{"type": "string"}, {"type": "object"}]}
        args = {"body": '{"a": 1}'}
        assert coerce_json_arguments(args, {"body": union}) == []
        assert args["body"] == '{"a": 1}'

    def test_an_undeclared_parameter_is_never_parsed(self):
        args = {"mystery": '{"a": 1}'}
        assert coerce_json_arguments(args, {}) == []
        assert args["mystery"] == '{"a": 1}'

    def test_a_scalar_json_string_is_not_promoted(self):
        """`"5"` parses as JSON but is not an object or an array."""
        args = {"patch": "5"}
        assert coerce_json_arguments(args, {"patch": OBJ}) == []
        assert args["patch"] == "5"


class TestItLeavesGenuineMistakesAlone:
    def test_unparseable_json_is_left_for_validation_to_report(self):
        """A truncated payload IS malformed; say so rather than paper over it."""
        args = {"patch": '{"doc": {"blocks": ['}
        assert coerce_json_arguments(args, {"patch": OBJ}) == []
        assert args["patch"] == '{"doc": {"blocks": ['

    def test_plain_prose_is_left_alone(self):
        args = {"patch": "not json at all"}
        assert coerce_json_arguments(args, {"patch": OBJ}) == []


class TestMixedArguments:
    def test_only_the_object_argument_moves(self):
        args = {"patch": json.dumps(DOC), "note": "hello", "post_id": "abc-123"}
        props = {"patch": OBJ, "note": STR, "post_id": STR}
        assert coerce_json_arguments(args, props) == ["patch"]
        assert args["patch"] == DOC
        assert args["note"] == "hello"
        assert args["post_id"] == "abc-123"


class TestThroughARealServer:
    """The unit tests above prove the parsing. This proves the wiring."""

    def test_a_stringified_object_reaches_the_tool_as_a_dict(self):
        import asyncio

        import pytest
        pytest.importorskip("fastmcp")
        from fastmcp import Client, FastMCP

        from tollbooth.json_args import build_json_arg_middleware

        mw = build_json_arg_middleware()
        assert mw is not None, "installed FastMCP should support middleware"

        mcp = FastMCP("json-args-test")
        mcp.add_middleware(mw)

        @mcp.tool
        async def patcher(patch: dict, note: str = "") -> dict:
            return {"blocks": len(patch["doc"]["blocks"]), "note": note}

        async def _run():
            async with Client(mcp) as c:
                # The exact failure from the report: a serialised object.
                r = await c.call_tool(
                    "patcher", {"patch": json.dumps(DOC), "note": '{"a": 1}'},
                )
                return r.data

        data = asyncio.run(_run())
        assert data["blocks"] == 9
        # ...and the declared string survived verbatim alongside it.
        assert data["note"] == '{"a": 1}'
