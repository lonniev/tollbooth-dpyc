"""One generation sent and read back: request shape, refusals, and the answer readers."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest

from tollbooth import llm_call
from tollbooth.llm_call import LlmAnswerError, ask, number, one_of, tagged, text, url, usage_of, whole

KEY = "sk-or-test-not-a-real-key"


def _client_answering(monkeypatch: pytest.MonkeyPatch, status: int, body: Any, seen: list[dict]) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append({"headers": dict(request.headers), "json": json.loads(request.content)})
        return httpx.Response(status, json=body)

    real = httpx.AsyncClient

    def factory(**kw: Any) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handle), timeout=kw.get("timeout"))

    monkeypatch.setattr(llm_call.httpx, "AsyncClient", factory)


ANSWER = {
    "model": "anthropic/claude-haiku",
    "content": [{"type": "text", "text": 'ok <draft>{"title": "A table"}</draft>'}],
    "usage": {"input_tokens": 10, "cache_read_input_tokens": 3000, "output_tokens": 200},
    "stop_reason": "end_turn",
}


async def test_ask_sends_effort_and_a_cached_system_block(monkeypatch, caplog):
    seen: list[dict] = []
    _client_answering(monkeypatch, 200, ANSWER, seen)
    caplog.set_level(logging.INFO, logger="tollbooth.llm_call")
    data = await ask(KEY, system="FIXED PROMPT", user="an idea", max_tokens=500,
                     model="vendor/small", effort="low", cache_system=True, purpose="draft")
    body = seen[0]["json"]
    assert body["model"] == "vendor/small"
    assert body["output_config"] == {"effort": "low"}
    assert body["system"] == [{"type": "text", "text": "FIXED PROMPT", "cache_control": {"type": "ephemeral"}}]
    assert "tools" not in body
    assert data["model"] == "anthropic/claude-haiku"
    # The key goes in the header and nowhere in the log; nor do the prompts.
    assert seen[0]["headers"]["x-api-key"] == KEY
    assert KEY not in caplog.text and "FIXED PROMPT" not in caplog.text and "an idea" not in caplog.text


async def test_ask_default_effort_sends_none_and_plain_system(monkeypatch):
    seen: list[dict] = []
    _client_answering(monkeypatch, 200, ANSWER, seen)
    await ask(KEY, system="S", user="U", max_tokens=100, model="vendor/small", searches=2)
    body = seen[0]["json"]
    assert "output_config" not in body
    assert body["system"] == "S"
    assert body["tools"][0]["name"] == "web_search" and body["tools"][0]["max_uses"] == 2


async def test_an_unfunded_provider_is_a_refund_without_the_key(monkeypatch, caplog):
    seen: list[dict] = []
    _client_answering(monkeypatch, 402, {"error": {"message": "Insufficient credits"}}, seen)
    with pytest.raises(LlmAnswerError) as e:
        await ask(KEY, system="S", user="U", max_tokens=100, model="vendor/small")
    assert isinstance(e.value, ValueError)
    assert "No fare was charged" in str(e.value)
    assert KEY not in caplog.text


def test_tagged_reads_the_last_block_and_refuses_clearly():
    data = {"content": [{"type": "text", "text": "<x>1</x> then <x> 2 </x>"}]}
    assert tagged(data, "x") == "2"
    with pytest.raises(LlmAnswerError, match="ran out of room"):
        tagged({"content": [], "stop_reason": "max_tokens"}, "x")
    with pytest.raises(LlmAnswerError, match="without the part"):
        tagged({"content": [{"type": "text", "text": "no tag"}]}, "x")


def test_usage_counts_cached_input():
    u = usage_of(ANSWER)
    assert u["input_tokens"] == 3010 and u["cached_input_tokens"] == 3000 and u["output_tokens"] == 200


def test_field_readers_drop_what_they_cannot_vouch_for():
    assert number(5, 0, 10) == 5.0 and number(True, 0, 10) is None and number(float("nan"), 0, 1) is None
    assert number(11, 0, 10) is None
    assert whole(2.6, 0, 10) == 3 and whole("3", 0, 10) is None
    assert text("  a \n b  ") == "a b" and text(3) == "" and len(text("x" * 999, 50)) == 50
    assert url("https://example.com/a") == "https://example.com/a" and url("javascript:alert(1)") == ""
    assert one_of("Assign", ("assign", "license")) == "assign" and one_of("sell", ("assign",)) is None
