"""fan_out — every relay asked at once; answers in registry order.

The primitive behind the bootstrap read, the courier's liveness probe and the
profile read/publish. What it must hold: input order regardless of arrival
order, one relay's failure as data, a bounded wait, and the settle window that
lets a slower relay still cast a newer vote after the first acceptable answer.
"""

from __future__ import annotations

import threading
import time

from tollbooth.relay_fanout import Outcome, fan_out


def _sleeper(delays: dict[str, float]):
    def fn(relay: str) -> str:
        time.sleep(delays.get(relay, 0.0))
        return f"answer:{relay}"
    return fn


def test_outcomes_keep_input_order_when_completion_order_is_reversed() -> None:
    relays = ["wss://slowest", "wss://middle", "wss://fastest"]
    outcomes = fan_out(relays, _sleeper({"wss://slowest": 0.15, "wss://middle": 0.08}), deadline=2)
    assert [o.relay for o in outcomes] == relays
    assert all(o.ok for o in outcomes)
    assert outcomes[2].value == "answer:wss://fastest"


def test_relays_run_concurrently_not_in_series() -> None:
    relays = [f"wss://r{i}" for i in range(5)]
    start = time.monotonic()
    outcomes = fan_out(relays, _sleeper(dict.fromkeys(relays, 0.2)), deadline=5)
    elapsed = time.monotonic() - start
    assert all(o.ok for o in outcomes)
    assert elapsed < 0.6, f"five 0.2s relays took {elapsed:.2f}s — that is a series"


def test_one_relay_raising_is_an_error_outcome_not_a_fault() -> None:
    def fn(relay: str) -> str:
        if relay == "wss://bad":
            raise ConnectionRefusedError("refused")
        return "ok"

    good, bad = fan_out(["wss://good", "wss://bad"], fn, deadline=1)
    assert good.ok and good.value == "ok"
    assert bad.state == "error" and bad.error == "refused" and bad.value is None


def test_an_exception_with_no_message_still_names_itself() -> None:
    def fn(_relay: str) -> None:
        raise TimeoutError()

    (o,) = fan_out(["wss://r"], fn, deadline=1)
    assert o.error == "TimeoutError"


def test_deadline_abandons_the_silent_relay_and_returns_on_time() -> None:
    relays = ["wss://fast", "wss://silent"]
    start = time.monotonic()
    fast, silent = fan_out(relays, _sleeper({"wss://silent": 2.0}), deadline=0.3)
    elapsed = time.monotonic() - start
    assert fast.ok
    assert silent.state == "abandoned" and "no answer" in (silent.error or "")
    assert not silent.ok and silent.value is None
    assert 0.25 < elapsed < 0.8


def test_settle_window_hears_the_slightly_slower_relay_but_not_the_straggler() -> None:
    delays = {"wss://first": 0.0, "wss://second": 0.1, "wss://straggler": 5.0}
    start = time.monotonic()
    outcomes = fan_out(
        list(delays), _sleeper(delays), deadline=10, settle=0.3, accept=lambda v: True,
    )
    elapsed = time.monotonic() - start
    states = {o.relay: o.state for o in outcomes}
    assert states == {"wss://first": "ok", "wss://second": "ok", "wss://straggler": "abandoned"}
    assert elapsed < 1.0, f"settle window did not cut the wait: {elapsed:.2f}s"


def test_accept_without_settle_stops_at_the_first_acceptable_answer() -> None:
    delays = {"wss://first": 0.0, "wss://later": 1.0}
    start = time.monotonic()
    outcomes = fan_out(list(delays), _sleeper(delays), deadline=10, accept=lambda v: True)
    assert [o.state for o in outcomes] == ["ok", "abandoned"]
    assert time.monotonic() - start < 0.5


def test_accept_that_says_no_keeps_waiting() -> None:
    delays = {"wss://empty": 0.0, "wss://has-it": 0.2}

    def fn(relay: str) -> str | None:
        time.sleep(delays[relay])
        return "config" if relay == "wss://has-it" else None

    outcomes = fan_out(list(delays), fn, deadline=2, settle=0.0, accept=lambda v: v is not None)
    assert [o.state for o in outcomes] == ["ok", "ok"]
    assert outcomes[1].value == "config"


def test_a_predicate_that_raises_counts_as_no() -> None:
    def accept(_v: str) -> bool:
        raise ValueError("boom")

    outcomes = fan_out(["wss://a", "wss://b"], _sleeper({"wss://b": 0.1}), deadline=2, accept=accept)
    assert [o.state for o in outcomes] == ["ok", "ok"]


def test_empty_input_is_empty_output() -> None:
    assert fan_out([], _sleeper({}), deadline=1) == []


def test_elapsed_is_the_relays_own_time() -> None:
    (o,) = fan_out(["wss://r"], _sleeper({"wss://r": 0.1}), deadline=2)
    assert 0.08 < o.elapsed < 0.5
    assert isinstance(o, Outcome)


def test_workers_run_off_the_calling_thread() -> None:
    seen: list[str] = []

    def fn(_relay: str) -> None:
        seen.append(threading.current_thread().name)

    fan_out(["wss://a", "wss://b"], fn, deadline=1)
    assert len(seen) == 2
    assert all(name.startswith("relay-fanout") for name in seen)
