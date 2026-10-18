"""Tests for tollbooth.agent_keyring.with_npub — naming the agent on each call.

The FastMCP proxy/middleware plumbing is FastMCP's, and the connection's NIP-98
signing is ``service_client``'s; the logic the keyring owns is naming the agent's
npub on a forwarded call that names none. No FastMCP import required.
"""

from tollbooth.agent_keyring import with_npub

AGENT = "npub1agent"


def test_names_the_agent_and_keeps_the_arguments():
    out = with_npub({"repo_name": "x", "issue_number": 1}, AGENT)
    assert out == {"repo_name": "x", "issue_number": 1, "npub": AGENT}


def test_no_arguments_still_names_the_agent():
    assert with_npub(None, AGENT) == {"npub": AGENT}


def test_an_explicit_npub_is_respected():
    assert with_npub({"npub": "npub1other"}, AGENT)["npub"] == "npub1other"


def test_the_callers_dict_is_not_mutated():
    args = {"a": 1}
    with_npub(args, AGENT)
    assert args == {"a": 1}
