"""Shared test configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_operator_state_writes(request):
    """A successful bootstrap records state in the background, over real sockets.

    Tests that exercise it mark themselves ``records_operator_state``.
    """
    if "records_operator_state" in request.keywords:
        yield
        return
    with patch("tollbooth.bootstrap._record_state_in_background"):
        yield
