# SPDX-License-Identifier: Apache-2.0
"""Prefetch-job cleanup when a request session ends."""

# Standard
from unittest.mock import MagicMock
import threading

# First Party
from lmcache.v1.multiprocess.modules.lookup import LookupModule


def _module() -> LookupModule:
    """Create a LookupModule with only prefetch-job state.

    Returns:
        A module whose context is a mock.
    """
    module = LookupModule.__new__(LookupModule)
    module._ctx = MagicMock()
    module._prefetch_jobs = {}
    module._prefetch_job_lock = threading.Lock()
    return module


def test_end_session_drops_prefetch_job() -> None:
    """END_SESSION must remove the request's prefetch job."""
    module = _module()
    module._prefetch_jobs["req-1"] = MagicMock()

    module.end_session("req-1")

    assert "req-1" not in module._prefetch_jobs


def test_session_destroy_listener_drops_prefetch_job() -> None:
    """TTL-driven session destruction must remove the prefetch job."""
    module = _module()
    module._prefetch_jobs["req-1"] = MagicMock()
    session = MagicMock()
    session.request_id = "req-1"

    module._on_session_destroyed(session)

    assert "req-1" not in module._prefetch_jobs
