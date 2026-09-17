# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the MP preemption model-based tests.

These tests drive the real vLLM ``Scheduler`` and the real
``LMCacheMPConnector`` (scheduler role) against a simulated LMCache
server/worker.  They need vLLM importable but no GPU and no LMCache server.
"""

# Standard
import os

# Third Party
import pytest

# The scheduler factory builds a ModelConfig for a tiny HF model. Never hit
# the network for it; the config is expected to be in the local HF cache.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

pytest.importorskip("vllm", reason="MP preemption tests require vLLM")


@pytest.fixture(autouse=True)
def _quiet_lmcache_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    # First Party
    import lmcache.integration.vllm.lmcache_mp_connector as connector_mod

    monkeypatch.setattr(connector_mod, "print_banner_once", lambda *_a, **_k: None)
