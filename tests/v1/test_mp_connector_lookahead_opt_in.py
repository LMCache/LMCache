# SPDX-License-Identifier: Apache-2.0
"""The MP connector opts into vLLM's EAGLE / MTP lookahead block hashes."""

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig  # noqa: E402

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

pytestmark = pytest.mark.no_shared_allocator


def test_connector_opts_into_lookahead_block_hashes() -> None:
    """vLLM keeps its last-block drop on local hits unless the connector opts in."""
    transfer_config = KVTransferConfig(
        kv_connector="LMCacheMPConnector", kv_role="kv_both"
    )
    assert LMCacheMPConnector.supports_lookahead_block_hashes(transfer_config)
