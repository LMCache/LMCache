# SPDX-License-Identifier: Apache-2.0
"""The vLLM Q ring registers its sliding window with the server.

With ``lmcache.mp.q.sw_size_tokens`` set, the ring's group carries the
window, so the server stores only each chunk's last ``sw_size_tokens`` query
rows (and, with object-group separation, reads only the trailing chunks).
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk.context import FULL_WINDOW
from lmcache.sdk.qringbuffer import QRingBufferAdapter

CHUNK = 256
BLOCKS_IN_CHUNK = 16  # 16-token ring blocks


def _register(
    sw_size_tokens: int, block_size: int = CHUNK // BLOCKS_IN_CHUNK
) -> MagicMock:
    """Register a small CPU Q ring and return the transfer context it used."""
    transfer_ctx = MagicMock()
    worker = SimpleNamespace(
        transfer_ctx=transfer_ctx,
        lmcache_tokens_per_chunk=CHUNK,
        blocks_in_chunk=BLOCKS_IN_CHUNK,
        world_size=1,
        _mq_timeout=1.0,
    )
    adapter = QRingBufferAdapter(worker, "m##query")  # type: ignore[arg-type]
    with patch("lmcache.sdk.qringbuffer.vllm_layout_hints", return_value={}):
        adapter.register_q_ring(
            num_layers=2,
            num_q_heads=2,
            head_size=4,
            dtype=torch.float32,
            num_ring_blocks=4,
            device=torch.device("cpu"),
            block_size=block_size,
            sw_size_tokens=sw_size_tokens,
        )
    return transfer_ctx


@pytest.mark.parametrize("sw_size_tokens", [FULL_WINDOW, 64, 512])
def test_ring_registers_its_window(sw_size_tokens: int):
    transfer_ctx = _register(sw_size_tokens)

    (group,) = transfer_ctx.register_q.call_args.kwargs["engine_group_infos"]
    assert group.sw_size_tokens == sw_size_tokens
    assert group.tokens_per_block == CHUNK // BLOCKS_IN_CHUNK


def test_ring_block_size_must_divide_the_chunk():
    """A hybrid's full-attention block (e.g. 544) must tile the chunk."""
    with pytest.raises(ValueError, match="must divide the LMCache chunk"):
        _register(FULL_WINDOW, block_size=100)


@pytest.mark.parametrize("sw_size_tokens", [0, -2, 40])
def test_ring_rejects_windows_off_the_block_grid(sw_size_tokens: int):
    """The server floors a partial block, so the window must be whole blocks."""
    with pytest.raises(ValueError, match="sw_size_tokens"):
        _register(sw_size_tokens)
