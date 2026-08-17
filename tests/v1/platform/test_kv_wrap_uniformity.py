# SPDX-License-Identifier: Apache-2.0
"""Tests for the mixed-rank guard in ``lmcache.v1.platform.kv_wrap``."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.multiprocess.posix_shm import shm_unlink
from lmcache.v1.platform.kv_wrap import wrap_kv_caches

ATTN = (2, 1, 4, 8)
INDEX = (2, 4, 4)


def _release(wrappers):
    for wrapper in wrappers:
        name = getattr(wrapper, "shm_name", None)
        if name is not None:
            shm_unlink(name)


def test_uniform_rank_batch_wraps():
    """A single-geometry batch still wraps."""
    caches = {
        f"layers.{i}.self_attn.attn": torch.zeros(ATTN, dtype=torch.float32)
        for i in range(2)
    }
    wrappers = wrap_kv_caches(caches)
    try:
        assert len(wrappers) == 2
    finally:
        _release(wrappers)


def test_mixed_rank_batch_raises_and_names_both_groups():
    """Mixed ranks raise, and the message identifies each group.

    Mirrors MiniMax-M3: rank-4 ``self_attn.attn`` alongside rank-3
    ``self_attn.attn.index_cache`` in one batch.
    """
    caches = {
        "layers.0.self_attn.attn": torch.zeros(ATTN, dtype=torch.bfloat16),
        "layers.1.self_attn.attn.index_cache": torch.zeros(
            INDEX, dtype=torch.bfloat16
        ),
    }
    with pytest.raises(ValueError, match="mixed tensor ranks") as excinfo:
        wrap_kv_caches(caches)
    message = str(excinfo.value)
    assert "rank 3" in message and "rank 4" in message
    assert "index_cache" in message


def test_list_entries_are_not_rank_compared():
    """Mamba-style list entries are left to the factories, not rank-checked."""
    caches = {
        "layers.0.self_attn.attn": torch.zeros(ATTN, dtype=torch.float32),
        "layers.1.mixer": [torch.zeros((2, 4)), torch.zeros((2, 4, 4))],
    }
    try:
        wrappers = wrap_kv_caches(caches)
    except ValueError as exc:
        assert "mixed tensor ranks" not in str(exc)
    except Exception:
        pass  # a factory may reject the list entry; out of scope here
    else:
        _release(wrappers)
