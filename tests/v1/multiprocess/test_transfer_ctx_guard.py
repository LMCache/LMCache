# SPDX-License-Identifier: Apache-2.0
"""
Tests for the transfer guard: context close must never release resources
while a transfer holds context-owned views (use-after-unmap hardening).
"""

# Standard
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.multiprocess.transfer_context.base import (
    ContextClosedError,
    EngineDrivenContext,
    EngineDrivenContextMetadata,
)


class _FakeContext(EngineDrivenContext):
    """Minimal concrete context recording when close releases resources."""

    def __init__(self) -> None:
        metadata = EngineDrivenContextMetadata(
            layout_desc=MemoryLayoutDesc(
                shapes=[torch.Size([1])], dtypes=[torch.float32]
            ),
            block_size=16,
            use_mla=False,
        )
        super().__init__(metadata, MagicMock(), mq_timeout=1.0)
        self.released = threading.Event()
        self.drain_timeout = 5.0

    def prepare_store(self, key, instance_id):
        return None

    def commit_store(self, key, instance_id, chunks):
        return True

    def prepare_retrieve(self, key, instance_id):
        return None

    def commit_retrieve(self, key, instance_id):
        return True

    def close(self) -> None:
        self._drain_transfers(timeout=self.drain_timeout)
        self.released.set()


def test_close_waits_for_inflight_transfer():
    ctx = _FakeContext()
    transfer_entered = threading.Event()
    release_transfer = threading.Event()

    def _transfer():
        with ctx.transfer_guard():
            transfer_entered.set()
            release_transfer.wait(timeout=5.0)

    t = threading.Thread(target=_transfer)
    t.start()
    assert transfer_entered.wait(timeout=5.0)

    closer = threading.Thread(target=ctx.close)
    closer.start()
    # Close must not release resources while the transfer is in flight.
    time.sleep(0.2)
    assert not ctx.released.is_set()

    release_transfer.set()
    closer.join(timeout=5.0)
    t.join(timeout=5.0)
    assert ctx.released.is_set()


def test_closing_context_rejects_new_transfers():
    ctx = _FakeContext()
    ctx.close()
    with pytest.raises(ContextClosedError):
        with ctx.transfer_guard():
            pass


def test_drain_timeout_proceeds_with_warning():
    ctx = _FakeContext()
    ctx.drain_timeout = 0.1
    stuck_entered = threading.Event()
    release_stuck = threading.Event()

    def _stuck_transfer():
        with ctx.transfer_guard():
            stuck_entered.set()
            release_stuck.wait(timeout=10.0)

    t = threading.Thread(target=_stuck_transfer)
    t.start()
    assert stuck_entered.wait(timeout=5.0)

    start = time.monotonic()
    ctx.close()
    elapsed = time.monotonic() - start
    # Bounded: close returned despite the stuck transfer.
    assert ctx.released.is_set()
    assert elapsed < 2.0

    release_stuck.set()
    t.join(timeout=5.0)


def test_guard_reentrant_across_sequential_transfers():
    ctx = _FakeContext()
    for _ in range(3):
        with ctx.transfer_guard():
            pass
    ctx.close()
    assert ctx.released.is_set()
