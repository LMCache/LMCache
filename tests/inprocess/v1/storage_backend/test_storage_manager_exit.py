# SPDX-License-Identifier: Apache-2.0
"""StorageManager must not block interpreter exit when close() is skipped.

A host engine can fail to forward shutdown to LMCache, in which case
StorageManager.close() never runs. The storage manager's event loop thread must
not keep the host process alive in that case.
"""

# Standard
import subprocess
import sys

# Third Party
import pytest

# A regression hangs forever rather than being merely slow, so this bound only
# has to exceed normal startup, which is a few seconds dominated by importing
# torch. Keep it tight enough that a failure does not stall CI.
CHILD_EXIT_TIMEOUT_SECONDS = 60.0

_CHILD_SOURCE = """
import torch

from lmcache.inprocess.v1.event_manager import EventManager
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend.storage_manager import StorageManager

# role="scheduler" skips the accelerator allocator backend so this runs on a
# CPU-only host. No storage backend is configured because the event loop
# thread is started before any backend exists.
config = LMCacheEngineConfig.from_defaults(
    chunk_size=256,
    local_cpu=False,
    lmcache_instance_id="storage_manager_exit_test",
)
metadata = LMCacheMetadata(
    model_name="exit_test_model",
    world_size=1,
    local_world_size=1,
    worker_id=0,
    local_worker_id=0,
    kv_dtype=torch.bfloat16,
    kv_shape=(28, 2, 256, 8, 128),
    role="scheduler",
)

manager = StorageManager(
    config=config,
    metadata=metadata,
    event_manager=EventManager(),
)
print("storage manager built", flush=True)
# close() is deliberately not called: this models a host that skips shutdown.
"""


def test_interpreter_exits_when_close_is_skipped() -> None:
    """A child that builds a StorageManager and skips close() must still exit."""
    # subprocess.run kills and reaps the child when the timeout expires and on
    # any other exception. Popen's context manager only waits, so a hanging
    # child would block an interrupted run.
    try:
        result = subprocess.run(
            [sys.executable, "-c", _CHILD_SOURCE],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=CHILD_EXIT_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        # Output read before the timeout is attached undecoded, even though
        # text=True decodes the output of a child that exits in time.
        partial = (exc.stdout or b"").decode("utf-8", errors="replace")
        pytest.fail(
            "Child process did not exit within %.0fs after building a "
            "StorageManager without calling close(); a background thread is "
            "blocking interpreter exit. Child output:\n%s"
            % (CHILD_EXIT_TIMEOUT_SECONDS, partial)
        )

    assert "storage manager built" in result.stdout, (
        "Child failed before the StorageManager was constructed, so this test "
        "did not exercise interpreter exit. Child output:\n%s" % result.stdout
    )
    assert result.returncode == 0, "Child exited with %s. Child output:\n%s" % (
        result.returncode,
        result.stdout,
    )
