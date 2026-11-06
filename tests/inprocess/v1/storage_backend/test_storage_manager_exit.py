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

# A regression hangs forever, so this just needs to exceed normal startup.
CHILD_EXIT_TIMEOUT_SECONDS = 60.0

_CHILD_SOURCE = """
import torch

from lmcache.inprocess.v1.event_manager import EventManager
from lmcache.inprocess.v1.config import LMCacheEngineConfig
from lmcache.inprocess.v1.metadata import LMCacheMetadata
from lmcache.inprocess.v1.storage_backend.storage_manager import StorageManager

# role="scheduler" allows this to run on a CPU-only host.
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
# close() is intentionally not called, to model a host that skips shutdown.
"""


def test_interpreter_exits_when_close_is_skipped() -> None:
    """A child that builds a StorageManager and skips close() must still exit."""
    # subprocess.run kills and reaps the child on timeout or any other exception.
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
        # Output captured before the timeout is undecoded.
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
