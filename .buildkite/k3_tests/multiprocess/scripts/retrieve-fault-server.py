# SPDX-License-Identifier: Apache-2.0
"""Run the normal MP server with a CI-only barrier before a retrieve RPC reply."""

# Standard
from functools import wraps
from pathlib import Path
from threading import Event, Lock
from unittest.mock import patch
import os
import sys

# First Party
from lmcache.cli.main import main
from lmcache.v1.common.custom_types import IPCCacheServerKey, PrepareRetrieveResponse
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)


def serve(fault_dir: Path, server_args: list[str]) -> None:
    """Run the LMCache CLI, pausing retrieves once the test arms the barrier.

    Args:
        fault_dir: Per-run directory shared with the test driver. The driver
            creates `armed`. The first paused retrieve publishes its PID, RPC name, and
            request ID in `retrieve-pending` before waiting for process death.
        server_args: The original LMCache CLI arguments, including `server`.
    """
    original_retrieve = LMCacheDrivenTransferModule.retrieve
    original_prepare = EngineDrivenTransferModule.prepare_retrieve
    marker_lock = Lock()
    armed = fault_dir / "armed"
    pending = fault_dir / "retrieve-pending"

    def pause_if_armed(operation: str, key: IPCCacheServerKey) -> None:
        if armed.is_file():
            with marker_lock:
                if not pending.exists():
                    marker = pending.with_suffix(".tmp")
                    marker.write_text(f"{os.getpid()} {operation} {key.request_id}\n")
                    marker.replace(pending)
            # Bound the wait so a failed test driver cannot leave a handler
            # stuck forever. Never send a successful reply for an armed call.
            Event().wait(timeout=120)
            raise RuntimeError("CI did not kill the MP server at the retrieve barrier")

    # RPC discovery reads the handler metadata and signature from the method.
    @wraps(original_retrieve)
    def retrieve(
        module: LMCacheDrivenTransferModule,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        pause_if_armed("RETRIEVE", key)
        return original_retrieve(
            module,
            key,
            instance_id,
            gpu_block_ids,
            event_ipc_handle,
            skip_first_n_tokens,
        )

    @wraps(original_prepare)
    def prepare_retrieve(
        module: EngineDrivenTransferModule, key: IPCCacheServerKey, instance_id: int
    ) -> PrepareRetrieveResponse:
        pause_if_armed("PREPARE_RETRIEVE", key)
        return original_prepare(module, key, instance_id)

    with (
        patch.object(LMCacheDrivenTransferModule, "retrieve", retrieve),
        patch.object(EngineDrivenTransferModule, "prepare_retrieve", prepare_retrieve),
    ):
        main(server_args)


if __name__ == "__main__":
    serve(Path(sys.argv[1]), sys.argv[2:])
