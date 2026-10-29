# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration of Mooncake process fencing and remote admission."""

# Standard
from dataclasses import dataclass, fields
from typing import Any
import math
import os
import sys

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.base import L2TaskId
from lmcache.v1.distributed.l2_adapters.mooncake_process_client import (
    ProcessNativeClient,
)
from lmcache.v1.distributed.l2_adapters.mooncake_resilience import (
    GuardConfig,
    GuardedNativeClient,
    SubprocessDataProbe,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)


@dataclass(frozen=True)
class MooncakeProcessConfig:
    """Native deadlines and capacity for one exclusively owned SDK child.

    All durations are positive finite seconds. The caller must keep each L1
    allocation pinned until terminal completion and use a fresh SHM name for
    each service generation.
    """

    operation_timeout: float = 8.0
    startup_timeout: float = 15.0
    kill_timeout: float = 10.0
    restart_interval: float = 2.0
    max_pending: int = 16

    def __post_init__(self) -> None:
        """Reject invalid options before allocating process resources."""
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or (name == "max_pending" and not isinstance(value, int))
            ):
                raise ValueError(f"{name} must be a positive finite number")


def parse_process_options(
    value: object,
) -> tuple[MooncakeProcessConfig, GuardConfig]:
    """Validate a process-mode option dictionary without starting resources.

    Args:
        value: Dictionary containing process fields and optional ``guard`` dict.

    Returns:
        Validated process and circuit policies.

    Raises:
        ValueError: Unknown keys, malformed dictionaries or invalid values.
    """
    if not isinstance(value, dict):
        raise ValueError("process must be a dictionary")
    options = dict(value)
    guard = options.pop("guard", {})
    if not isinstance(guard, dict):
        raise ValueError("process.guard must be a dictionary")
    if options.keys() - {f.name for f in fields(MooncakeProcessConfig)}:
        raise ValueError("Unknown Mooncake process option")
    if guard.keys() - {f.name for f in fields(GuardConfig)}:
        raise ValueError("Unknown Mooncake guard option")
    return MooncakeProcessConfig(**options), GuardConfig(**guard)


def create_process_adapter(
    setup_config: dict[str, str],
    num_workers: int,
    per_op_workers: dict[str, int] | None,
    descriptor: L1MemoryDesc | None,
    process_config: MooncakeProcessConfig,
    guard_config: GuardConfig,
) -> "MooncakeProcessAdapter":
    """Create an SDK-free parent bridge over an exact named shared L1 arena.

    Raises:
        ValueError: The descriptor is missing, anonymous or invalid.
        RuntimeError: Process isolation is unavailable on this platform.
        OSError: The actual mapping does not match its backing file.

    Failed construction closes every resource acquired by this factory.
    """
    if sys.platform != "linux":
        raise RuntimeError("Process Mooncake transport requires Linux")
    if (
        descriptor is None
        or not descriptor.shm_name
        or descriptor.ptr <= 0
        or descriptor.size <= 0
    ):
        raise ValueError("Process Mooncake requires an allocator-owned named SHM arena")
    native_notifier = create_event_notifier()
    try:
        native = ProcessNativeClient(
            setup_config=setup_config,
            num_workers=num_workers,
            per_op_workers=per_op_workers,
            l1_base=descriptor.ptr,
            l1_size=descriptor.size,
            shm_name=descriptor.shm_name,
            notifier=native_notifier,
            **vars(process_config),
        )
    except BaseException:
        native_notifier.close()
        raise
    notifier = None
    probe = None
    guard = None
    try:
        notifier = create_event_notifier()
        probe = SubprocessDataProbe(
            setup_config, guard_config.probe_timeout, process_config.kill_timeout
        )
        guard = GuardedNativeClient(lambda: native, probe, notifier, guard_config)
        return MooncakeProcessAdapter(guard, native)
    except BaseException:
        if guard is not None:
            try:
                guard.close()
            finally:
                native.close()
        else:
            if probe is not None:
                probe.close()
            if notifier is not None:
                notifier.close()
            native.close()
        raise


class MooncakeProcessAdapter(NativeConnectorL2Adapter):
    """Fence all native access before reporting a failed data submission.

    Controllers release their I/O pins when submission raises. An exception
    after native admission or bridge bookkeeping must therefore stop native
    access first. Normal deadline failures remain asynchronous and recoverable.
    """

    def __init__(self, guard: GuardedNativeClient, native: ProcessNativeClient) -> None:
        """Take ownership of a process-backed guard and start completion delivery."""
        self._guard = guard
        self._native_process = native
        super().__init__(
            guard, MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)
        )

    def submit_store_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Submit STORE; any raised exception guarantees no native buffer access."""
        try:
            return super().submit_store_task(keys, objects)
        except BufferError:
            raise
        except Exception:
            self._fence_submission_error()
            raise

    def submit_load_task(
        self, keys: list[ObjectKey], objects: list[MemoryObj]
    ) -> L2TaskId:
        """Submit GET; any raised exception guarantees no native buffer access."""
        try:
            return super().submit_load_task(keys, objects)
        except BufferError:
            raise
        except Exception:
            self._fence_submission_error()
            raise

    def report_status(self) -> dict[str, Any]:
        """Return bridge status and the live circuit/worker health snapshot."""
        return {
            **super().report_status(),
            "resilience": self._guard.report_status(),
            "process": self._native_process.report_status(),
        }

    def close(self) -> None:
        """Close the native process even before the guard adopts its factory result."""
        try:
            super().close()
        finally:
            self._native_process.close()

    def _fence_submission_error(self) -> None:
        logger.exception("Mooncake data submission failed; fencing native access")
        try:
            self._guard.quiesce()
        except BaseException:
            logger.exception("Cannot confirm Mooncake I/O fencing; exiting")
            os._exit(78)
