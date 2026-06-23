# SPDX-License-Identifier: Apache-2.0
"""Shared path-sharding logic for multi-path storage backends.

A :class:`PathSharder` takes a comma-separated list of directory paths
and a sharding strategy, then assigns one or more paths to the current
worker.  Both :class:`LocalDiskBackend` and :class:`GdsBackend` delegate
path selection to this module so the policy lives in one place.

Only the ``"by_gpu"`` strategy exists. Its behavior depends on whether
the caller supplies local-worker metadata:

* **Without** ``local_rank`` / ``local_world_size`` (e.g. the GDS
  backend): legacy single-path mapping keyed by the CUDA device index,
  ``paths[device_id % len(paths)]``.
* **With** ``local_rank`` / ``local_world_size`` (e.g. the local disk
  backend): worker-aware assignment keyed by the worker's local rank.
  When there are more storage devices than workers, a worker is assigned
  a contiguous *subset* of paths (e.g. 2 GPUs x 4 NVMe drives -> 2 drives
  per GPU) so that no device sits idle.

The two branches coincide on a single-node tensor-parallel deployment,
where ``local_rank == device_id`` -- the path layout is unchanged.  They
diverge only when the physical device index does not match the logical
worker rank (e.g. ``CUDA_VISIBLE_DEVICES`` masking or data-parallel
replicas), in which case the worker-aware branch is the coherent choice.
"""

# Standard
import os

# First Party
from lmcache import torch_dev
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

logger = init_logger(__name__)


def _mix_u64(value: int) -> int:
    """Mix an integer into a stable 64-bit hash value.

    SplitMix64 provides a cheap deterministic mixer that spreads low-bit
    patterns before modulo-based path selection, avoiding skew when a
    worker is assigned only a small number of paths.

    Args:
        value: Arbitrary integer to mix.

    Returns:
        A well-distributed 64-bit unsigned integer derived from *value*.
    """
    value &= 0xFFFFFFFFFFFFFFFF
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    value ^= value >> 31
    return value


def _resolve_device_id(dst_device: str) -> int:
    """Derive an integer device index from *dst_device*.

    Args:
        dst_device: Device string such as ``"cuda:2"``, ``"cuda"``,
            or ``"cpu"``.

    Returns:
        Integer device index.  Falls back to
        :func:`torch_dev.current_device` when the string carries no
        explicit index, or ``0`` when the accelerator is unavailable.
    """
    if ":" in dst_device:
        try:
            return int(dst_device.split(":", 1)[1])
        except ValueError:
            logger.warning(f"Invalid device index in '{dst_device}', falling back.")
    if torch_dev.is_available() and dst_device != "cpu":
        return torch_dev.current_device()
    return 0


class PathSharder:
    """Assign one or more paths from a comma-separated list to a worker.

    Args:
        raw_csv: Comma-separated directory paths (e.g.
            ``"/mnt/nvme0/cache,/mnt/nvme1/cache"``).
        strategy: Sharding strategy name.  Only ``"by_gpu"`` is supported.
        dst_device: Device string used to derive the worker index when no
            local-rank metadata is supplied (e.g. ``"cuda:0"``, ``"cuda"``,
            ``"cpu"``).
        local_rank: Local worker rank on the current host.  When provided
            together with *local_world_size*, path assignment is keyed by
            this rank instead of the device index, enabling subset
            assignment.  When ``None``, the legacy device-index mapping is
            used.
        local_world_size: Number of local workers on the current host.
            Must be provided together with *local_rank*.
        create_dirs: If ``True``, create **all** directories in the list at
            construction time (not just the assigned ones).

    Raises:
        ValueError: If *raw_csv* is empty or contains no valid paths.
        ValueError: If *strategy* is not a supported sharding mode.
        ValueError: If exactly one of *local_rank* / *local_world_size* is
            provided.
        ValueError: If the path count and worker count are not equal or an
            exact multiple of each other.

    Example::

        sharder = PathSharder(
            "/mnt/nvme0/cache,/mnt/nvme1/cache",
            strategy="by_gpu",
            dst_device="cuda:1",
        )
        sharder.selected   # "/mnt/nvme1/cache"
        sharder.all_paths  # ["/mnt/nvme0/cache", "/mnt/nvme1/cache"]

        # Worker-aware subset (2 workers, 4 paths):
        sharder = PathSharder(
            "/d0,/d1,/d2,/d3",
            strategy="by_gpu",
            dst_device="cuda:1",
            local_rank=1,
            local_world_size=2,
        )
        sharder.assigned_paths  # ["/d2", "/d3"]
    """

    _SUPPORTED_STRATEGIES = ("by_gpu",)

    def __init__(
        self,
        raw_csv: str,
        strategy: str,
        dst_device: str,
        local_rank: int | None = None,
        local_world_size: int | None = None,
        create_dirs: bool = False,
    ) -> None:
        paths = [os.path.normpath(p.strip()) for p in raw_csv.split(",") if p.strip()]
        if not paths:
            raise ValueError("At least one path must be provided")

        if strategy not in self._SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unsupported path sharding strategy '{strategy}'. "
                f"Supported: {', '.join(self._SUPPORTED_STRATEGIES)}"
            )

        if (local_rank is None) != (local_world_size is None):
            raise ValueError(
                "local_rank and local_world_size must be provided together "
                f"(got local_rank={local_rank}, "
                f"local_world_size={local_world_size})"
            )

        self._all_paths: list[str] = paths
        self._strategy: str = strategy

        if local_rank is None or local_world_size is None:
            # Legacy mapping: keep behavior identical to the original
            # device-index single-path selection.  This is the branch the
            # GDS backend takes, so it must never gain new validation or
            # subset behavior.
            device_id = _resolve_device_id(dst_device)
            assigned_paths = [paths[device_id % len(paths)]]
        else:
            assigned_paths = self._assign_paths_for_worker(
                paths=paths,
                local_rank=local_rank,
                local_world_size=local_world_size,
            )

        self._assigned_paths: list[str] = assigned_paths
        self._selected: str = assigned_paths[0]

        if create_dirs:
            for p in paths:
                os.makedirs(p, exist_ok=True)

    def resolve_path_for_key(self, key: CacheEngineKey) -> str:
        """Resolve the storage path for ``key`` on the current worker.

        Args:
            key: Cache key whose ``chunk_hash`` selects among the worker's
                assigned paths when more than one path is assigned.

        Returns:
            Absolute path to the directory where ``key`` should be stored
            or read from.
        """
        if len(self._assigned_paths) == 1:
            return self._assigned_paths[0]

        path_index = _mix_u64(int(key.chunk_hash)) % len(self._assigned_paths)
        return self._assigned_paths[path_index]

    # -- public read-only properties -----------------------------------------

    @property
    def selected(self) -> str:
        """The first assigned path, kept for backward compatibility."""
        return self._selected

    @property
    def all_paths(self) -> list[str]:
        """All configured paths (unmodified order)."""
        return list(self._all_paths)

    @property
    def assigned_paths(self) -> list[str]:
        """Paths assigned to the current worker (length >= 1)."""
        return list(self._assigned_paths)

    @property
    def strategy(self) -> str:
        """Name of the active sharding strategy."""
        return self._strategy

    def _assign_paths_for_worker(
        self,
        paths: list[str],
        local_rank: int,
        local_world_size: int,
    ) -> list[str]:
        """Assign a subset of *paths* to the worker identified by *local_rank*.

        Three cases are handled:

        * **workers == paths**: one-to-one mapping; each worker owns exactly
          one device (e.g. 4 GPUs, 4 NVMe drives).
        * **workers > paths**: multiple workers share devices; allowed only
          when the worker count is an exact multiple of the path count
          (e.g. 4 GPUs, 2 NVMe drives -> GPU 0 and 2 share drive 0, GPU 1
          and 3 share drive 1).
        * **workers < paths**: each worker owns a contiguous block of
          devices; *num_paths* must be divisible by *local_world_size* to
          guarantee even distribution (e.g. 2 GPUs, 4 NVMe drives -> GPU 0
          owns drives 0-1, GPU 1 owns drives 2-3).

        Args:
            paths: Ordered list of all configured storage paths.
            local_rank: Zero-based index of this worker on the current host.
            local_world_size: Total number of workers on the current host.

        Returns:
            List of paths assigned to this worker (length >= 1).

        Raises:
            ValueError: If *local_world_size* is not positive, if
                *local_rank* is out of range, or if the worker count and
                path count are not equal and neither is an exact multiple
                of the other.
        """
        num_paths = len(paths)
        num_workers = local_world_size

        if num_workers <= 0:
            raise ValueError(
                "by_gpu worker-aware assignment requires local_world_size to "
                f"be positive (got {num_workers})"
            )

        if local_rank < 0 or local_rank >= num_workers:
            raise ValueError(
                "by_gpu worker-aware assignment requires local_rank to be in "
                f"the range [0, local_world_size) (got local_rank={local_rank}, "
                f"local_world_size={num_workers})"
            )

        if (
            num_workers != num_paths
            and max(num_workers, num_paths) % min(num_workers, num_paths) != 0
        ):
            raise ValueError(
                "by_gpu worker-aware assignment requires local_world_size and "
                "num_paths to be equal or exact multiples of each other "
                f"(got {num_paths} paths and {num_workers} local workers)"
            )

        if num_workers >= num_paths:
            return [paths[local_rank % num_paths]]

        paths_per_worker = num_paths // num_workers
        start = local_rank * paths_per_worker
        end = start + paths_per_worker
        return paths[start:end]
