# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for platform cache contexts.

Defines the common interface shared by :class:`GPUCacheContext` and
:class:`CPUCacheContext`.  Concrete subclasses provide
device-specific implementations of stream / buffer / copy primitives
while the base class owns layout-agnostic helpers (shape calculation,
status reporting, block-ID staging).
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar
import array

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.utils import (
    get_attention_backend,
    get_concrete_engine_kv_shape_from_shape_desc,
    get_engine_kv_shape_description,
    is_mla,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

if TYPE_CHECKING:
    # First Party
    import lmcache.c_ops as lmc_ops


class TransferLane:
    """One retrieve transfer queue on a cache context: a stream + private
    staging. Lane 0 aliases the context's own stream/buffers (``buffers`` /
    ``block_ids_buffer`` None => use the context). ``guarded`` (context
    runs >1 lanes) arms consumers' same-lane staging reuse guards."""

    __slots__ = (
        "index",
        "stream",
        "cupy_stream",
        "buffers",
        "block_ids_buffer",
        "guarded",
    )

    def __init__(
        self,
        index: int,
        stream: Any,
        cupy_stream: Any,
        buffers: Any = None,
        block_ids_buffer: "torch.Tensor | None" = None,
        guarded: bool = False,
    ) -> None:
        self.index = index
        self.stream = stream
        self.cupy_stream = cupy_stream
        self.buffers = buffers
        self.block_ids_buffer = block_ids_buffer
        self.guarded = guarded


class BaseCacheContext(ABC):
    """Abstract base for GPU and CPU cache contexts.

    Subclasses call :meth:`__init__` after computing the common
    layout parameters and before setting up device-specific state.
    All keyword arguments are required so the contract is explicit.

    Concrete subclasses MUST set :attr:`device_type` to the
    ``torch.device.type`` string they handle (``"cuda"``, ``"cpu"``,
    ...). The platform-agnostic :func:`create_cache_context` factory
    uses this attribute (via the platform registry) to pick the right
    subclass without any ``isinstance`` / ``if-elif`` chain.
    """

    #: ``torch.device.type`` string the subclass handles. Concrete
    #: subclasses MUST override this.
    device_type: ClassVar[str] = ""

    def __init__(
        self,
        *,
        kv_caches: list[torch.Tensor],
        device: torch.device,
        num_layers: int,
        kv_layer_groups_manager: KVLayerGroupsManager,
        block_ids_buffer: torch.Tensor,
        lmcache_tokens_per_chunk: int,
    ) -> None:
        self.kv_caches_ = kv_caches
        self.device_ = device
        self.num_layers_ = num_layers
        self.kv_layer_groups_manager_ = kv_layer_groups_manager
        self.block_ids_buffer_ = block_ids_buffer
        self.lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        # Retrieve scatter lanes (see next_retrieve_lane); the count is
        # resolved once at first use, lanes created lazily or by
        # prewarm_retrieve_lanes.
        self.retrieve_lanes_: "list[TransferLane | None]" = []
        self.retrieve_lane_next_ = 0
        self.retrieve_lane_count_: "int | None" = None

    # ------------------------------------------------------------------
    # Abstract -- subclasses MUST implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def stream(self) -> Any:
        """Returns the device-specific stream for async operations."""
        ...

    @property
    @abstractmethod
    def cupy_stream(self) -> Any:
        """Returns the cupy ExternalStream wrapping *stream*."""
        ...

    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Returns the maximum number of concurrent batches."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release device-specific resources (GDS staging buffers, etc.)."""
        ...

    @abstractmethod
    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Returns the KV-cache pointer tensor for *kernel_group_idx*."""
        ...

    @abstractmethod
    def get_temp_kernel_group_buffer(
        self, batch_idx: int, kernel_group_idx: int
    ) -> torch.Tensor:
        """Returns a typed temp-buffer view for a (batch, kernel-group)
        pair."""
        ...

    @abstractmethod
    def get_temp_object_group_buffer(
        self, batch_idx: int, object_group_idx: int
    ) -> torch.Tensor:
        """Returns a flat uint8 temp-buffer view for a (batch, object-group)
        pair."""
        ...

    @abstractmethod
    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Returns ``(shape, dtype)`` for *kernel_group_idx*."""
        ...

    @abstractmethod
    def cache_size_per_token(self) -> int:
        """Returns cache size per logical token in bytes (all groups)."""
        ...

    # ------------------------------------------------------------------
    # Concrete -- shared implementations
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """Returns the device where KV-cache tensors live."""
        return self.device_

    @property
    def kv_tensors(self) -> list[torch.Tensor]:
        """Returns the list of per-layer KV cache tensors."""
        return self.kv_caches_

    @property
    def num_layers(self) -> int:
        """Returns the number of layers in the model."""
        return self.num_layers_

    @property
    def num_blocks(self) -> int:
        """Returns the number of blocks in the KV cache.

        Sourced from the kernel groups (one shared block-id space), not a
        representative-format computation.
        """
        return self.kv_layer_groups_manager_.num_blocks

    @property
    def hidden_dim_sizes(self) -> list[int]:
        """Returns hidden dimension sizes per KV layer group."""
        return [
            group.hidden_dim_size
            for group in self.kv_layer_groups_manager.kernel_groups
        ]

    @property
    def kv_layer_groups_manager(self) -> KVLayerGroupsManager:
        """Returns the KV layer groups manager."""
        return self.kv_layer_groups_manager_

    def calculate_num_blocks(self, num_tokens: int, kernel_group_idx: int) -> int:
        """Calculate the number of blocks for *num_tokens* in a kernel
        group."""
        return self.kv_layer_groups_manager.calculate_num_blocks(
            kernel_group_idx, num_tokens
        )

    def get_shape_desc(self, group_idx: int) -> "lmc_ops.PageBufferShapeDesc":
        """Returns the PageBufferShapeDesc for *group_idx*."""
        return self.kv_layer_groups_manager.get_shape_desc(group_idx)

    def get_engine_kv_format(self, kernel_group_idx: int) -> "lmc_ops.EngineKVFormat":
        """Returns the Engine KV format of kernel *kernel_group_idx*.

        Raises:
            ValueError: If the group has no format (a bookkeeping group built by
                ``parse_kvcache_shape_spec`` should never reach the transfer
                path; detection-built groups always carry one).
        """
        groups = self.kv_layer_groups_manager.kernel_groups
        engine_kv_format = groups[kernel_group_idx].engine_kv_format
        if engine_kv_format is None:
            raise ValueError(
                f"kernel group {kernel_group_idx} has no engine_kv_format; a "
                "formatless bookkeeping group reached the transfer path"
            )
        return engine_kv_format

    def engine_kv_formats(self) -> list["lmc_ops.EngineKVFormat"]:
        """Returns the Engine KV format of each kernel group, in group order."""
        num_groups = len(self.kv_layer_groups_manager.kernel_groups)
        return [self.get_engine_kv_format(idx) for idx in range(num_groups)]

    def engine_kv_format_per_layer(self) -> list["lmc_ops.EngineKVFormat | None"]:
        """Returns each layer's Engine KV format, indexed by layer index.

        Formats differ across layers for a mixed-format model. ``None`` marks a
        layer in no kernel group (a cross-layer KV-sharing layer).
        """
        formats: list["lmc_ops.EngineKVFormat | None"] = [None] * len(self.kv_caches_)
        for kernel_group_idx, group in enumerate(
            self.kv_layer_groups_manager.kernel_groups
        ):
            fmt = self.get_engine_kv_format(kernel_group_idx)
            for layer_idx in group.layer_indices:
                formats[layer_idx] = fmt
        return formats

    def get_slots_per_chunk_in_sw(self, kernel_group_idx: int) -> int:
        """Returns the number of slots per lmcache chunk for D/H
        transfer."""
        return self.kv_layer_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)

    def get_kv_buffer_shape(
        self, logical_num_tokens: int, group_idx: int = 0
    ) -> torch.Size:
        """Returns the KV buffer shape for *logical_num_tokens*."""
        group = self.kv_layer_groups_manager.kernel_groups[group_idx]
        compress_ratio = group.tokens_per_block // group.slots_per_block
        if logical_num_tokens % compress_ratio != 0:
            raise ValueError(
                "logical_num_tokens (%d) is not a multiple of "
                "compress_ratio (%d) for group %d"
                % (logical_num_tokens, compress_ratio, group_idx)
            )
        num_slots = logical_num_tokens // compress_ratio
        sd = group.shape_desc
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def stage_block_ids(
        self,
        block_ids_per_group: list[list[int]],
        out: "torch.Tensor | None" = None,
    ) -> list[torch.Tensor]:
        """Stage per-group block IDs into the shared staging buffer, or into
        ``out``: retrieve lanes pass their private buffer so one lane's
        staging cannot overwrite block IDs another lane's queued kernels
        still read.

        Returns one non-overlapping view per LMCache group.
        """
        buffer = self.block_ids_buffer_ if out is None else out
        offsets = [0]
        flat: array.array = array.array("q")
        for view_block_ids in block_ids_per_group:
            flat.extend(view_block_ids)
            offsets.append(len(flat))

        total = offsets[-1]
        if total > buffer.shape[0]:
            raise ValueError(
                "block ID total %d exceeds the pre-allocated buffer "
                "size %d" % (total, buffer.shape[0])
            )
        if total:
            cpu_tensor = torch.frombuffer(flat, dtype=torch.long)
            buffer[:total].copy_(cpu_tensor, non_blocking=True)

        return [
            buffer[offsets[i] : offsets[i + 1]] for i in range(len(block_ids_per_group))
        ]

    # ------------------------------------------------------------------
    # Retrieve scatter lanes
    # ------------------------------------------------------------------
    # Concurrent retrieves serialize on the single per-context stream:
    # request N's completion event drains behind requests 1..N-1. Lanes are
    # extra transfer streams, each with private staging, so requests'
    # copies overlap; per-lane stream ordering keeps staging reuse safe
    # without locks. 1 lane = stock behavior.

    def supports_retrieve_lanes(self) -> bool:
        """Whether this platform can provision extra transfer streams +
        staging. Base: single-lane only (lane 0 is always safe)."""
        return False

    def _make_retrieve_lane(self, index: int, guarded: bool) -> TransferLane:
        """Build lane ``index``. Lane 0 aliases the context's own stream and
        staging (nothing allocated); platforms answering True to
        :meth:`supports_retrieve_lanes` must provision lanes >= 1."""
        if index == 0:
            return TransferLane(0, self.stream, self.cupy_stream, guarded=guarded)
        raise NotImplementedError(
            f"{type(self).__name__} does not provision retrieve lanes >= 1"
        )

    def _retrieve_lane_slots(self, num_lanes: int) -> int:
        """Resolve the lane count once per context (first provisioning
        wins) and size the slot list, so consumers with different resolved
        counts share one stable round-robin modulus."""
        if self.retrieve_lane_count_ is None:
            self.retrieve_lane_count_ = (
                max(1, num_lanes) if self.supports_retrieve_lanes() else 1
            )
            self.retrieve_lanes_ = [None] * self.retrieve_lane_count_
        return self.retrieve_lane_count_

    def next_retrieve_lane(self, num_lanes: int) -> TransferLane:
        """Round-robin the next retrieve lane, creating it on first use.

        ``num_lanes`` is the consumer-resolved count (policy lives with the
        consumers); 1 => always lane 0, i.e. the stock single-stream
        behavior. A context's transfers run on its one affinity thread, so
        the counter needs no lock.
        """
        n = self._retrieve_lane_slots(num_lanes)
        idx = self.retrieve_lane_next_ % n
        self.retrieve_lane_next_ = (idx + 1) % n
        lane = self.retrieve_lanes_[idx]
        if lane is None:
            lane = self._make_retrieve_lane(idx, guarded=n > 1)
            self.retrieve_lanes_[idx] = lane
        return lane

    def prewarm_retrieve_lanes(
        self, num_lanes: int
    ) -> "tuple[list[TransferLane], int]":
        """Create every lane now, off the retrieve critical path. Leaves
        the round-robin counter untouched.

        Returns:
            ``(lanes, created)``: all lanes in index order and how many this
            call created (0 => already warm, e.g. a re-registration).
        """
        n = self._retrieve_lane_slots(num_lanes)
        created = 0
        for idx in range(n):
            if self.retrieve_lanes_[idx] is None:
                self.retrieve_lanes_[idx] = self._make_retrieve_lane(idx, guarded=n > 1)
                created += 1
        return [lane for lane in self.retrieve_lanes_ if lane is not None], created

    # ------------------------------------------------------------------
    # Derived properties (pure helpers)
    # ------------------------------------------------------------------

    @property
    def concrete_engine_kv_shape(self) -> str:
        """Returns the engine KV shape with actual numeric values."""
        group = self.kv_layer_groups_manager.kernel_groups[0]
        return get_concrete_engine_kv_shape_from_shape_desc(
            group.shape_desc, group.engine_kv_format
        )

    # ------------------------------------------------------------------
    # Shared report_status
    # ------------------------------------------------------------------

    def _build_group_report_map(self) -> dict[int, int]:
        """Map each kernel-group index to its owning object-group index."""
        return {
            kg_idx: og_idx
            for og_idx, og in enumerate(self.kv_layer_groups_manager.object_groups)
            for kg_idx in og.kernel_group_indices
        }

    def _build_single_group_report(
        self,
        kernel_group_idx: int,
        group: Any,
        group_map: dict[int, int],
    ) -> dict:
        """Build a status dict for a single kernel group.

        Override this in subclasses to inject extra per-group fields
        without duplicating the whole :meth:`report_status` method.
        """
        engine_kv_format = self.get_engine_kv_format(kernel_group_idx)
        return {
            "kernel_group_idx": kernel_group_idx,
            "engine_group_idx": group.engine_group_idx,
            "object_group_idx": group_map.get(kernel_group_idx, 0),
            "num_layers": group.num_layers,
            "layer_indices": list(group.layer_indices),
            "tokens_per_block": group.tokens_per_block,
            "slots_per_block": group.slots_per_block,
            "dtype": str(group.dtype),
            "engine_kv_concrete_shape": (
                get_concrete_engine_kv_shape_from_shape_desc(
                    group.shape_desc, engine_kv_format
                )
            ),
            "is_mla": is_mla(engine_kv_format),
            "engine_kv_format": engine_kv_format.name,
            "engine_kv_shape": get_engine_kv_shape_description(engine_kv_format),
            "attention_backend": get_attention_backend(engine_kv_format),
        }

    def report_status(self) -> dict:
        """Return this context's KV cache layout metadata."""
        manager = self.kv_layer_groups_manager
        kernel_groups = manager.kernel_groups
        group_map = self._build_group_report_map()

        group_reports = [
            self._build_single_group_report(kernel_group_idx, group, group_map)
            for kernel_group_idx, group in enumerate(kernel_groups)
        ]

        return {
            "num_layers": self.num_layers,
            "num_blocks": self.num_blocks,
            "cache_size_per_token": self.cache_size_per_token(),
            "kernel_groups": group_reports,
        }
