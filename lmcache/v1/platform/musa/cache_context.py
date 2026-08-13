# SPDX-License-Identifier: Apache-2.0
"""MUSA cache context for LMCache-driven multiprocess transfer."""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.lmcache_native import EngineKVFormat
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.types import DiscoverableKVCache
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    get_device,
    get_group_data_ptrs,
    normalize_and_discover_per_layer_formats,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.group_view import engine_group_layer_indices
from lmcache.v1.platform.base.cache_context import BaseCacheContext

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)


def unwrap_kv_cache_tensors(kv_caches: KVCache) -> list[torch.Tensor]:
    """Return tensors reconstructed from MUSA IPC wrappers.

    Args:
        kv_caches: MUSA IPC wrappers received over the multiprocess wire.

    Returns:
        The reconstructed MUSA KV-cache tensors.
    """
    return [wrapper.to_tensor() for wrapper in kv_caches]


class _MUSAHostCallbackStream:
    """Small adapter for stream-ordered callback call sites.

    CUDA contexts expose a CuPy stream as ``cupy_stream``. MUSA does not use
    CuPy here, so this adapter provides ordered callback and synchronization
    operations over a TorchMUSA stream and exposes its pointer through the
    existing platform stream contract.
    """

    def __init__(self, stream: object) -> None:
        self._stream = stream

    def synchronize(self) -> None:
        """Wait for all work submitted to the wrapped MUSA stream."""
        synchronize = getattr(self._stream, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError("MUSA stream does not support synchronization")
        synchronize()

    @property
    def ptr(self) -> int:
        """Return the wrapped MUSA stream pointer for platform call sites."""
        pointer = getattr(self._stream, "musa_stream", None)
        if pointer is None:
            pointer = getattr(self._stream, "ptr", None)
        if pointer is None:
            raise RuntimeError("MUSA stream does not expose a stream pointer")
        return int(pointer)

    def launch_host_func(self, callback: Any, arg: Any = None) -> None:
        """Schedule or run ``callback(arg)``.

        If the backend stream exposes ``launch_host_func`` directly, delegate
        to it. Otherwise synchronize the stream before running the callback on
        the current thread.
        """
        launch_host_func = getattr(self._stream, "launch_host_func", None)
        if callable(launch_host_func):
            launch_host_func(callback, arg)
            return
        self.synchronize()
        callback(arg)


class _TempMUSABuffer:
    """Owns MUSA staging buffers for MP block-transfer batches."""

    def __init__(
        self,
        kv_layer_groups_manager: KVLayerGroupsManager,
        lmcache_tokens_per_chunk: int,
        device: torch.device,
        max_batch_size: int = 4,
    ) -> None:
        self._kv_groups_manager = kv_layer_groups_manager
        self._lmcache_tokens_per_chunk = lmcache_tokens_per_chunk
        self._max_batch_size = max_batch_size

        self._temp_buffer = torch.empty(
            self._get_size_for_single_batch() * max_batch_size,
            dtype=torch.uint8,
            device=device,
        )
        self._offset_map_kernel_group_only: dict[tuple[int, int], tuple[int, int]] = {}
        self._offset_map_object_group_only: dict[tuple[int, int], tuple[int, int]] = {}

        offset = 0
        for batch_idx in range(max_batch_size):
            for object_group_idx in range(self._kv_groups_manager.num_object_groups):
                object_group_start_offset = offset
                object_group_size = 0
                object_group = self._kv_groups_manager.object_groups[object_group_idx]
                for kernel_group_idx in object_group.kernel_group_indices:
                    size = self._get_size_for_kernel_group(kernel_group_idx)
                    self._offset_map_kernel_group_only[
                        (batch_idx, kernel_group_idx)
                    ] = (
                        offset,
                        size,
                    )
                    offset += size
                    object_group_size += size

                self._offset_map_object_group_only[(batch_idx, object_group_idx)] = (
                    object_group_start_offset,
                    object_group_size,
                )

        self._shape_cache_kernel_group: dict[int, tuple[torch.Size, torch.dtype]] = {}
        for kernel_group_idx in range(self._kv_groups_manager.num_kernel_groups):
            shape = self._get_shape_for_kernel_group(
                self._lmcache_tokens_per_chunk,
                kernel_group_idx,
            )
            group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
            self._shape_cache_kernel_group[kernel_group_idx] = (shape, group.dtype)

    @property
    def max_batch_size(self) -> int:
        """Return the number of chunks that fit in the staging buffer."""
        return self._max_batch_size

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return a typed staging view for a batch/kernel-group pair."""
        key = (batch_idx, kernel_group_idx)
        if key not in self._offset_map_kernel_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or kernel_group_idx {kernel_group_idx}"
            )

        offset, size = self._offset_map_kernel_group_only[key]
        shape, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._temp_buffer[offset : offset + size].view(dtype).view(shape)

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return a flat ``uint8`` staging view for a batch/object-group pair."""
        key = (batch_idx, object_group_idx)
        if key not in self._offset_map_object_group_only:
            raise ValueError(
                f"Invalid batch_idx {batch_idx} or object_group_idx {object_group_idx}"
            )

        offset, size = self._offset_map_object_group_only[key]
        return self._temp_buffer[offset : offset + size]

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return ``(shape, dtype)`` for a kernel group and token count."""
        _, dtype = self._shape_cache_kernel_group[kernel_group_idx]
        return self._get_shape_for_kernel_group(num_tokens, kernel_group_idx), dtype

    def get_cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._get_size_for_single_batch() // self._lmcache_tokens_per_chunk

    def _get_shape_for_kernel_group(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> torch.Size:
        if num_tokens % self._lmcache_tokens_per_chunk != 0:
            raise ValueError(
                f"num_tokens ({num_tokens}) must be a multiple of "
                f"lmcache_tokens_per_chunk ({self._lmcache_tokens_per_chunk})"
            )

        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        sd = group.shape_desc
        num_chunks = num_tokens // self._lmcache_tokens_per_chunk
        num_slots = (
            self._kv_groups_manager.get_slots_per_chunk_in_sw(kernel_group_idx)
            * num_chunks
        )
        if group.engine_kv_format == EngineKVFormat.NL_X_NB_BS_HS:
            return torch.Size((group.num_layers, num_slots, group.hidden_dim_size))
        return torch.Size(
            (sd.kv_size, group.num_layers, num_slots, group.hidden_dim_size)
        )

    def _get_size_for_kernel_group(self, kernel_group_idx: int) -> int:
        shape = self._get_shape_for_kernel_group(
            self._lmcache_tokens_per_chunk,
            kernel_group_idx,
        )
        group = self._kv_groups_manager.kernel_groups[kernel_group_idx]
        return shape.numel() * group.dtype.itemsize

    def _get_size_for_object_group(self, object_group_idx: int) -> int:
        object_group = self._kv_groups_manager.object_groups[object_group_idx]
        return sum(
            self._get_size_for_kernel_group(kernel_group_idx)
            for kernel_group_idx in object_group.kernel_group_indices
        )

    def _get_size_for_single_batch(self) -> int:
        return sum(
            self._get_size_for_object_group(object_group_idx)
            for object_group_idx in range(self._kv_groups_manager.num_object_groups)
        )


class MUSACacheContext(BaseCacheContext):
    """Cache context for MUSA-backed KV tensors in MP handle mode."""

    device_type = "musa"

    def __init__(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int = 256,
        layout_hints: LayoutHints | None = None,
        engine_group_infos: "Sequence[EngineGroupInfo]" = (),
        engine_type: EngineType = EngineType.VLLM,
        separate_object_groups: bool = True,
        full_sw_kv: bool = False,
    ) -> None:
        """Build a MUSA cache context from IPC-wrapped KV tensors.

        Args:
            kv_caches: MUSA IPC wrappers received during server registration.
            lmcache_tokens_per_chunk: Tokens per LMCache object.
            layout_hints: Optional KV layout hints from the serving engine.
            engine_group_infos: Engine-neutral group metadata.
            engine_type: Serving engine that produced the KV cache.
            separate_object_groups: Whether to split object groups by window.
            full_sw_kv: Whether sliding-window groups transfer their full KV.

        Raises:
            ValueError: If reconstructed tensors are not MUSA tensors.
        """
        self._ipc_wrappers = tuple(kv_caches)
        try:
            self._initialize(
                kv_caches,
                lmcache_tokens_per_chunk,
                layout_hints,
                engine_group_infos,
                engine_type,
                separate_object_groups,
                full_sw_kv,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        kv_caches: KVCache,
        lmcache_tokens_per_chunk: int,
        layout_hints: LayoutHints | None,
        engine_group_infos: "Sequence[EngineGroupInfo]",
        engine_type: EngineType,
        separate_object_groups: bool,
        full_sw_kv: bool,
    ) -> None:
        """Initialize reconstructed tensors and MUSA transfer resources."""
        unwrapped = cast(DiscoverableKVCache, unwrap_kv_cache_tensors(kv_caches))
        discovered, engine_kv_formats = normalize_and_discover_per_layer_formats(
            unwrapped,
            engine_group_layer_indices(engine_group_infos),
            engine_type,
            layout_hints,
        )
        if not isinstance(discovered, list) or not all(
            isinstance(tensor, torch.Tensor) for tensor in discovered
        ):
            raise ValueError("MUSACacheContext requires one tensor per KV layer")
        kv_caches_norm = cast(list[torch.Tensor], discovered)
        normalized_discoverable = cast(DiscoverableKVCache, kv_caches_norm)
        self.device_ = get_device(normalized_discoverable)
        if self.device_.type != "musa":
            raise ValueError(
                f"MUSACacheContext expected MUSA tensors, got {self.device_.type!r}"
            )
        num_layers_val = len(engine_kv_formats)

        kv_layer_groups_manager = KVLayerGroupsManager(
            normalized_discoverable,
            engine_kv_formats=engine_kv_formats,
            engine_group_infos=engine_group_infos,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            separate_object_groups=separate_object_groups,
        )
        if full_sw_kv:
            kv_layer_groups_manager.enable_full_sw_kv()

        block_ids_buffer = torch.empty(
            1 << 20,
            dtype=torch.long,
            device=self.device_,
        )

        super().__init__(
            kv_caches=kv_caches_norm,
            device=self.device_,
            num_layers=num_layers_val,
            kv_layer_groups_manager=kv_layer_groups_manager,
            block_ids_buffer=block_ids_buffer,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
        )

        self.group_kv_pointers_: list[torch.Tensor] = []
        for group_idx, group in enumerate(self.kv_layer_groups_manager_.kernel_groups):
            pointers = get_group_data_ptrs(
                self.kv_caches_,
                self.get_engine_kv_format(group_idx),
                group.layer_indices,
            )
            self.group_kv_pointers_.append(
                torch.tensor(pointers, dtype=torch.int64, device=self.device_)
            )

        self._temp_buffer = _TempMUSABuffer(
            kv_layer_groups_manager=self.kv_layer_groups_manager_,
            lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
            device=self.device_,
            max_batch_size=4,
        )
        self.stream_ = torch_dev.Stream(device=self.device_)
        self.host_callback_stream_ = _MUSAHostCallbackStream(self.stream_)

        logger.debug(
            "MUSACacheContext: %d layers, %d blocks, dtype=%s",
            self.num_layers_,
            self.num_blocks,
            self.kv_caches_[0].dtype,
        )

    def close(self) -> None:
        """Synchronize transfers and release receiver-side IPC owners.

        Returns:
            None.

        Raises:
            RuntimeError: If the MUSA stream cannot be synchronized.
        """
        wrappers = self._ipc_wrappers
        if not wrappers:
            return

        stream = getattr(self, "stream_", None)
        synchronize = getattr(stream, "synchronize", None)
        if callable(synchronize):
            synchronize()

        kv_tensors = getattr(self, "kv_caches_", None)
        if isinstance(kv_tensors, list):
            kv_tensors.clear()

        for wrapper in wrappers:
            close = getattr(wrapper, "close", None)
            if callable(close):
                close()
        self._ipc_wrappers = ()

    @property
    def stream(self) -> Any:
        """Return the MUSA stream used for transfer work."""
        return self.stream_

    @property
    def cupy_stream(self) -> _MUSAHostCallbackStream:
        """Return a host-callback stream adapter for shared MP code paths."""
        return self.host_callback_stream_

    def get_kernel_group_kv_pointers(self, kernel_group_idx: int) -> torch.Tensor:
        """Return packed, process-local MUSA pointers for a kernel group.

        Args:
            kernel_group_idx: Index of the requested kernel group.

        Returns:
            A one-dimensional ``int64`` tensor containing one pointer per
            layer in kernel order.
        """
        return self.group_kv_pointers_[kernel_group_idx]

    def get_temp_kernel_group_buffer(
        self,
        batch_idx: int,
        kernel_group_idx: int,
    ) -> torch.Tensor:
        """Return the MUSA staging buffer for a batch/kernel-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            kernel_group_idx: Index of the kernel group to transfer.

        Returns:
            A typed view into the MUSA staging allocation.
        """
        return self._temp_buffer.get_temp_kernel_group_buffer(
            batch_idx,
            kernel_group_idx,
        )

    @property
    def max_batch_size(self) -> int:
        """Return the maximum number of chunks transferred per batch."""
        return self._temp_buffer.max_batch_size

    def get_temp_object_group_buffer(
        self,
        batch_idx: int,
        object_group_idx: int,
    ) -> torch.Tensor:
        """Return the MUSA staging buffer for a batch/object-group pair.

        Args:
            batch_idx: Index within the current transfer batch.
            object_group_idx: Index of the object group to transfer.

        Returns:
            A flat view covering the object's MUSA staging allocation.
        """
        return self._temp_buffer.get_temp_object_group_buffer(
            batch_idx,
            object_group_idx,
        )

    def get_kernel_group_shape_dtype(
        self,
        num_tokens: int,
        kernel_group_idx: int,
    ) -> tuple[torch.Size, torch.dtype]:
        """Return the shape and dtype for a kernel-group allocation.

        Args:
            num_tokens: Number of tokens represented by the allocation.
            kernel_group_idx: Index of the kernel group.

        Returns:
            The allocation shape and element dtype.
        """
        return self._temp_buffer.get_kernel_group_shape_dtype(
            num_tokens,
            kernel_group_idx,
        )

    def cache_size_per_token(self) -> int:
        """Return total cache bytes per logical token across all groups."""
        return self._temp_buffer.get_cache_size_per_token()
