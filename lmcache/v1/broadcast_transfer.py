# SPDX-License-Identifier: Apache-2.0
"""Bounded, synchronous KV transfers into a serving engine's existing pages."""

# Standard
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from math import gcd, lcm
from typing import Any
from weakref import WeakValueDictionary
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)

logger = init_logger(__name__)

RetrievalChunk = tuple[MemoryObj, int, int]
WriteWindow = Callable[[list[MemoryObj], list[int], list[int]], None]
_ALIGNMENT = 64


def _tile_metadata(metadata: MemoryObjMetadata, num_tokens: int) -> MemoryObjMetadata:
    """Describe a contiguous token slice without inheriting source ownership."""
    if metadata.fmt not in (MemoryFormat.KV_2LTD, MemoryFormat.KV_MLA_FMT):
        raise ValueError(f"Unsupported retrieval format: {metadata.fmt}")
    shapes = metadata.shapes or [metadata.shape]
    if (metadata.shapes is None) != (metadata.dtypes is None):
        raise ValueError("Grouped KV metadata requires both shapes and dtypes")
    dtypes = metadata.dtypes
    if dtypes is None:
        if metadata.dtype is None:
            raise ValueError("Retrieval requires a KV dtype")
        dtypes = [metadata.dtype]
    if len(shapes) != len(dtypes) or not shapes:
        raise ValueError("KV shapes and dtypes must describe the same groups")
    sliced_shapes = []
    for shape, dtype in zip(shapes, dtypes, strict=True):
        if len(shape) != 4 or min(shape) <= 0 or dtype is None:
            raise ValueError("Retrieval requires nonempty, typed KV tensors")
        sliced_shapes.append(torch.Size((*shape[:2], num_tokens, shape[3])))
    result = replace(
        metadata,
        shape=sliced_shapes[0],
        dtype=dtypes[0],
        shapes=sliced_shapes if metadata.shapes is not None else None,
        address=0,
        phy_size=0,
        ref_count=1,
        pin_count=0,
        cached_positions=None,
    )
    result.phy_size = result.get_size()
    return result


def _token_alignment(metadata: MemoryObjMetadata) -> int:
    """Keep each concatenated group's byte offset aligned to its dtype."""
    offset = 0
    alignment = 1
    for shape, dtype in zip(metadata.shapes or [], metadata.dtypes or [], strict=True):
        alignment = lcm(alignment, dtype.itemsize // gcd(dtype.itemsize, offset))
        offset += shape.numel() // shape[2] * dtype.itemsize
    return alignment


class _SynchronizationError(RuntimeError):
    """A device fence failed; its buffers must not be treated as reusable."""


class RetrievalBuffer:
    """One reusable allocation shared by retrievals on a device in this process.

    Use :meth:`get` to share the allocation between engines. A lease serializes
    complete retrievals, including their collective sequence. Callers must finish
    every operation using the buffer before returning the lease. Serving engines
    must schedule distributed retrievals in the same order on all participating
    ranks, as required by the underlying collective communicator.
    """

    _buffers: WeakValueDictionary[torch.device, "RetrievalBuffer"] = (
        WeakValueDictionary()
    )
    _registry_lock = threading.Lock()

    def __init__(self, device: torch.device, size_bytes: int) -> None:
        self.size_bytes = size_bytes
        self._lock = threading.Lock()
        self._buffer = torch.empty(size_bytes, dtype=torch.uint8, device=device)
        self._failed = False

    @classmethod
    def get(cls, device: torch.device, size_bytes: int) -> "RetrievalBuffer":
        """Reserve or reuse the device's retrieval buffer.

        Args:
            device: Explicit local device, including its index for accelerators.
            size_bytes: Positive allocation size in bytes. All engines sharing
                this device must use the same budget.

        Returns:
            The shared buffer. It is allocated here, before retrieval starts.

        Raises:
            ValueError: If the budget is nonpositive or conflicts with an existing
                reservation, or an accelerator device has no explicit index.
            torch.OutOfMemoryError: If the device cannot reserve the buffer.
        """
        if size_bytes <= 0:
            raise ValueError("The retrieval buffer size must be positive")
        if device.type != "cpu" and device.index is None:
            raise ValueError("Retrieval requires an explicit local device index")
        if device.type == "cpu":
            device = torch.device("cpu")
        with cls._registry_lock:
            existing = cls._buffers.get(device)
            if existing is not None:
                if existing.size_bytes != size_bytes:
                    raise ValueError("Engines sharing a device must share its budget")
                return existing
            buffer = cls(device, size_bytes)
            cls._buffers[device] = buffer
            return buffer

    @contextmanager
    def lease(self) -> Iterator[torch.Tensor | None]:
        """Borrow the buffer exclusively until the caller's operations finish.

        Yields:
            The reserved flat uint8 tensor, or None after a fatal transfer error.
            This method does not allocate memory. A failed lease is never reused.
        """
        with self._lock:
            try:
                yield None if self._failed else self._buffer
            except BaseException:
                self._failed = True
                raise


class BroadcastTransfer:
    """Stream KV chunks through a fixed buffer and publish only complete results.

    The source iterator owns its memory objects; this class never pins, releases,
    or modifies them. The caller must close that iterator on early termination.
    Transfer callbacks may enqueue work. ``synchronize`` must wait for all copies,
    broadcasts, and destination writes before buffer reuse or source release.
    Device/communication failures raised by synchronization or broadcasting are
    fatal and propagate; they are not converted into cache misses.

    Args:
        buffer: Shared reservation, or None if initialization failed on this rank.
        is_source: Whether this rank supplies the source iterator.
        source_rank: Source rank in the broadcast callbacks' rank space.
        broadcast: Send/receive a device tensor using the supplied source rank.
        broadcast_object: Send/receive host metadata and return the source object.
        agree: Host-side collective returning True only if every rank voted True.
            It must not allocate device memory and must return the same result
            on every rank.
        synchronize: Fence all work that can still access the borrowed buffers.
    """

    def __init__(
        self,
        buffer: RetrievalBuffer | None,
        *,
        is_source: bool,
        source_rank: int,
        broadcast: Callable[[torch.Tensor, int], None],
        broadcast_object: Callable[[Any, int], Any],
        agree: Callable[[bool], bool],
        synchronize: Callable[[], None],
    ) -> None:
        self._buffer = buffer
        self._is_source = is_source
        self._source_rank = source_rank
        self._broadcast = broadcast
        self._broadcast_object = broadcast_object
        self._agree = agree
        self._synchronize_fn = synchronize

    def retrieve(
        self,
        chunks: Iterator[RetrievalChunk],
        num_tokens: int,
        write: WriteWindow,
        *,
        request_id: str | None = None,
    ) -> torch.Tensor:
        """Fill destination pages and return their committed token mask on CPU.

        Args:
            chunks: Contiguous source chunks in token order; unused on receivers.
                Each object remains owned by the iterator until its next advance.
            num_tokens: Length of the result mask.
            write: Write a bounded batch into the caller's reserved final KV pages.
                It must not retain temporary objects after synchronization.
            request_id: Request identity to check across ranks before any writes.
                Integrations should supply it when available.

        Returns:
            A mask of completely loaded tokens. Any source, validation, packing,
            or destination-write failure returns an all-false mask on every rank.

        Raises:
            Exception: Fatal collective or synchronization failures. The caller
                must recover the affected communicator/device before continuing.
        """
        result = torch.zeros(num_tokens, dtype=torch.bool, device="cpu")
        lease = self._buffer.lease() if self._buffer is not None else nullcontext()
        with lease as buffer:
            capacity = buffer.numel() if buffer is not None else 0
            reservation = (capacity, num_tokens, request_id)
            source_reservation = self._broadcast_object(
                reservation if self._is_source else None, self._source_rank
            )
            if not self._agree(capacity > 0 and reservation == source_reservation):
                return result
            assert buffer is not None
            try:
                return self._retrieve(chunks, result, buffer, write)
            finally:
                # Also fence writes enqueued before a callback raised. A failed
                # fence propagates: these pages cannot safely be reused.
                self._synchronize()

    def _retrieve(
        self,
        chunks: Iterator[RetrievalChunk],
        result: torch.Tensor,
        buffer: torch.Tensor,
        write: WriteWindow,
    ) -> torch.Tensor:
        windows = self._source_windows(chunks, buffer)
        previous_end = None
        while True:
            header: Any = None
            if self._is_source:
                try:
                    header = next(windows, None)
                except _SynchronizationError:
                    raise
                except Exception:
                    logger.exception("Failed to prepare a KV retrieval window")
                    header = False
            # Drain partial packing before notifying peers or releasing a source.
            self._synchronize()
            header = self._broadcast_object(header, self._source_rank)
            if header is None:
                return result
            if header is False:
                return result.zero_()

            ready = True
            memory_objs: list[MemoryObj] = []
            starts: list[int] = []
            ends: list[int] = []
            used_bytes = 0
            try:
                for start, end, description in header:
                    metadata = MemoryObjMetadata.from_dict(description)
                    _tile_metadata(metadata, end - start)
                    shapes = metadata.shapes or [metadata.shape]
                    offset = (used_bytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
                    used_bytes = offset + metadata.get_size()
                    if (
                        not 0 <= start < end <= len(result)
                        or (previous_end is not None and start != previous_end)
                        or any(shape[2] != end - start for shape in shapes)
                        or used_bytes > buffer.numel()
                    ):
                        raise ValueError("Invalid KV retrieval window")
                    memory_objs.append(
                        TensorMemoryObj(buffer[offset:used_bytes], metadata, None)
                    )
                    starts.append(start)
                    ends.append(end)
                    previous_end = end
                if not memory_objs:
                    raise ValueError("A retrieval window must contain KV data")
            except Exception:
                logger.exception("Invalid KV retrieval metadata")
                ready = False
            if not self._agree(ready):
                return result.zero_()

            self._broadcast(buffer[:used_bytes], self._source_rank)
            self._synchronize()
            written = True
            try:
                write(memory_objs, starts, ends)
            except Exception:
                logger.exception("Failed to write a KV retrieval window")
                written = False
            self._synchronize()
            if not self._agree(written):
                return result.zero_()
            for start, end in zip(starts, ends, strict=True):
                result[start:end] = True

    def _source_windows(
        self, chunks: Iterator[RetrievalChunk], buffer: torch.Tensor
    ) -> Iterator[list[tuple[int, int, dict]]]:
        window: list[tuple[int, int, dict]] = []
        used_bytes = 0
        for source, start, end in chunks:
            metadata = source.metadata
            token_bytes = _tile_metadata(metadata, 1).get_size()
            alignment = _token_alignment(metadata)
            shapes = metadata.shapes or [metadata.shape]
            if (
                token_bytes * alignment > buffer.numel()
                or (end - start) % alignment != 0
                or any(shape[2] != end - start for shape in shapes)
            ):
                raise ValueError("KV chunk cannot be sliced within the reservation")
            begin = start
            while begin < end:
                offset = (used_bytes + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
                tile_tokens = (buffer.numel() - offset) // token_bytes
                tile_tokens = tile_tokens // alignment * alignment
                if tile_tokens <= 0:
                    yield window
                    window = []
                    used_bytes = 0
                    continue
                finish = min(begin + tile_tokens, end)
                tile_metadata = _tile_metadata(metadata, finish - begin)
                buffer[used_bytes:offset].zero_()
                used_bytes = offset + tile_metadata.get_size()
                tile = TensorMemoryObj(buffer[offset:used_bytes], tile_metadata, None)
                self._pack(source, tile, begin - start, finish - begin)
                window.append((begin, finish, tile_metadata.to_dict()))
                begin = finish
            # Advancing chunks may release/reuse its source allocation.
            self._synchronize()
        if window:
            yield window

    def _pack(
        self, source: MemoryObj, target: MemoryObj, offset: int, length: int
    ) -> None:
        if source.metadata.shapes is None:
            tensors = [(source.tensor, target.tensor)]
        else:
            tensors = [
                (source.get_tensor(index), target.get_tensor(index))
                for index in range(len(source.metadata.shapes))
            ]
        for source_tensor, target_tensor in tensors:
            if source_tensor is None or target_tensor is None:
                raise ValueError("KV retrieval requires valid tensor storage")
            target_tensor.copy_(source_tensor.narrow(2, offset, length))

    def _synchronize(self) -> None:
        try:
            self._synchronize_fn()
        except Exception as exc:
            raise _SynchronizationError("KV retrieval synchronization failed") from exc
