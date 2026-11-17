# SPDX-License-Identifier: Apache-2.0
"""Serializer/Deserializer implementations for accelerated KV compression."""

# Standard
from dataclasses import dataclass
from typing import cast
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend
from lmcache.v1.distributed.compress_adapters.blocked import (
    BlockPlan,
    as_bytes,
    compressed_offsets,
    pack_header,
    plan_blocks,
    unpack_header,
)
from lmcache.v1.distributed.pending_fill import FillTracker
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import Deserializer, SerdeTaskId, Serializer
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)


@dataclass(frozen=True)
class DecompressedGroup:
    """A decompressed group's byte range within the destination buffer."""

    index: int
    offset: int
    length: int


class AccelCompressSerializer(Serializer):
    """Serializer that compresses objects with an accelerated backend.

    Output uses the blocked container format, so each of ``num_groups``
    byte-range groups decompresses independently. The source is not modified.

    Args:
        backend: Accelerated compression backend (e.g. IaxlBackend).
        num_groups: Number of independently decodable groups per object. Set
            this to the number of layer groups so that, under a layer-major
            layout, one group is one N-layer slice.
    """

    def __init__(
        self,
        backend: AccelCompressBackend,
        num_groups: int = 1,
    ) -> None:
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        self._backend = backend
        self._num_groups = num_groups
        # Per-thread staging buffer for serialize_to_bytes.
        self._scratch = threading.local()

    def serialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> int:
        """Compress src into dst.

        Returns:
            Number of bytes written to dst, including the header.
        """
        dst_buf = as_bytes(cast(memoryview, dst.byte_array))
        _, plan, staged, sizes = self._compress_staged(
            as_bytes(cast(memoryview, src.byte_array)), dst_buf
        )

        # Compact the blocks against the header, closing the staging gaps.
        write_offset = plan.header_size
        for stage_offset, size in zip(staged, sizes, strict=True):
            if stage_offset != write_offset:
                dst_buf[write_offset : write_offset + size] = bytes(
                    dst_buf[stage_offset : stage_offset + size]
                )
            write_offset += size

        pack_header(dst_buf, plan, sizes)
        return write_offset

    def serialize_to_bytes(self, src: MemoryObj) -> bytes:
        """Compress src into a new bytes object of exactly the output size.

        Blocks are compressed into a per-thread scratch buffer, then the
        header and blocks are copied out once.
        """
        buf, plan, staged, sizes = self._compress_staged(
            as_bytes(cast(memoryview, src.byte_array)), None
        )
        pack_header(buf, plan, sizes)
        parts = [buf[: plan.header_size]]
        parts.extend(
            buf[offset : offset + size]
            for offset, size in zip(staged, sizes, strict=True)
        )
        return b"".join(parts)

    def _compress_staged(
        self, src_buf: memoryview, dst_buf: memoryview | None
    ) -> tuple[memoryview, BlockPlan, list[int], list[int]]:
        """Compress the blocks of src_buf into dst_buf, after the header space.

        A block's compressed size is not known until the backend returns, so
        blocks are staged at worst-case spacing. If dst_buf is None, this
        thread's scratch buffer is used, grown as needed.

        Returns:
            The buffer written, the plan, each block's staged offset, and
            each block's compressed size.
        """
        plan = plan_blocks(len(src_buf), self._num_groups, self._backend.max_block_size)
        capacities = [
            self._backend.max_compressed_length(n) for n in plan.block_lengths
        ]
        staged: list[int] = []
        cursor = plan.header_size
        for capacity in capacities:
            staged.append(cursor)
            cursor += capacity

        if dst_buf is None:
            scratch: bytearray | None = getattr(self._scratch, "buf", None)
            if scratch is None or len(scratch) < cursor:
                scratch = bytearray(cursor)
                self._scratch.buf = scratch
            dst_buf = memoryview(scratch)
        elif len(dst_buf) < cursor:
            raise ValueError(
                f"destination too small for worst-case output: "
                f"{len(dst_buf)} < {cursor}"
            )

        srcs = [
            src_buf[offset : offset + length]
            for offset, length in zip(
                plan.block_offsets, plan.block_lengths, strict=True
            )
        ]
        dsts = [
            dst_buf[offset : offset + capacity]
            for offset, capacity in zip(staged, capacities, strict=True)
        ]
        sizes = self._backend.compress_blocks(srcs, dsts)
        return dst_buf, plan, staged, sizes

    def estimate_serialized_size(self, layout_desc: MemoryLayoutDesc) -> int:
        """Upper bound on compressed output size, including the header."""
        raw_size = 0
        for shape, dtype in zip(layout_desc.shapes, layout_desc.dtypes, strict=False):
            numel = 1
            for dim in shape:
                numel *= dim
            raw_size += numel * dtype.itemsize
        if raw_size <= 0:
            return 0

        plan = plan_blocks(raw_size, self._num_groups, self._backend.max_block_size)
        return plan.header_size + sum(
            self._backend.max_compressed_length(n) for n in plan.block_lengths
        )


class GroupwiseDecode:
    """Decompresses a batch of serialized objects one group index at a time.

    Round ``g`` decompresses group ``g`` of every object in one backend call.
    Groups are byte ranges in order, so after a round the leading bytes of
    each object up to the end of its group ``g`` are final.

    Args:
        backend: Accelerated compression backend.
        srcs: Serialized objects.
        dsts: Full-size destinations, one per source.

    Raises:
        ValueError: If a header is invalid or a destination is too small.
    """

    def __init__(
        self,
        backend: AccelCompressBackend,
        srcs: list[MemoryObj],
        dsts: list[MemoryObj],
    ) -> None:
        self._backend = backend
        # The byte views below do not keep their objects alive.
        self._keepalive = (list(srcs), list(dsts))
        self._objects = []
        for src, dst in zip(srcs, dsts, strict=True):
            src_buf = as_bytes(cast(memoryview, src.byte_array))
            dst_buf = as_bytes(cast(memoryview, dst.byte_array))
            plan, sizes = unpack_header(src_buf)
            if len(dst_buf) < plan.total_size:
                raise ValueError(
                    f"destination too small: {len(dst_buf)} < {plan.total_size}"
                )
            offsets = compressed_offsets(plan, sizes)
            self._objects.append((src_buf, dst_buf, plan, sizes, offsets))
        self.num_rounds = max((obj[2].num_groups for obj in self._objects), default=0)
        """Rounds to run; the largest group count among the objects."""

    def run_round(self, group_index: int) -> int:
        """Decompress group ``group_index`` of every object that has it.

        Returns:
            Bytes from the start that are final in every object.
        """
        srcs: list[memoryview] = []
        dsts: list[memoryview] = []
        filled = None
        for src_buf, dst_buf, plan, sizes, offsets in self._objects:
            if group_index < plan.num_groups:
                for i in plan.group_blocks(group_index):
                    srcs.append(src_buf[offsets[i] : offsets[i] + sizes[i]])
                    start = plan.block_offsets[i]
                    dsts.append(dst_buf[start : start + plan.block_lengths[i]])
            if group_index + 1 < plan.num_groups:
                offset, length = plan.group_range(group_index)
                end = offset + length
            else:
                end = plan.total_size
            filled = end if filled is None else min(filled, end)
        if srcs:
            self._backend.decompress_blocks(srcs, dsts)
        return filled or 0


class AccelCompressDeserializer(Deserializer):
    """Deserializer that decompresses objects with an accelerated backend.

    Args:
        backend: Accelerated compression backend (e.g. IaxlBackend).
    """

    def __init__(self, backend: AccelCompressBackend) -> None:
        self._backend = backend

    def deserialize(self, src: MemoryObj, dst: MemoryObj, key: ObjectKey) -> None:
        """Decompress every group of src into dst."""
        src_buf = as_bytes(cast(memoryview, src.byte_array))
        dst_buf = as_bytes(cast(memoryview, dst.byte_array))

        plan, sizes = unpack_header(src_buf)
        if len(dst_buf) < plan.total_size:
            raise ValueError(
                f"destination too small: {len(dst_buf)} < {plan.total_size}"
            )

        self._decompress_blocks(src_buf, dst_buf, plan, sizes, range(plan.num_blocks))

    def groupwise(
        self, srcs: list[MemoryObj], dsts: list[MemoryObj]
    ) -> GroupwiseDecode:
        """Prepare decompressing ``srcs`` into ``dsts`` group by group.

        Raises:
            ValueError: If a header is invalid or a destination is too small.
        """
        return GroupwiseDecode(self._backend, srcs, dsts)

    def group_count(self, src: MemoryObj) -> int:
        """Return the number of independently decodable groups in src."""
        plan, _ = unpack_header(as_bytes(cast(memoryview, src.byte_array)))
        return plan.num_groups

    def deserialize_group(
        self, src: MemoryObj, dst: MemoryObj, key: ObjectKey, group_index: int
    ) -> DecompressedGroup:
        """Decompress one group into its natural offset within dst.

        Only the blocks belonging to ``group_index`` are touched, so groups may
        be decoded in any order or concurrently into disjoint ranges of dst.

        Args:
            src: Serialized object.
            dst: Full-size destination for the whole object; the group is
                written at its own offset within it.
            key: Object key (unused; kept for interface symmetry).
            group_index: Which group to decompress.

        Returns:
            The byte range written.
        """
        src_buf = as_bytes(cast(memoryview, src.byte_array))
        dst_buf = as_bytes(cast(memoryview, dst.byte_array))

        plan, sizes = unpack_header(src_buf)
        if len(dst_buf) < plan.total_size:
            raise ValueError(
                f"destination too small: {len(dst_buf)} < {plan.total_size}"
            )

        offset, length = plan.group_range(group_index)
        self._decompress_blocks(
            src_buf, dst_buf, plan, sizes, plan.group_blocks(group_index)
        )

        return DecompressedGroup(index=group_index, offset=offset, length=length)

    def _decompress_blocks(
        self,
        src_buf: memoryview,
        dst_buf: memoryview,
        plan,
        sizes,
        block_indices,
    ) -> None:
        """Decompress the selected blocks from src_buf into dst_buf."""
        offsets = compressed_offsets(plan, sizes)
        srcs = [src_buf[offsets[i] : offsets[i] + sizes[i]] for i in block_indices]
        dsts = [
            dst_buf[
                plan.block_offsets[i] : plan.block_offsets[i] + plan.block_lengths[i]
            ]
            for i in block_indices
        ]
        if srcs:
            self._backend.decompress_blocks(srcs, dsts)


class AccelCompressSerdeProcessor(AsyncSerdeProcessor):
    """Async processor for the accelerated compression serde.

    With ``pipelined_load``, a load's objects are decompressed group by group
    across the whole batch: group 0 of every object, then group 1, and so on,
    advancing a :class:`FillTracker` after each. A layer-wise retrieve can
    then copy a layer group to the GPU while later groups are still being
    decompressed.

    Args:
        serializer: Compressing serializer.
        deserializer: Decompressing deserializer.
        max_workers: Thread pool size.
        pipelined_load: Offer :meth:`submit_pipelined_deserialize`.
    """

    def __init__(
        self,
        serializer: AccelCompressSerializer,
        deserializer: AccelCompressDeserializer,
        max_workers: int = 1,
        pipelined_load: bool = False,
    ) -> None:
        super().__init__(serializer, deserializer, max_workers=max_workers)
        self._compressor = serializer
        self._groupwise = deserializer
        self._pipelined_load = pipelined_load

    @property
    def supports_bytes_io(self) -> bool:
        return True

    def submit_serialize_to_bytes(
        self,
        src_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> tuple[SerdeTaskId, list[bytes]]:
        """Submit a compression into exact-size bytes; see the base class."""
        del keys
        task_id = self._alloc_task_id()
        payloads: list[bytes] = []
        self._pool.submit(
            self._run_serialize_to_bytes, task_id, list(src_objs), payloads
        )
        return task_id, payloads

    def _run_serialize_to_bytes(
        self, task_id: SerdeTaskId, src_objs: list[MemoryObj], payloads: list[bytes]
    ) -> None:
        ok = True
        try:
            for src in src_objs:
                payloads.append(self._compressor.serialize_to_bytes(src))
        except Exception:
            logger.exception("Serialize-to-bytes task %d failed", task_id)
            payloads.clear()
            ok = False
        with self._lock:
            self._completed_serialize[task_id] = ok
        try:
            self._serialize_efd.notify()
        except OSError:
            logger.exception("Failed to signal notifier for serde task %d", task_id)

    def submit_pipelined_deserialize(
        self,
        src_objs: list[MemoryObj],
        dst_objs: list[MemoryObj],
        keys: list[ObjectKey],
    ) -> tuple[SerdeTaskId, FillTracker] | None:
        """Submit a group-major decompression; see the base class.

        Returns:
            The task ID and its tracker, or None without ``pipelined_load``.

        Raises:
            ValueError: If a header is invalid or a destination too small.
        """
        del keys
        if not self._pipelined_load:
            return None
        decode = self._groupwise.groupwise(src_objs, dst_objs)
        fill = FillTracker(dst_objs)
        task_id = self._alloc_task_id()
        self._pool.submit(self._run_pipelined, task_id, decode, fill)
        return task_id, fill

    def _run_pipelined(
        self, task_id: SerdeTaskId, decode: GroupwiseDecode, fill: FillTracker
    ) -> None:
        ok = True
        try:
            for group_index in range(decode.num_rounds):
                fill.advance(decode.run_round(group_index))
        except Exception:
            logger.exception("Pipelined deserialize task %d failed", task_id)
            ok = False
        fill.settle(ok)
        with self._lock:
            self._completed_deserialize[task_id] = ok
        try:
            self._deserialize_efd.notify()
        except OSError:
            logger.exception("Failed to signal notifier for serde task %d", task_id)
