# SPDX-License-Identifier: Apache-2.0
"""Tests for the blocked, group-addressable accelerated-compression format.

Uses a zlib-based mock backend, so no accelerator is required.
"""

# Standard
import os
import time
import zlib

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed import pending_fill
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend
from lmcache.v1.distributed.compress_adapters.blocked import (
    _HEADER,
    compressed_offsets,
    pack_header,
    plan_blocks,
    unpack_header,
)
from lmcache.v1.distributed.compress_adapters.serde import (
    AccelCompressDeserializer,
    AccelCompressSerdeProcessor,
    AccelCompressSerializer,
)
from lmcache.v1.distributed.pending_fill import FillFailedError
from lmcache.v1.memory_management import (
    BytesBufferMemoryObj,
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)

_KEY = ObjectKey(chunk_hash=b"blocked", model_name="m", kv_rank=0)


class ZlibBackend(AccelCompressBackend):
    """zlib-based mock satisfying AccelCompressBackend."""

    def compress(self, src: memoryview, dst: memoryview) -> int:
        out = zlib.compress(bytes(src), level=1)
        dst_cast = dst.cast("B") if dst.format != "B" else dst
        dst_cast[: len(out)] = out
        return len(out)

    def decompress(self, src: memoryview, dst: memoryview) -> int:
        out = zlib.decompress(bytes(src))
        dst_cast = dst.cast("B") if dst.format != "B" else dst
        dst_cast[: len(out)] = out
        return len(out)

    def max_compressed_length(self, src_size: int) -> int:
        return src_size + (src_size // 100) + 64

    def close(self) -> None:
        pass


class BoundedZlibBackend(ZlibBackend):
    """Mimics a codec that can only encode one bounded block per call.

    The blocked container must then split each group into bounded blocks and
    record every block's compressed size to decode it again.
    """

    def __init__(self, max_block: int) -> None:
        self._max_block = max_block

    @property
    def max_block_size(self) -> int | None:
        return self._max_block

    def compress(self, src: memoryview, dst: memoryview) -> int:
        if len(src) > self._max_block:
            raise AssertionError(
                f"serde submitted a {len(src)}-byte block, "
                f"over the {self._max_block}-byte limit"
            )
        return super().compress(src, dst)


def _obj(data: bytes) -> TensorMemoryObj:
    tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return _wrap(tensor)


def _empty(size: int) -> TensorMemoryObj:
    return _wrap(torch.zeros(size, dtype=torch.uint8))


def _wrap(tensor: torch.Tensor) -> TensorMemoryObj:
    metadata = MemoryObjMetadata(
        shape=torch.Size([tensor.numel()]),
        dtype=torch.uint8,
        address=0,
        phy_size=tensor.numel(),
        fmt=MemoryFormat.KV_2LTD,
        ref_count=1,
    )
    return TensorMemoryObj(tensor, metadata, parent_allocator=None)


def _roundtrip(data: bytes, num_groups: int, backend: AccelCompressBackend):
    serializer = AccelCompressSerializer(backend=backend, num_groups=num_groups)
    des = AccelCompressDeserializer(backend=backend)
    src = _obj(data)
    dst = _empty(len(data) * 2 + 8192)
    written = serializer.serialize(src, dst, _KEY)
    out = _empty(len(data))
    des.deserialize(_obj(bytes(dst.byte_array[:written])), out, _KEY)
    return bytes(out.byte_array), written


class TestBlockPlan:
    def test_groups_partition_the_buffer_exactly(self):
        plan = plan_blocks(1000, num_groups=4)
        assert plan.num_groups == 4
        covered = [plan.group_range(g) for g in range(plan.num_groups)]
        assert covered[0][0] == 0
        assert sum(length for _, length in covered) == 1000
        for (off, length), (next_off, _) in zip(covered, covered[1:], strict=False):
            assert off + length == next_off

    def test_blocks_never_straddle_a_group_boundary(self):
        plan = plan_blocks(1000, num_groups=4, max_block_size=99)
        for g in range(plan.num_groups):
            g_off, g_len = plan.group_range(g)
            for b in plan.group_blocks(g):
                b_off = plan.block_offsets[b]
                b_end = b_off + plan.block_lengths[b]
                assert g_off <= b_off < b_end <= g_off + g_len

    def test_block_size_respects_backend_limit(self):
        plan = plan_blocks(10_000, num_groups=2, max_block_size=1024)
        assert max(plan.block_lengths) <= 1024
        assert sum(plan.block_lengths) == 10_000

    def test_header_roundtrip(self):
        plan = plan_blocks(5000, num_groups=3, max_block_size=700)
        sizes = [10 + i for i in range(plan.num_blocks)]
        buf = memoryview(bytearray(plan.header_size))
        assert pack_header(buf, plan, sizes) == plan.header_size

        parsed, parsed_sizes = unpack_header(buf)
        assert parsed == plan
        assert list(parsed_sizes) == sizes

    def test_compressed_offsets_are_cumulative(self):
        plan = plan_blocks(300, num_groups=3)
        sizes = [7, 11, 13]
        offsets = compressed_offsets(plan, sizes)
        assert offsets[0] == plan.header_size
        assert offsets[1] == plan.header_size + 7
        assert offsets[2] == plan.header_size + 18

    def test_rejects_bad_magic(self):
        plan = plan_blocks(100, num_groups=1)
        buf = bytearray(plan.header_size)
        pack_header(memoryview(buf), plan, [42])
        buf[0:4] = b"XXXX"
        with pytest.raises(ValueError, match="bad magic"):
            unpack_header(memoryview(buf))

    def test_rejects_truncated_buffer(self):
        with pytest.raises(ValueError, match="too small"):
            unpack_header(memoryview(bytearray(_HEADER.size - 1)))

    def test_rejects_invalid_group_index(self):
        plan = plan_blocks(100, num_groups=2)
        with pytest.raises(IndexError):
            plan.group_range(2)


class TestRoundtrip:
    @pytest.mark.parametrize("num_groups", [1, 2, 4, 7, 32])
    def test_roundtrip_preserves_data(self, num_groups):
        data = os.urandom(64 * 1024)
        out, _ = _roundtrip(data, num_groups, ZlibBackend())
        assert out == data

    def test_roundtrip_compressible_data_shrinks(self):
        data = b"\x00" * (256 * 1024)
        out, written = _roundtrip(data, 8, ZlibBackend())
        assert out == data
        assert written < len(data)

    def test_multi_block_roundtrip_over_the_backend_limit(self):
        """The case the old single-buffer split path could not decode."""
        backend = BoundedZlibBackend(max_block=1 << 20)
        data = os.urandom(3 * (1 << 20) + 12345)
        out, _ = _roundtrip(data, num_groups=4, backend=backend)
        assert out == data

    def test_single_group_still_splits_into_bounded_blocks(self):
        backend = BoundedZlibBackend(max_block=4096)
        data = os.urandom(50_000)
        out, _ = _roundtrip(data, num_groups=1, backend=backend)
        assert out == data

    def test_estimate_is_an_upper_bound(self):
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(backend=backend, num_groups=4)
        data = os.urandom(48 * 1024)
        # Layout mirroring the real buffer: one uint8 tensor of len(data).
        layout = type(
            "L", (), {"shapes": [torch.Size([len(data)])], "dtypes": [torch.uint8]}
        )()
        estimate = serializer.estimate_serialized_size(layout)
        src = _obj(data)
        dst = _empty(estimate)
        assert serializer.serialize(src, dst, _KEY) <= estimate

    def test_rejects_num_groups_below_one(self):
        with pytest.raises(ValueError, match="num_groups"):
            AccelCompressSerializer(backend=ZlibBackend(), num_groups=0)


class TestGroupAddressing:
    def _prepare(self, data: bytes, num_groups: int):
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(backend=backend, num_groups=num_groups)
        des = AccelCompressDeserializer(backend=backend)
        dst = _empty(len(data) * 2 + 8192)
        written = serializer.serialize(_obj(data), dst, _KEY)
        return des, _obj(bytes(dst.byte_array[:written]))

    def test_single_group_decodes_without_touching_others(self):
        data = os.urandom(40 * 1024)
        des, blob = self._prepare(data, num_groups=8)

        out = _empty(len(data))
        group = des.deserialize_group(blob, out, _KEY, 3)

        written = bytes(out.tensor.numpy().tobytes())
        assert (
            written[group.offset : group.offset + group.length]
            == (data[group.offset : group.offset + group.length])
        )
        # Everything outside the group is untouched.
        assert written[: group.offset] == b"\x00" * group.offset
        assert written[group.offset + group.length :] == b"\x00" * (
            len(data) - group.offset - group.length
        )

    def test_all_groups_in_any_order_reconstruct_the_object(self):
        data = os.urandom(40 * 1024)
        des, blob = self._prepare(data, num_groups=8)
        out = _empty(len(data))
        for g in reversed(range(des.group_count(blob))):
            des.deserialize_group(blob, out, _KEY, g)
        assert bytes(out.tensor.numpy().tobytes()) == data

    def test_group_count_matches_request(self):
        data = os.urandom(8192)
        des, blob = self._prepare(data, num_groups=8)
        assert des.group_count(blob) == 8


def _blobs(payloads: list[bytes], num_groups: int, backend: AccelCompressBackend):
    serializer = AccelCompressSerializer(backend=backend, num_groups=num_groups)
    blobs = []
    for data in payloads:
        dst = _empty(len(data) * 2 + 8192)
        written = serializer.serialize(_obj(data), dst, _KEY)
        blobs.append(_obj(bytes(dst.byte_array[:written])))
    return blobs


class TestGroupwiseDecode:
    def test_rounds_fill_every_object_front_to_back(self):
        backend = ZlibBackend()
        payloads = [os.urandom(40 * 1024) for _ in range(3)]
        outs = [_empty(len(d)) for d in payloads]
        decode = AccelCompressDeserializer(backend).groupwise(
            _blobs(payloads, 4, backend), outs
        )

        assert decode.num_rounds == 4
        filled = 0
        for group_index in range(decode.num_rounds):
            previous, filled = filled, decode.run_round(group_index)
            assert filled > previous
            for data, out in zip(payloads, outs, strict=True):
                written = bytes(out.tensor.numpy().tobytes())
                assert written[:filled] == data[:filled]
                assert written[filled:] == bytes(len(data) - filled)
        assert filled == 40 * 1024

    def test_watermark_is_the_least_filled_object(self):
        backend = ZlibBackend()
        payloads = [os.urandom(8192), os.urandom(8192)]
        blobs = _blobs(payloads[:1], 4, backend) + _blobs(payloads[1:], 1, backend)
        outs = [_empty(8192), _empty(8192)]
        decode = AccelCompressDeserializer(backend).groupwise(blobs, outs)

        assert decode.num_rounds == 4
        # The single-group object is whole after round 0; the other is not.
        assert decode.run_round(0) == 2048
        assert bytes(outs[1].tensor.numpy().tobytes()) == payloads[1]
        assert [decode.run_round(g) for g in (1, 2, 3)] == [4096, 6144, 8192]
        assert bytes(outs[0].tensor.numpy().tobytes()) == payloads[0]

    def test_rejects_a_small_destination(self):
        backend = ZlibBackend()
        blobs = _blobs([os.urandom(8192)], 2, backend)
        with pytest.raises(ValueError, match="destination too small"):
            AccelCompressDeserializer(backend).groupwise(blobs, [_empty(4096)])


def _wait_deserialize_result(processor, task_id) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = processor.query_deserialize_result(task_id)
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("deserialize task never completed")


class TestPipelinedProcessor:
    def _processor(self, pipelined_load: bool = True):
        backend = ZlibBackend()
        return backend, AccelCompressSerdeProcessor(
            AccelCompressSerializer(backend=backend, num_groups=4),
            AccelCompressDeserializer(backend),
            pipelined_load=pipelined_load,
        )

    def test_disabled_offers_no_pipelined_path(self):
        backend, processor = self._processor(pipelined_load=False)
        try:
            blobs = _blobs([os.urandom(8192)], 4, backend)
            outs = [_empty(8192)]
            assert processor.submit_pipelined_deserialize(blobs, outs, [_KEY]) is None
        finally:
            processor.close()

    def test_fill_settles_after_the_last_round(self):
        backend, processor = self._processor()
        try:
            payloads = [os.urandom(40 * 1024) for _ in range(2)]
            outs = [_empty(len(d)) for d in payloads]
            submitted = processor.submit_pipelined_deserialize(
                _blobs(payloads, 4, backend), outs, [_KEY, _KEY]
            )
            assert submitted is not None
            task_id, fill = submitted

            fill.wait()
            assert pending_fill.find(outs[0]) is None
            assert _wait_deserialize_result(processor, task_id) is True
            assert [bytes(o.tensor.numpy().tobytes()) for o in outs] == payloads
        finally:
            processor.close()

    def test_failed_round_fails_the_rest_and_keeps_the_prefix(self):
        backend, processor = self._processor()
        try:
            data = os.urandom(8192)
            (blob,) = _blobs([data], 4, backend)
            # Corrupt the tail: only the last group's block fails to inflate.
            blob.tensor[-8:] ^= 0xFF
            out = _empty(len(data))
            submitted = processor.submit_pipelined_deserialize([blob], [out], [_KEY])
            assert submitted is not None
            task_id, fill = submitted

            assert _wait_deserialize_result(processor, task_id) is False
            assert fill.failed
            assert pending_fill.find(out) is fill
            fill.wait(6144)
            with pytest.raises(FillFailedError):
                fill.wait()
            assert bytes(out.tensor.numpy().tobytes())[:6144] == data[:6144]
            fill.release()
        finally:
            processor.close()

    def test_bad_header_raises_at_submit(self):
        _, processor = self._processor()
        try:
            blob = _obj(b"not a compressed object" * 8)
            out = _empty(4096)
            with pytest.raises(ValueError):
                processor.submit_pipelined_deserialize([blob], [out], [_KEY])
            assert pending_fill.find(out) is None
        finally:
            processor.close()


class TestSerializeToBytes:
    @pytest.mark.parametrize("num_groups", [1, 4])
    def test_matches_serialize(self, num_groups):
        serializer = AccelCompressSerializer(
            backend=BoundedZlibBackend(max_block=8192), num_groups=num_groups
        )
        data = os.urandom(50_000)
        dst = _empty(len(data) * 2 + 8192)
        written = serializer.serialize(_obj(data), dst, _KEY)

        payload = serializer.serialize_to_bytes(_obj(data))

        assert type(payload) is bytes
        assert payload == bytes(dst.byte_array[:written])

    def test_round_trips_from_bytes_sources(self):
        backend = ZlibBackend()
        serializer = AccelCompressSerializer(backend=backend, num_groups=4)
        des = AccelCompressDeserializer(backend)
        # Shrinking and then growing the input reuses, then regrows, the
        # scratch buffer.
        for size in (64 * 1024, 4096, 128 * 1024):
            data = os.urandom(size)
            payload = serializer.serialize_to_bytes(_obj(data))
            out = _empty(size)
            des.deserialize(BytesBufferMemoryObj(payload), out, _KEY)
            assert bytes(out.byte_array) == data


def _wait_serialize_result(processor, task_id) -> bool:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = processor.query_serialize_result(task_id)
        if result is not None:
            return result
        time.sleep(0.01)
    raise AssertionError("serialize task never completed")


class _FailingZlibBackend(ZlibBackend):
    def compress(self, src: memoryview, dst: memoryview) -> int:
        raise RuntimeError("compress failed")


class TestBytesProcessor:
    def _processor(self, backend: AccelCompressBackend):
        return AccelCompressSerdeProcessor(
            AccelCompressSerializer(backend=backend, num_groups=4),
            AccelCompressDeserializer(backend),
            pipelined_load=True,
        )

    def test_serialized_bytes_decode_group_by_group(self):
        processor = self._processor(ZlibBackend())
        try:
            assert processor.supports_bytes_io
            payloads = [os.urandom(40 * 1024) for _ in range(2)]
            task_id, blobs = processor.submit_serialize_to_bytes(
                [_obj(d) for d in payloads], [_KEY, _KEY]
            )
            assert _wait_serialize_result(processor, task_id) is True
            assert len(blobs) == 2

            outs = [_empty(len(d)) for d in payloads]
            submitted = processor.submit_pipelined_deserialize(
                [BytesBufferMemoryObj(b) for b in blobs], outs, [_KEY, _KEY]
            )
            assert submitted is not None
            task_id, fill = submitted
            fill.wait()
            assert _wait_deserialize_result(processor, task_id) is True
            assert [bytes(o.byte_array) for o in outs] == payloads
        finally:
            processor.close()

    def test_failed_serialize_leaves_no_payloads(self):
        processor = self._processor(_FailingZlibBackend())
        try:
            task_id, blobs = processor.submit_serialize_to_bytes(
                [_obj(os.urandom(8192))], [_KEY]
            )
            assert _wait_serialize_result(processor, task_id) is False
            assert blobs == []
        finally:
            processor.close()
