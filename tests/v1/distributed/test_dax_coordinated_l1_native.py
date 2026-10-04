# SPDX-License-Identifier: Apache-2.0
"""Reader preservation and write exclusion through the public native API.

Anonymous shared mappings exercise two and four participants without hardware access.
"""

# Future
from __future__ import annotations

# Standard
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from multiprocessing.connection import Connection
from threading import Barrier
from typing import Any, Callable, Generator, TypeVar
import ctypes
import mmap
import multiprocessing
import struct
import time

# Third Party
import pytest

lmcache_native = pytest.importorskip(
    "lmcache.lmcache_dax_coordinated_l1",
    reason="build with BUILD_WITH_DAX_COORDINATED_L1=1",
)


_T = TypeVar("_T")
Result = lmcache_native.DaxCoordinatedL1Result


def _parameters(
    slot_count: int = 1024,
    buckets: tuple[int, ...] = (17, 19, 23, 29, 31),
    visibility: int = 2,
    participant_0_slots: int | None = None,
    *,
    payload_bytes: int = 64,
    payload_capacity: int | None = None,
    participant_count: int = 2,
) -> Any:
    return lmcache_native.dax_coordinated_l1_parameters(
        19,
        list(buckets),
        payload_bytes,
        slot_count,
        4096,
        visibility,
        slot_count // participant_count
        if participant_0_slots is None
        else participant_0_slots,
        bytes([5]) * 32,
        bytes([6]) * 32,
        bytes([7]) * 32,
        slot_count * payload_bytes if payload_capacity is None else payload_capacity,
        participant_count,
    )


def _attach(
    address: int,
    size: int,
    participant: int,
    parameters: Any,
    **kwargs: Any,
) -> Any:
    kwargs.setdefault(
        "payload_ranges",
        [(0, parameters.layout.required_payload_bytes // 64, address + size, 0)],
    )
    return lmcache_native.DevDaxBucketIndexCore(
        address,
        size,
        participant,
        parameters,
        **kwargs,
    )


def _four_participant_parameters(
    count: int = 4, first_slots: int = 16, buckets: tuple[int, ...] = (17, 19)
) -> Any:
    return _parameters(
        slot_count=64,
        buckets=buckets,
        visibility=1,
        participant_0_slots=first_slots,
        participant_count=count,
    )


def _allocate_delete(address: int, size: int, participant: int, barrier: Any) -> None:
    """Repeatedly reserve the only bucket, publish a new key and reclaim it."""
    core = _attach(
        address, size, participant, _four_participant_parameters(buckets=(1,))
    )
    try:
        barrier.wait(timeout=15)
        deadline = time.monotonic() + 30
        for iteration in range(200):
            key = bytes([participant + 1]) + iteration.to_bytes(31, "little")
            while True:
                write = core.reserve_write(key, 64, 1)
                if write.result == Result.SUCCESS:
                    break
                assert write.result in (Result.WRITER_BUSY, Result.NO_FREE_BUCKET)
                assert time.monotonic() < deadline, "tournament failed to make progress"
            # A separate test counter measures exclusion while the sole bucket
            # is reserved. It is not a published KV object or an update path.
            counter = ctypes.c_uint64.from_address(address + size + 4096)
            previous = counter.value
            time.sleep(0.0001)
            counter.value = previous + 1
            pointer = address + size + write.payload_offset
            ctypes.memset(pointer, participant + 1, 64)
            assert core.finish_writes([write.token]) == [Result.SUCCESS]
            read = core.reserve_reads([key], 1)[0]
            assert read.result == Result.SUCCESS
            assert ctypes.string_at(address + size + read.payload_offset, 64) == (
                bytes([participant + 1]) * 64
            )
            assert core.finish_reads([read.token], [1]) == [Result.SUCCESS]
            assert core.delete_key(key).result == Result.SUCCESS
        assert core.report_status().active_write_reservations == 0
        assert core.report_status().used_slots == 0
    finally:
        core.close()


def _attach_when_formatted(
    address: int,
    region_size: int,
    slot_count: int,
    visibility: int,
    connection: Connection,
) -> None:
    """Observe publication from another process, then immediately attach."""
    connection.send("watching")
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline:
            lmcache_native.dax_coordinated_l1_refresh(address, 512)
            if ctypes.string_at(address, 8) == b"LMCDAX1\x00":
                core = _attach(
                    address,
                    region_size,
                    1,
                    _parameters(slot_count, (17, 19), visibility),
                )
                core.close()
                connection.send("attached")
                return
        connection.send("format publication timed out")
    except Exception as error:
        connection.send(str(error))
    finally:
        connection.close()


@pytest.mark.parametrize("visibility_mode", [1, 2])
def test_attach_waits_for_complete_format_publication(visibility_mode: int) -> None:
    """Seeing format magic must imply that all attach metadata is ready."""
    if (
        visibility_mode == 2
        and not lmcache_native.dax_coordinated_l1_cpu_profile()["has_clflushopt"]
    ):
        pytest.skip("CLFLUSHOPT is unavailable")
    slot_count = 1 << 20
    buckets = [17, 19]
    layout = _parameters(slot_count, tuple(buckets), visibility_mode).layout
    size = layout.required_metadata_bytes
    with mmap.mmap(-1, size + layout.required_payload_bytes) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        context = multiprocessing.get_context("fork")
        parent, child = context.Pipe()
        process = context.Process(
            target=_attach_when_formatted,
            args=(address, size, slot_count, visibility_mode, child),
        )
        process.start()
        child.close()
        try:
            assert parent.poll(20), "attach observer did not start"
            assert parent.recv() == "watching"
            lmcache_native.format_dax_coordinated_l1_region(
                address, size, _parameters(slot_count, tuple(buckets), visibility_mode)
            )
            assert parent.poll(20), "attach observer did not finish"
            assert parent.recv() == "attached"
            process.join(timeout=5)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            parent.close()


def test_failed_format_does_not_publish_completion() -> None:
    """Invalid ownership must leave the arena unformatted for peers."""
    layout = _parameters(16, (17,), 1).layout
    size = layout.required_metadata_bytes
    with mmap.mmap(-1, size + layout.required_payload_bytes) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        with pytest.raises(ValueError, match="invalid participant ownership boundary"):
            lmcache_native.format_dax_coordinated_l1_region(
                address, size, _parameters(16, (17,), 1, 17)
            )
        with pytest.raises(RuntimeError, match="region is not formatted"):
            _attach(address, size, 1, _parameters(16, (17,), 1))


@dataclass
class _CorePair:
    mapping: mmap.mmap
    core_0: Any
    core_1: Any
    metadata_size: int

    @property
    def payload(self) -> memoryview:
        """View the explicit payload range after the test metadata mapping."""
        return memoryview(self.mapping)[self.metadata_size :]


def _make_pair(skip_flush: bool = False, parameters: Any = None) -> _CorePair:
    if parameters is None:
        parameters = _parameters()
    layout = parameters.layout
    region_size = layout.required_metadata_bytes
    mapping = mmap.mmap(-1, region_size + layout.required_payload_bytes)
    address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
    lmcache_native.format_dax_coordinated_l1_region(address, region_size, parameters)
    cores = [
        _attach(
            address,
            region_size,
            participant_id,
            parameters,
            skip_payload_flush=skip_flush,
        )
        for participant_id in (0, 1)
    ]
    return _CorePair(mapping, cores[0], cores[1], region_size)


@pytest.fixture
def pair() -> Generator[_CorePair, None, None]:
    value = _make_pair()
    with closing(value.mapping), closing(value.core_0), closing(value.core_1):
        yield value


def test_compact_superblock_keeps_metadata_cache_line_aligned() -> None:
    """The superblock uses six cache lines and metadata sections stay aligned."""
    layout = _parameters(16, (17, 19), 1).layout
    assert layout.participant_registry_offset == 384
    offsets = [
        layout.participant_registry_offset,
        *layout.bucket_metadata_offsets,
        layout.bucket_peterson_offset,
        layout.reader_activity_offset,
        layout.payload_header_offset,
    ]
    assert all(offset % 64 == 0 for offset in offsets)


@pytest.mark.parametrize("version", [0, 1, 0xFFFFFFFF])
def test_unsupported_format_is_rejected_without_modifying_region(version: int) -> None:
    """Attaching an unsupported format must leave the region untouched."""
    pair = _make_pair()
    pair.core_0.close()
    pair.core_1.close()
    with closing(pair.mapping):
        assert struct.unpack_from("<II", pair.mapping, 8) == (2, 384)
        struct.pack_into("<I", pair.mapping, 8, version)
        address = ctypes.addressof(ctypes.c_char.from_buffer(pair.mapping))
        lmcache_native.dax_coordinated_l1_publish(address, 64)
        before = pair.mapping[:]
        with pytest.raises(RuntimeError, match="explicitly reformat"):
            _attach(address, pair.metadata_size, 0, _parameters())
        assert pair.mapping[:] == before


def _reserve_new(pair: _CorePair, key: bytes, value: int) -> Any:
    reservation = pair.core_0.reserve_write(key, 64, 1)
    assert reservation.result == Result.SUCCESS
    pair.payload[reservation.payload_offset : reservation.payload_offset + 64] = (
        bytes([value]) * 64
    )
    assert pair.core_0.finish_writes([reservation.token])[0] == Result.SUCCESS
    return reservation


def _read_payload(pair: _CorePair, core: Any, key: bytes) -> bytes:
    reservation = core.reserve_reads([key], 1)[0]
    assert reservation.result == Result.SUCCESS
    observed = bytes(
        pair.payload[reservation.payload_offset : reservation.payload_offset + 64]
    )
    assert core.finish_reads([reservation.token], [1])[0] == Result.SUCCESS
    return observed


def _start_together(left: Callable[[], _T], right: Callable[[], _T]) -> tuple[_T, _T]:
    barrier = Barrier(3)

    def run(operation: Callable[[], _T]) -> _T:
        barrier.wait()
        return operation()

    with ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(run, left)
        right_future = executor.submit(run, right)
        barrier.wait()
        return left_future.result(timeout=5), right_future.result(timeout=5)


def test_read_read_keeps_both_participant_activity_until_release(
    pair: _CorePair,
) -> None:
    key = bytes([11]) * 32
    _reserve_new(pair, key, 0xA5)

    read_0, read_1 = _start_together(
        lambda: pair.core_0.reserve_reads([key], 1)[0],
        lambda: pair.core_1.reserve_reads([key], 1)[0],
    )

    # Expected: read/read is shareable.  Both participants must receive the
    # same READY generation and independently publish reader activity.
    assert read_0.result == Result.SUCCESS
    assert read_1.result == Result.SUCCESS
    assert read_0.global_payload_slot_id == read_1.global_payload_slot_id
    assert read_0.bucket_generation == read_1.bucket_generation
    assert bytes(pair.payload[read_0.payload_offset : read_0.payload_offset + 64]) == (
        bytes([0xA5]) * 64
    )

    # Expected: releasing participant 0 must not clear participant 1's
    # in-flight reader activity. Owner deletion stays blocked until both
    # activity banks are clear. This is synchronization state, not eviction
    # frequency or recency tracking.
    assert pair.core_1.delete_key(key).result == Result.OWNER_MISMATCH
    assert pair.core_0.delete_key(key).result == Result.ACTIVE_READER
    assert _read_payload(pair, pair.core_1, key) == bytes([0xA5]) * 64
    assert pair.core_0.finish_reads([read_0.token], [1])[0] == Result.SUCCESS
    assert pair.core_0.delete_key(key).result == Result.ACTIVE_READER
    assert pair.core_1.finish_reads([read_1.token], [1])[0] == Result.SUCCESS
    assert pair.core_0.delete_key(key).result == Result.SUCCESS


@pytest.mark.parametrize("active_reader", [False, True])
def test_existing_key_rejects_writes_without_changing_payload(
    pair: _CorePair, active_reader: bool
) -> None:
    key = bytes([13]) * 32
    original = _reserve_new(pair, key, 0xA5)
    reader = pair.core_1.reserve_reads([key], 1)[0] if active_reader else None
    if reader is not None:
        assert reader.result == Result.SUCCESS
    for core in (pair.core_0, pair.core_1):
        writer = core.reserve_write(key, 64, 1)
        assert writer.result == Result.WRITER_BUSY
        assert writer.token == 0
        assert core.report_status().active_write_reservations == 0
        assert _read_payload(pair, core, key) == bytes([0xA5]) * 64
    if reader is not None:
        # Rejected writes and deletion must leave the object open to new reads.
        assert pair.core_0.delete_key(key).result == Result.ACTIVE_READER
        assert _read_payload(pair, pair.core_0, key) == bytes([0xA5]) * 64
        assert pair.core_1.finish_reads([reader.token], [1]) == [Result.SUCCESS]
    assert pair.core_0.delete_key(key).result == Result.SUCCESS
    replacement = _reserve_new(pair, key, 0xB6)
    assert replacement.bucket_generation != original.bucket_generation
    assert _read_payload(pair, pair.core_1, key) == bytes([0xB6]) * 64
    assert pair.core_0.memcheck()


def test_new_write_never_exposes_partial_payload(pair: _CorePair) -> None:
    key = bytes([14]) * 32
    writer = pair.core_0.reserve_write(key, 64, 1)
    assert writer.result == Result.SUCCESS
    pair.payload[writer.payload_offset : writer.payload_offset + 32] = (
        bytes([0xB6]) * 32
    )
    reader = pair.core_1.reserve_reads([key], 1)[0]
    assert reader.result == Result.NOT_FOUND
    assert reader.token == 0
    assert pair.core_1.report_status().active_read_reservations == 0
    pair.payload[writer.payload_offset + 32 : writer.payload_offset + 64] = (
        bytes([0xB6]) * 32
    )
    assert pair.core_0.finish_writes([writer.token]) == [Result.SUCCESS]
    assert _read_payload(pair, pair.core_1, key) == bytes([0xB6]) * 64


def test_duplicate_batch_reads_preserve_partial_reader_counts(pair: _CorePair) -> None:
    """Duplicate keys acquire independent tokens and retain every reader count."""
    result = lmcache_native.DaxCoordinatedL1Result
    key = bytes([43]) * 32
    _reserve_new(pair, key, 0xA5)
    reads = pair.core_1.reserve_reads([key, key], 2)
    assert all(read.result == result.SUCCESS for read in reads)
    assert reads[0].token != reads[1].token
    assert pair.core_1.finish_reads([read.token for read in reads], [1, 2]) == [
        result.SUCCESS,
        result.SUCCESS,
    ]
    assert pair.core_0.delete_key(key).result == result.ACTIVE_READER
    assert pair.core_1.finish_reads([reads[0].token], [1])[0] == result.SUCCESS
    assert pair.core_0.delete_key(key).result == result.SUCCESS
    assert pair.core_1.report_status().active_read_reservations == 0


@pytest.mark.parametrize("skip_flush", [False, True])
@pytest.mark.parametrize("visibility_mode", [1, 2])
def test_both_payload_flush_policies_preserve_payload_lifecycle(
    skip_flush: bool, visibility_mode: int
) -> None:
    """Writes and peer reads retain their lifecycle under either flush policy."""
    pair = _make_pair(
        skip_flush=skip_flush, parameters=_parameters(visibility=visibility_mode)
    )
    with closing(pair.mapping), closing(pair.core_1), closing(pair.core_0):
        key = bytes([42]) * 32
        _reserve_new(pair, key, 0xA5)
        assert _read_payload(pair, pair.core_1, key) == bytes([0xA5]) * 64


@pytest.mark.parametrize(
    "begin,count,rank",
    [(0, 0, 0), (1, 1023, 0), (0, 1023, 0), (0, 1025, 0), (0, 1024, 8)],
)
def test_invalid_payload_placement_is_rejected_without_metadata_changes(
    pair: _CorePair, begin: int, count: int, rank: int
) -> None:
    """Native attach rejects gaps, capacity overflow and unsupported ranks."""
    before = bytes(pair.mapping)
    address = ctypes.addressof(ctypes.c_char.from_buffer(pair.mapping))
    payload = address + pair.metadata_size
    with pytest.raises(ValueError, match="payload placement"):
        _attach(
            address,
            pair.metadata_size,
            0,
            _parameters(),
            payload_ranges=[(begin, count, payload, rank)] if count else [],
        )
    assert bytes(pair.mapping) == before


@pytest.mark.parametrize("reserve_read", [False, True])
def test_completion_rejects_a_reclaimed_payload_slot(
    pair: _CorePair, reserve_read: bool
) -> None:
    """Both completion paths must detect a slot reclaimed during a write."""
    key = bytes([61]) * 32
    write = pair.core_0.reserve_write(key, 64, 1)
    assert write.result == Result.SUCCESS
    # The persistent ABI places allocation_state at byte 24 of a slot header.
    offset = (
        _parameters().layout.payload_header_offset + 64 * write.global_payload_slot_id
    )
    pair.mapping[offset + 24] = 0
    address = ctypes.addressof(ctypes.c_char.from_buffer(pair.mapping))
    lmcache_native.dax_coordinated_l1_publish(address + offset, 64)
    result = (
        pair.core_0.finish_write_and_reserve_read(write.token, 1).result
        if reserve_read
        else pair.core_0.finish_writes([write.token])[0]
    )
    assert result == Result.GENERATION_MISMATCH
    assert pair.core_0.report_status().active_write_reservations == 0
    assert not pair.core_0.memcheck()


@pytest.mark.parametrize(
    "slots,buckets,payload_bytes",
    [
        (1 << 63, (17,), 64),
        (16, (1 << 63,), 64),
        (16, (1 << 56,) * 3, 64),
        (16, (17,), 1 << 63),
    ],
)
def test_layout_rejects_overflow_without_allocating(
    slots: int, buckets: tuple[int, ...], payload_bytes: int
) -> None:
    """Metadata sections and payload byte products must fit uint64."""
    with pytest.raises(OverflowError, match="size overflow"):
        _ = _parameters(
            slots, buckets, payload_bytes=payload_bytes, payload_capacity=(1 << 64) - 1
        ).layout


@pytest.mark.parametrize(
    "parameters",
    [_parameters(512), _parameters(buckets=(17, 19)), _parameters(visibility=1)],
)
def test_attach_checks_geometry_even_when_digests_match(
    pair: _CorePair, parameters: Any
) -> None:
    """A digest alone cannot authorize different slot, bucket or visibility fields."""
    before = bytes(pair.mapping)
    address = ctypes.addressof(ctypes.c_char.from_buffer(pair.mapping))
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _attach(address, pair.metadata_size, 1, parameters)
    assert bytes(pair.mapping) == before


@pytest.mark.parametrize("memcheck_on_attach", [False, True])
@pytest.mark.parametrize(
    "offset,value,message",
    [
        (0, 0, "region is not formatted"),
        (16, 20, "contract mismatch"),
        (384 + 64, 99, "participant contract mismatch"),
    ],
)
def test_attach_always_checks_static_contract(
    pair: _CorePair, memcheck_on_attach: bool, offset: int, value: int, message: str
) -> None:
    """Disabling the index scan must not bypass magic, epoch or ownership checks."""
    pair.core_0.close()
    pair.core_1.close()
    struct.pack_into("<Q", pair.mapping, offset, value)
    address = ctypes.addressof(ctypes.c_char.from_buffer(pair.mapping))
    lmcache_native.dax_coordinated_l1_publish(address + offset, 8)
    with pytest.raises(RuntimeError, match=message):
        _attach(
            address,
            pair.metadata_size,
            1,
            _parameters(),
            memcheck_on_attach=memcheck_on_attach,
        )


def test_attach_scan_is_optional_but_read_validation_is_not(pair: _CorePair) -> None:
    """A bad slot link is diagnosed on demand and still rejected by normal reads."""
    key = bytes([93]) * 32
    write = _reserve_new(pair, key, 42)
    pair.core_1.close()
    address = ctypes.addressof(ctypes.c_char.from_buffer(pair.mapping))
    # Inject a bad back-reference in the persistent slot-header ABI.
    offset = (
        _parameters().layout.payload_header_offset
        + 64 * write.global_payload_slot_id
        + 32
    )
    original = pair.mapping[offset : offset + 8]
    struct.pack_into("<Q", pair.mapping, offset, (1 << 64) - 1)
    lmcache_native.dax_coordinated_l1_publish(address + offset, 8)
    try:
        with closing(_attach(address, pair.metadata_size, 1, _parameters())) as peer:
            assert not peer.memcheck()
            assert (
                peer.reserve_reads([key], 1)[0].result == Result.CORRUPT_FORWARD_BACKREF
            )
        with pytest.raises(RuntimeError, match="requires recovery before attach"):
            _attach(
                address, pair.metadata_size, 1, _parameters(), memcheck_on_attach=True
            )
    finally:
        pair.mapping[offset : offset + 8] = original
        lmcache_native.dax_coordinated_l1_publish(address + offset, 8)
    with closing(
        _attach(address, pair.metadata_size, 1, _parameters(), memcheck_on_attach=True)
    ) as peer:
        assert peer.memcheck()
        assert _read_payload(pair, peer, key) == bytes([42]) * 64


def test_collisions_use_all_contiguous_bucket_levels() -> None:
    """Three one-bucket levels must retain independent payloads and reclamation."""
    value = _make_pair(parameters=_parameters(buckets=(1, 1, 1)))
    with closing(value.mapping), closing(value.core_0), closing(value.core_1):
        keys = [bytes([byte]) * 32 for byte in (71, 72, 73)]
        for byte, key in enumerate(keys):
            _reserve_new(value, key, byte)
        assert (
            value.core_0.reserve_write(bytes([74]) * 32, 64, 1).result
            == Result.NO_FREE_BUCKET
        )
        for byte, key in enumerate(keys):
            assert _read_payload(value, value.core_1, key) == bytes([byte]) * 64
            assert value.core_0.delete_key(key).result == Result.SUCCESS
        assert value.core_0.report_status().used_slots == 0


def test_four_process_tournament_preserves_allocation_exclusion() -> None:
    """Four processes must exclusively reserve and reclaim the same bucket."""
    parameters = _four_participant_parameters(buckets=(1,))
    size = parameters.layout.required_metadata_bytes
    with mmap.mmap(-1, size + 4096 + 64) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        lmcache_native.format_dax_coordinated_l1_region(address, size, parameters)
        context = multiprocessing.get_context("fork")
        barrier = context.Barrier(4)
        processes = [
            context.Process(target=_allocate_delete, args=(address, size, pid, barrier))
            for pid in range(4)
        ]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=40)
                assert process.exitcode == 0
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        assert struct.unpack_from("<Q", mapping, size + 4096)[0] == 800
        for pid in range(4):
            core = _attach(address, size, pid, parameters)
            try:
                assert core.report_status().free_slots == 16
                assert core.memcheck()
            finally:
                core.close()


@pytest.mark.parametrize("first_slots", [16, 64])
def test_four_owner_ranges_and_independent_reader_banks(first_slots: int) -> None:
    """Each participant retains its ownership and independent read protection."""
    parameters = _four_participant_parameters(first_slots=first_slots)
    size = parameters.layout.required_metadata_bytes
    with ExitStack() as stack:
        mapping = stack.enter_context(mmap.mmap(-1, size + 4096))
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        lmcache_native.format_dax_coordinated_l1_region(address, size, parameters)
        cores = [_attach(address, size, pid, parameters) for pid in range(4)]
        for core in cores:
            stack.callback(core.close)
        assert [core.report_status().free_slots for core in cores] == (
            [16] * 4 if first_slots == 16 else [64, 0, 0, 0]
        )
        for pid, core in enumerate(cores if first_slots == 16 else cores[:1]):
            key = bytes([80 + pid]) * 32
            write = core.reserve_write(key, 64, 1)
            assert write.result == Result.SUCCESS
            if first_slots == 16:
                assert pid * 16 <= write.global_payload_slot_id < (pid + 1) * 16
            ctypes.memset(address + size + write.payload_offset, 80 + pid, 64)
            assert core.finish_writes([write.token]) == [Result.SUCCESS]
            reads = [peer.reserve_reads([key], 1)[0] for peer in cores]
            for read in reads:
                assert read.result == Result.SUCCESS
                assert (
                    bytes(
                        mapping[
                            size + read.payload_offset : size + read.payload_offset + 64
                        ]
                    )
                    == bytes([80 + pid]) * 64
                )
            for reader, (peer, read) in enumerate(zip(cores, reads, strict=True)):
                assert core.delete_key(key).result == Result.ACTIVE_READER
                assert peer.finish_reads([read.token], [1]) == [Result.SUCCESS]
                if reader != 3:
                    assert core.delete_key(key).result == Result.ACTIVE_READER
            assert core.delete_key(key).result == Result.SUCCESS
        assert all(core.memcheck() for core in cores)


def test_participant_count_mismatch_rejects_attach_without_writes() -> None:
    """Invalid participant topology must leave existing metadata untouched."""
    parameters = _four_participant_parameters()
    size = parameters.layout.required_metadata_bytes
    with mmap.mmap(-1, size + 4096) as mapping:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapping))
        lmcache_native.format_dax_coordinated_l1_region(address, size, parameters)
        before = bytes(mapping)
        with pytest.raises(RuntimeError, match="contract mismatch"):
            _attach(
                address, size, 0, _four_participant_parameters(count=2, first_slots=32)
            )
        with pytest.raises(ValueError, match="ownership boundary"):
            _attach(address, size, 4, parameters)
        assert bytes(mapping) == before
