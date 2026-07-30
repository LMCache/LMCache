# SPDX-License-Identifier: Apache-2.0
"""Focused tests for mapped Device-DAX views and visibility ordering."""

# Standard
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import mmap

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import L1MemoryManagerConfig, SharedL1Config
from lmcache.v1.distributed.shared_l1.client import SharedL1Client
from lmcache.v1.distributed.shared_l1.pool import SharedL1Pool
import lmcache.v1.distributed.shared_l1.client as client_module

_CAPACITY = 4096
_ALIGNMENT = 64
_MAPPING_OFFSET = mmap.PAGESIZE


class _Visibility:
    def __init__(self, fail_operation: int | None = None) -> None:
        self.calls: list[tuple[int, int, int, int]] = []
        self.fail_operation = fail_operation

    @property
    def granularity(self) -> int:
        return _ALIGNMENT

    def apply(
        self,
        operation: int,
        _device_fd: int,
        _mapped_address: int,
        device_offset: int,
        length: int,
        generation: int,
    ) -> None:
        self.calls.append((operation, device_offset, length, generation))
        if operation == self.fail_operation:
            raise RuntimeError("injected visibility failure")


def _key(seed: int = 1) -> ObjectKey:
    return ObjectKey(seed.to_bytes(4, "big"), "model", 0)


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([4, 4])], [torch.float16])


def _configs(path: Path) -> tuple[SharedL1Config, L1MemoryManagerConfig]:
    return (
        SharedL1Config(
            coordinator_host="127.0.0.1",
            coordinator_port=9301,
            authkey_file="/unused/authkey",
            region_id="region",
            layout_id="layout",
            mapping_offset=_MAPPING_OFFSET,
            visibility_library_path="/unused/visibility.so",
        ),
        L1MemoryManagerConfig(
            size_in_bytes=_CAPACITY,
            use_lazy=False,
            align_bytes=_ALIGNMENT,
            shm_name="",
            devdax_path=str(path),
        ),
    )


def _region(tmp_path: Path) -> Path:
    path = tmp_path / "region.bin"
    with path.open("wb") as region:
        region.truncate(_MAPPING_OFFSET + _CAPACITY)
    return path


def _client(
    path: Path,
    pool: SharedL1Pool,
    visibility: _Visibility,
) -> SharedL1Client:
    shared, memory = _configs(path)
    return SharedL1Client(
        shared,
        memory,
        pool=pool,
        visibility=visibility,
        register_cuda=False,
    )


def test_two_clients_share_one_physical_tensor_with_exact_visibility_range(
    tmp_path: Path,
) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    producer_visibility = _Visibility()
    consumer_visibility = _Visibility()
    producer = _client(path, pool, producer_visibility)
    consumer = _client(path, pool, consumer_visibility)
    try:
        key = _key()
        write_obj = producer.reserve_writes([key], _layout())[0]
        assert write_obj is not None
        expected = torch.arange(write_obj.get_size(), dtype=torch.uint8)
        write_obj.raw_data.copy_(expected)
        producer.finish_writes([key])

        read_obj = consumer.reserve_reads([key])[0]
        assert read_obj is not None
        assert torch.equal(read_obj.raw_data, expected)
        consumer.finish_reads([key])
        assert consumer.reserve_reads([key])[0] is read_obj
        consumer.finish_reads([key])

        snapshot = pool.snapshot()
        objects = cast(dict[str, dict[str, Any]], snapshot["objects"])
        handle = next(iter(objects.values()))["handle"]
        expected_call = (
            _MAPPING_OFFSET + handle.offset,
            handle.length,
            handle.generation,
        )
        assert producer_visibility.calls == [(1, *expected_call)]
        assert consumer_visibility.calls == [
            (2, *expected_call),
            (2, *expected_call),
        ]
        assert objects[next(iter(objects))]["active_readers"] == 0
    finally:
        consumer.close()
        producer.close()


def test_contract_mismatch_fails_before_mapping(tmp_path: Path) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    shared, memory = _configs(path)
    wrong = SharedL1Config(
        **{
            **shared.__dict__,
            "region_id": "wrong-region",
        }
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        SharedL1Client(
            wrong,
            memory,
            pool=pool,
            visibility=_Visibility(),
            register_cuda=False,
        )


@pytest.mark.parametrize("operation", [1, 2])
def test_visibility_failure_keeps_object_unpublished_or_releases_read(
    tmp_path: Path,
    operation: int,
) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    key = _key()
    producer = _client(path, pool, _Visibility(fail_operation=operation))
    try:
        producer.reserve_writes([key], _layout())
        if operation == 1:
            with pytest.raises(RuntimeError, match="visibility failure"):
                producer.finish_writes([key])
            assert pool.snapshot()["objects"] == {}
            return
        producer.finish_writes([key])
        with pytest.raises(RuntimeError, match="visibility failure"):
            producer.reserve_reads([key])
        objects = cast(dict[str, dict[str, Any]], pool.snapshot()["objects"])
        assert next(iter(objects.values()))["active_readers"] == 0
    finally:
        producer.close()


def test_layout_length_mismatch_releases_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    client = _client(path, pool, _Visibility())
    key = _key()
    try:
        client.reserve_writes([key], _layout())
        client.finish_writes([key])
        reserve_reads = pool.reserve_reads

        def malformed_reserve(keys: list[str]):
            reservations = reserve_reads(keys)
            assert reservations[0] is not None
            reservations[0] = replace(
                reservations[0],
                layout=MemoryLayoutDesc([torch.Size([1])], [torch.float16]),
            )
            return reservations

        monkeypatch.setattr(pool, "reserve_reads", malformed_reserve)
        with pytest.raises(ValueError, match="length does not match"):
            client.reserve_reads([key])
        objects = cast(dict[str, dict[str, Any]], pool.snapshot()["objects"])
        assert next(iter(objects.values()))["active_readers"] == 0
    finally:
        client.close()


def test_failed_read_release_can_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    client = _client(path, pool, _Visibility())
    try:
        key = _key()
        client.reserve_writes([key], _layout())
        client.finish_writes([key])
        client.reserve_reads([key])
        finish_reads = pool.finish_reads
        monkeypatch.setattr(
            pool,
            "finish_reads",
            lambda _items: (_ for _ in ()).throw(RuntimeError("rpc failed")),
        )
        with pytest.raises(RuntimeError, match="rpc failed"):
            client.finish_reads([key])
        monkeypatch.setattr(pool, "finish_reads", finish_reads)
        client.finish_reads([key])
        objects = cast(dict[str, dict[str, Any]], pool.snapshot()["objects"])
        assert next(iter(objects.values()))["active_readers"] == 0
    finally:
        client.close()


def test_cuda_registration_failure_is_not_staged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    shared, memory = _configs(path)
    unpinned: list[int] = []
    monkeypatch.setattr(
        client_module,
        "current_device_spec",
        SimpleNamespace(
            pin_memory=lambda _pointer, _length: False,
            unpin_memory=lambda pointer: unpinned.append(pointer),
        ),
    )
    with pytest.raises(RuntimeError, match="pageable staging is not accepted"):
        SharedL1Client(
            shared,
            memory,
            pool=pool,
            visibility=_Visibility(),
        )
    assert unpinned == []


def test_cuda_unregistration_failure_keeps_mapping_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _region(tmp_path)
    pool = SharedL1Pool("region", _CAPACITY, _ALIGNMENT, "layout")
    shared, memory = _configs(path)
    unpin_succeeds = False
    monkeypatch.setattr(
        client_module,
        "current_device_spec",
        SimpleNamespace(
            pin_memory=lambda _pointer, _length: True,
            unpin_memory=lambda _pointer: unpin_succeeds,
        ),
    )
    client = SharedL1Client(
        shared,
        memory,
        pool=pool,
        visibility=_Visibility(),
    )

    with pytest.raises(RuntimeError, match="host unregistration failed"):
        client.close()
    assert client.memcheck()

    unpin_succeeds = True
    client.close()
    assert not client.memcheck()
