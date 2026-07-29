# SPDX-License-Identifier: Apache-2.0
"""Contract tests for Engine-driven executors bound to shared transfer plans."""

# Standard
from contextlib import contextmanager
from dataclasses import replace
from threading import Lock
from typing import Iterator
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.server_transfer import (
    PickleTransferStrategy,
    ShmTransferStrategy,
)
from lmcache.v1.multiprocess.protocols.engine import RegisterEngineDrivenContextResponse
from lmcache.v1.multiprocess.transfer_context import worker_transfer
from lmcache.v1.multiprocess.transfer_context.base import EngineDrivenContextMetadata
from lmcache.v1.multiprocess.transfer_context.worker_transfer import (
    EngineDrivenTransferContext,
)
from lmcache.v1.multiprocess.transfer_plan import (
    KernelGroupTransferMetadata,
    KVTransferMetadata,
    ObjectGroupTransferMetadata,
    TransferPlan,
    TransferPlanDirection,
    build_engine_driven_object_group_layout_desc,
    build_transfer_plan_from_kernel_group_block_ids,
    build_transfer_plan_without_block_ids,
)
import lmcache.c_ops as lmc_ops


def _transfer_metadata() -> KVTransferMetadata:
    """Build heterogeneous two-object-group metadata for plan binding tests."""
    return KVTransferMetadata(
        num_chunks_in_sw=(-1, 1),
        tokens_per_chunk=8,
        kernel_groups=(
            KernelGroupTransferMetadata(
                kernel_group_id=0,
                engine_group_id=0,
                layer_indices=(0,),
                blocks_per_chunk=2,
                blocks_per_window=1,
                slots_per_chunk_in_window=4,
                kv_size=2,
                num_layers=1,
                hidden_dim_size=1,
                slots_per_block=4,
                tokens_per_block=4,
                dtype=torch.float32,
                engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            ),
            KernelGroupTransferMetadata(
                kernel_group_id=1,
                engine_group_id=1,
                layer_indices=(1,),
                blocks_per_chunk=2,
                blocks_per_window=1,
                slots_per_chunk_in_window=4,
                kv_size=2,
                num_layers=1,
                hidden_dim_size=1,
                slots_per_block=4,
                tokens_per_block=4,
                dtype=torch.float32,
                engine_kv_format=lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            ),
        ),
        object_groups=(
            ObjectGroupTransferMetadata(
                object_group_id=0,
                kernel_group_ids=(0,),
                sw_size_chunks=-1,
            ),
            ObjectGroupTransferMetadata(
                object_group_id=1,
                kernel_group_ids=(1,),
                sw_size_chunks=1,
            ),
        ),
    )


def test_kernel_group_block_id_adapter_rejects_conflicting_repeated_groups() -> None:
    """Repeated kernel groups must agree before an executor binds a plan."""
    metadata = _transfer_metadata()
    repeated_engine_metadata = replace(
        metadata,
        kernel_groups=(
            metadata.kernel_groups[0],
            replace(metadata.kernel_groups[1], engine_group_id=0),
        ),
    )

    with pytest.raises(ValueError, match="conflicting block IDs"):
        build_transfer_plan_from_kernel_group_block_ids(
            repeated_engine_metadata,
            [[0, 1], [2, 3]],
            TransferPlanDirection.STORE,
        )


class _CompletedFuture(MessagingFuture[RegisterEngineDrivenContextResponse]):
    """Minimal completed future used by worker registration."""

    def __init__(self) -> None:
        """Initialize the future with a no-SHM registration response."""
        super().__init__()
        self.set_result(RegisterEngineDrivenContextResponse())


class _Event:
    """No-op event implementing the worker transfer event protocol."""

    def wait(self, _stream: object | None = None) -> None:
        """Provide ordering without requiring a device event."""


class _WorkerContext:
    """Fake pickle context that records retrieve-plan transport arguments."""

    def __init__(self, payload: list[list[list[torch.Tensor]]]) -> None:
        self._payload = payload
        self.retrieve_skips: list[int] = []

    def prepare_retrieve(
        self,
        _key: IPCCacheServerKey,
        _instance_id: int,
        skip_first_n_tokens: int = 0,
        _transfer_plan: TransferPlan | None = None,
    ) -> list[list[list[torch.Tensor]]]:
        """Record the prefix forwarded to the server and return planned chunks."""
        self.retrieve_skips.append(skip_first_n_tokens)
        return self._payload

    def commit_retrieve(self, _key: IPCCacheServerKey, _instance_id: int) -> bool:
        """Complete the fake retrieve lifecycle."""
        return True

    def close(self) -> None:
        """Release no resources."""


def test_worker_retrieve_binds_window_and_prefix_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public worker context scatters payloads in shared-plan order."""
    transfer_metadata = _transfer_metadata()
    layouts = [
        build_engine_driven_object_group_layout_desc(transfer_metadata, 8, group_id)
        for group_id in range(2)
    ]
    payload = [
        [[torch.full((2, 1, 4, 1), float(index))] for index in range(3)],
        [[torch.full((2, 1, 4, 1), 9.0)]],
    ]
    fake_context = _WorkerContext(payload)

    monkeypatch.setattr(
        worker_transfer,
        "compute_kv_layout",
        lambda *_args, **_kwargs: (
            4,
            2,
            1,
            "float32",
            lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS,
            2,
        ),
    )
    monkeypatch.setattr(
        worker_transfer,
        "_build_multi_group_wire_fields",
        lambda *_args, **_kwargs: (
            [],
            [],
            [],
            [],
            layouts,
            AttnWindowDesc(num_chunks_in_sw=[-1, 1]),
            transfer_metadata,
        ),
    )
    monkeypatch.setattr(
        worker_transfer,
        "create_engine_driven_context",
        lambda *_args, **_kwargs: fake_context,
    )
    scattered: list[tuple[list[int], int, int]] = []

    def _scatter(
        _kv_caches: dict[str, torch.Tensor],
        block_ids: list[int],
        _chunks: list[torch.Tensor],
        blocks_per_chunk: int,
        skip_first_n_tokens: int = 0,
        **_kwargs: object,
    ) -> None:
        scattered.append((block_ids, blocks_per_chunk, skip_first_n_tokens))

    monkeypatch.setattr(worker_transfer, "scatter_cpu_to_paged_kv", _scatter)
    context = EngineDrivenTransferContext()
    context.register(
        instance_id=1,
        kv_caches={
            "layer_0": torch.zeros(1),
            "layer_1": torch.zeros(1),
        },
        model_name="model",
        world_size=1,
        blocks_in_chunk=2,
        mq_client=MagicMock(),
        mq_timeout=1.0,
        send_request=lambda *_args: _CompletedFuture(),
        engine_group_infos=(
            EngineGroupInfo(engine_group_id=0, layer_indices=(0,)),
            EngineGroupInfo(engine_group_id=1, layer_indices=(1,)),
        ),
    )

    result = context.submit_retrieve(
        "request",
        IPCCacheServerKey.from_token_ids(
            "model", 1, 1, list(range(24)), request_id="request"
        ),
        1,
        {"layer_0": torch.zeros(1), "layer_1": torch.zeros(1)},
        [[0, 1, 2, 3, 4, 5], [10, 11, 12, 13, 14, 15]],
        _Event(),
        blocks_in_chunk=2,
        skip_first_n_tokens=4,
    )

    assert result.result(timeout=1.0) is True
    assert fake_context.retrieve_skips == [4]
    assert scattered == [([1, 3, 5], 1, 0), ([15], 1, 0)]


class _MemoryObject:
    """Small in-memory object exposing the storage tensors required by a strategy."""

    def __init__(self, value: float, offset: int) -> None:
        self._tensor = torch.full((2, 1, 4, 1), value)
        self.shm_offset = offset
        self.shm_byte_length = self._tensor.numel() * self._tensor.element_size()
        self.data_ptr = self._tensor.data_ptr()

    def get_tensor(self, tensor_index: int) -> torch.Tensor:
        """Return the single object-group tensor."""
        if tensor_index != 0:
            raise IndexError(tensor_index)
        return self._tensor


def test_server_pickle_and_shm_bind_window_and_prefix_plan() -> None:
    """Pickle and SHM storage executors select the same planned object keys."""
    transfer_metadata = _transfer_metadata()
    layouts = [
        build_engine_driven_object_group_layout_desc(transfer_metadata, 8, group_id)
        for group_id in range(2)
    ]
    metadata = EngineDrivenContextMetadata(
        layout_desc=layouts[0],
        block_size=4,
        use_mla=False,
        object_group_layout_descs=layouts,
        attn_desc=AttnWindowDesc(num_chunks_in_sw=[-1, 1]),
        transfer_metadata=transfer_metadata,
    )
    transfer_plan = build_transfer_plan_without_block_ids(
        transfer_metadata,
        num_chunks=3,
        direction=TransferPlanDirection.RETRIEVE,
        skip_first_n_tokens=4,
    )
    key = IPCCacheServerKey.from_token_ids("model", 1, 1, list(range(24)))
    object_keys: list[list[ObjectKey]] = [
        [ObjectKey(f"g0-c{index}".encode(), "model", 0, 0) for index in range(3)],
        [ObjectKey(f"g1-c{index}".encode(), "model", 0, 1) for index in range(3)],
    ]
    objects = {
        object_key: _MemoryObject(float(index), index * 64)
        for index, object_key in enumerate(
            [key for group_keys in object_keys for key in group_keys]
        )
    }

    pickle_storage = MagicMock()
    pickle_reads: list[list[ObjectKey]] = []

    @contextmanager
    def _read_prefetched(
        requested_keys: list[ObjectKey],
    ) -> Iterator[list[_MemoryObject]]:
        pickle_reads.append(requested_keys)
        yield [objects[object_key] for object_key in requested_keys]

    pickle_storage.read_prefetched_results.side_effect = _read_prefetched
    pickle_strategy = PickleTransferStrategy(pickle_storage)
    pickle_response = pickle_strategy.prepare_retrieve(
        key,
        1,
        metadata,
        lambda _key: object_keys,
        transfer_plan,
    )
    assert pickle_response.success is True
    assert pickle_reads == [
        object_keys[0],
        [object_keys[1][2]],
    ]

    shm_storage = MagicMock()
    shm_reads: list[list[ObjectKey]] = []

    def _unsafe_read(
        requested_keys: list[ObjectKey],
    ) -> tuple[list[ObjectKey], list[_MemoryObject]]:
        shm_reads.append(requested_keys)
        return requested_keys, [objects[object_key] for object_key in requested_keys]

    shm_storage.unsafe_read.side_effect = _unsafe_read
    shm_strategy = ShmTransferStrategy(
        storage_manager=shm_storage,
        pending_writes={},
        pending_reads={},
        pending_lock=Lock(),
        transfer_key_factory=lambda transfer_key, instance_id: (
            instance_id,
            transfer_key,
        ),
        fallback_strategy=PickleTransferStrategy(shm_storage),
    )
    shm_response = shm_strategy.prepare_retrieve(
        key,
        1,
        metadata,
        lambda _key: object_keys,
        transfer_plan,
    )
    assert shm_response.success is True
    assert shm_reads == pickle_reads
    assert [
        group["chunk_indices"] for group in shm_response.context["object_groups"]
    ] == [[0, 1, 2], [2]]
