# SPDX-License-Identifier: Apache-2.0
"""Regression tests for per-kernel sliding-window retrieve trimming."""

# Standard
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.platform.base.cache_context import BaseCacheContext


class FakeTransferDirection:
    H2D = "h2d"
    D2H = "d2h"


class FakeKernelGroupSpec:
    def __init__(self, *_args: Any) -> None:
        pass


@dataclass
class FakeLaunchVar:
    spec_idx: int
    start_block_pos: int
    num_blocks: int
    batch_size: int
    skip_blocks: int


@dataclass
class FakeBatchStep:
    staging: Any
    launch_vars: list[FakeLaunchVar]


class FakeOps:
    TransferDirection = FakeTransferDirection
    KernelGroupSpec = FakeKernelGroupSpec
    LaunchVar = FakeLaunchVar
    BatchStep = FakeBatchStep

    def __init__(self) -> None:
        self.native_skips: list[list[int]] = []
        self.fallback_skips: list[int] = []

    def execute_object_group_transfer(
        self,
        _direction: str,
        _device: torch.device,
        _pin_chunk_size: int,
        _kernel_group_specs: list[FakeKernelGroupSpec],
        batch_steps: list[FakeBatchStep],
    ) -> None:
        self.native_skips.extend(
            [
                [launch.skip_blocks for launch in step.launch_vars]
                for step in batch_steps
            ]
        )

    def multi_layer_block_kv_transfer(self, *args: Any) -> None:
        self.fallback_skips.append(cast(int, args[-1]))


class FakeAttnDesc:
    def __init__(self, is_full_attention: bool) -> None:
        self._is_full_attention = is_full_attention
        self.num_chunks_in_sw = [-1 if is_full_attention else 1]

    def is_full_attention(self, _object_group_id: int) -> bool:
        return self._is_full_attention


class FakeKVLayerGroupsManager:
    def __init__(
        self,
        full_sw_kv: bool = False,
        separate_object_groups: bool = False,
    ) -> None:
        self.full_sw_kv = full_sw_kv
        self.separate_object_groups = separate_object_groups
        kernel_group_indices = [1] if separate_object_groups else [0, 1]
        self.object_groups = [
            SimpleNamespace(kernel_group_indices=kernel_group_indices)
        ]
        self.kernel_groups = [
            SimpleNamespace(sw_size_tokens=-1),
            SimpleNamespace(sw_size_tokens=64),
        ]

    def get_subchunk_sw_size_tokens(self, kernel_group_id: int) -> int:
        return [256, 64][kernel_group_id]

    def get_sw_size_chunks(self, kernel_group_id: int) -> int:
        if self.full_sw_kv:
            return -1
        return [-1, 1][kernel_group_id]

    def get_attn_desc(self) -> FakeAttnDesc:
        return FakeAttnDesc(not self.separate_object_groups)


class FakeCacheContext:
    lmcache_tokens_per_chunk = 256
    max_batch_size = 2
    device = torch.device("cpu")

    def __init__(
        self,
        full_sw_kv: bool = False,
        separate_object_groups: bool = False,
    ) -> None:
        self.kv_layer_groups_manager = FakeKVLayerGroupsManager(
            full_sw_kv,
            separate_object_groups,
        )
        self._buffers = [[torch.zeros(1), torch.zeros(1)] for _ in range(2)]

    def calculate_num_blocks(self, num_tokens: int, _kernel_group_id: int) -> int:
        return num_tokens // 32

    def get_kernel_group_kv_pointers(self, kernel_group_id: int) -> torch.Tensor:
        return self._buffers[kernel_group_id][0]

    def get_temp_kernel_group_buffer(
        self, slot: int, kernel_group_id: int
    ) -> torch.Tensor:
        return self._buffers[kernel_group_id][slot]

    def get_shape_desc(self, _kernel_group_id: int) -> object:
        return object()

    def get_slots_per_chunk_in_sw(self, _kernel_group_id: int) -> int:
        return 1

    def get_engine_kv_format(self, _kernel_group_id: int) -> str:
        return "fake"

    def get_temp_object_group_buffer(
        self, _slot: int, _object_group_id: int
    ) -> torch.Tensor:
        return torch.zeros(1)


def run_transfer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_native: bool,
    direction: str,
    skip_first_n_tokens: int = 0,
    full_sw_kv: bool = False,
    separate_object_groups: bool = False,
) -> list[list[int]]:
    fake_ops = FakeOps()
    monkeypatch.setattr(transfer, "lmc_ops", fake_ops)
    monkeypatch.setattr(
        transfer,
        "_HAS_NATIVE_OBJECT_GROUP_TRANSFER",
        use_native,
    )
    monkeypatch.setattr(
        transfer,
        "build_staging_copies",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        transfer,
        "lmcache_memcpy_async_h2d",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        transfer,
        "lmcache_memcpy_async_d2h",
        lambda *_args, **_kwargs: None,
    )

    memory_objs = cast(list[MemoryObj], [object() for _ in range(4)])
    transfer.transfer_kv_per_object_group(
        cast(
            BaseCacheContext,
            FakeCacheContext(full_sw_kv, separate_object_groups),
        ),
        [
            torch.arange(32, dtype=torch.int64),
            torch.arange(8, dtype=torch.int64),
        ],
        memory_objs,
        object_group_id=0,
        batch_size=2,
        skip_first_n_tokens=skip_first_n_tokens,
        direction=direction,
    )

    if use_native:
        return fake_ops.native_skips
    num_kernel_groups = 1 if separate_object_groups else 2
    return [
        fake_ops.fallback_skips[index : index + num_kernel_groups]
        for index in range(
            0,
            len(fake_ops.fallback_skips),
            num_kernel_groups,
        )
    ]


@pytest.mark.parametrize("use_native", [False, True])
def test_h2d_trims_sliding_window_group_in_mixed_object_group(
    monkeypatch: pytest.MonkeyPatch,
    use_native: bool,
) -> None:
    """Retrieve skips old SW chunks while loading every full-attention chunk."""
    skips = run_transfer(
        monkeypatch,
        use_native=use_native,
        direction=FakeTransferDirection.H2D,
    )

    assert skips == [[0, 4], [0, 2]]


@pytest.mark.parametrize("use_native", [False, True])
def test_d2h_does_not_trim_sliding_window_group(
    monkeypatch: pytest.MonkeyPatch,
    use_native: bool,
) -> None:
    """Store preserves all chunks for future requests."""
    skips = run_transfer(
        monkeypatch,
        use_native=use_native,
        direction=FakeTransferDirection.D2H,
    )

    assert skips == [[0, 0], [0, 0]]


@pytest.mark.parametrize("use_native", [False, True])
def test_full_sw_kv_does_not_trim_sliding_window_group(
    monkeypatch: pytest.MonkeyPatch,
    use_native: bool,
) -> None:
    """Full-SW mode preserves all chunks during retrieve."""
    skips = run_transfer(
        monkeypatch,
        use_native=use_native,
        direction=FakeTransferDirection.H2D,
        full_sw_kv=True,
    )

    assert skips == [[0, 0], [0, 0]]


@pytest.mark.parametrize("use_native", [False, True])
def test_separated_sw_group_uses_object_level_trim_only(
    monkeypatch: pytest.MonkeyPatch,
    use_native: bool,
) -> None:
    """Separated SW objects keep the existing object-level trimming path."""
    skips = run_transfer(
        monkeypatch,
        use_native=use_native,
        direction=FakeTransferDirection.H2D,
        separate_object_groups=True,
    )

    assert skips == [[0]]


@pytest.mark.parametrize("use_native", [False, True])
def test_window_trim_does_not_double_count_existing_prefix_skip(
    monkeypatch: pytest.MonkeyPatch,
    use_native: bool,
) -> None:
    """Overlapping APC and window prefixes use the larger skip, not their sum."""
    skips = run_transfer(
        monkeypatch,
        use_native=use_native,
        direction=FakeTransferDirection.H2D,
        skip_first_n_tokens=736,
    )

    assert skips == [[7, 2]]
