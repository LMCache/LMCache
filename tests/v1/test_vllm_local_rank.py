# SPDX-License-Identifier: Apache-2.0
"""
:func:`lmcache.integration.vllm.utils.calculate_local_rank_and_world_size`
must return the device index vLLM binds the worker to.

Each vLLM data-parallel rank is its own engine whose TP/PCP/PP world starts
at rank 0, so ``parallel_config.rank`` alone repeats on every data-parallel
rank. vLLM offsets the device by the local data-parallel rank in
``Worker.init_device``; LMCache's ``local_worker_id`` must match it, because
``CreateGPUConnector`` calls ``set_device(local_worker_id)``.

The tests stub ``torch_dev`` and ``vllm.platforms`` so they run without
accelerators or vLLM.
"""

# Standard
from types import SimpleNamespace
from typing import Any, Callable
import sys

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.utils import calculate_local_rank_and_world_size
import lmcache


def _vllm_config(
    *,
    rank: int,
    tp: int = 1,
    pp: int = 1,
    pcp: int = 1,
    dp_local_rank: int | None = 0,
    dp_rank: int = 0,
    executor_backend: str = "mp",
    dp_backend: str = "mp",
    nnodes_within_dp: int = 1,
) -> Any:
    parallel_config = SimpleNamespace(
        rank=rank,
        world_size=tp * pp * pcp,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
        data_parallel_rank_local=dp_local_rank,
        data_parallel_index=dp_rank,
        data_parallel_rank=dp_rank,
        distributed_executor_backend=executor_backend,
        data_parallel_backend=dp_backend,
        nnodes_within_dp=nnodes_within_dp,
    )
    return SimpleNamespace(parallel_config=parallel_config)


@pytest.fixture
def platform(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Stub a node with ``device_count`` devices and a device-id mapping."""

    def install(
        device_count: int = 8, to_visible: Callable[[int], int] = lambda i: i
    ) -> None:
        monkeypatch.setattr(
            lmcache, "torch_dev", SimpleNamespace(device_count=lambda: device_count)
        )
        current_platform = SimpleNamespace(
            logical_device_id_to_visible_device_id=to_visible
        )
        monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace())
        monkeypatch.setitem(
            sys.modules,
            "vllm.platforms",
            SimpleNamespace(current_platform=current_platform),
        )

    return install


@pytest.mark.parametrize("dp_local_rank", [0, 1, 3])
def test_data_parallel_ranks_get_distinct_devices(
    platform: Callable[..., None], dp_local_rank: int
) -> None:
    """With TP=1, data-parallel rank N is bound to device N, not device 0."""
    platform()
    config = _vllm_config(rank=0, dp_local_rank=dp_local_rank)
    assert calculate_local_rank_and_world_size(config) == (dp_local_rank, 1)


def test_data_parallel_offset_scales_with_tp(platform: Callable[..., None]) -> None:
    """TP rank 1 of data-parallel rank 1 with TP=2 is device 1 * 2 + 1."""
    platform()
    config = _vllm_config(rank=1, tp=2, dp_local_rank=1)
    assert calculate_local_rank_and_world_size(config) == (3, 2)


def test_falls_back_to_data_parallel_index(platform: Callable[..., None]) -> None:
    """Without a local data-parallel rank, vLLM uses the data-parallel index."""
    platform()
    config = _vllm_config(rank=0, dp_local_rank=None, dp_rank=2)
    assert calculate_local_rank_and_world_size(config) == (2, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"executor_backend": "ray"},
        {"executor_backend": "external_launcher"},
        {"dp_backend": "ray"},
        {"nnodes_within_dp": 2},
    ],
)
def test_no_data_parallel_offset_where_vllm_skips_it(
    platform: Callable[..., None], overrides: dict[str, Any]
) -> None:
    """Backends where vLLM does not offset by the data-parallel rank."""
    platform()
    config = _vllm_config(rank=0, dp_local_rank=1, **overrides)
    assert calculate_local_rank_and_world_size(config) == (0, 1)


def test_applies_vllm_visible_device_mapping(platform: Callable[..., None]) -> None:
    """The index goes through vLLM's logical-to-visible device mapping."""
    platform(to_visible=lambda i: i + 4)
    config = _vllm_config(rank=0, dp_local_rank=1)
    assert calculate_local_rank_and_world_size(config) == (5, 1)
