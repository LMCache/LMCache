# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace

# First Party
from lmcache.integration.vllm.utils import (
    compute_dp_device_offset,
    get_dp_local_rank,
)


def _make_parallel_config(
    *,
    data_parallel_rank_local=None,
    data_parallel_index=None,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    nnodes_within_dp: int = 1,
    distributed_executor_backend: str | None = "mp",
    data_parallel_backend: str | None = "mp",
) -> SimpleNamespace:
    """Build a duck-typed stand-in for vLLM's ``ParallelConfig``.

    Only the attributes consulted by ``get_dp_local_rank`` and
    ``compute_dp_device_offset`` need to exist.
    """
    return SimpleNamespace(
        data_parallel_rank_local=data_parallel_rank_local,
        data_parallel_index=data_parallel_index,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        nnodes_within_dp=nnodes_within_dp,
        distributed_executor_backend=distributed_executor_backend,
        data_parallel_backend=data_parallel_backend,
    )


# -- get_dp_local_rank --------------------------------------------------------


def test_get_dp_local_rank_prefers_explicit_attr() -> None:
    cfg = _make_parallel_config(data_parallel_rank_local=3, data_parallel_index=7)
    assert get_dp_local_rank(cfg) == 3


def test_get_dp_local_rank_falls_back_to_index() -> None:
    cfg = _make_parallel_config(data_parallel_rank_local=None, data_parallel_index=5)
    assert get_dp_local_rank(cfg) == 5


def test_get_dp_local_rank_defaults_to_zero() -> None:
    cfg = _make_parallel_config(data_parallel_rank_local=None, data_parallel_index=None)
    assert get_dp_local_rank(cfg) == 0


def test_get_dp_local_rank_missing_attributes() -> None:
    # ``ParallelConfig`` on older vLLM may not expose either attribute at all.
    cfg = SimpleNamespace()
    assert get_dp_local_rank(cfg) == 0


# -- compute_dp_device_offset ------------------------------------------------


def test_dp_offset_zero_when_dp_disabled() -> None:
    # dp_local_rank == 0 implies a zero offset regardless of TP/PP.
    cfg = _make_parallel_config(
        data_parallel_rank_local=0,
        tensor_parallel_size=4,
        pipeline_parallel_size=2,
    )
    assert compute_dp_device_offset(cfg) == 0


def test_dp_offset_with_tp_only() -> None:
    # DP1 with TP=2 → DP offset = 1 * 2 * 1 = 2
    cfg = _make_parallel_config(
        data_parallel_rank_local=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
    )
    assert compute_dp_device_offset(cfg) == 2


def test_dp_offset_with_tp_and_pp() -> None:
    # DP2 with TP=2, PP=2 → DP offset = 2 * 2 * 2 = 8
    cfg = _make_parallel_config(
        data_parallel_rank_local=2,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )
    assert compute_dp_device_offset(cfg) == 8


def test_dp_offset_uses_index_fallback() -> None:
    # When ``data_parallel_rank_local`` is None, fall back to
    # ``data_parallel_index`` (matches vLLM gpu_worker logic).
    cfg = _make_parallel_config(
        data_parallel_rank_local=None,
        data_parallel_index=1,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )
    assert compute_dp_device_offset(cfg) == 1


def test_dp_offset_zero_for_ray_executor() -> None:
    # Ray actors already isolate each worker to a single visible device.
    cfg = _make_parallel_config(
        data_parallel_rank_local=1,
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
    )
    assert compute_dp_device_offset(cfg) == 0


def test_dp_offset_zero_for_external_launcher() -> None:
    # torchrun / external launchers handle GPU pinning via LOCAL_RANK.
    cfg = _make_parallel_config(
        data_parallel_rank_local=1,
        tensor_parallel_size=2,
        distributed_executor_backend="external_launcher",
    )
    assert compute_dp_device_offset(cfg) == 0


def test_dp_offset_zero_for_ray_dp_backend() -> None:
    # Ray DP coordinator also implies Ray-managed GPU placement.
    cfg = _make_parallel_config(
        data_parallel_rank_local=1,
        tensor_parallel_size=2,
        data_parallel_backend="ray",
    )
    assert compute_dp_device_offset(cfg) == 0


def test_dp_offset_zero_for_multi_node_dp() -> None:
    # Multi-node DP: ``dp_local_rank * tp * pp`` is no longer a valid local
    # device index, so vLLM skips the adjustment and so do we.
    cfg = _make_parallel_config(
        data_parallel_rank_local=1,
        tensor_parallel_size=2,
        nnodes_within_dp=2,
    )
    assert compute_dp_device_offset(cfg) == 0


def test_dp_offset_handles_missing_optional_attributes() -> None:
    # Older vLLM versions may not expose ``nnodes_within_dp`` /
    # ``data_parallel_backend``; the helper must default safely.
    cfg = SimpleNamespace(
        data_parallel_rank_local=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        distributed_executor_backend="mp",
    )
    assert compute_dp_device_offset(cfg) == 2
