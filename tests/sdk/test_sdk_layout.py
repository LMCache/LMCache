# SPDX-License-Identifier: Apache-2.0
"""Unit tests for how the SDK mirrors an engine's registered layout.

The kernel-group entries mirror what the LMCache server's ``/status`` reports:

- for Qwen3-8B on vLLM, a fused-K/V KV cache whose concrete shape the server
  cannot render, and an MLA-like query ring whose concrete shape it can;
- for Qwen3.5-0.8B (18 GDN + 6 attention layers), a hybrid KV cache whose
  attention pages vLLM re-pages at 32-token kernel pages.
"""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.sdk import context as sdk_context
from lmcache.sdk.cache_kind import LMCacheSDKCacheKind
from lmcache.sdk.context import (
    FULL_WINDOW,
    LMCacheSDKError,
    _kv_layout,
    _layer_labels,
    _pool_layer_shape,
    _resolve_device,
    _resolve_window,
    _server_pool_chunks,
)
from lmcache.sdk.hybrid_layout import (
    ATTENTION,
    RECURRENT,
    HybridLayoutError,
    layer_roles,
    plan_hybrid_groups,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

# Dense pool layout and windows

SERVER_Q_KEY = "sdk.q_sw_size_tokens"

KV_GROUP = {
    "tokens_per_block": 16,
    "slots_per_block": 16,
    "engine_kv_format": "NL_X_NB_BS_NH_CS",
    "engine_kv_shape": "NL x [NB, BS, NH, CS]",
    "engine_kv_concrete_shape": "Unknown (EngineKVFormat.NL_X_NB_BS_NH_CS)",
}
Q_GROUP = {
    "tokens_per_block": 16,
    "slots_per_block": 16,
    "engine_kv_format": "NL_X_NB_BS_HS",
    "engine_kv_shape": "NL x [NB, BS, HS]",
    "engine_kv_concrete_shape": "36 x [1024, 16, 4096]",
}


@pytest.fixture
def qwen3_8b_heads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Stand in for Qwen3-8B's Hugging Face config: 8 KV heads of dim 128."""
    calls: list[tuple[str, int]] = []

    def fake(hf_model_name: str, world_size: int) -> dict[str, int]:
        calls.append((hf_model_name, world_size))
        return {"NH": 8 // world_size, "HS": 128, "CS": 256}

    monkeypatch.setattr(sdk_context, "_hf_head_sizes", fake)
    return calls


def test_layer_labels_reads_one_layer_of_the_legend():
    assert _layer_labels("NL x [NB, BS, NH, CS]") == ["NB", "BS", "NH", "CS"]
    assert _layer_labels("NL x [NB, 2, NH, BS, HS]") == ["NB", "2", "NH", "BS", "HS"]


@pytest.mark.parametrize("legend", ["2 x NL x [NB, BS, NH, HS]", "NB, BS, HS"])
def test_layer_labels_rejects_formats_without_one_tensor_per_layer(legend: str):
    with pytest.raises(LMCacheSDKError, match="one tensor per layer"):
        _layer_labels(legend)


def test_query_ring_shape_comes_from_its_concrete_shape():
    """The ring's reported [1024, 16, 4096] keeps its sizes; only NB shrinks."""
    shape = _pool_layer_shape(
        Q_GROUP,
        _layer_labels(Q_GROUP["engine_kv_shape"]),
        LMCacheSDKCacheKind.QUERY,
        num_blocks=64,
        world_size=1,
        hf_model_name="unused",
    )
    assert shape == (64, 16, 4096)


def test_unrenderable_kv_shape_falls_back_to_hf_head_sizes(qwen3_8b_heads):
    """Fused K/V packs K and V per head, so CS is twice the head dim."""
    shape = _pool_layer_shape(
        KV_GROUP,
        _layer_labels(KV_GROUP["engine_kv_shape"]),
        LMCacheSDKCacheKind.KV,
        num_blocks=32,
        world_size=2,
        hf_model_name="Qwen/Qwen3-8B",
    )
    assert shape == (32, 16, 4, 256)
    assert qwen3_8b_heads == [("Qwen/Qwen3-8B", 2)]


def test_unrenderable_split_kv_shape_keeps_literal_axes(qwen3_8b_heads):
    group = {**KV_GROUP, "engine_kv_shape": "NL x [NB, 2, NH, BS, HS]"}
    shape = _pool_layer_shape(
        group,
        _layer_labels(group["engine_kv_shape"]),
        LMCacheSDKCacheKind.KV,
        num_blocks=32,
        world_size=1,
        hf_model_name="Qwen/Qwen3-8B",
    )
    assert shape == (32, 2, 8, 16, 128)


def test_unrenderable_query_shape_is_rejected(qwen3_8b_heads):
    """The HF config describes KV heads, not the query ring, so never guess."""
    group = {**Q_GROUP, "engine_kv_concrete_shape": "Unknown (x)"}
    with pytest.raises(LMCacheSDKError, match="query ring"):
        _pool_layer_shape(
            group,
            _layer_labels(group["engine_kv_shape"]),
            LMCacheSDKCacheKind.QUERY,
            num_blocks=32,
            world_size=1,
            hf_model_name="Qwen/Qwen3-8B",
        )
    assert qwen3_8b_heads == []


def test_block_size_must_match_tokens_per_block():
    group = {**Q_GROUP, "engine_kv_concrete_shape": "36 x [1024, 32, 4096]"}
    with pytest.raises(LMCacheSDKError, match="tokens_per_block"):
        _pool_layer_shape(
            group,
            _layer_labels(group["engine_kv_shape"]),
            LMCacheSDKCacheKind.QUERY,
            num_blocks=32,
            world_size=1,
            hf_model_name="unused",
        )


@pytest.mark.parametrize(
    ("legend", "layout"),
    [
        ("NL x [NB, BS, NH, CS]", "NHD"),
        ("NL x [NB, NH, BS, CS]", "HND"),
        ("NL x [NB, 2, NH, BS, HS]", "HND"),
        ("NL x [NB, BS, HS]", "NHD"),
    ],
)
def test_kv_layout_follows_the_heads_axis(legend: str, layout: str):
    assert _kv_layout(_layer_labels(legend)) == layout


def test_resolve_device_rejects_cpu():
    """The server's lmcache-driven path cannot map a CPU pool on GPU hosts."""
    with pytest.raises(LMCacheSDKError, match="accelerator"):
        _resolve_device("cpu")


def test_resolve_device_keeps_an_explicit_index(monkeypatch: pytest.MonkeyPatch):
    """An explicit index must never consult the current (default) device."""

    def fail() -> int:
        raise AssertionError("current_device() consulted")

    monkeypatch.setattr(sdk_context, "torch_dev", SimpleNamespace(current_device=fail))
    assert _resolve_device("cuda:1") == torch.device("cuda", 1)


def test_resolve_device_defaults_to_the_current_index(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        sdk_context, "torch_dev", SimpleNamespace(current_device=lambda: 3)
    )
    assert _resolve_device("cuda") == torch.device("cuda", 3)


def _mp_conf(extra: dict[str, object]) -> dict[str, object]:
    """The ``mp`` section of ``/config`` with a given plugin extra config."""
    return {"chunk_size": 256, "runtime_plugin_config": {"extra_config": extra}}


def test_server_pool_chunks_reads_the_runtime_plugin_config():
    """``--runtime-plugin-config '{"sdk.pool_chunks": 8}'`` sizes the pool."""
    assert _server_pool_chunks(_mp_conf({"sdk.pool_chunks": 8})) == 8


def test_server_pool_chunks_defaults_when_unset():
    assert _server_pool_chunks(_mp_conf({})) == 4
    assert _server_pool_chunks({"chunk_size": 256}) == 4


@pytest.mark.parametrize("value", [0, -1, "8", 2.5, True])
def test_server_pool_chunks_rejects_non_positive_integers(value: object):
    with pytest.raises(LMCacheSDKError, match="sdk.pool_chunks"):
        _server_pool_chunks(_mp_conf({"sdk.pool_chunks": value}))


def _mp(extra: dict[str, object], separate: bool = False) -> dict[str, object]:
    """``/config``'s ``mp`` section with a plugin extra config and grouping."""
    return {
        "chunk_size": 256,
        "separate_object_groups": separate,
        "runtime_plugin_config": {"extra_config": extra},
    }


def test_query_window_defers_to_the_server():
    """``--runtime-plugin-config '{"sdk.q_sw_size_tokens": 64}'`` sets it."""
    window = _resolve_window(
        LMCacheSDKCacheKind.QUERY, None, _mp({SERVER_Q_KEY: 64}), 256
    )
    assert window == 64


def test_query_window_defaults_to_full():
    window = _resolve_window(LMCacheSDKCacheKind.QUERY, None, _mp({}), 256)
    assert window == FULL_WINDOW


def test_explicit_window_overrides_the_server():
    window = _resolve_window(
        LMCacheSDKCacheKind.QUERY, 32, _mp({SERVER_Q_KEY: 64}), 256
    )
    assert window == 32


def test_kv_ignores_the_server_query_window():
    """The server key is for the query ring; KV stays full."""
    window = _resolve_window(LMCacheSDKCacheKind.KV, None, _mp({SERVER_Q_KEY: 64}), 256)
    assert window == FULL_WINDOW


def test_kv_window_is_rejected():
    """KV shares its layout registration with the engine, which stays full."""
    with pytest.raises(LMCacheSDKError, match="only the QUERY kind"):
        _resolve_window(LMCacheSDKCacheKind.KV, 64, _mp({}), 256)


def test_whole_chunk_window_needs_object_group_separation():
    """Without separation the server would silently read every chunk."""
    with pytest.raises(LMCacheSDKError, match="--separate-object-groups"):
        _resolve_window(LMCacheSDKCacheKind.QUERY, 512, _mp({}), 256)
    window = _resolve_window(
        LMCacheSDKCacheKind.QUERY, 512, _mp({}, separate=True), 256
    )
    assert window == 512


def test_invalid_window_is_rejected():
    with pytest.raises(LMCacheSDKError, match="sw_size_tokens"):
        _resolve_window(LMCacheSDKCacheKind.QUERY, -5, _mp({}), 256)
    with pytest.raises(LMCacheSDKError, match="sw_size_tokens"):
        _resolve_window(LMCacheSDKCacheKind.QUERY, 0, _mp({}), 256)


# Hybrid group planning

SUBPAGED = "6 x [5830, 2, 544, 1, 512]"


def _group(idx: int, object_idx: int, shape: str = SUBPAGED) -> dict[str, object]:
    return {
        "kernel_group_idx": idx,
        "engine_group_idx": idx,
        "object_group_idx": object_idx,
        "num_layers": 6,
        "layer_indices": list(range(6 * idx, 6 * idx + 6)),
        "tokens_per_block": 544,
        "slots_per_block": 544,
        "dtype": "torch.bfloat16",
        "engine_kv_concrete_shape": shape,
        "engine_kv_format": "NL_X_NB_TWO_BS_NH_HS",
        "engine_kv_shape": "NL x [NB, 2, BS, NH, HS]",
    }


# Three GDN groups share object group 0; attention is object group 1.
QWEN35_GROUPS = [_group(0, 0), _group(1, 0), _group(2, 0), _group(3, 1)]


def _qwen35_config() -> SimpleNamespace:
    text = SimpleNamespace(
        layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 6,
        num_hidden_layers=24,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=256,
        hidden_size=1024,
    )
    return SimpleNamespace(text_config=text)


def test_qwen35_groups_mirror_vllm_registration():
    plans = plan_hybrid_groups(
        QWEN35_GROUPS, _qwen35_config(), world_size=1, kernel_block_override=32
    )
    # The attention pages are sub-paged at the given kernel size.
    assert plans[3].kernel_block_size == 32

    assert [p.role for p in plans] == [RECURRENT] * 3 + [ATTENTION]
    # vLLM's rules: align-mode Mamba is a one-block window, attention full.
    assert plans[0].engine_group_info() == EngineGroupInfo(
        engine_group_id=0,
        layer_indices=tuple(range(6)),
        tokens_per_block=544,
        sw_size_tokens=544,
        recurrent_state=True,
    )
    assert plans[3].engine_group_info() == EngineGroupInfo(
        engine_group_id=3,
        layer_indices=tuple(range(18, 24)),
        tokens_per_block=544,
        sw_size_tokens=-1,
    )


def test_subpaged_attention_needs_the_kernel_page_size():
    """The registration does not report it, so the SDK does not guess it."""
    with pytest.raises(HybridLayoutError, match="sdk.kernel_block_size"):
        plan_hybrid_groups(QWEN35_GROUPS, _qwen35_config(), world_size=1)


@pytest.mark.parametrize("kernel", [544, 100])
def test_kernel_page_size_must_divide_the_block(kernel: int):
    with pytest.raises(HybridLayoutError, match="proper divisor"):
        plan_hybrid_groups(
            QWEN35_GROUPS, _qwen35_config(), world_size=1, kernel_block_override=kernel
        )


def test_real_heads_mean_the_page_is_not_subpaged():
    groups = QWEN35_GROUPS[:3] + [_group(3, 1, "6 x [5830, 2, 544, 2, 256]")]
    plans = plan_hybrid_groups(groups, _qwen35_config(), world_size=1)
    assert plans[3].kernel_block_size == 544


def test_heads_matching_neither_view_are_rejected():
    groups = QWEN35_GROUPS[:3] + [_group(3, 1, "6 x [5830, 2, 544, 4, 128]")]
    with pytest.raises(HybridLayoutError, match="matching neither"):
        plan_hybrid_groups(groups, _qwen35_config(), world_size=1)


def test_one_object_group_cannot_separate_the_roles():
    """Without --separate-object-groups every group shares object group 0."""
    groups = [_group(i, 0) for i in range(4)]
    with pytest.raises(HybridLayoutError, match="--separate-object-groups"):
        plan_hybrid_groups(groups, _qwen35_config(), world_size=1)


def test_equal_layer_counts_are_ambiguous():
    config = _qwen35_config()
    config.text_config.layer_types = ["linear_attention", "full_attention"] * 6
    groups = [_group(0, 0), _group(1, 1)]
    with pytest.raises(HybridLayoutError, match="cannot be told apart"):
        plan_hybrid_groups(groups, config, world_size=1)


def test_layer_count_mismatch_is_rejected():
    with pytest.raises(HybridLayoutError, match="decoder layers"):
        plan_hybrid_groups(QWEN35_GROUPS[:3], _qwen35_config(), world_size=1)


def test_roles_derive_from_the_full_attention_interval():
    config = SimpleNamespace(num_hidden_layers=8, full_attention_interval=4)
    assert layer_roles(config) == ([RECURRENT] * 3 + [ATTENTION]) * 2


def test_unsupported_layer_types_are_rejected():
    config = SimpleNamespace(layer_types=["sliding_attention", "full_attention"])
    with pytest.raises(HybridLayoutError, match="not supported"):
        layer_roles(config)
