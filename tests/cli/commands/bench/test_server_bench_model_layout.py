# SPDX-License-Identifier: Apache-2.0
"""Public layout contracts; vLLM numeric anchors from 63ff748f (not engine E2E)."""

# Standard
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal
import argparse

# Third Party
import pytest
import torch
import yaml

# First Party
from lmcache.cli.commands.bench.server_bench.command import add_server_arguments
from lmcache.cli.commands.bench.server_bench.config import parse_args_to_config
from lmcache.cli.commands.bench.server_bench.model_layout import (
    DSV4,
    KDA,
    MHA,
    MLA,
    Allocation,
    Layout,
    Model,
    ModelLayout,
    Parallel,
    allocate_layout,
    load_model_layout,
    resolve_rank_local_tensor_specs,
)
from lmcache.utils import EngineType

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/server_bench"


def _spec(
    layers: list, tp: int = 1, mode: Literal["basic", "vllm"] = "basic"
) -> ModelLayout:
    definitions = {str(i): layer for i, layer in enumerate(layers)}
    dtype: Literal["bfloat16", "fp8_ds_mla"] | None = (
        None
        if mode == "basic"
        else ("fp8_ds_mla" if isinstance(layers[0], DSV4) else "bfloat16")
    )
    return ModelLayout(
        Model(definitions, list(definitions)),
        Allocation(257),
        Parallel(tp),
        Layout(mode, 256, dtype),
    )


@pytest.mark.parametrize(
    "tp,conv,state", [(1, 432, 6144), (2, 216, 3072), (8, 56, 768)]
)
def test_basic_geometry_and_tp(tp: int, conv: int, state: int) -> None:
    spec = _spec([KDA(96), MLA()], tp)
    views, engine = resolve_rank_local_tensor_specs(spec)
    assert engine == EngineType.MOCK
    assert [(t.shape, t.dtype, t.sharded) for t in views] == [
        ((256, conv), torch.bfloat16, True),
        ((256, state), torch.float32, True),
        ((256, 576), torch.bfloat16, False),
    ]
    assert len({t.pool for t in views}) == 3
    layer = spec.model.layer_definitions["0"]
    assert isinstance(layer, KDA) and layer.num_heads == 96


@pytest.mark.parametrize(
    "kv_heads,window,width,block,sharded",
    [(32, None, 512, 256, True), (8, 128, 128, 128, True), (1, None, 128, 256, False)],
)
def test_basic_mha_variants(
    kv_heads: int,
    window: int | None,
    width: int,
    block: int,
    sharded: bool,
) -> None:
    views, _ = resolve_rank_local_tensor_specs(
        _spec([MHA(32, kv_heads, 128, window)], 8)
    )
    assert [t.name.rsplit(".", 1)[-1] for t in views] == ["key", "value"]
    assert [(t.shape, t.sharded) for t in views] == [
        ((block, width), sharded),
        ((block, width), sharded),
    ]
    assert {t.group.tokens_per_block for t in views} == {block}
    assert {t.group.sw_size_tokens for t in views} == {window or -1}


def test_mha_yaml_kind(tmp_path: Path) -> None:
    data = yaml.safe_load((EXAMPLES / "basic.yaml").read_text())
    data["model"] = {
        "layer_definitions": {
            "attention": {
                "kind": "mha",
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            }
        },
        "layer_types": ["attention"],
    }
    path = tmp_path / "mha.yaml"
    path.write_text(yaml.safe_dump(data))
    layer = load_model_layout(path).model.layer_definitions["attention"]
    assert isinstance(layer, MHA)


@pytest.mark.parametrize(
    "layers",
    [
        [MLA()],
        [MHA(32, 8, 128)],
        [KDA(8)],
        [KDA(8), KDA(16), MLA(), DSV4(4), DSV4(128)],
    ],
)
def test_composition_restore_and_corruption(layers: list) -> None:
    spec = _spec(layers, 2)
    views, engine = resolve_rank_local_tensor_specs(spec)
    cache = allocate_layout(views, 257, 256, "cpu", engine)
    cache.fill(512, 0, 0)
    original = cache.checksum()
    retained = list(cache.restore_views(512, 0, 0, 512, 256))
    saved = [t.clone() for t in retained]
    for tensor in retained:
        tensor.zero_()
    assert cache.checksum() != original
    for tensor, data in zip(retained, saved, strict=True):
        tensor.copy_(data)
    assert cache.checksum() == original
    retained[-1].zero_()  # A missed restore must fail even if other components match.
    assert cache.checksum() != original
    cache.fill(512, 0, 1)
    assert (cache.checksum() == original) == all(isinstance(x, MLA) for x in layers)


def test_presets_and_namespace(tmp_path: Path) -> None:
    kimi = load_model_layout(EXAMPLES / "kimi_k3.yaml")
    ds = load_model_layout(EXAMPLES / "deepseek_v4_flash.yaml")
    assert Counter(kimi.model.layer_types) == {"linear": 69, "latent": 24}
    assert Counter(ds.model.layer_types) == {"swa": 2, "c4": 21, "c128": 20}
    for spec, counts in [
        (kimi, [23, 23, 23, 24]),
        (ds, [21, 21, 20, 22, 21, 21, 21, 20]),
    ]:
        views, _ = resolve_rank_local_tensor_specs(spec)
        assert (
            list(
                Counter(
                    (t.group.engine_group_id, t.shape, t.dtype) for t in views
                ).values()
            )
            == counts
        )
    identity = kimi.cache_name
    kimi.model.layer_definitions["unused"] = KDA(7)
    kimi.allocation.num_blocks = 99
    assert kimi.cache_name == identity
    resolve_rank_local_tensor_specs(
        kimi
    )  # Unused incompatible TP does not affect the layout.
    kimi.parallel.tp_size = 2
    assert kimi.cache_name != identity


@pytest.mark.parametrize("nb", [1, 2])
def test_packed_geometry_and_registration(nb: int) -> None:
    spec = _spec([DSV4(1), DSV4(4), DSV4(128)], 8, "vllm")
    views, engine = resolve_rank_local_tensor_specs(spec)
    expected = [
        ("1.indexer", (64, 132), 0, 256, 0),
        ("1.compressed", (64, 584), 8640, 256, 0),
        ("2.compressed", (2, 584), 46080, 256, 0),
        ("0.swa", (64, 584), 0, 64, 1),
        ("1.swa", (64, 584), 0, 64, 2),
        ("2.swa", (64, 584), 0, 64, 3),
        ("1.indexer_state", (4, 512), 0, 4, 4),
        ("1.compressor_state", (4, 2048), 8640, 4, 4),
        ("2.compressor_state", (8, 1024), 0, 8, 5),
    ]
    assert [
        (
            t.name.removeprefix("model.layers."),
            t.shape,
            t.offset,
            t.group.tokens_per_block,
            t.group.engine_group_id,
        )
        for t in views
    ] == expected
    cache = allocate_layout(views, nb, 256, "cpu", engine)
    assert len(cache.owners) == 1
    for tensor, t in zip(cache.caches.values(), views, strict=True):
        assert tensor.shape == (nb, *t.shape)
        assert tensor.stride(0) * tensor.element_size() == 47808
        assert tensor.untyped_storage().data_ptr() == cache.owners[0].data_ptr()
    assert len(cache.manager.kernel_groups) == 9


def test_unified_geometry_and_address_capacity() -> None:
    views, engine = resolve_rank_local_tensor_specs(
        _spec([KDA(96)] * 3 + [MLA()], 8, "vllm")
    )
    assert [t.shape for t in views] == [(768, 1, 1152)] * 3 + [(768, 576)]
    assert {t.stride for t in views} == {884736}
    cache = allocate_layout(views, 9, 768, "cpu", engine)
    assert cache.block_ids(1536, 0, 0, 1536) == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert cache.block_ids(1536, 0, 768, 1536) == [[2], [4], [6], [8]]
    with pytest.raises(ValueError, match="capacity"):
        cache.block_ids(2304, 0, 0, 2304)
    for bounds in [(1536, 1, 768), (1536, 0, 1537), (-1, 0, 0)]:
        with pytest.raises(ValueError):
            cache.block_ids(bounds[0], 0, *bounds[1:])
    with pytest.raises(ValueError, match="chunk"):
        allocate_layout(views, 9, 256, "cpu", engine)
    with pytest.raises(ValueError, match="bounds"):
        allocate_layout([replace(views[0], offset=1)], 9, 768, "cpu", engine)


@pytest.mark.parametrize(
    "patch",
    [
        {"parallel": {"tp_size": 0}},
        {"parallel": {"tp_size": 5}},
        {"allocation": {"num_blocks": 0}},
        {"layout": {"block_size": 0}},
        {"model": {"layer_definitions": {}, "layer_types": []}},
        {"model": {"layer_definitions": {}, "layer_types": ["missing"]}},
        {"model": {"preset": "unknown"}},
        {"layout": {"mode": "sglang"}},
        {"layout": {"kv_cache_dtype": "bfloat16"}},
        {"family": "kimi"},
        {"model": {"preset": "kimi_k3", "layer_types": ["latent"]}},
    ],
)
def test_invalid_yaml(tmp_path: Path, patch: dict) -> None:
    data = yaml.safe_load((EXAMPLES / "basic.yaml").read_text())
    data.update(patch)
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        load_model_layout(path)


@pytest.mark.parametrize("text", ["model: [", "model: {}\nmodel: {}", "[]"])
def test_malformed_yaml(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(text)
    with pytest.raises(ValueError):
        load_model_layout(path)


@pytest.mark.parametrize(
    "layers",
    [
        [KDA(8)],
        [MHA(32, 8, 128)],
        [MLA()],
        [KDA(8), KDA(16), MLA()],
        [DSV4(1)],
        [DSV4(4), MLA()],
    ],
)
def test_vllm_rejects_unsupported_combinations(layers: list) -> None:
    with pytest.raises(ValueError):
        resolve_rank_local_tensor_specs(_spec(layers, 1, "vllm"))


@pytest.mark.parametrize(
    "layer,tp",
    [(MHA(30, 8, 128), 1), (MHA(32, 2, 128), 4)],
)
def test_mha_rejects_invalid_head_geometry(layer: MHA, tp: int) -> None:
    with pytest.raises(ValueError):
        resolve_rank_local_tensor_specs(_spec([layer], tp))


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--tp-size", "2"],
        ["--num-blocks", "3"],
        ["--block-size", "256"],
        ["--use-mla"],
        ["--mode", "cpu"],
        ["--transfer-mode", "engine_driven"],
    ],
)
def test_cli_file_ownership(extra: list[str]) -> None:
    parser = argparse.ArgumentParser()
    add_server_arguments(parser)
    args = parser.parse_args(["--model-layout", str(EXAMPLES / "basic.yaml"), *extra])
    if extra:
        with pytest.raises(ValueError):
            parse_args_to_config(args)
    else:
        config = parse_args_to_config(args)
        assert config.tp_size == 2
        assert config.model_layout is not None
        assert config.model_layout.layout.mode == "basic"


@pytest.mark.parametrize(
    "layers,writers,world", [([KDA(8), MLA()], 2, 2), ([MLA()], 1, 1)]
)
def test_client_routing_identity_and_probe_cleanup(
    monkeypatch: pytest.MonkeyPatch, layers: list, writers: int, world: int
) -> None:
    # Standard
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # First Party
    from lmcache.cli.commands.bench.server_bench import client as client_module
    from lmcache.cli.commands.bench.server_bench import helpers, model_layout
    from lmcache.cli.commands.bench.server_bench.config import BenchConfig
    from lmcache.v1.multiprocess import transfer_context
    from lmcache.v1.multiprocess.custom_types import SKIP_L2_REQUEST_CONFIG_KEY
    from lmcache.v1.multiprocess.transport.factory import RequestClientFactory

    caches, contexts = [], []
    rpc = MagicMock()
    rpc.lookup.return_value.result.return_value = None
    rpc.query_prefetch_status.return_value.result.side_effect = [0, 2]

    def allocate(specs: list, nb: int, chunk: int, device: str, engine: EngineType):
        cache = allocate_layout(specs, nb, chunk, "cpu", engine)
        caches.append(cache)
        return cache

    def context(*args: Any, **kwargs: Any) -> MagicMock:
        ctx = MagicMock()
        ctx.create_recorded_event.return_value = None
        ctx.submit_store.return_value.result.return_value = True
        ctx.submit_retrieve.return_value.result.return_value = True
        contexts.append(ctx)
        return ctx

    monkeypatch.setattr(model_layout, "allocate_layout", allocate)
    monkeypatch.setattr(
        client_module, "torch_dev", SimpleNamespace(is_available=lambda: True)
    )
    monkeypatch.setattr(helpers, "_get_chunk_size", lambda _: 256)
    monkeypatch.setattr(RequestClientFactory, "create", lambda *a, **k: rpc)
    monkeypatch.setattr(transfer_context, "create_transfer_context", context)
    spec = _spec(layers, 2)
    spec.allocation.num_blocks = 3
    config = BenchConfig(
        "ipc:///tmp/model-test", "", "gpu", "auto", 2, False, 511, "", 3, 256, spec
    )
    client = client_module.ServerBenchClient(config, lambda _: None)
    client.start()
    try:
        cold = client.create_request(0, request_id="cold", request_kind="cold")
        assert cold is not None
        before = client.compute_checksums(cold, 0, 512)
        assert before and len(before) == 2
        result = client.store(cold, 0, 512)
        assert result and result.attempted_worker_ranks == tuple(range(writers))
        client.wait_store_visible(cold)
        assert rpc.free_lookup_locks.call_count == rpc.end_session.call_count == 2
        for call in rpc.lookup.call_args_list:
            key, tp = call.args
            assert key.model_name == spec.cache_name and key.world_size == world
            assert key.num_kv_readers == (2 if world == 1 else 1) and tp == 2
            assert key.request_configs == {SKIP_L2_REQUEST_CONFIG_KEY: True}
        rpc.wait_prefetch_status.assert_not_called()
        warm = client.create_request(0, request_id="warm", request_kind="warm")
        assert warm is not None
        assert client.compute_checksums(warm, 0, 512) == before
        client.zero_destination(warm, 0, 512)
        after = client.compute_checksums(warm, 0, 512)
        assert after and all(a != b for a, b in zip(before, after, strict=True))
        result = client.retrieve(warm, 0, 512)
        assert result and result.attempted_worker_ranks == (0, 1)
        for ctx in contexts:
            assert ctx.register.call_args.kwargs["engine_type"] == EngineType.MOCK
            key = ctx.submit_retrieve.call_args.args[1]
            assert key.model_name == spec.cache_name and key.world_size == world
        # Even a failed unlock must still attempt to close its completed probe.
        rpc.query_prefetch_status.return_value.result.side_effect = [2]
        rpc.free_lookup_locks.return_value.result.side_effect = RuntimeError("unlock")
        with pytest.raises(RuntimeError, match="unlock"):
            client.wait_store_visible(cold)
        assert rpc.end_session.call_count == 3

        # A broken L1-only contract closes the session and aborts instead of
        # starting another probe whose locks could overlap the first one.
        rpc.free_lookup_locks.return_value.result.side_effect = None
        rpc.query_prefetch_status.return_value.result.side_effect = [None]
        with pytest.raises(RuntimeError, match="did not complete"):
            client.wait_store_visible(cold)
        assert rpc.free_lookup_locks.call_count == 3
        assert rpc.end_session.call_count == 4
    finally:
        client.close()
        client.close()
    assert all(context.unregister.call_count == 1 for context in contexts)
    assert all(ctx.close.call_count == 1 for ctx in contexts)
