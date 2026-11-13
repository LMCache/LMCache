# SPDX-License-Identifier: Apache-2.0
"""Store-time byte-coverage check: engine-spec-derived vs staged bytes.

The check family targets the "silently stores 1 of N kernel pages" bug
class: a registration-geometry bug makes LMCache size each stored chunk
to a fraction of the engine-spec KV payload while internal bookkeeping
stays consistent (high hit rates, no error). The invariant implemented
here derives the expected bytes ONLY from the engine's KV cache spec
(leaf ``page_size_bytes / block_size``, summed over member layers),
carries it through ``EngineGroupInfo``/``KernelGroupInfo``, and compares
it against the reserved host object's size at store reservation time.

All tests run on CPU without vLLM installed; the vLLM spec classes are
doubled with the same class names (sliding-window / mamba / attention
specs are detected by class name).
"""

# Standard
from dataclasses import dataclass, field
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    _spec_bytes_per_token,
    create_engine_group_infos_from_vllm,
)
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    byte_coverage_check_enabled,
    verify_store_byte_coverage,
)
import lmcache.lmcache_native as lmcache_native

# ---------------------------------------------------------------------------
# Test doubles for the vLLM KV cache spec classes (same class names; the
# conversion module detects kinds by class name).
# ---------------------------------------------------------------------------


@dataclass
class AttentionSpec:
    block_size: int
    page_size_bytes: int


@dataclass
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int


@dataclass
class MambaSpec:
    block_size: int
    mamba_cache_mode: str = "align"


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: "dict[str, object]" = field(default_factory=dict)


@dataclass
class MockKVCacheGroup:
    layer_names: list[str]
    kv_cache_spec: object


@dataclass
class MockKVCacheConfig:
    kv_cache_groups: list[MockKVCacheGroup]


class TestSpecBytesPerToken:
    def test_rate_is_page_over_block(self):
        spec = SimpleNamespace(page_size_bytes=128, block_size=64)
        assert _spec_bytes_per_token(spec) == 2

    def test_rate_keeps_slot_compressed_page_semantics(self):
        # Slot packing re-parameterizes the page interior; the page still
        # covers ``block_size`` logical tokens.
        spec = SimpleNamespace(page_size_bytes=4096, block_size=256, tokens_per_state=4)
        assert _spec_bytes_per_token(spec) == 16

    def test_non_divisible_page_is_none(self):
        spec = SimpleNamespace(page_size_bytes=100, block_size=64)
        assert _spec_bytes_per_token(spec) is None

    def test_missing_fields_are_none(self):
        assert _spec_bytes_per_token(SimpleNamespace(block_size=64)) is None
        assert _spec_bytes_per_token(object()) is None

    def test_turboquant_is_none(self):
        spec = SimpleNamespace(page_size_bytes=128, block_size=64, tq_slot_size=8)
        assert _spec_bytes_per_token(spec) is None

    def test_non_positive_fields_are_none(self):
        spec = SimpleNamespace(page_size_bytes=0, block_size=64)
        assert _spec_bytes_per_token(spec) is None


class TestConversionStamping:
    """``create_engine_group_infos_from_vllm`` stamps the spec-derived rate."""

    def _dense_caches(self, names):
        return {n: torch.randn(2, 32, 16, 8, 64, dtype=torch.float16) for n in names}

    def test_stamps_expected_bytes_per_token(self):
        names = ["layer.0", "layer.1", "layer.2", "layer.3"]
        caches = self._dense_caches(names)
        # One block of 16 tokens covers kv2 x 16 x 8 heads x 64 x 2B = 32768.
        spec = AttentionSpec(block_size=16, page_size_bytes=32768)
        infos = create_engine_group_infos_from_vllm(
            MockKVCacheConfig(kv_cache_groups=[MockKVCacheGroup(names, spec)]),
            caches,
        )
        assert len(infos) == 1
        # The rate sums over the group's member layers (the stored object
        # covers all four layers).
        assert infos[0].expected_bytes_per_token == 4 * (32768 // 16)

    def test_missing_page_size_is_none(self):
        names = ["layer.0"]
        caches = self._dense_caches(names)
        spec = SimpleNamespace(block_size=16)
        infos = create_engine_group_infos_from_vllm(
            MockKVCacheConfig(kv_cache_groups=[MockKVCacheGroup(names, spec)]),
            caches,
        )
        assert infos[0].expected_bytes_per_token is None

    def test_dcp_is_none(self):
        names = ["layer.0"]
        caches = self._dense_caches(names)
        spec = AttentionSpec(block_size=16, page_size_bytes=32768)
        infos = create_engine_group_infos_from_vllm(
            MockKVCacheConfig(kv_cache_groups=[MockKVCacheGroup(names, spec)]),
            caches,
            dcp_size=2,
        )
        assert infos[0].expected_bytes_per_token is None

    def test_sliding_window_is_none(self):
        names = ["layer.0"]
        caches = self._dense_caches(names)
        spec = SlidingWindowSpec(block_size=16, page_size_bytes=32768, sliding_window=8)
        infos = create_engine_group_infos_from_vllm(
            MockKVCacheConfig(kv_cache_groups=[MockKVCacheGroup(names, spec)]),
            caches,
        )
        assert infos[0].expected_bytes_per_token is None

    def test_recurrent_mamba_is_none(self):
        names = ["layer.0"]
        caches = self._dense_caches(names)
        spec = MambaSpec(block_size=16)
        infos = create_engine_group_infos_from_vllm(
            MockKVCacheConfig(kv_cache_groups=[MockKVCacheGroup(names, spec)]),
            caches,
        )
        assert infos[0].expected_bytes_per_token is None

    def test_rates_are_per_transferred_plane(self):
        """A mixed registration (K+V main cache + key-only MLA-style cache)
        splits into two transfer identities; each carries only its own
        plane's rate, so a dual-plane model never double-counts."""
        caches = {
            # rank-5 K+V (kv_size 2): kv2 x 16 x 8 x 64 x 2B per 16-token block
            "layer.0": torch.randn(2, 32, 16, 8, 64, dtype=torch.float16),
            # rank-3 key-only: 16 x 128 x 2B per 16-token block
            "layer.1": torch.randn(32, 16, 128, dtype=torch.bfloat16),
        }
        spec = UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={
                "layer.0": AttentionSpec(block_size=16, page_size_bytes=32768),
                "layer.1": AttentionSpec(block_size=16, page_size_bytes=4096),
            },
        )
        infos = create_engine_group_infos_from_vllm(
            MockKVCacheConfig(kv_cache_groups=[MockKVCacheGroup(list(caches), spec)]),
            caches,
        )
        by_layers = {info.layer_indices: info for info in infos}
        assert by_layers[(0,)].expected_bytes_per_token == 32768 // 16
        assert by_layers[(1,)].expected_bytes_per_token == 4096 // 16


def _nhd_tensors(num_layers, bs):
    return [
        torch.randn(2, 32, bs, 2, 8, dtype=torch.float16) for _ in range(num_layers)
    ]


def _coverage_manager(tensors, infos, lmcache_tokens_per_chunk=256):
    return KVLayerGroupsManager(
        tensors,
        engine_kv_formats=[lmcache_native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS]
        * len(tensors),
        engine_group_infos=infos,
        lmcache_tokens_per_chunk=lmcache_tokens_per_chunk,
    )


class TestExpectedObjectGroupBytes:
    def test_healthy_rate_matches_staged_layout(self):
        # 8 layers, kv2 x nh2 x hs8 x fp16 => 64 B/token/layer; the group
        # (all 8 layers) stamps 512 B/token.
        tensors = _nhd_tensors(8, bs=64)
        infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(8)),
                tokens_per_block=64,
                expected_bytes_per_token=512,
            )
        ]
        manager = _coverage_manager(tensors, infos)
        # Full-chunk staging: kv2 x nl8 x slots256 x hidden16 x 2B = 131072.
        assert manager.expected_object_group_bytes_per_chunk(0, 256) == 512 * 256

    def test_unknown_group_rate_is_none(self):
        tensors = _nhd_tensors(2, bs=64)
        infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1),
                tokens_per_block=64,
                expected_bytes_per_token=None,
            )
        ]
        manager = _coverage_manager(tensors, infos)
        assert manager.expected_object_group_bytes_per_chunk(0, 256) is None

    def test_out_of_range_object_group_is_none(self):
        manager = _coverage_manager(_nhd_tensors(1, bs=64), [])
        assert manager.expected_object_group_bytes_per_chunk(1, 256) is None


class TestVerifyStoreByteCoverage:
    """The fail-closed comparison, including the family's 1-of-N equation."""

    @staticmethod
    def _fake_ctx(manager):
        return SimpleNamespace(kv_layer_groups_manager=manager)

    @staticmethod
    def _reserved(size, count=2):
        return {i: SimpleNamespace(get_size=lambda s=size: s) for i in range(count)}

    def test_healthy_reservation_passes(self):
        tensors = _nhd_tensors(8, bs=64)
        infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(8)),
                tokens_per_block=64,
                expected_bytes_per_token=512,
            )
        ]
        manager = _coverage_manager(tensors, infos)
        ctx = self._fake_ctx(manager)
        # No raise: staged bytes equal the spec-derived expectation.
        verify_store_byte_coverage(ctx, 0, self._reserved(131072), 256)

    def test_one_of_n_truncation_raises(self):
        # 16:1 family geometry: logical tokens_per_block 1024 over 64-slot
        # kernel pages. Without a layout fix the registration geometry
        # stages only 1/16 of the chunk's KV payload per object.
        tensors = _nhd_tensors(8, bs=64)
        infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(8)),
                tokens_per_block=1024,
                expected_bytes_per_token=512,
            )
        ]
        manager = _coverage_manager(tensors, infos, lmcache_tokens_per_chunk=1024)
        ctx = self._fake_ctx(manager)
        # Staged: kv2 x nl8 x slots64 x hidden16 x 2B = 32768, i.e. 1/16 of
        # the spec-derived 512 B/token x 1024 tokens.
        reserved = self._reserved(32768)
        with pytest.raises(ValueError, match="32768") as exc_info:
            verify_store_byte_coverage(ctx, 0, reserved, 1024)
        assert "524288" in str(exc_info.value)

    def test_unknown_expectation_noops(self):
        tensors = _nhd_tensors(2, bs=64)
        infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0, 1),
                tokens_per_block=64,
                expected_bytes_per_token=None,
            )
        ]
        manager = _coverage_manager(tensors, infos)
        # No raise even though the staged size is arbitrary.
        verify_store_byte_coverage(self._fake_ctx(manager), 0, self._reserved(1), 256)

    def test_empty_reservation_noops(self):
        tensors = _nhd_tensors(1, bs=64)
        infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=(0,),
                tokens_per_block=64,
                expected_bytes_per_token=64,
            )
        ]
        manager = _coverage_manager(tensors, infos)
        verify_store_byte_coverage(self._fake_ctx(manager), 0, {}, 256)


class TestByteCoverageEnvSwitch:
    def test_default_on(self, monkeypatch):
        monkeypatch.delenv("LMCACHE_STORE_BYTE_COVERAGE_CHECK", raising=False)
        assert byte_coverage_check_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "FALSE", " Off "])
    def test_disable_values(self, monkeypatch, value):
        monkeypatch.setenv("LMCACHE_STORE_BYTE_COVERAGE_CHECK", value)
        assert byte_coverage_check_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_enable_values(self, monkeypatch, value):
        monkeypatch.setenv("LMCACHE_STORE_BYTE_COVERAGE_CHECK", value)
        assert byte_coverage_check_enabled() is True
