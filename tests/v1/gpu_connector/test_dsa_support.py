# SPDX-License-Identifier: Apache-2.0
"""Tests for DSA (Deepseek Sparse Attention) indexer support.

These tests cover the SGLang DSA detection path: the indexer buffer is a
flat list of 2-D uint8 tensors whose last axis is a multiple of 132
(128 B fp8 key + 4 B fp32 scale per token).  Detection reshapes each
tensor to 3-D ``(NB, BS, 132)`` and reports ``NL_X_NB_BSV_BSS``.

The mixed-list tests verify that ``normalize_and_discover_per_layer_formats``
splits a DSA dual-buffer ``kv_caches`` list (NL MLA latent + NL DSA indexer)
into two format groups.  The ``_build_engine_group_infos`` tests verify the
daemon-side engine-group metadata for the same split.  The
``get_shapes`` tests verify per-group shape reporting through
``KVLayerGroupsManager``.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format import detect_format
from lmcache.v1.gpu_connector.utils import normalize_and_discover_per_layer_formats
import lmcache.c_ops as lmc_ops

NB, NL, BS, NH, HS = 7, 5, 4, 2, 4
DT = torch.float16
F = lmc_ops.EngineKVFormat

# DSA indexer constants
INDEX_DIM = 132  # 128 B fp8 key + 4 B fp32 scale per token


def _t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=DT)


def _dsa_t(num_pages: int, page_size: int = BS) -> torch.Tensor:
    """Create a DSA indexer tensor: 2-D uint8 (num_pages, page_size * 132)."""
    return torch.zeros(num_pages, page_size * INDEX_DIM, dtype=torch.uint8)


# ---------------------------------------------------------------------------
# SGLang DSA detector
# ---------------------------------------------------------------------------


class TestSGLangDSADetection:
    """SGLang DSA indexer format detection via ``detect_format``."""

    def test_dsa_indexer_detected(self):
        """A flat list of 2-D uint8 tensors with last axis % 132 == 0 is
        detected as ``NL_X_NB_BSV_BSS`` and reshaped to 3-D."""
        kv = [_dsa_t(NB) for _ in range(NL)]
        fmt, out = detect_format(kv, EngineType.SGLANG, {"tokens_per_block": BS})
        assert fmt == F.NL_X_NB_BSV_BSS
        assert len(out) == NL
        for t in out:
            assert t.dim() == 3
            assert tuple(t.shape) == (NB, BS, INDEX_DIM)

    def test_dsa_indexer_preserves_data_ptr(self):
        """The reshape is a view -- the data pointer must not change."""
        kv = [_dsa_t(NB) for _ in range(NL)]
        _, out = detect_format(kv, EngineType.SGLANG, {"tokens_per_block": BS})
        assert out[0].data_ptr() == kv[0].data_ptr()

    def test_dsa_indexer_larger_multiple_of_132(self):
        """A last axis that is a larger multiple of 132 (e.g. 2 * 132) is
        also valid -- the detector infers page_size from the quotient."""
        kv = [
            torch.zeros(NB, 2 * INDEX_DIM, dtype=torch.uint8) for _ in range(NL)
        ]
        fmt, out = detect_format(kv, EngineType.SGLANG, {"tokens_per_block": BS})
        assert fmt == F.NL_X_NB_BSV_BSS
        assert tuple(out[0].shape) == (NB, 2, INDEX_DIM)

    def test_dsa_not_confused_with_mla(self):
        """A 2-D uint8 tensor must not match the MLA branch (which expects
        3-D with shape[1] == 1).  DSA detection runs first and catches it."""
        kv = [_dsa_t(NB) for _ in range(NL)]
        fmt, _ = detect_format(kv, EngineType.SGLANG, {})
        assert fmt == F.NL_X_NB_BSV_BSS

    def test_dsa_does_not_require_tokens_per_block_hint(self):
        """DSA detection works without the ``tokens_per_block`` layout hint
        -- it infers page_size from ``shape[-1] // 132``."""
        kv = [_dsa_t(NB) for _ in range(NL)]
        fmt, out = detect_format(kv, EngineType.SGLANG, {})
        assert fmt == F.NL_X_NB_BSV_BSS
        assert tuple(out[0].shape) == (NB, BS, INDEX_DIM)


# ---------------------------------------------------------------------------
# normalize_and_discover_per_layer_formats: mixed DSA dual-buffer
# ---------------------------------------------------------------------------


class TestNormalizeMixedDSA:
    """``normalize_and_discover_per_layer_formats`` splits a DSA dual-buffer
    list (NL MLA latent + NL DSA indexer) into two format groups."""

    def test_mixed_mla_dsa_with_explicit_groups(self):
        """A flat list of NL MLA latent tensors followed by NL DSA indexer
        tensors is split into two groups with distinct formats."""
        mla = [_t(NB * BS, 1, HS) for _ in range(NL)]
        dsa = [_dsa_t(NB) for _ in range(NL)]
        kv = mla + dsa
        normalized, formats = normalize_and_discover_per_layer_formats(
            kv,
            [list(range(NL)), list(range(NL, 2 * NL))],
            EngineType.SGLANG,
            {"tokens_per_block": BS},
        )
        assert len(normalized) == 2 * NL
        assert len(formats) == 2 * NL
        # First NL layers: MLA latent
        assert formats[:NL] == [F.NL_X_NBBS_ONE_HS] * NL
        # Last NL layers: DSA indexer
        assert formats[NL:] == [F.NL_X_NB_BSV_BSS] * NL
        # DSA tensors are reshaped to 3-D
        for t in normalized[NL:]:
            assert t.dim() == 3
            assert tuple(t.shape) == (NB, BS, INDEX_DIM)

    def test_mixed_mla_dsa_no_explicit_groups(self):
        """Without explicit ``layer_index_groups``, the function still
        splits by tensor shape -- MLA and DSA have different shapes."""
        mla = [_t(NB * BS, 1, HS) for _ in range(NL)]
        dsa = [_dsa_t(NB) for _ in range(NL)]
        kv = mla + dsa
        normalized, formats = normalize_and_discover_per_layer_formats(
            kv, (), EngineType.SGLANG, {"tokens_per_block": BS}
        )
        assert formats[:NL] == [F.NL_X_NBBS_ONE_HS] * NL
        assert formats[NL:] == [F.NL_X_NB_BSV_BSS] * NL


# ---------------------------------------------------------------------------
# _build_engine_group_infos
# ---------------------------------------------------------------------------


class TestBuildEngineGroupInfos:
    """``LMCacheMPConnector._build_engine_group_infos`` returns two groups
    for DSA (MLA latent + DSA indexer) and empty for MLA/MHA."""

    @staticmethod
    def _make_connector(kvcaches, use_mla, num_layers, page_size):
        """Create an ``LMCacheMPConnector`` without calling ``__init__``."""
        # First Party
        from lmcache.integration.sglang.multi_process_adapter import (
            LMCacheMPConnector,
        )

        conn = LMCacheMPConnector.__new__(LMCacheMPConnector)
        conn.kvcaches = kvcaches
        conn.use_mla = use_mla
        conn.num_layers = num_layers
        conn.page_size = page_size
        return conn

    def test_dsa_returns_two_groups(self):
        """DSA (use_mla=True, num_kv_tensors > num_layers) returns two
        ``EngineGroupInfo`` entries, both with ``engine_group_id=0``."""
        kv = [_t(NB * BS, 1, HS) for _ in range(NL)] + [
            _dsa_t(NB) for _ in range(NL)
        ]
        conn = self._make_connector(
            kv, use_mla=True, num_layers=NL, page_size=BS
        )
        infos = conn._build_engine_group_infos()
        assert len(infos) == 2
        # Both share engine_group_id=0 (same page address space)
        assert all(info.engine_group_id == 0 for info in infos)
        # First group: MLA latent layers [0, NL)
        assert tuple(infos[0].layer_indices) == tuple(range(NL))
        assert infos[0].tokens_per_block == BS
        # Second group: DSA indexer layers [NL, 2*NL)
        assert tuple(infos[1].layer_indices) == tuple(range(NL, 2 * NL))
        assert infos[1].tokens_per_block == BS

    def test_mla_returns_empty(self):
        """MLA (use_mla=True, num_kv_tensors == num_layers) returns empty."""
        kv = [_t(NB * BS, 1, HS) for _ in range(NL)]
        conn = self._make_connector(
            kv, use_mla=True, num_layers=NL, page_size=BS
        )
        infos = conn._build_engine_group_infos()
        assert infos == []

    def test_mha_returns_empty(self):
        """MHA (use_mla=False) returns empty regardless of tensor count."""
        kv = [_t(NB * BS, NH, HS) for _ in range(2 * NL)]
        conn = self._make_connector(
            kv, use_mla=False, num_layers=NL, page_size=BS
        )
        infos = conn._build_engine_group_infos()
        assert infos == []


# ---------------------------------------------------------------------------
# metadata.get_shapes()
# ---------------------------------------------------------------------------


class TestGetShapes:
    """``LMCacheMetadata.get_shapes`` returns per-group shapes when a
    ``KVLayerGroupsManager`` is attached."""

    def test_fallback_single_shape(self):
        """Without a ``KVLayerGroupsManager``, ``get_shapes`` returns a
        single shape derived from the legacy ``kv_shape`` field."""
        # First Party
        from lmcache.v1.metadata import LMCacheMetadata

        meta = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(NL, 2, 256, NH, HS),
        )
        shapes = meta.get_shapes()
        assert len(shapes) == 1
        assert tuple(shapes[0]) == (2, NL, 256, NH * HS)

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="KVLayerGroupsManager requires CUDA build",
    )
    def test_dsa_two_group_shapes(self):
        """With a ``KVLayerGroupsManager`` for DSA, ``get_shapes`` returns
        two shapes: one for MLA latent and one for DSA indexer."""
        # First Party
        from lmcache.v1.kv_layer_groups import KVLayerGroupsManager
        from lmcache.v1.metadata import LMCacheMetadata
        from lmcache.v1.multiprocess.group_view import EngineGroupInfo

        mla = [_t(NB * BS, 1, HS) for _ in range(NL)]
        dsa = [_dsa_t(NB) for _ in range(NL)]
        kv = mla + dsa
        normalized, formats = normalize_and_discover_per_layer_formats(
            kv,
            [list(range(NL)), list(range(NL, 2 * NL))],
            EngineType.SGLANG,
            {"tokens_per_block": BS},
        )
        # engine_group_infos provides tokens_per_block so that
        # group_layers_by_identity can fall back to it for fused formats
        # (NL_X_NBBS_ONE_HS) whose block_size() is undefined.
        engine_group_infos = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(NL)),
                tokens_per_block=BS,
            ),
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(NL, 2 * NL)),
                tokens_per_block=BS,
            ),
        ]
        manager = KVLayerGroupsManager(
            normalized,
            engine_kv_formats=formats,
            engine_group_infos=engine_group_infos,
        )
        meta = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(NL, 1, 256, 1, HS),
            use_mla=True,
            kv_layer_groups_manager=manager,
        )
        shapes = meta.get_shapes(num_tokens=256)
        assert len(shapes) == 2
        # Group 0: MLA latent (kv_size=1, NL layers, 256 tokens, HS hidden)
        assert tuple(shapes[0]) == (1, NL, 256, HS)
        # Group 1: DSA indexer (kv_size=1, NL layers, 256 tokens, 132 hidden)
        assert tuple(shapes[1]) == (1, NL, 256, INDEX_DIM)


# ---------------------------------------------------------------------------
# Layerwise MLA/DSA guard
# ---------------------------------------------------------------------------


class TestLayerwiseMLAGuard:
    """``CreateGPUConnector`` rejects ``use_layerwise=True`` for MLA/DSA."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CreateGPUConnector calls torch_dev.set_device (requires CUDA)",
    )
    def test_layerwise_mla_raises(self):
        """When ``metadata.use_mla`` is True and ``config.use_layerwise``
        is True, ``CreateGPUConnector`` raises ``ValueError``."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.gpu_connector import CreateGPUConnector
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(use_layerwise=True)
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(NL, 1, 256, 1, HS),
            use_mla=True,
        )
        with pytest.raises(ValueError, match="Layerwise mode.*not yet supported.*MLA/DSA"):
            CreateGPUConnector(
                config=config,
                metadata=metadata,
                engine=EngineType.SGLANG,
                layout_hints={"tokens_per_block": BS},
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CreateGPUConnector calls torch_dev.set_device (requires CUDA)",
    )
    def test_layerwise_mha_allowed(self):
        """When ``metadata.use_mla`` is False, ``use_layerwise=True`` is
        allowed (the layerwise connector is constructed for MHA)."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig
        from lmcache.v1.gpu_connector import CreateGPUConnector
        from lmcache.v1.gpu_connector.gpu_connectors import (
            SGLangLayerwiseGPUConnector,
        )
        from lmcache.v1.metadata import LMCacheMetadata

        config = LMCacheEngineConfig.from_defaults(use_layerwise=True)
        metadata = LMCacheMetadata(
            model_name="test",
            world_size=1,
            local_world_size=1,
            worker_id=0,
            local_worker_id=0,
            kv_dtype=torch.float16,
            kv_shape=(NL, 2, 256, NH, HS),
            use_mla=False,
        )
        connector = CreateGPUConnector(
            config=config,
            metadata=metadata,
            engine=EngineType.SGLANG,
            layout_hints={"tokens_per_block": BS},
        )
        assert isinstance(connector, SGLangLayerwiseGPUConnector)
