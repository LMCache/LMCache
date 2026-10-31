# SPDX-License-Identifier: Apache-2.0
"""Server-free contract tests for the SGLang unified KV adapter."""

# Future
from __future__ import annotations

# Standard
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import Mock, patch
import importlib.util
import sys
import unittest

# Third Party
import torch


class _ComponentType(Enum):
    """Minimal SGLang component enum used by adapter contract tests."""

    FULL = auto()
    SWA = auto()
    MAMBA = auto()
    C128 = auto()


class _AttentionArch(Enum):
    """Minimal SGLang attention architecture enum."""

    MHA = auto()
    MLA = auto()


class _DeepSeekV4TokenToKVPool:
    """Test double used to exercise DeepSeek V4 public dispatch."""

    _unified_kv: bool
    full_size: int
    kv_pools: dict[int, Any]
    index_pools: dict[int, Any]
    swa_kv_pool: Any
    compress_state_pools: tuple[Any, ...]
    indexer_compress_state_pools: tuple[Any, ...]


class _IndexerPool:
    """Expose page-native indexer buffers like SGLang's index pools."""

    def __init__(self, page_size: int, buffers: list[torch.Tensor]) -> None:
        self.page_size = page_size
        self._buffers = buffers

    def contiguous_page_row_buffers(self) -> list[torch.Tensor]:
        """Return contiguous physical page rows."""
        return self._buffers


def _is_deepseek_v4(hf_config: object) -> bool:
    """Return the synthetic DeepSeek V4 marker from a test config."""
    return bool(getattr(hf_config, "is_deepseek_v4", False))


def _fake_sglang_modules() -> dict[str, ModuleType]:
    """Build the small SGLang module surface imported by the adapter."""
    package_names = (
        "sglang",
        "sglang.srt",
        "sglang.srt.configs",
        "sglang.srt.mem_cache",
        "sglang.srt.mem_cache.unified_cache",
    )
    modules: dict[str, ModuleType] = {}
    for name in package_names:
        module = ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        modules[name] = module

    model_config = ModuleType("sglang.srt.configs.model_config")
    model_config.AttentionArch = _AttentionArch  # type: ignore[attr-defined]
    model_config.is_deepseek_v4 = _is_deepseek_v4  # type: ignore[attr-defined]
    modules[model_config.__name__] = model_config

    components = ModuleType("sglang.srt.mem_cache.unified_cache.components")
    components.ComponentType = _ComponentType  # type: ignore[attr-defined]
    modules[components.__name__] = components

    dsv4_pool = ModuleType("sglang.srt.mem_cache.deepseek_v4_memory_pool")
    dsv4_pool.DeepSeekV4TokenToKVPool = (  # type: ignore[attr-defined]
        _DeepSeekV4TokenToKVPool
    )
    modules[dsv4_pool.__name__] = dsv4_pool
    return modules


def _load_adapter_type() -> type[Any]:
    """Load the adapter against the synthetic SGLang module surface."""
    module_path = (
        Path(__file__).parents[2]
        / "lmcache"
        / "integration"
        / "sglang"
        / "unified_kv_adapter.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_lmcache_test_unified_kv_adapter", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SGLang adapter from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SGLangUnifiedKVAdapter


class TestSGLangUnifiedKVAdapter(unittest.TestCase):
    """Verify SGLang layout translation without requiring SGLang installation."""

    adapter_type: ClassVar[type[Any]]
    sglang_modules_patch: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        """Load the adapter while its third-party type surface is available."""
        super().setUpClass()
        cls.sglang_modules_patch = patch.dict(sys.modules, _fake_sglang_modules())
        cls.sglang_modules_patch.start()
        cls.adapter_type = _load_adapter_type()

    @classmethod
    def tearDownClass(cls) -> None:
        """Restore the process module table after the adapter tests."""
        cls.sglang_modules_patch.stop()
        super().tearDownClass()

    def _make_adapter(
        self,
        *,
        kv_pool: object,
        tree_components: tuple[_ComponentType, ...],
        page_size: int = 4,
        sliding_window_size: int | None = None,
        mamba_pool: object | None = None,
        mamba_checkpoint_grid: int = 4,
        allocator: object | None = None,
        req_to_token_pool: object | None = None,
    ) -> Any:
        """Construct an adapter with small public-behavior collaborators."""
        if allocator is None:
            allocator = SimpleNamespace(get_kvcache=lambda: kv_pool)
        if req_to_token_pool is None:
            req_to_token_pool = SimpleNamespace(mamba_pool=mamba_pool)
        return self.adapter_type(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=req_to_token_pool,
            tree_components=tree_components,
            mamba_component=SimpleNamespace(
                mamba_checkpoint_grid=mamba_checkpoint_grid
            ),
            page_size=page_size,
            sliding_window_size=sliding_window_size,
        )

    def test_constructor_rejects_unsupported_component_sequences(self) -> None:
        """Component groups must be nonempty, FULL-first, and exclude C128."""
        cases: tuple[tuple[tuple[_ComponentType, ...], type[Exception], str], ...] = (
            ((), ValueError, "must not be empty"),
            ((_ComponentType.SWA,), ValueError, "must start with FULL"),
            (
                (_ComponentType.FULL, _ComponentType.C128),
                NotImplementedError,
                "C128 is not supported",
            ),
        )
        for components, exception_type, message in cases:
            with self.subTest(components=components):
                with self.assertRaisesRegex(exception_type, message):
                    self._make_adapter(
                        kv_pool=object(),
                        tree_components=components,
                    )

    def test_resolve_registered_groups_maps_mha_full_and_swa_pools(self) -> None:
        """FULL and SWA groups preserve their independent address spaces."""
        full_k = torch.arange(8 * 2).reshape(8, 2)
        full_v = torch.arange(8 * 3).reshape(8, 3)
        swa_k = torch.arange(12 * 2).reshape(12, 2)
        swa_v = torch.arange(12 * 3).reshape(12, 3)
        kv_pool = SimpleNamespace(
            full_kv_pool=SimpleNamespace(
                k_buffer=[full_k],
                v_buffer=[full_v],
                kv_cache_layout="nhd",
            ),
            swa_kv_pool=SimpleNamespace(
                k_buffer=[swa_k],
                v_buffer=[swa_v],
                kv_cache_layout="nhd",
            ),
        )
        adapter = self._make_adapter(
            kv_pool=kv_pool,
            tree_components=(_ComponentType.FULL, _ComponentType.SWA),
            sliding_window_size=6,
        )

        groups = adapter.resolve_registered_groups()

        self.assertEqual([group.name for group in groups], ["full", "swa"])
        self.assertIs(groups[0].kv_tensors[0], full_k)
        self.assertIs(groups[0].kv_tensors[1], full_v)
        self.assertIs(groups[1].kv_tensors[0], swa_k)
        self.assertIs(groups[1].kv_tensors[1], swa_v)
        self.assertEqual(groups[0].tensor_rows_per_block, (4, 4))
        self.assertEqual(groups[1].sliding_window_size, 8)

    def test_resolve_registered_groups_keeps_dsa_in_full_group(self) -> None:
        """A DSA indexer remains a sidecar in FULL's block-ID space."""
        full = torch.arange(8 * 4).reshape(8, 4)
        indexer = torch.arange(2 * 6).reshape(2, 6)
        kv_pool = SimpleNamespace(
            kv_buffer=[full],
            use_dsa=True,
            index_k_with_scale_buffer=[indexer],
        )
        adapter = self._make_adapter(
            kv_pool=kv_pool,
            tree_components=(_ComponentType.FULL,),
        )

        groups = adapter.resolve_registered_groups()

        self.assertEqual(len(groups), 1)
        self.assertIs(groups[0].kv_tensors[0], full)
        self.assertIs(groups[0].kv_tensors[1], indexer)
        self.assertEqual(groups[0].tensor_rows_per_block, (4, 1))

    def test_resolve_registered_groups_accepts_kimi_conv_slice_axis(self) -> None:
        """Kimi conv state uses one complete slot despite TP slice_axis=1."""
        full = torch.empty(8, 1, 4)
        conv = torch.arange(5 * 3 * 7).reshape(5, 3, 7)
        mamba_pool = SimpleNamespace(
            _unified_buffer=None,
            _sub_pool_name=None,
            _iter_transfer_state_entries=lambda: iter([("conv", conv, 1, 0)]),
        )
        adapter = self._make_adapter(
            kv_pool=SimpleNamespace(kv_buffer=[full]),
            tree_components=(_ComponentType.FULL, _ComponentType.MAMBA),
            mamba_pool=mamba_pool,
        )

        groups = adapter.resolve_registered_groups()

        self.assertEqual([group.name for group in groups], ["full", "mamba"])
        self.assertEqual(tuple(groups[1].kv_tensors[0].shape), (5, 1, 21))
        self.assertEqual(groups[1].kv_tensors[0].data_ptr(), conv.data_ptr())
        self.assertTrue(groups[1].recurrent_state)

    def test_resolve_registered_groups_maps_unified_mamba_buffer(self) -> None:
        """Unified Mamba storage is exposed as a zero-copy slot tensor."""
        full = torch.empty(8, 1, 4)
        raw = torch.arange(5 * 12, dtype=torch.uint8)
        unified_buffer = SimpleNamespace(
            _raw=raw,
            anchor_bytes=lambda name: 0,
            mamba_spec=lambda name: SimpleNamespace(entry_bytes=lambda: 12),
        )
        mamba_pool = SimpleNamespace(
            _unified_buffer=unified_buffer,
            _sub_pool_name="linear",
            _max_size=4,
        )
        adapter = self._make_adapter(
            kv_pool=SimpleNamespace(kv_buffer=[full]),
            tree_components=(_ComponentType.FULL, _ComponentType.MAMBA),
            mamba_pool=mamba_pool,
        )

        groups = adapter.resolve_registered_groups()

        mamba_tensor = groups[1].kv_tensors[0]
        self.assertEqual(tuple(mamba_tensor.shape), (5, 1, 12))
        self.assertEqual(mamba_tensor.data_ptr(), raw.data_ptr())

    def test_device_indices_by_group_translates_mamba_checkpoints(self) -> None:
        """Physical block IDs are generated in each component address space."""
        full_translation = Mock(return_value=torch.arange(10, 18))
        swa_translation = Mock(return_value=torch.arange(20, 28))
        allocator = SimpleNamespace(
            get_kvcache=lambda: object(),
            translate_kv_indices_for_transfer=full_translation,
            translate_loc_from_full_to_swa=swa_translation,
        )
        mamba_translation = Mock(return_value=torch.tensor([7]))
        req_to_token_pool = SimpleNamespace(
            mamba_pool=object(),
            translate_mamba_indices=mamba_translation,
        )
        adapter = self._make_adapter(
            kv_pool=object(),
            tree_components=(
                _ComponentType.FULL,
                _ComponentType.SWA,
                _ComponentType.MAMBA,
            ),
            allocator=allocator,
            req_to_token_pool=req_to_token_pool,
        )
        full_indices = torch.arange(8)
        mamba_value = torch.tensor([[42]])

        indices = adapter.device_indices_by_group(
            full_indices,
            mamba_value=mamba_value,
            mamba_transfer_tokens=8,
        )

        self.assertTrue(torch.equal(indices[0], torch.arange(10, 18)))
        self.assertTrue(torch.equal(indices[1], torch.arange(20, 28)))
        self.assertTrue(torch.equal(indices[2], torch.tensor([0, 7])))
        full_translation.assert_called_once_with(full_indices)
        swa_translation.assert_called_once_with(full_indices)
        self.assertTrue(
            torch.equal(mamba_translation.call_args_list[0].args[0], torch.tensor([42]))
        )

    def test_device_indices_by_group_validates_mamba_geometry(self) -> None:
        """Mamba ranges and translated state slots must match checkpoint geometry."""
        allocator = SimpleNamespace(
            get_kvcache=lambda: object(),
            translate_kv_indices_for_transfer=lambda indices: indices,
        )
        req_to_token_pool = SimpleNamespace(
            mamba_pool=object(),
            translate_mamba_indices=lambda value: torch.tensor([3, 4]),
        )
        adapter = self._make_adapter(
            kv_pool=object(),
            tree_components=(_ComponentType.FULL, _ComponentType.MAMBA),
            allocator=allocator,
            req_to_token_pool=req_to_token_pool,
        )

        with self.assertRaisesRegex(ValueError, "align to checkpoint grid"):
            adapter.device_indices_by_group(torch.arange(6), mamba_transfer_tokens=6)
        with self.assertRaisesRegex(ValueError, "exactly one state slot"):
            adapter.device_indices_by_group(
                torch.arange(8),
                mamba_value=torch.tensor([1]),
                mamba_transfer_tokens=8,
            )

    def test_reset_mamba_checkpoint_metadata_clears_present_fields(self) -> None:
        """ReplaySSM cursors are cleared only for the released request rows."""
        write_pos = torch.ones(4, 2, dtype=torch.int64)
        cache_base = torch.ones(4, 2, dtype=torch.int64)
        mamba_pool = SimpleNamespace(
            replayssm_write_pos=write_pos,
            replayssm_cache_base=cache_base,
        )
        adapter = self._make_adapter(
            kv_pool=object(),
            tree_components=(_ComponentType.FULL,),
            mamba_pool=mamba_pool,
        )

        adapter.reset_mamba_checkpoint_metadata(torch.tensor([1, 3]))

        self.assertTrue(
            torch.equal(write_pos[[1, 3]], torch.zeros_like(write_pos[[1, 3]]))
        )
        self.assertTrue(
            torch.equal(cache_base[[1, 3]], torch.zeros_like(cache_base[[1, 3]]))
        )
        self.assertTrue(
            torch.equal(write_pos[[0, 2]], torch.ones_like(write_pos[[0, 2]]))
        )

    def test_resolve_registered_groups_maps_dsv4_full_and_swa(self) -> None:
        """DeepSeek V4.1 sidecars share FULL pages while SWA stays separate."""
        c4 = torch.arange(4 * 2, dtype=torch.uint8).reshape(4, 2)
        c4_index = torch.arange(4 * 3, dtype=torch.uint8).reshape(4, 3)
        c128 = torch.zeros(130, 4, dtype=torch.uint8)
        c1 = torch.arange(3 * 5, dtype=torch.uint8).reshape(3, 5)
        c1_index = torch.arange(13 * 7, dtype=torch.uint8).reshape(13, 7)
        c2 = torch.arange(4 * 6, dtype=torch.uint8).reshape(4, 6)
        c2_index = torch.arange(7 * 8, dtype=torch.uint8).reshape(7, 8)
        swa = torch.arange(3 * 9, dtype=torch.uint8).reshape(3, 9)
        kv_pool = _DeepSeekV4TokenToKVPool()
        kv_pool._unified_kv = False
        kv_pool.full_size = 512
        kv_pool.kv_pools = {
            4: SimpleNamespace(kv_buffer=[c4]),
            128: SimpleNamespace(kv_buffer=[c128]),
            1: SimpleNamespace(kv_buffer=[c1]),
            2: SimpleNamespace(kv_buffer=[c2]),
        }
        kv_pool.index_pools = {
            4: _IndexerPool(64, [c4_index]),
            1: _IndexerPool(64, [c1_index]),
            2: _IndexerPool(64, [c2_index]),
        }
        kv_pool.swa_kv_pool = SimpleNamespace(kv_buffer=[swa])
        kv_pool.compress_state_pools = ()
        kv_pool.indexer_compress_state_pools = ()
        adapter = self._make_adapter(
            kv_pool=kv_pool,
            tree_components=(_ComponentType.FULL, _ComponentType.SWA),
            page_size=256,
            sliding_window_size=300,
        )

        groups = adapter.resolve_registered_groups()

        self.assertEqual([group.name for group in groups], ["full", "swa"])
        self.assertEqual(
            [tuple(tensor.shape) for tensor in groups[0].kv_tensors],
            [(3, 2), (3, 3), (3, 4), (3, 5), (3, 28), (3, 6), (3, 16)],
        )
        self.assertEqual(groups[0].tensor_rows_per_block, (1,) * 7)
        self.assertIs(groups[1].kv_tensors[0], swa)
        self.assertEqual(groups[1].sliding_window_size, 512)

    def test_is_mla_enabled_accepts_mla_and_deepseek_v4(self) -> None:
        """Both MLA architecture and DeepSeek V4 select replicated-KV mode."""
        adapter = self._make_adapter(
            kv_pool=object(),
            tree_components=(_ComponentType.FULL,),
        )

        self.assertTrue(
            adapter.is_mla_enabled(
                SimpleNamespace(
                    attention_arch=_AttentionArch.MLA,
                    hf_config=SimpleNamespace(is_deepseek_v4=False),
                )
            )
        )
        self.assertTrue(
            adapter.is_mla_enabled(
                SimpleNamespace(
                    attention_arch=_AttentionArch.MHA,
                    hf_config=SimpleNamespace(is_deepseek_v4=True),
                )
            )
        )
        self.assertFalse(
            adapter.is_mla_enabled(
                SimpleNamespace(
                    attention_arch=_AttentionArch.MHA,
                    hf_config=SimpleNamespace(is_deepseek_v4=False),
                )
            )
        )
