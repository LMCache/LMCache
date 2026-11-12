# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LMCache-side TRT-LLM integration components.

Covers ``EngineType.TRTLLM``, the new ``NB_NL_TWO_NH_BS_HS`` GPU KV
format, format-aware accessors in ``common/utils.py``, and the
``RawCudaIPCWrapper`` subclass relationship — without requiring TensorRT-LLM
to be installed.
"""

# Third Party
import pytest
import torch

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.utils import EngineType
import lmcache.lmcache_native as lmcache_native

# TRT-LLM connector path is CUDA-only.
pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(
        not (torch_dev.is_available() and torch_device_type == "cuda"),
        reason="Requires CUDA backend",
    ),
]


def _has_cuda_ops() -> bool:
    try:
        # First Party
        import lmcache.cuda_ops  # noqa: F401

        return True
    except ImportError:
        return False


class TestEngineType:
    def test_trtllm_exists(self) -> None:
        assert hasattr(EngineType, "TRTLLM")
        assert EngineType.TRTLLM.value == "trtllm"

    def test_all_engine_types(self) -> None:
        expected = {"vllm", "atom", "sglang", "trtllm", "mock"}
        actual = {e.value for e in EngineType}
        assert expected == actual


class TestAssertContiguous:
    def test_contiguous_tensor_passes(self) -> None:
        # First Party
        from lmcache.v1.common.utils import assert_contiguous

        t = torch.zeros(2, 3, 4)
        assert_contiguous(t)  # no raise

    def test_non_contiguous_raises(self) -> None:
        # First Party
        from lmcache.v1.common.utils import assert_contiguous

        t = torch.zeros(2, 3, 4).transpose(0, 1)
        with pytest.raises(ValueError, match="not contiguous"):
            assert_contiguous(t)

    def test_nonzero_storage_offset_raises(self) -> None:
        # First Party
        from lmcache.v1.common.utils import assert_contiguous

        base = torch.zeros(8)
        view = base[2:6]  # storage_offset = 2
        with pytest.raises(ValueError, match="storage_offset"):
            assert_contiguous(view)


@pytest.mark.skipif(not _has_cuda_ops(), reason="lmcache C ops not built")
class TestGPUKVFormatEnum:
    def test_nb_nl_two_nh_bs_hs_exists(self) -> None:
        # First Party

        assert hasattr(lmcache_native.EngineKVFormat, "NB_NL_TWO_NH_BS_HS")


@pytest.mark.skipif(not _has_cuda_ops(), reason="lmcache C ops not built")
class TestNormalizeTRTLLM:
    """``normalize_kv_and_discover_format`` for ``EngineType.TRTLLM``."""

    def test_4d_tensor_reshape_to_6d(self) -> None:
        # First Party
        from lmcache.v1.common.utils import (
            LayoutHints,
            normalize_kv_and_discover_format,
        )

        nb, nl, kv = 4, 2, 2
        nh, bs, hs = 8, 16, 64
        flat = nh * bs * hs

        t = torch.zeros(nb, nl, kv, flat, dtype=torch.bfloat16)
        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": nh,
            "tokens_per_block": bs,
            "head_dim": hs,
        }

        fmt, normalized = normalize_kv_and_discover_format(
            t, EngineType.TRTLLM, layout_hints=layout_hints
        )

        assert fmt == lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS
        assert isinstance(normalized, torch.Tensor)
        assert tuple(normalized.shape) == (nb, nl, kv, nh, bs, hs)

    def test_collapses_one_element_list(self) -> None:
        # First Party
        from lmcache.v1.common.utils import (
            LayoutHints,
            normalize_kv_and_discover_format,
        )

        nb, nl, kv, nh, bs, hs = 2, 2, 2, 4, 8, 32
        t = torch.zeros(nb, nl, kv, nh * bs * hs)
        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": nh,
            "tokens_per_block": bs,
            "head_dim": hs,
        }
        fmt, normalized = normalize_kv_and_discover_format(
            [t], EngineType.TRTLLM, layout_hints=layout_hints
        )
        assert fmt == lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS
        assert isinstance(normalized, torch.Tensor)
        assert normalized.shape == (nb, nl, kv, nh, bs, hs)

    def test_missing_layout_hints_raises(self) -> None:
        # First Party
        from lmcache.v1.common.utils import normalize_kv_and_discover_format

        t = torch.zeros(2, 2, 2, 16)
        with pytest.raises(ValueError, match="num_kv_heads"):
            normalize_kv_and_discover_format(t, EngineType.TRTLLM, layout_hints={})

    def test_flat_dim_mismatch_raises(self) -> None:
        # First Party
        from lmcache.v1.common.utils import (
            LayoutHints,
            normalize_kv_and_discover_format,
        )

        t = torch.zeros(2, 2, 2, 17)  # not divisible by 4*2*2 = 16
        layout_hints: LayoutHints = {
            "kv_layout": "HND",
            "num_kv_heads": 4,
            "tokens_per_block": 2,
            "head_dim": 2,
        }
        with pytest.raises(ValueError, match="flat dim"):
            normalize_kv_and_discover_format(
                t, EngineType.TRTLLM, layout_hints=layout_hints
            )


@pytest.mark.skipif(not _has_cuda_ops(), reason="lmcache C ops not built")
class TestAccessorsTRTLLM:
    """Format accessors for ``NB_NL_TWO_NH_BS_HS``."""

    def _tensor(
        self,
        nb: int = 4,
        nl: int = 3,
        kv: int = 2,
        nh: int = 8,
        bs: int = 16,
        hs: int = 64,
    ) -> torch.Tensor:
        return torch.zeros(nb, nl, kv, nh, bs, hs, dtype=torch.bfloat16)

    def test_get_num_layers(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_num_layers

        t = self._tensor()
        assert get_num_layers(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 3

    def test_get_num_blocks(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_num_blocks

        t = self._tensor()
        assert get_num_blocks(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 4

    def test_get_block_size(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_block_size

        t = self._tensor()
        assert get_block_size(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 16

    def test_get_num_heads(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_num_heads

        t = self._tensor()
        assert get_num_heads(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 8

    def test_get_head_size(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_head_size

        t = self._tensor()
        assert get_head_size(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS) == 64

    def test_get_hidden_dim_size(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_hidden_dim_size

        t = self._tensor()
        assert (
            get_hidden_dim_size(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS)
            == 8 * 64
        )

    def test_get_page_buffer_size(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_page_buffer_size

        t = self._tensor()
        assert (
            get_page_buffer_size(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS)
            == 4 * 16
        )

    def test_get_dtype(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_dtype

        t = self._tensor()
        assert (
            get_dtype(t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS)
            == torch.bfloat16
        )

    def test_get_group_data_ptrs_returns_single_base_pointer(self) -> None:
        # First Party
        from lmcache.v1.common.utils import get_group_data_ptrs

        t = self._tensor()
        ptrs = get_group_data_ptrs(
            t, lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS, list(range(3))
        )
        assert len(ptrs) == 1
        assert ptrs[0] == t.data_ptr()

    def test_shape_description_strings(self) -> None:
        # First Party
        from lmcache.v1.common.utils import (
            get_attention_backend,
            get_concrete_engine_kv_shape,
            get_engine_kv_shape_description,
        )

        fmt = lmcache_native.EngineKVFormat.NB_NL_TWO_NH_BS_HS
        assert get_engine_kv_shape_description(fmt) == "[NB, NL, 2, NH, BS, HS]"
        assert "TRT-LLM" in get_attention_backend(fmt)
        assert get_concrete_engine_kv_shape(self._tensor(), fmt) == (
            "[4, 3, 2, 8, 16, 64]"
        )


class TestRawCudaIPCWrapperType:
    """Subclass relationship — without invoking CUDA IPC syscalls."""

    def test_is_subclass(self) -> None:
        # First Party
        from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
        from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper

        assert issubclass(RawCudaIPCWrapper, DeviceIPCWrapper)

    def test_kvcache_typing_unchanged(self) -> None:
        """``KVCache = list[DeviceIPCWrapper]`` should accept concrete wrappers
        — load-bearing for msgspec ext-code reuse over ZMQ.
        """
        # First Party
        from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper, KVCache
        from lmcache.v1.platform.devices.cuda.ipc_wrapper import RawCudaIPCWrapper

        assert KVCache == list[DeviceIPCWrapper]
        # Static check substitute: a concrete wrapper fits the list type.
        assert issubclass(RawCudaIPCWrapper, DeviceIPCWrapper)

    def test_serde_uses_shared_ext_code(self) -> None:
        """Ext code 1 dispatches via ``DeviceIPCWrapper`` for all wrappers."""
        # First Party
        from lmcache.v1.multiprocess import custom_types

        registry = custom_types._CUSTOMERIZED_SERIALIZERS  # noqa: PLC2701
        assert custom_types.DeviceIPCWrapper in registry
        assert registry[custom_types.DeviceIPCWrapper].code == 1
