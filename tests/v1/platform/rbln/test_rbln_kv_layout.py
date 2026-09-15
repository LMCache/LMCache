# SPDX-License-Identifier: Apache-2.0
"""Tests for the native RBLN KV layout as a first-class ``EngineKVFormat``.

The multiprocess path never goes through a connector: ``compute_kv_layout``,
``gather_paged_kv_to_cpu`` and ``scatter_cpu_to_paged_kv`` all resolve layouts
through ``normalize_kv_and_discover_format``. What these cover is that the
native 6-D layout is detected as ``NL_X_TWO_NB_NH_ONE_BS_HS`` from its shape
alone -- no device spec consulted, no reshape applied -- and that the squeeze
happens only in the ops backend that indexes the bytes.

``torch_device_type`` is patched rather than requiring an NPU, so these run
anywhere.
"""

# Standard
from typing import cast
from unittest.mock import patch

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detectors import vllm as vllm_detector
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)
from lmcache.v1.gpu_connector.utils import (
    get_block_size,
    get_head_size,
    get_num_heads,
    get_num_layers,
    normalize_kv_and_discover_format,
)
from lmcache.v1.platform.rbln.kv_layout import (
    is_rbln_kv_layout,
    squeeze_singleton_axis,
)
import lmcache.lmcache_native as lmcache_native

NUM_LAYERS = 2
NUM_BLOCKS = 8
NUM_HEADS = 2
BLOCK_SIZE = 4
HEAD_SIZE = 16

_NATIVE_FORMAT = lmcache_native.EngineKVFormat.NL_X_TWO_NB_NH_ONE_BS_HS


def _native_kv() -> list[torch.Tensor]:
    """Per-layer tensors in the native RBLN 6-D layout."""
    torch.manual_seed(3)
    shape = (2, NUM_BLOCKS, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE)
    return [torch.randn(shape) for _ in range(NUM_LAYERS)]


def _discover(
    kv_caches: list[torch.Tensor],
    device_type: str = "rbln",
    layout_hints: "LayoutHints | None" = None,
):
    """Run discovery against a chosen device type."""
    with patch.object(vllm_detector, "torch_device_type", device_type):
        return normalize_kv_and_discover_format(
            cast("DiscoverableKVCache", kv_caches),
            EngineType.VLLM,
            layout_hints=layout_hints,
        )


# ---------------------------------------------------------------------------
# Layout predicates
# ---------------------------------------------------------------------------


def test_recognizes_the_native_layout() -> None:
    """Only 6-D, K/V-first, singleton-at-3 qualifies."""
    assert is_rbln_kv_layout(_native_kv()[0]) is True


@pytest.mark.parametrize(
    "shape",
    [
        (2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE, HEAD_SIZE),
        (2, NUM_BLOCKS, NUM_HEADS, 2, BLOCK_SIZE, HEAD_SIZE),
        (NUM_BLOCKS, 2, NUM_HEADS, 1, BLOCK_SIZE, HEAD_SIZE),
    ],
    ids=["5d", "non-singleton-axis", "blocks-first"],
)
def test_rejects_other_layouts(shape: tuple[int, ...]) -> None:
    """Anything else is not the RBLN layout."""
    assert is_rbln_kv_layout(torch.zeros(shape)) is False


def test_squeeze_is_a_free_view() -> None:
    """Dropping the singleton must not copy the KV cache."""
    native = _native_kv()
    for view, tensor in zip(squeeze_singleton_axis(native), native, strict=True):
        assert view.data_ptr() == tensor.data_ptr()
        assert tuple(view.shape) == (
            2,
            NUM_BLOCKS,
            NUM_HEADS,
            BLOCK_SIZE,
            HEAD_SIZE,
        )


def test_squeeze_rejects_a_foreign_layout() -> None:
    """A mismatched rank fails loudly instead of mis-transferring."""
    with pytest.raises(ValueError, match=r"\[2, NB, NH, 1, BS, HS\]"):
        squeeze_singleton_axis([torch.zeros(2, NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE)])


# ---------------------------------------------------------------------------
# Discovery integration
# ---------------------------------------------------------------------------


def test_native_layout_is_its_own_format() -> None:
    """6-D input resolves to the native format with its rank intact."""
    fmt, normalized = _discover(_native_kv())
    assert int(fmt) == int(_NATIVE_FORMAT)
    assert [t.ndim for t in normalized] == [6] * NUM_LAYERS


def test_detection_does_not_reshape_the_cache() -> None:
    """The returned structure aliases the input, singleton axis and all."""
    native = _native_kv()
    _, normalized = _discover(native)
    for out, src in zip(normalized, native, strict=True):
        assert out.data_ptr() == src.data_ptr()
        assert tuple(out.shape) == tuple(src.shape)


@pytest.mark.parametrize(
    "device_type", ["rbln", "cuda", "cpu"], ids=["rbln", "cuda", "cpu"]
)
def test_detection_is_device_independent(device_type: str) -> None:
    """The shape identifies the format, so no device spec is consulted.

    A format keyed off the registered shape means detection is a pure function
    of ``(kv_caches, hints)`` -- it cannot depend on which accelerator the
    process happens to have resolved.
    """
    fmt, _ = _discover(_native_kv(), device_type=device_type)
    assert int(fmt) == int(_NATIVE_FORMAT)


@pytest.mark.parametrize(
    "hints",
    [None, {"kv_layout": "NHD"}, {"kv_layout": "HND"}],
    ids=["absent", "nhd", "hnd"],
)
def test_the_reported_layout_is_never_consulted(hints: "LayoutHints | None") -> None:
    """vLLM-RBLN stores HND but does not report it, so the hint cannot be trusted.

    ``get_kv_cache_layout()`` is unset on vllm-rbln and defaults to NHD, so
    honouring it would silently pick the wrong axis order. The format is HND by
    definition, which is why the hint plays no part.
    """
    fmt, _ = _discover(_native_kv(), layout_hints=hints)
    assert int(fmt) == int(_NATIVE_FORMAT)


def test_geometry_reads_past_the_singleton() -> None:
    """The spec must report the real dims, not the ones shifted by the axis."""
    fmt, normalized = _discover(_native_kv())
    assert get_num_layers(normalized, fmt) == NUM_LAYERS
    assert get_num_heads(normalized, fmt) == NUM_HEADS
    assert get_block_size(normalized, fmt) == BLOCK_SIZE
    assert get_head_size(normalized, fmt) == HEAD_SIZE
