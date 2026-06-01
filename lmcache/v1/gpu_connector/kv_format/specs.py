# SPDX-License-Identifier: Apache-2.0
"""Concrete :class:`KVFormatSpec` implementations and the format registry.

There is one spec class per ``EngineKVFormat`` and one flat registry dict
mapping the enum to its spec class. Adding a new format is two edits:
write a spec class here (declaring its ``engine_kv_format``) and add it to
``_ALL_SPECS``. There is no inheritance taxonomy, no auto-discovery, and
no plugin machinery -- the set of formats is closed and lives in the C++
enum.

Spec classes are named by their **layout geometry**, never by a serving
engine: a format describes how a tensor is laid out, and detection (not
the spec) is the engine-aware layer. A single ``EngineKVFormat`` may be
produced by many (engine, attention-backend) combinations, so a spec
carries no engine/backend identity at all; any engine named in a
docstring is a non-authoritative example. A human-readable backend label
for diagnostics lives in the ``utils`` facade (``get_attention_backend``),
not here.

Every spec's indexing logic mirrors exactly one branch of the historical
``utils.py`` accessor functions; the golden test in
``tests/v1/gpu_connector/test_kv_format_specs.py`` pins them.
"""

# ``kv_caches`` is a ``DiscoverableKVCache`` (Tensor | nested list); each spec
# indexes it according to its format, so the per-format ``.shape`` / ``[...]``
# access is well-defined even though mypy cannot prove it. This mirrors the
# suppression the historical ``utils.py`` accessors carried.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.base import KVFormatSpec
from lmcache.v1.gpu_connector.types import DiscoverableKVCache
import lmcache.c_ops as lmc_ops

# Raised when a caller asks for an attribute that the detected format
# does not expose (e.g. ``num_blocks`` on an NBBS-fused SGLang layout).
_ATTRIBUTE_NOT_EXIST_ERROR = (
    "trying to access an attribute of the GPU KV Cache "
    "that does not exist for the format detected {format}. "
    "A misalignment with the EngineKVFormat must be resolved"
)


# ---------------------------------------------------------------------------
# Cross-layer: a single bare tensor packs all layers along dim-1.
# ---------------------------------------------------------------------------
class CrossLayerNhdSpec(KVFormatSpec):
    """Cross-layer, NHD: ``[NB, NL, 2, BS, NH, HS]`` (e.g. vLLM CROSS_LAYER)."""

    engine_kv_format = lmc_ops.EngineKVFormat.NB_NL_TWO_BS_NH_HS
    is_cross_layer = True
    shape_desc = "[NB, NL, 2, BS, NH, HS]"

    def num_layers(self) -> int:
        return self.kv_caches.shape[1]

    def num_blocks(self) -> int:
        return self.kv_caches.shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3]

    def page_buffer_size(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[3]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4] * self.kv_caches.shape[5]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[5]

    def tokens_per_layer(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[3]

    def elements_per_layer(self) -> int:
        t = self.kv_caches
        return t.shape[0] * 2 * t.shape[3] * t.shape[4] * t.shape[5]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches.dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        tensor = cast(torch.Tensor, self.kv_caches)
        return [tensor.data_ptr()]

    def concrete_shape_str(self) -> str:
        return (
            f"[{self.num_blocks()}, {self.num_layers()}, 2, "
            f"{self.block_size()}, {self.num_heads()}, {self.head_size()}]"
        )


class CrossLayerHndSpec(KVFormatSpec):
    """Cross-layer, HND: ``[NB, NL, 2, NH, BS, HS]`` (e.g. TRT-LLM)."""

    engine_kv_format = lmc_ops.EngineKVFormat.NB_NL_TWO_NH_BS_HS
    is_cross_layer = True
    is_hnd = True
    shape_desc = "[NB, NL, 2, NH, BS, HS]"

    def num_layers(self) -> int:
        return self.kv_caches.shape[1]

    def num_blocks(self) -> int:
        return self.kv_caches.shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[4]

    def page_buffer_size(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[4]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[3] * self.kv_caches.shape[5]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches.shape[5]

    def tokens_per_layer(self) -> int:
        return self.kv_caches.shape[0] * self.kv_caches.shape[4]

    def elements_per_layer(self) -> int:
        t = self.kv_caches
        return t.shape[0] * 2 * t.shape[3] * t.shape[4] * t.shape[5]

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches.dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        tensor = cast(torch.Tensor, self.kv_caches)
        return [tensor.data_ptr()]

    def concrete_shape_str(self) -> str:
        return (
            f"[{self.num_blocks()}, {self.num_layers()}, 2, "
            f"{self.num_heads()}, {self.block_size()}, {self.head_size()}]"
        )


# ---------------------------------------------------------------------------
# Per-layer non-MLA: a list[NL] of a 5-D tensor. "KvFirst" = the K/V (size-2)
# axis leads the per-layer tensor ([2, NB, ...]); "BlockFirst" = num_blocks
# leads and the K/V axis follows ([NB, 2, ...]).
# ---------------------------------------------------------------------------
class PerLayerKvFirstNhdSpec(KVFormatSpec):
    """Per-layer, NHD, K/V-axis first: ``NL x [2, NB, BS, NH, HS]``.

    Produced e.g. by vLLM non-MLA flash attention.
    """

    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    shape_desc = "NL x [2, NB, BS, NH, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[1]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[1] * self.kv_caches[0].shape[2]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[3]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[3] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        k = self.kv_caches[0][0].shape
        return k[0] * k[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].shape.numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        return (
            f"{self.num_layers()} x [2, {self.num_blocks()}, "
            f"{self.block_size()}, {self.num_heads()}, {self.head_size()}]"
        )


class PerLayerBlockFirstNhdSpec(KVFormatSpec):
    """Per-layer, NHD, num_blocks first: ``NL x [NB, 2, BS, NH, HS]``.

    Produced e.g. by vLLM non-MLA flash infer.
    """

    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS
    shape_desc = "NL x [NB, 2, BS, NH, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[2]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[3]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[3] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        k = self.kv_caches[0][:, 0].shape
        return k[0] * k[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][:, 0].shape.numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        return (
            f"{self.num_layers()} x [{self.num_blocks()}, 2, "
            f"{self.block_size()}, {self.num_heads()}, {self.head_size()}]"
        )


class PerLayerKvFirstHndSpec(KVFormatSpec):
    """Per-layer, HND, K/V-axis first: ``NL x [2, NB, NH, BS, HS]``.

    Produced e.g. by vLLM non-MLA flash attention (HND layout).
    """

    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_TWO_NB_NH_BS_HS
    is_hnd = True
    shape_desc = "NL x [2, NB, NH, BS, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[1]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[3]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[1] * self.kv_caches[0].shape[3]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[2] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        k = self.kv_caches[0][0].shape
        return k[0] * k[2]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].shape.numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        return (
            f"{self.num_layers()} x [2, {self.num_blocks()}, "
            f"{self.num_heads()}, {self.block_size()}, {self.head_size()}]"
        )


class PerLayerBlockFirstHndSpec(KVFormatSpec):
    """Per-layer, HND, num_blocks first: ``NL x [NB, 2, NH, BS, HS]``.

    Produced e.g. by vLLM non-MLA flash infer (HND layout).
    """

    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_TWO_NH_BS_HS
    is_hnd = True
    shape_desc = "NL x [NB, 2, NH, BS, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[3]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[3]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[2] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        k = self.kv_caches[0][:, 0].shape
        return k[0] * k[2]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][:, 0].shape.numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        return (
            f"{self.num_layers()} x [{self.num_blocks()}, 2, "
            f"{self.num_heads()}, {self.block_size()}, {self.head_size()}]"
        )


# ---------------------------------------------------------------------------
# Per-layer MLA: a list[NL] of a 3-D tensor; K and V share a latent
# (num_heads == 1).
# ---------------------------------------------------------------------------
class PerLayerMlaSpec(KVFormatSpec):
    """Per-layer MLA: ``NL x [NB, BS, HS]`` (e.g. vLLM MLA)."""

    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_BS_HS
    is_mla = True
    shape_desc = "NL x [NB, BS, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[1]

    def num_heads(self, layer_idx: int = 0) -> int:
        return 1

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0].numel()

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        return (
            f"{self.num_layers()} x [{self.num_blocks()}, "
            f"{self.block_size()}, {self.head_size()}]"
        )


# ---------------------------------------------------------------------------
# Two-list MHA: ``[K_layers, V_layers]``, each a list[NL] of a per-layer
# tensor. "FusedPbs" folds num_blocks*block_size into one axis (3-D inner);
# "SplitNbBs" keeps them separate (4-D inner).
# ---------------------------------------------------------------------------
class TwoListFusedPbsSpec(KVFormatSpec):
    """Two-list MHA, fused PBS: ``2 x NL x [PBS, NH, HS]`` (e.g. SGLang MHA)."""

    engine_kv_format = lmc_ops.EngineKVFormat.TWO_X_NL_X_NBBS_NH_HS
    shape_desc = "2 x NL x [PBS, NH, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches[0])

    def num_blocks(self) -> int:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=self.engine_kv_format))

    def block_size(self, layer_idx: int = 0) -> int:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=self.engine_kv_format))

    def page_buffer_size(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        inner = self.kv_caches[0][layer_idx]
        return inner.shape[1] * inner.shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[-1]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[0][layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        k, v = cast(list[list[torch.Tensor]], self.kv_caches)
        return [k[i].data_ptr() for i in layer_indices] + [
            v[i].data_ptr() for i in layer_indices
        ]

    def concrete_shape_str(self) -> str:
        return (
            f"2 x {self.num_layers()} x [{self.page_buffer_size()}, "
            f"{self.num_heads()}, {self.head_size()}]"
        )


class TwoListSplitNbBsSpec(KVFormatSpec):
    """Two-list MHA, split NB/BS: ``2 x NL x [NB, BS, NH, HS]`` (SGLang MP daemon)."""

    engine_kv_format = lmc_ops.EngineKVFormat.TWO_X_NL_X_NB_BS_NH_HS
    shape_desc = "2 x NL x [NB, BS, NH, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches[0])

    def num_blocks(self) -> int:
        return self.kv_caches[0][0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][0].shape[1]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[2]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        inner = self.kv_caches[0][layer_idx]
        return inner.shape[2] * inner.shape[3]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[0][layer_idx].shape[-1]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0][0].shape[0] * self.kv_caches[0][0].shape[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][0].numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[0][layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        k, v = cast(list[list[torch.Tensor]], self.kv_caches)
        return [k[i].data_ptr() for i in layer_indices] + [
            v[i].data_ptr() for i in layer_indices
        ]

    def concrete_shape_str(self) -> str:
        return (
            f"2 x {self.num_layers()} x [{self.num_blocks()}, "
            f"{self.block_size()}, {self.num_heads()}, {self.head_size()}]"
        )


# ---------------------------------------------------------------------------
# Per-layer MLA, fused PBS: list[NL] of a 3-D tensor with a singleton
# head axis; num_blocks*block_size folded into the first axis.
# ---------------------------------------------------------------------------
class PerLayerMlaFusedPbsSpec(KVFormatSpec):
    """Per-layer MLA, fused PBS: ``NL x [PBS, 1, HS]`` (e.g. SGLang MLA)."""

    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NBBS_ONE_HS
    is_mla = True
    shape_desc = "NL x [PBS, 1, HS]"

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=self.engine_kv_format))

    def block_size(self, layer_idx: int = 0) -> int:
        raise ValueError(_ATTRIBUTE_NOT_EXIST_ERROR.format(format=self.engine_kv_format))

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0]

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[1]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def tokens_per_layer(self) -> int:
        return self.kv_caches[0].shape[0]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0].numel()

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]

    def concrete_shape_str(self) -> str:
        return (
            f"{self.num_layers()} x [{self.page_buffer_size()}, 1, {self.head_size()}]"
        )


# ---------------------------------------------------------------------------
# Registry: the single, flat mapping from enum -> spec class, derived from
# each spec's own ``engine_kv_format`` so identity lives in exactly one place.
# ---------------------------------------------------------------------------
_ALL_SPECS: list[type[KVFormatSpec]] = [
    CrossLayerNhdSpec,
    CrossLayerHndSpec,
    PerLayerKvFirstNhdSpec,
    PerLayerBlockFirstNhdSpec,
    PerLayerKvFirstHndSpec,
    PerLayerBlockFirstHndSpec,
    PerLayerMlaSpec,
    TwoListFusedPbsSpec,
    TwoListSplitNbBsSpec,
    PerLayerMlaFusedPbsSpec,
]

_SPECS: dict["lmc_ops.EngineKVFormat", type[KVFormatSpec]] = {
    spec_cls.engine_kv_format: spec_cls for spec_cls in _ALL_SPECS
}


def get_spec_class(
    engine_kv_format: "lmc_ops.EngineKVFormat",
) -> type[KVFormatSpec]:
    """Return the spec class for *engine_kv_format*.

    Use this for static, value-independent geometry facts (``is_mla``,
    ``is_hnd``, ``is_cross_layer``, ``shape_desc``). For geometry that
    depends on actual tensors, use :func:`get_spec`.

    Args:
        engine_kv_format: The format to look up.

    Returns:
        The :class:`KVFormatSpec` subclass describing the format.

    Raises:
        ValueError: If *engine_kv_format* has no registered spec.
    """
    spec_cls = _SPECS.get(engine_kv_format)
    if spec_cls is None:
        raise ValueError(f"Unknown GPU KV Format: {engine_kv_format}")
    return spec_cls


def get_spec(
    kv_caches: DiscoverableKVCache,
    engine_kv_format: "lmc_ops.EngineKVFormat",
) -> KVFormatSpec:
    """Return a spec instance wrapping *kv_caches* for geometry queries.

    Args:
        kv_caches: A normalized :data:`DiscoverableKVCache` value of
            *engine_kv_format* (as returned by ``detect_format``).
        engine_kv_format: The format of *kv_caches*.

    Returns:
        A :class:`KVFormatSpec` instance bound to *kv_caches*.

    Raises:
        ValueError: If *engine_kv_format* has no registered spec.
    """
    return get_spec_class(engine_kv_format)(kv_caches)
