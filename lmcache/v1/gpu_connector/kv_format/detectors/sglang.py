# SPDX-License-Identifier: Apache-2.0
"""SGLang KV cache detector."""

# Standard
from typing import ClassVar, Optional

# Third Party
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.kv_format.detection_base import (
    EngineDetector,
    descend_to_tensor,
    list_depth_tensor_dim,
)
from lmcache.v1.gpu_connector.kv_format.types import (
    DiscoverableKVCache,
    LayoutHints,
)
import lmcache.c_ops as lmc_ops


class SGLangDetector(EngineDetector):
    """Detector for SGLang serving engine KV cache layouts."""

    abstract: ClassVar[bool] = False
    engine: ClassVar = EngineType.SGLANG

    def normalize(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> DiscoverableKVCache:
        """Reshape SGLang's MP-daemon KV cache to the canonical
        depth-2 / 4-D-inner form.

        SGLang MP hands us a flat ``list[Tensor]`` of length
        ``2 * num_layers`` (first half K layers, second half V layers)
        so the wire payload fits ``KVCache = list[CudaIPCWrapper]``.
        We restore the canonical depth-2 ``[K_layers, V_layers]``
        shape, and reshape each per-layer tensor from
        ``(page_buffer_size, num_heads, head_size)`` to
        ``(num_blocks, block_size, num_heads, head_size)`` using the
        engine-supplied ``tokens_per_block`` (same field TRT-LLM uses
        to drive its reshape). After this, format detection lands on
        the dedicated ``TWO_X_NL_X_NB_BS_NH_HS`` enum (4-D inner) and
        ``num_blocks`` / ``block_size`` become readable as
        ``shape[0]`` / ``shape[1]``.

        Triggers structurally on a depth-1 list of an even number of
        3-D tensors with ``shape[1] > 1`` (which excludes SGLang MLA)
        plus a ``tokens_per_block`` hint. The depth-2 in-process
        layout fails the ``isinstance(kv_caches[0], torch.Tensor)``
        check and passes through untouched.

        Args:
            kv_caches: SGLang KV cache structure.
            layout_hints: Must contain ``tokens_per_block`` for the
                MP path to fire.

        Returns:
            The depth-2 / 4-D-inner canonical form for the MP path,
            otherwise the unmodified input.

        Raises:
            ValueError: ``page_buffer_size`` is not divisible by
                ``tokens_per_block``.
        """
        if (
            isinstance(kv_caches, list)
            and len(kv_caches) > 0
            and len(kv_caches) % 2 == 0
            and isinstance(kv_caches[0], torch.Tensor)
            and kv_caches[0].dim() == 3
            and kv_caches[0].shape[1] > 1
            and "tokens_per_block" in layout_hints
        ):
            block_size = layout_hints["tokens_per_block"]
            half = len(kv_caches) // 2
            reshaped: list[list[torch.Tensor]] = []
            for layers in (kv_caches[:half], kv_caches[half:]):
                inner: list[torch.Tensor] = []
                for t in layers:
                    pbs = t.shape[0]
                    if pbs % block_size != 0:
                        raise ValueError(
                            f"SGLang KV page_buffer_size {pbs} not "
                            f"divisible by tokens_per_block {block_size}"
                        )
                    inner.append(t.view(pbs // block_size, block_size, *t.shape[1:]))
                reshaped.append(inner)
            return reshaped
        return kv_caches

    def detect(
        self,
        kv_caches: DiscoverableKVCache,
        layout_hints: LayoutHints,
    ) -> Optional["lmc_ops.GPUKVFormat"]:
        """Identify the GPU KV format for SGLang.

        Args:
            kv_caches: Normalized KV cache structure.
            layout_hints: Engine-supplied layout hints (unused for SGLang).

        Returns:
            The matching ``GPUKVFormat`` enum value, or ``None`` if the
            structure does not match any known SGLang layout.
        """
        list_depth, tensor_dim = list_depth_tensor_dim(kv_caches)
        if list_depth == 1:
            probe = descend_to_tensor(kv_caches, 1)
            if probe.shape[1] == 1:
                return lmc_ops.GPUKVFormat.NL_X_NBBS_ONE_HS
        elif list_depth == 2:
            if tensor_dim == 4:
                # MP path: reshaped per-layer tensor exposes
                # block_size as ``shape[1]`` and num_blocks as
                # ``shape[0]``.
                return lmc_ops.GPUKVFormat.TWO_X_NL_X_NB_BS_NH_HS
            return lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS
        return None
