# SPDX-License-Identifier: Apache-2.0
"""Per-format geometry interface for GPU KV caches.

Each concrete :class:`KVFormatSpec` describes exactly one
``lmc_ops.EngineKVFormat``. The class-level attributes carry the static
facts about the format (identity, MLA/HND/cross-layer nature, the
symbolic shape legend); the instance methods read concrete geometry
(layer count, head size, device pointers, ...) off a normalized
``kv_caches`` value.

The format enum is the single source of truth for *which* formats
exist; a spec class adds the Python-side knowledge of how to index a
value of that format. There is intentionally no engine identity here
-- engine awareness lives only in the detection layer.
"""

# Standard
from abc import ABC, abstractmethod
from typing import ClassVar

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.types import DiscoverableKVCache
import lmcache.c_ops as lmc_ops


class KVFormatSpec(ABC):
    """Pure geometry accessors for a single ``EngineKVFormat``.

    A spec is constructed around an already-normalized ``kv_caches``
    value (see :func:`lmcache.v1.gpu_connector.kv_format.detect_format`)
    and answers geometry questions about it. Instances are cheap and
    hold no state beyond the wrapped value, so create one per query.

    A spec describes *only* layout geometry. It carries no engine or
    attention-backend identity: a single ``EngineKVFormat`` may be produced
    by many (engine, attention-backend) combinations, so that mapping
    belongs to a higher-level type that *owns* a ``EngineKVFormat`` (and to
    the engine-aware detection layer), never to this geometry object.

    Class attributes:
        engine_kv_format: The format this spec describes. One spec class
            per enum value; this is the identity.
        is_mla: ``True`` for Multi-head Latent Attention layouts, where
            K and V share a single latent tensor (``num_heads == 1``).
        is_hnd: ``True`` for HND physical layouts (heads before
            block-size), ``False`` for NHD.
        is_cross_layer: ``True`` if all layers are packed into one bare
            tensor rather than a per-layer list.
        shape_desc: Symbolic shape legend using the enum naming
            convention (NB=num_blocks, NL=num_layers, BS=block_size,
            NH=num_heads, HS=head_size, PBS=page_buffer_size). Purely
            geometric -- no engine information.
    """

    engine_kv_format: ClassVar["lmc_ops.EngineKVFormat"]
    is_mla: ClassVar[bool] = False
    is_hnd: ClassVar[bool] = False
    is_cross_layer: ClassVar[bool] = False
    shape_desc: ClassVar[str]

    def __init__(self, kv_caches: DiscoverableKVCache) -> None:
        """Wrap a normalized ``kv_caches`` value for geometry queries.

        Args:
            kv_caches: A :data:`DiscoverableKVCache` already normalized
                to this format's canonical structure (the value
                returned by ``detect_format``). The spec does not
                validate the structure; passing a value of a different
                format yields undefined results.
        """
        self.kv_caches = kv_caches

    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers in ``kv_caches``."""

    @abstractmethod
    def num_blocks(self) -> int:
        """Return the number of paged blocks.

        Raises:
            ValueError: For NBBS-fused formats (SGLang MHA/MLA) whose
                per-layer tensor folds ``num_blocks`` and ``block_size``
                into a single ``page_buffer_size`` axis.
        """

    @abstractmethod
    def block_size(self, layer_idx: int = 0) -> int:
        """Return the block size (tokens per block) for ``layer_idx``.

        ``layer_idx`` matters only for per-layer formats where the
        block size may differ across layers (mixed-compression pools);
        cross-layer formats ignore it.

        Raises:
            ValueError: For NBBS-fused formats with no separate
                ``block_size`` axis.
        """

    @abstractmethod
    def page_buffer_size(self) -> int:
        """Return ``num_blocks * block_size`` (or the fused PBS axis)."""

    @abstractmethod
    def num_heads(self, layer_idx: int = 0) -> int:
        """Return the number of KV heads for ``layer_idx`` (1 for MLA)."""

    @abstractmethod
    def hidden_dim(self, layer_idx: int = 0) -> int:
        """Return the hidden dimension (``num_heads * head_size``) for a layer."""

    @abstractmethod
    def head_size(self, layer_idx: int = 0) -> int:
        """Return the per-head dimension for ``layer_idx``."""

    @abstractmethod
    def tokens_per_layer(self) -> int:
        """Return the token capacity per layer (``num_blocks * block_size``)."""

    @abstractmethod
    def elements_per_layer(self) -> int:
        """Return the element count per layer, including both K and V for non-MLA."""

    @abstractmethod
    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        """Return the tensor dtype for ``layer_idx``."""

    @abstractmethod
    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        """Return device pointers for ``layer_indices`` in kernel-expected order.

        The pointer-array shape is a property of the format, not the
        caller (see ``csrc/mp_mem_kernels.cu``):

        - Per-layer formats: one pointer per requested layer, in order.
        - SGLang two-list MHA: all K pointers then all V pointers.
        - Cross-layer formats: a single base pointer; ``layer_indices``
          is ignored (the kernel walks layers internally).

        Args:
            layer_indices: 0-based layer indices, in the order the
                kernel should iterate them.

        Returns:
            Device pointers as ints, in kernel-expected order.
        """

    @abstractmethod
    def concrete_shape_str(self) -> str:
        """Return :attr:`shape_desc` with concrete numeric dims substituted.

        For example ``NL x [2, NB, BS, NH, HS]`` becomes
        ``80 x [2, 2048, 128, 8, 128]``.
        """
