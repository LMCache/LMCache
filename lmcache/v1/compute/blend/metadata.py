# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional, Sequence, Any

# Third Party
import torch

BLEND_MODES = ("direct_reuse", "topk", "codecsight", "vlcache", "random")

@dataclass
class LMCBlendCommonMetadata:
    """
    CommonMetadata (fixed hyperparams) for blending operations in LMCache.
    """

    check_layers: List[int]
    recomp_ratios: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None
    blend_mode: str = "codecsight"
    GOP: int = 8
    vlcache_recompute_ratio: float = 0.05

@dataclass
class LMCBlendMetadata:
    """
    Metadata (determined during runtime) for blending operations in LMCache.
    """

    imp_indices: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    tokens_per_frame: Optional[int] = None
    mm_positions: Optional[Sequence[Any]] = None
    selection_effective_len: Optional[int] = None
    is_full_selection: bool = False
    image_grid_thw: Optional[List[List[int]]] = None
    input_ids: Optional[List[int]] = None

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
        self.tokens_per_frame = None
        self.mm_positions = None
        self.selection_effective_len = None
        self.is_full_selection = False
        self.image_grid_thw = None
        self.input_ids = None
