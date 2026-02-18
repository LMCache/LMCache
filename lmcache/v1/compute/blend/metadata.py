# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional, Sequence, Any

# Third Party
import torch


@dataclass
class LMCBlendCommonMetadata:
    """
    CommonMetadata (fixed hyperparams) for blending operations in LMCache.
    """

    check_layers: List[int]
    recomp_ratios: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None
    is_costream: bool = True
    GOP: int = 8

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

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
        self.tokens_per_frame = None
        self.mm_positions = None
        self.selection_effective_len = None
        self.is_full_selection = False
