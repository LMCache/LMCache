# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional

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

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
        self.tokens_per_frame = None
