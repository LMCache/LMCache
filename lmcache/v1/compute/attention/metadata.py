# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass

# Third Party
import torch


@dataclass
class LMCAttnMetadata:
    pass


@dataclass
class LMCFlashAttnMetadata(LMCAttnMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_query_len: torch.Tensor
    max_seq_len: torch.Tensor
