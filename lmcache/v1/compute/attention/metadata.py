# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Third Party
import torch


@dataclass
class LMCAttnMetadata(metaclass=abc.ABCMeta):
    
    @abstractmethod
    def update_from_topk(self, top_k: int):
        raise NotImplementedError(
            "This method should be implemented in subclasses.")


@dataclass
class LMCFlashAttnMetadata(LMCAttnMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_query_len: torch.Tensor
    max_seq_len: torch.Tensor

    def update_from_topk(self, top_k_num: int):
        self.max_query_len = top_k_num
        device = self.query_start_loc.device
        dtype = self.query_start_loc.dtype
        self.query_start_loc = torch.tensor(
            [0, top_k_num], dtype=dtype, device=device
        )

@dataclass
class LMCFlashInferSparseMetadata(LMCAttnMetadata):

    def update_from_topk(self, top_k: int):
        pass
