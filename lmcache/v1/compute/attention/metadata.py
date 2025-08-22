# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import abstractmethod
from dataclasses import dataclass

# Third Party
import torch

from lmcache.v1.compute.attention.flash_infer_sparse import HackBSAWrapper

@dataclass
class LMCAttnMetadata(metaclass=abc.ABCMeta):
    @abstractmethod
    def update_from_top_indices(self, top_indices: torch.Tensor):
        raise NotImplementedError("This method should be implemented in subclasses.")


@dataclass
class LMCFlashAttnMetadata(LMCAttnMetadata):
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_query_len: torch.Tensor
    max_seq_len: torch.Tensor

    def update_from_top_indices(self, top_indices: torch.Tensor):
        top_k_num = len(top_indices)
        self.max_query_len = top_k_num
        device = self.query_start_loc.device
        dtype = self.query_start_loc.dtype
        self.query_start_loc = torch.tensor([0, top_k_num], dtype=dtype, device=device)


@dataclass
class LMCFlashInferSparseMetadata(LMCAttnMetadata):
    wrapper: Optional[HackBSAWrapper]
    seq_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    block_col_sizes: torch.Tensor
    sparse_blk_row_size: int = 32 # TODO(Jiayi): make this tunable
    sparse_blk_col_size: int = 32 # TODO(Jiayi): make this tunable
    is_causal: bool = True

    def update_from_top_indices(self, top_indices: torch.Tensor):
        device = top_indices.device
        top_k_num = len(top_indices)
        num_block_row = top_k_num // self.sparse_blk_row_size
        block_row_sizes = torch.tensor([sparse_blk_row_size] * num_block_row, device=device)
        
        block_mask_map = torch.zeros(top_k_num, seq_len, dtype=torch.bool, device=device)
        cols = torch.arange(block_mask_map.size(1)).expand(block_mask_map.size(0), -1)
        mask = cols < top_indices.unsqueeze(1)
        block_mask_map[mask] = 1

        self.wrapper.plan(
            block_mask_map,
            block_row_sizes,
            self.block_col_sizes,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
        )
