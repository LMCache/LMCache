# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING
import abc

# Third Party
import torch

if TYPE_CHECKING:
    # First Party
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
    wrapper: "HackBSAWrapper"
    seq_len: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    block_col_sizes: torch.Tensor
    sparse_blk_row_size: int = 32  # TODO(Jiayi): make this tunable
    sparse_blk_col_size: int = 32  # TODO(Jiayi): make this tunable
    is_causal: bool = True
    q_data_dtype: torch.dtype = torch.bfloat16  # TODO(Jiayi): remove hardcode

    def update_from_top_indices(self, top_indices: torch.Tensor):
        # self.is_causal = False
        device = top_indices.device
        top_k_num = len(top_indices)
        num_block_row = top_k_num // self.sparse_blk_row_size
        block_row_sizes = torch.tensor(
            [self.sparse_blk_row_size] * num_block_row, device=device
        )
        block_row_sizes[-1] += top_k_num % self.sparse_blk_row_size

        block_mask_map = torch.zeros(
            num_block_row, len(self.block_col_sizes), dtype=torch.bool, device=device
        )
        cols = torch.arange(block_mask_map.size(1), device=device).expand(
            block_mask_map.size(0), -1
        )

        # NOTE(Jiayi): select every `sparse_blk_row_size`-th index from top_indices
        # to approximate the attention mask at block level.
        top_indices_slice = top_indices[
            self.sparse_blk_row_size - 1 :: self.sparse_blk_row_size
        ]
        top_indices_slice //= self.sparse_blk_col_size
        mask = cols < top_indices_slice.unsqueeze(1)
        block_mask_map[mask] = 1
        self.wrapper.plan(
            block_mask_map.expand(self.num_kv_heads, -1, -1),
            block_row_sizes.expand(self.num_kv_heads, -1),
            self.block_col_sizes.expand(self.num_kv_heads, -1),
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            q_data_type=self.q_data_dtype,
        )
