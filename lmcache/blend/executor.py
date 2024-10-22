from typing import List, Tuple
import torch

from lmcache.blend.interfaces import BlendExecutor, BlendOutput
from lmcache.logging import init_logger

logger = init_logger(__name__)

# TODO: add configuration item 

def mask_to_indices(mask):
    indices = mask.nonzero(as_tuple=True)[0]
    return indices

def indices_to_mask(indices, size):
    mask = torch.zeros(size, dtype=torch.long)
    mask[indices] = 1
    return mask

def create_index(ndims, target_dim, index):
    index_obj = [slice(None)] * ndims
    index_obj[target_dim] = index
    return tuple(index_obj)

class CacheBlendImpl(BlendExecutor):
    def __init__(self, recompute_ratio: float):
        self.recompute_ratio = recompute_ratio
        self.indexes_in_kv = None # Indexes in the retrieved_kv of the tokens from the fresh_q

    def _select_tokens_single_query(
            self,
            rk: torch.Tensor,
            rv: torch.Tensor,
            valid: torch.Tensor,
            fq: torch.Tensor,
            fk: torch.Tensor,
            fv: torch.Tensor,
            token_dim: int
        ) -> torch.Tensor:
        """
        Input: retrieved KV, valid_mask, and fresh QKV for a single query
        Output: selected tokens indices
        """
        # We compare the retrieved KVs with the fresh KVs and keep the 
        # following tokens:
        #  1. Invalid tokens
        #  2. Token with top difference in the fresh KV, if the token is 
        #     valid. Based on previous CacheBlend implementation, we only
        #     use V to compare the difference. The number of tokens to 
        #     keep is determined by the `recompute_ratio`
        assert fk.shape == rk.shape
        assert fv.shape == rv.shape

        # Find the top different tokens
        dims_to_average = [i for i in range(fv.dim()) if i != token_dim]
        diff_per_token = torch.mean(
                    (fv - rv) ** 2, 
                    dims_to_average)
        diff_per_token = diff_per_token * valid.to(diff_per_token.device)

        num_valid_tokens = valid.sum()
        num_selected_tokens = int(num_valid_tokens * self.recompute_ratio)
        top_indices = torch.topk(diff_per_token, 
                                num_selected_tokens).indices
        #logger.debug(f"Indices of the top differences: {top_indices}")

        # Merge the positions with the invalid tokens
        top_mask = indices_to_mask(top_indices, valid.shape[0])
        total_selected_mask = (1 - valid) + top_mask

        local_indices = mask_to_indices(total_selected_mask)
        #logger.debug(f"Local indices of the selected tokens: {local_indices}")
        return local_indices


    def blend(
        self,
        layer_id: int,
        retrieved_k: torch.Tensor,
        retrieved_v: torch.Tensor,
        valid_mask: torch.Tensor,
        fresh_q: torch.Tensor,
        fresh_k: torch.Tensor,
        fresh_v: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        token_dim: int,
    ) -> BlendOutput:
        """This function blends the retrieved KV with fresh KVs, and
        returns the short Q + long KV (blended) + positions of the tokens in Q

        :param int layer_id: The layer id
        :param torch.Tensor retrieved_k: The retrieved K layer, in shape
            [num_tokens, hidden_dims]
        :param torch.Tensor retrieved_v: The retrieved V layer, in shape
            [num_tokens, hidden_dims]
        :param torch.Tensor valid_mask: A CPU tensor returned from the 
            retriever indicating whether the KV is valid. 
        :param torch.Tensor fresh_q: The fresh Q tensor from QKV split,
            in shape [num_tokens, hidden_dims]
        :param torch.Tensor fresh_k: The fresh K tensor from QKV split,
            in shape [num_tokens, hidden_dims]
        :param torch.Tensor fresh_v: The fresh V tensor from QKV split,
            in shape [num_tokens, hidden_dims]
        :param torch.Tensor positions: The positions in the input of the
            tokens in the fresh_q
        :param torch.Tensor query_start_loc: The start location of the query if
            input_tokens has multiple requests in a batch. The length should be
            the number of requests in the batch + 1
        :param int token_dim: The token dimension  

        :return: The blended Q, K, V, and positions
        """
        # We should convert the shape of KV to [num_elems, hidden_dimensions]
        assert valid_mask.is_cpu, "valid_mask should be on CPU"

        if layer_id == 0:
            self.indexes_in_kv = torch.tensor([], 
                                              dtype = torch.long, 
                                              device = "cpu")
            return BlendOutput(fresh_q, fresh_k, fresh_v, positions, 
                               torch.arange(fresh_q.shape[token_dim], 
                                            device="cpu", dtype = torch.long),
                               query_start_loc)

        if layer_id == 1:
            new_query_start_locs = [0]
            for qstart, qend in zip(query_start_loc[:-1], query_start_loc[1:]):
                # Select the tokens for each query
                local_indices = self._select_tokens_single_query(
                    retrieved_k[qstart:qend], 
                    retrieved_v[qstart:qend], 
                    valid_mask[qstart:qend], 
                    fresh_q[qstart:qend], 
                    fresh_k[qstart:qend], 
                    fresh_v[qstart:qend], 
                    token_dim
                )

                new_query_start_locs.append(
                        new_query_start_locs[-1] + len(local_indices))

                self.indexes_in_kv = torch.cat(
                    (self.indexes_in_kv, local_indices + qstart.item())
                )

            new_q = fresh_q[self.indexes_in_kv]
            new_positions = positions[self.indexes_in_kv]
            new_query_start_locs = torch.tensor(new_query_start_locs,
                                                device = query_start_loc.device,
                                                dtype = query_start_loc.dtype)
            return BlendOutput(new_q, fresh_k, fresh_v, new_positions, 
                               self.indexes_in_kv, new_query_start_locs)

            
        if layer_id > 1:
            assert len(self.indexes_in_kv) == fresh_k.shape[token_dim]
            index_obj = create_index(fresh_k.dim(), token_dim, self.indexes_in_kv)
            retrieved_k[index_obj] = fresh_k
            retrieved_v[index_obj] = fresh_v

            return BlendOutput(fresh_q, retrieved_k, retrieved_v, positions, 
                               self.indexes_in_kv, query_start_loc)
