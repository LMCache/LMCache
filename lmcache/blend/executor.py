# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Callable, Optional, Tuple

# Third Party
import torch

# First Party
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


PositionalEncoder = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]


class CacheBlendImpl(BlendExecutor):
    def __init__(
        self,
        recompute_ratio: float,
        all_reduce_function=None,
    ):
        self.recompute_ratio = recompute_ratio

        # Indexes in the retrieved_kv of the tokens from the fresh_q
        self.indexes_in_kv = torch.tensor([], dtype=torch.long, device="cpu")

        self.positional_encoder: Optional[PositionalEncoder] = None
        self.reverse_positional_encoder: Optional[PositionalEncoder] = None
        self.all_reduce_function = all_reduce_function

    def set_positional_encoder(self, positional_encoder: PositionalEncoder):
        self.positional_encoder = positional_encoder

    def set_reverse_positional_encoder(
        self, reverse_positional_encoder: PositionalEncoder
    ):
        self.reverse_positional_encoder = reverse_positional_encoder

    def _select_tokens_single_query(
        self,
        rk: torch.Tensor,
        rv: torch.Tensor,
        valid: torch.Tensor,
        fq: torch.Tensor,
        fk: torch.Tensor,
        fv: torch.Tensor,
        token_dim: int,
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
        diff_per_token = torch.mean((fv - rv) ** 2, dims_to_average)
        diff_per_token = diff_per_token * valid.to(diff_per_token.device)

        num_valid_tokens = valid.sum()
        num_selected_tokens = int(num_valid_tokens * self.recompute_ratio)
        top_indices = torch.topk(diff_per_token, num_selected_tokens).indices
        # logger.debug(f"Indices of the top differences: {top_indices}")

        # Merge the positions with the invalid tokens
        top_mask = indices_to_mask(top_indices, valid.shape[0])
        total_selected_mask = (1 - valid) + top_mask

        local_indices = mask_to_indices(total_selected_mask)
        # logger.debug(f"Local indices of the selected tokens: {local_indices}")
        return local_indices

    def _build_positions(self, query_start_loc: torch.Tensor, device) -> torch.Tensor:
        """Rebuild the positions based on the query start locs"""
        # ret = torch.arange(int(query_start_loc[-1]), device=device)
        ret = torch.arange(query_start_loc[-1], device=device)  # type: ignore
        for start, end in zip(query_start_loc[:-1], query_start_loc[1:], strict=False):
            ret[start:end] -= start
        return ret.long()

    def _select_tokens_all_queries(
        self,
        rk: torch.Tensor,
        rv: torch.Tensor,
        valid: torch.Tensor,
        fq: torch.Tensor,
        fk: torch.Tensor,
        fv: torch.Tensor,
        token_dim: int,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input: retrieved KV, valid_mask, and fresh QKV for a single query,
        and query_start_loc
        Output: new_query_start_locs
        """
        # Consider TP here.
        # But we cannot couple it with serving engine,
        # so pass a all_reduce_function.

        # We compare the retrieved KVs with the fresh KVs and keep the
        # following tokens:
        #  1. Invalid tokens
        #  2. Token with top difference in the fresh KV, if the token is
        #     valid. Based on previous CacheBlend implementation, we only
        #     use V to compare the difference. The number of tokens to
        #     keep is determined by the `recompute_ratio`
        assert fk.shape == rk.shape
        assert fv.shape == rv.shape
        new_query_start_locs = [0]

        # Find the top different tokens
        dims_to_average = [i for i in range(fv.dim()) if i != token_dim]
        diff_per_token = torch.mean((fv - rv) ** 2, dims_to_average)
        # NOTE(Sixian): Here I assume valid mask is the same across TPs.
        # As TP runs in lock-step, we should guarantee this in evictor.
        diff_per_token = diff_per_token * valid.to(diff_per_token.device)
        if self.all_reduce_function is not None:
            diff_per_token = self.all_reduce_function(diff_per_token)
        for qstart, qend in zip(
            query_start_loc[:-1], query_start_loc[1:], strict=False
        ):
            local_valid = valid[qstart:qend]
            num_valid_tokens = local_valid.sum()
            num_selected_tokens = int(num_valid_tokens * self.recompute_ratio)
            top_indices = torch.topk(
                diff_per_token[qstart:qend], num_selected_tokens
            ).indices
            top_mask = indices_to_mask(top_indices, local_valid.shape[0])
            total_selected_mask = (1 - local_valid) + top_mask
            local_indices = mask_to_indices(total_selected_mask)
            new_query_start_locs.append(new_query_start_locs[-1] + len(local_indices))
            self.indexes_in_kv = torch.cat(
                (self.indexes_in_kv, local_indices + int(qstart))
            )
        return torch.tensor(
            new_query_start_locs,
            device=query_start_loc.device,
            dtype=query_start_loc.dtype,
        )

    def blend(
        self,
        layer_id: int,
        retrieved_k: torch.Tensor,
        retrieved_v: torch.Tensor,
        valid_mask: torch.Tensor,
        original_positions: torch.Tensor,
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
        :param torch.Tensor original_positions: The original positions of the
            tokens in the retrieved KV
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
            the number of requests in the batch + 1. Note this will NOT be
            changed after token selection.
        :param int token_dim: The token dimension

        :return: The blended Q, K, V, and positions
        """
        # We should convert the shape of KV to [num_elems, hidden_dimensions]
        assert valid_mask.is_cpu, "valid_mask should be on CPU"

        if layer_id == 0:
            return BlendOutput(
                fresh_q,
                fresh_k,
                fresh_v,
                positions,
                torch.arange(fresh_q.shape[token_dim], device="cpu", dtype=torch.long),
                query_start_loc=None,
            )

        elif layer_id == 1:
            query_start_locs_tensor = self._select_tokens_all_queries(
                retrieved_k,
                retrieved_v,
                valid_mask,
                fresh_q,
                fresh_k,
                fresh_v,
                token_dim,
                query_start_loc,
            )
            new_q = fresh_q[self.indexes_in_kv]
            new_positions = positions[self.indexes_in_kv]
            logger.info(
                f"Selected {len(self.indexes_in_kv)} tokens out of "
                f"{len(retrieved_k)} tokens to blend"
            )
            return BlendOutput(
                new_q,
                fresh_k,
                fresh_v,
                new_positions,
                self.indexes_in_kv,
                query_start_locs_tensor,
            )

        else:
            assert len(self.indexes_in_kv) == fresh_k.shape[token_dim]
            index_obj = create_index(fresh_k.dim(), token_dim, self.indexes_in_kv)

            if (
                self.positional_encoder is not None
                and self.reverse_positional_encoder is not None
            ):
                # Clear the positional encoding
                dumb_q = torch.zeros(
                    retrieved_k.shape,
                    device=fresh_q.device,
                    dtype=fresh_q.dtype,
                )
                dumb_q, rk_no_position = self.reverse_positional_encoder(
                    original_positions.to(device=retrieved_k.device, dtype=torch.long),
                    dumb_q,
                    retrieved_k,
                )

                # Re-apply positional encodings based on query_start_loc
                new_positions = self._build_positions(
                    query_start_loc, device=fresh_q.device
                )
                dumb_q, rk_with_position = self.positional_encoder(
                    new_positions, dumb_q, rk_no_position
                )
            else:
                logger.warning(
                    "Positional encoder and reverse positional "
                    "encoder is not set. This may lead to "
                    "incorrect results."
                )
                rk_with_position = retrieved_k

            rk_with_position[index_obj] = fresh_k
            retrieved_v[index_obj] = fresh_v

            return BlendOutput(
                fresh_q,
                rk_with_position,
                retrieved_v,
                positions,
                self.indexes_in_kv,
                None,
            )
