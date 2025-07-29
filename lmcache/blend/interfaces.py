# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional
import abc

# Third Party
import torch


@dataclass
class BlendOutput:
    """The output of the cacheblend module

    :ivar torch.Tensor q: The short Q tensor with selected tokens
    :ivar torch.Tensor k: The long K tensor with the updated values
    :ivar torch.Tensor v: The long V tensor with the updated values
    :ivar torch.Tensor positions: The positions of the selected Q tokens in
        the input sequence
    :ivar torch.Tensor local_indices: The positions of the selected Q tokens in
        fresh q
    :ivar Optional[torch.Tensor] query_start_loc: The modified query_start_loc
        if token selection has happened. Will be None if no selection has
        happened.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    positions: torch.Tensor
    local_indices: torch.Tensor
    query_start_loc: Optional[torch.Tensor]


@dataclass
class BlendRetrieverResult:
    """The result of the cacheblend retriever

    :ivar torch.Tensor k: The K tensor of a single layer, will be None if
        nothing is retrieved
    :ivar torch.Tensor v: The V tensor of a single layer, will be None if
        nothing is retrieved
    :ivar torch.Tensor valid_mask: The valid mask on CPU
    :ivar torch.Tensor original_positions: The original positions of the
        retrieved KV in the input sequence. If the corresponding KV is not
        valid, the position will be 0. This tensor will be on the same
        device as K and V.
    """

    k: Optional[torch.Tensor]
    v: Optional[torch.Tensor]
    valid_mask: torch.Tensor
    original_positions: torch.Tensor


class BlendRetrieverTask(metaclass=abc.ABCMeta):
    """The KV retrieval task created by the BlendRetriever"""

    @abc.abstractmethod
    def result(self, layer_id: int) -> BlendRetrieverResult:
        """Blocking function to get a single layer of K and V tensor.
        The returned the K and V tensor should match the length of the input
        tokens passed to the `BlendRetriever.new_request` function.
        If the KV of a token is not available, the `vaild_mask` will be 0,
        and the corresponding values in the KV tensor will be undefined.

        :param int layer_id: the layer id
        :return: The BlendRetrieverResult object
        :rtype: BlendRetrieverResult
        """
        pass


class BlendRetriever(metaclass=abc.ABCMeta):
    """The interface for the cacheblend retriever to retrieve the KV caches

    It takes in input tokens and ROI as input, and launch some tasks (maybe
    async), and return a BlendRetrieverTask to retrieve the KV caches.
    """

    @abc.abstractmethod
    def new_request(
        self,
        full_prompts: List[torch.Tensor],
        indices: List[List[int]],
    ) -> BlendRetrieverTask:
        """Create a new BlendRetrieverTask to retrieve the KV caches.
        It may launch async tasks in the background during the retrieval.

        :param List[torch.Tensor] full_prompts: The full prompts for each
        request in this batch.
        :param List[List[int]] indices: The indices of where the
        segmengted requests start in the full prompts.

        :return: The retriever task to retrieve the KV caches
        :rtype: BlendRetrieverTask
        """
        pass


class BlendExecutor(metaclass=abc.ABCMeta):
    """The interface for the cacheblend executor to blend the retrieved KV
    with fresh KVs
    """

    # TODO: consider changing "(retrieved_k, retrieved_v, valid_mask,
    #       original_positions)" to BlendRetrieverResult
    @abc.abstractmethod
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
        :param torch.Tensor retrieved_k: The retrieved K tensor
        :param torch.Tensor retrieved_v: The retrieved V tensor
        :param torch.Tensor valid_mask: A CPU tensor returned from the
            retriever indicating whether the KV is valid.
        :param torch.Tensor original_positions: The original positions of the
            tokens in the retrieved KV
        :param torch.Tensor fresh_q: The fresh Q tensor from QKV split
        :param torch.Tensor fresh_k: The fresh K tensor from QKV split
        :param torch.Tensor fresh_v: The fresh V tensor from QKV split
        :param torch.Tensor positions: The positions in the input of the
            tokens in the fresh_q
        :param torch.Tensor query_start_loc: The start location of the query if
            input_tokens has multiple requests in a batch. The length should be
            the number of requests in the batch + 1
        :param int token_dim: The token dimension

        :return: The blended Q, K, V, and positions
        """
        pass
