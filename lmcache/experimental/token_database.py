import abc
import hashlib
from typing import Iterable, Optional, Tuple

import torch

from lmcache.config import LMCacheEngineMetadata
from lmcache.experimental.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey


class TokenDatabase(metaclass=abc.ABCMeta):
    """TokenDatabase is used to convert input tokens into list of
    cache engine keys. There are multiple ways to implement this:

    - ChunkedTokenDatabase: It processes tokens into chunks and convert 
    each chunk into a cache engine key using prefix hash.

    - RadixTokenDatabase: more advanced implementation using radix tree.
    """

    @abc.abstractmethod
    def process_tokens(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Iterable[Tuple[int, int, CacheEngineKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param torch.Tensor tokens: The tokens to process, in 1-D CPU tensor.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.
        """

        raise NotImplementedError


class ChunkedTokenDatabase(TokenDatabase):

    def __init__(self, config: LMCacheEngineConfig,
                 metadata: LMCacheEngineMetadata):
        self.chunk_size = config.chunk_size
        self.metadata = metadata

    def _make_key_by_hash(self, chunk_hash: str):
        return CacheEngineKey(self.metadata.fmt, self.metadata.model_name,
                              self.metadata.world_size,
                              self.metadata.worker_id, chunk_hash)

    def _get_init_hash(self) -> str:
        return ""

    def _hash(
        self,
        tokens: torch.Tensor,
        prefix_hash: str,
    ) -> str:
        # TODO: change it to a more efficient hash function
        return hashlib.sha256(
            prefix_hash.encode("ascii") +
            tokens.cpu().numpy().tobytes()).hexdigest()

    def _chunk_tokens(
        self,
        tokens: torch.Tensor,
    ) -> Iterable[torch.Tensor]:
        """
        Chunk the tokens into chunks of size self.chunk_size.

        :param tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        :return: a generator of chunks of tokens, each with 
                shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[torch.Tensor],
    ) -> Iterable[str]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def process_tokens(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Iterable[Tuple[int, int, CacheEngineKey]]:
        """Process the tokens and return the corresponding cache engine keys.

        :param torch.Tensor tokens: The tokens to process, in 1-D CPU tensor.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should 
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched, 
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :returns: A iterable of tuples with three elements. The first element
            is the start index of the tokens for the key. The second element
            is the end index of the tokens for the key. The third element is
            the cache engine key for the tokens.

        :raises: ValueError if the number of Falses in the mask is not a 
            multiple of the chunk size.
        """
        if mask is not None:
            num_falses = mask.numel() - mask.long().sum()
        else:
            num_falses = 0

        if num_falses % self.chunk_size != 0:
            raise ValueError("The number of Falses in the mask is not a "
                             "multiple of the chunk size.")
        total_len = len(tokens)

        token_chunks = self._chunk_tokens(tokens)
        prefix_hashes = self._prefix_hash(token_chunks)

        start_idx = 0
        for chunk_id, hash_val in enumerate(prefix_hashes):
            start_idx = chunk_id * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, total_len)
            if start_idx < num_falses:
                continue
            else:
                yield start_idx, end_idx, self._make_key_by_hash(hash_val)
