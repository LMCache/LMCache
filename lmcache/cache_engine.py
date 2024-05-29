import torch
import hashlib
from typing import Tuple, List, Union, Iterator

# FIXME: currently is v0.1: store the kv cache in CPU memory in a dictionary

# TODO: store to redis
# TODO: configuration class

KVCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]
class LMCacheEngine:
    def __init__(self, chunk_size: int = 256):
        self.chunk_size = chunk_size
        self.dict = {}

    def _chunk_tokens(self, tokens: torch.Tensor, device = 'cpu') -> Iterator[torch.Tensor]:
        """
        Chunk the tokens into chunks of size self.chunk_size.
        
        Input:
            tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        Output:
            a generator of chunks of tokens, each with shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i+self.chunk_size].cpu()

    def _chunk_kv(self, 
                  kv_tensors: KVCache,
                  fmt: str,
                  device = 'cpu') -> Iterator[KVCache]:
        """
        Chunk the kv cache into chunks of size self.chunk_size.

        Input:
            tokens: the input tokens, with shape [seq_len]
            kv_tensors: the kv cache of the tokens, in the format of nested tuples
            fmt: either 'huggingface' or 'vllm'

        Output:
            a generator of tuples, each tuple is a chunk of tokens and the corresponding kv cache.
        """
        match fmt:
            case "huggingface":
                num_heads, num_tokens, head_size = kv_tensors[0][0].shape
                for i in range(0, num_tokens, self.chunk_size):
                    yield tuple((kv[0][:, i:i+self.chunk_size, :].to(device), 
                                 kv[1][:, i:i+self.chunk_size, :].to(device)) 
                                for kv in kv_tensors)

            case "vllm":
                num_tokens, num_heads, head_size = kv_tensors[0][0].shape
                for i in range(0, num_tokens, self.chunk_size):
                    yield tuple((kv[0][i:i+self.chunk_size, :, :].to(device), 
                                 kv[1][i:i+self.chunk_size, :, :].to(device)) 
                                for kv in kv_tensors)

            case _:
                raise ValueError(f"Invalid format: {fmt}")

    def _make_chunks(self, 
                     tokens: torch.Tensor,
                     kv_tensors: KVCache,
                     fmt: str,
                     device = 'cpu') -> Iterator[Tuple[torch.Tensor, KVCache]]:
        return self._chunk_tokens(tokens, device), self._chunk_kv(kv_tensors, fmt, device)


    def _get_init_hash(self) -> str:
        return ""

    def _hash(self, tokens: torch.Tensor, prefix_hash: str) -> str:
        # TODO: change it to a more efficient hash function
        return hashlib.sha256(prefix_hash.encode("ascii") + tokens.numpy().tobytes()).hexdigest()

    def store(self, 
              tokens: torch.Tensor,
              kv_tensors: KVCache,
              fmt: str
              ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            tokens: the input tokens, with shape [seq_len]
            kv_tensors: the kv cache of the tokens, in the format of nested tuples
            format: either 'huggingface' or 'vllm'
                    For huggingface, it should have the shape of [num_heads, num_tokens, head_size]
                    For vllm, it should have the shape of [num_tokens, num_heads, head_size]

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """

        # TODO: check shapes

        ''' chunk the tokens and the kv caches '''
        token_chunks, kv_chunks = self._make_chunks(tokens, kv_tensors, fmt, device='cpu')

        ''' store them into the dictionary '''
        prefix_hash = self._get_init_hash()
        for token_chunk, kv_chunk in zip(token_chunks, kv_chunks):
            token_hash = self._hash(token_chunk, prefix_hash)
            self.dict[(token_hash, fmt)] = kv_chunk
            prefix_hash = token_hash


    def retrive(self,
                tokens: torch.Tensor,
                fmt: str,
                device: str = 'cuda'
                ) -> Tuple[KVCache, int]:
        """
        Retrive the KV cache of the tokens from the cache engine. The retrived KV cache 
        should be a prefix of the input tokens.

        Input:
            tokens: the input tokens, with shape [seq_len]
            format: either 'huggingface' or 'vllm'
                    For huggingface, it should have the shape of [num_heads, num_tokens, head_size]
                    For vllm, it should have the shape of [num_tokens, num_heads, head_size]

        Output: 
            kv_tensors: the kv cache of the tokens, in the format of nested tuples
            num_tokens: the number of tokens in the kv cache
        """
        token_chunks = self._chunk_tokens(tokens, device='cpu')
        prefix_hash = self._get_init_hash()
        retrived_kv_chunks: List[KVCache] = []

        ''' retrive the kv cache '''
        for token_chunk in token_chunks:
            token_hash = self._hash(token_chunk, prefix_hash)
            prefix_hash = token_hash
            if (token_hash, fmt) in self.dict:
                retrived_kv_chunks.append(self.dict[(token_hash, fmt)])
            else:
                break

        ''' concatenate the kv cache '''
        dim = None
        match fmt:
            case "huggingface":
                dim = 1
            case 'vllm':
                dim = 0
            case _:
                raise ValueError(f"Invalid format: {fmt}")

        ret = []
        for kv_layer in zip(*retrived_kv_chunks):
            klist, vlist = zip(*kv_layer)
            klayer = torch.cat(klist, dim=dim).to(device)
            vlayer = torch.cat(vlist, dim=dim).to(device)
            ret.append((klayer, vlayer))

        return tuple(ret), ret[0][0].shape[dim]


