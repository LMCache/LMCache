import torch
import yaml
import time
import os
import hashlib
from typing import Tuple, List, Union, Iterator, Optional
from dataclasses import dataclass
import logging

from lmcache.storage_backend import CreateStorageBackend
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.utils import KVCache, CacheEngineKey
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate

# TODO: (functionality) configuration class for backend implementations
# TODO: (functionality) the model name and the distributed rank should also be the key
# TODO: (functionality) the chunk size should also be related to the key

logger = init_logger(__name__)

class LMCacheEngine:
    def __init__(
            self, 
            config: LMCacheEngineConfig,
            metadata: LMCacheEngineMetadata,
        ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current configuration
        """

        self.config = config
        self.metadata = metadata
        self.chunk_size = config.chunk_size 

        match self.config.local_device:
            case "cpu":
                self.device = "cpu"
            case _:
                self.device = "cuda"
        logger.info("Using device: %s", self.device)
        self.engine_ = CreateStorageBackend(config, metadata)
        logger.debug(f"Current storage backend type {type(self.engine_)}")

    def _make_key(
            self,
            chunk_hash: str,
            fmt: str
        ) -> CacheEngineKey:
        return CacheEngineKey(fmt, self.metadata.model_name, self.metadata.world_size, self.metadata.worker_id, chunk_hash)

    def _num_tokens_in_kv(
            self,
            kv_tensors: KVCache,
            fmt: str
        ) -> int:
        if fmt == "huggingface":
            return kv_tensors[0][0].shape[1]
        elif fmt == "vllm":
            return kv_tensors[0][0].shape[0]
        else:
            raise ValueError(f"Invalid format: {fmt}")

    def _get_init_hash(self) -> str:
        return ""

    def _hash(
            self, 
            tokens: torch.Tensor, 
            prefix_hash: str,
        ) -> str:
        # TODO: change it to a more efficient hash function
        return hashlib.sha256(prefix_hash.encode("ascii") + tokens.cpu().numpy().tobytes()).hexdigest()

    def _chunk_tokens(
            self, 
            tokens: torch.Tensor, 
            device
        ) -> Iterator[torch.Tensor]:
        """
        Chunk the tokens into chunks of size self.chunk_size.
        
        Input:
            tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        Output:
            a generator of chunks of tokens, each with shape [chunk_size]
        """
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i+self.chunk_size].to(device)

    def _prefix_hash(
            self, 
            token_chunks: Iterator[torch.Tensor]
        ) -> Iterator[str]:
        prefix_hash = self._get_init_hash()
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            yield prefix_hash

    def _tuple_kv_to_blob(
            self,
            kv_tensor: KVCache,
        ) -> torch.Tensor:
        """
        Convert the nested tuple of kv tensors to a single big tensor with 2 extra dimensions
        """
        return torch.stack([torch.stack(inner_tuple, dim=0) for inner_tuple in kv_tensor], dim=0)

    def _blob_to_tuple_kv(
            self,
            blob: torch.Tensor,
        ) -> KVCache:
        """
        Convert a single big tensor to the nested tuple of kv tensors
        """
        outer_unbound = torch.unbind(blob, dim=0)
        return tuple((inner_tensor[0], inner_tensor[1]) for inner_tensor in outer_unbound)


    def _slice_kv_at(
            self,
            start_idx: int,
            end_idx: int,
            kv_tensors: KVCache,
            fmt: str,
            device
        ) -> KVCache:
        """
        Slice the kv cache of tokens between [start_idx:end_idx]
        """
        match fmt:
            case "huggingface":
                return tuple((kv[0][:, start_idx:end_idx, :].to(device), 
                              kv[1][:, start_idx:end_idx, :].to(device)) 
                             for kv in kv_tensors)
            case "vllm":
                return tuple((kv[0][start_idx:end_idx, :, :].to(device),
                              kv[1][start_idx:end_idx, :, :].to(device))
                             for kv in kv_tensors)
            case _:
                raise ValueError(f"Invalid format: {fmt}")

    def _chunk_kv(
            self, 
            kv_tensors: KVCache,
            fmt: str,
            device
        ) -> Iterator[KVCache]:
        """
        Chunk the kv cache into chunks of size self.chunk_size.

        Input:
            tokens: the input tokens, with shape [seq_len]
            kv_tensors: the kv cache of the tokens, in the format of nested tuples
            fmt: either 'huggingface' or 'vllm'

        Output:
            a generator of tuples, each tuple is a chunk of tokens and the corresponding kv cache.
        """
        num_tokens = self._num_tokens_in_kv(kv_tensors, fmt)

        for i in range(0, num_tokens, self.chunk_size):
            yield self._slice_kv_at(i, i+self.chunk_size, kv_tensors, fmt, device)

    def _make_chunks_skip_exsiting(
            self, 
            tokens: torch.Tensor,
            kv_tensors: KVCache,
            fmt: str,
            device
        ) -> Iterator[Tuple[torch.Tensor, KVCache]]:
        """
        Skip the existing chunks and return the rest of the chunks
        """
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens, device))
        num_tokens = self._num_tokens_in_kv(kv_tensors, fmt)

        for chunk_hash, idx in zip(chunk_hashes, range(0, num_tokens, self.chunk_size)):
            if not self.engine_.contains(self._make_key(chunk_hash, fmt)):
                yield chunk_hash, self._slice_kv_at(idx, idx+self.chunk_size, kv_tensors, fmt, device)

            #if (chunk_hash, fmt) not in self.dict:
            #    yield chunk_hash, self._slice_kv_at(idx, idx+self.chunk_size, kv_tensors, fmt, device)


    def _make_chunks(
            self, 
            tokens: torch.Tensor,
            kv_tensors: KVCache,
            fmt: str,
            device,
            skip_existing = True,
        ) -> Iterator[Tuple[torch.Tensor, KVCache]]:
        """
        Returns a generator of zipped (chunk_hash, chunk_kv) tuples
        """
        if skip_existing:
            return self._make_chunks_skip_exsiting(tokens, kv_tensors, fmt, device)
        else:
            return zip(self._prefix_hash(self._chunk_tokens(tokens, device)), self._chunk_kv(kv_tensors, fmt, device))

    def store(
            self, 
            tokens: torch.Tensor,
            kv_tensors: KVCache,
            skip_existing = True,
            blocking = True,
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
        start_time = time.perf_counter()
        fmt = self.metadata.fmt

        assert len(tokens.shape) == 1, f"Invalid shape of tokens: {tokens.shape}"
        assert len(kv_tensors) > 0, "Empty kv_tensors"
        assert len(tokens) == self._num_tokens_in_kv(kv_tensors, fmt), "Number of tokens in the kv cache does not match the input tokens"

        # TODO: check shapes

        ''' chunk the tokens and the kv caches '''
        chunk_hashes_and_kvs = self._make_chunks(tokens, kv_tensors, fmt, device=self.device, skip_existing=skip_existing)

        ''' Issue all the exists() query first if we are doing non-blocking '''
        if not blocking:
            chunk_hashes_and_kvs = list(chunk_hashes_and_kvs)
        end_make_chunks = time.perf_counter()

        ''' store them into the dictionary '''
        n_chunks = self.engine_.batched_put(
                ((
                    self._make_key(chunk_hash, fmt), 
                    self._tuple_kv_to_blob(kv_chunk)
                ) for chunk_hash, kv_chunk in chunk_hashes_and_kvs), 
                blocking=blocking
            )

        end_time = time.perf_counter()
        logger.info(f"Stored/updated {n_chunks} chunks, total time {end_time - start_time:.2f}s, make chunks time {end_make_chunks - start_time:.2f}s")

    @_lmcache_nvtx_annotate
    def retrive(self,
                tokens: torch.Tensor,
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
            kv_tensors: the kv cache of the tokens, in the format of nested tuples.
                        Will be an empty tuple if no kv cache is retrived.
            num_tokens: the number of tokens in the kv cache
        """
        st = time.perf_counter()
        fmt = self.metadata.fmt
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens, device=self.device))
        

        retrival_iterator = self.engine_.batched_get(
                (self._make_key(chunk_hash, fmt) for chunk_hash in chunk_hashes),
            )
        
        retrived_kv_chunks = []
        for chunk in retrival_iterator:
            if chunk is None:
                break
            retrived_kv_chunks.append(chunk)
        #retrived_kv_chunks: List[KVCache] = []

        #''' retrive the kv cache '''
        #for chunk_hash in chunk_hashes:
        #    if self.engine_.contains(self._make_key(chunk_hash, fmt)):
        #        blob_kv = self.engine_.get(self._make_key(chunk_hash, fmt))
        #        retrived_kv_chunks.append(blob_kv)
        #    else:
        #        break
        #    #if (chunk_hash, fmt) in self.dict:
        #    #    retrived_kv_chunks.append(self.dict[(chunk_hash, fmt)])
        #    #else:
        #    #    break

        ''' concatenate the kv cache '''
        dim = None
        match fmt:
            case "huggingface":
                dim = 1
            case 'vllm':
                dim = 0
            case _:
                raise ValueError(f"Invalid format: {fmt}")

        if len(retrived_kv_chunks) == 0:
            logging.info("Retrived 0 chunks")
            return (), 0

        st2 = time.perf_counter()
        ret = self._blob_to_tuple_kv(torch.cat(retrived_kv_chunks, dim=dim + 2).to(device))
        ed2 = time.perf_counter()
        logger.info(f"Concatenated {len(retrived_kv_chunks)} chunks -- elapsed time {ed2 - st2}")
        retrived_token_count = 0 if len(ret) == 0 else ret[0][0].shape[dim]
        ed = time.perf_counter()
        logger.info(f"Retrived {len(retrived_kv_chunks)} chunks ({retrived_token_count} tokens in total) -- elapsed time {ed - st}")
        return ret, retrived_token_count

    def persist(self):
        """
        Temporary function of persisting
        """
        self.engine_.persist()

    def close(self):
        self.engine_.close()

class LMCacheEngineBuilder:
    _instances = {}
    _cfgs = {}
    _metadatas = {}

    @classmethod
    def get_or_create(
            cls, 
            instance_id: str,
            config: LMCacheEngineConfig, 
            metadata: LMCacheEngineMetadata
        ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the given ID.

        Raises:
            ValueError if the instance already exists with a different configuration.
        """
        if instance_id not in cls._instances:
            engine = LMCacheEngine(config, metadata)
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            return engine
        else:
            if cls._cfgs[instance_id] != config or cls._metadatas[instance_id] != metadata:
                raise ValueError(f"Instance {instance_id} already exists with a different configuration or metadata.")
            return cls._instances[instance_id]

    @classmethod
    def get(cls, 
            instance_id: str
        ) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID, or None if not found."""
        return cls._instances.get(instance_id)
