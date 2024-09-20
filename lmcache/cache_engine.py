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
        ) -> Iterator[torch.Tensor]:
        """
        Chunk the tokens into chunks of size self.chunk_size.
        
        Input:
            tokens: the input tokens, with shape [seq_len]
            device: the target device after chunking

        Output:
            a generator of chunks of tokens, each with shape [chunk_size]
        """
        # TODO(Jiayi): the following step can be parallelized
        for i in range(0, len(tokens), self.chunk_size):
            yield tokens[i:i+self.chunk_size]#.to(device)

    def _prefix_hash(
            self, 
            token_chunks: Iterator[torch.Tensor]
        ) -> List[str]:
        prefix_hash = self._get_init_hash()
        prefix_hashes = []
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            prefix_hashes.append(prefix_hash)
        return prefix_hashes

    def _tuple_kv_to_blob(
            self,
            kv_tensors: KVCache,
        ) -> torch.Tensor:
        """
        Convert the nested tuple of kv tensors to a single big tensor with 2 extra dimensions
        """
        k_temp = []
        v_temp = []
        for kv_layer in kv_tensors:
            k_temp.append(kv_layer[0])
            v_temp.append(kv_layer[1])
        k_tensor_blob = torch.stack(k_temp)
        v_tensor_blob = torch.stack(v_temp)
        
        # kv_tensors: [num_layer, 2, num_tok, num_kv_head, head_size]
        kv_tensors = torch.stack((k_tensor_blob, v_tensor_blob))
        kv_tensors = kv_tensors.permute([1, 0, 2, 3, 4])
        
        return kv_tensors

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
        kv_tensors: torch.Tensor,
        fmt: str,
    ) -> List[torch.Tensor]: 
        """
        vllm format: [num_layer, 2, num_tokens, num_kv_head, head_size]
        huggingface format: [num_layer, 2, num_kv_head, num_tokens, head_size]
        """
        match fmt:
            case "vllm":
                return [x.contiguous() for x in list(torch.split(kv_tensors[:, :, start_idx:, ...], self.chunk_size, dim=2))]
            case "huggingface":
                return [x.contiguous() for x in list(torch.split(kv_tensors[:, :, :, start_idx:, ...], self.chunk_size, dim=3))]
            case _:
                raise ValueError(f"Invalid format: {fmt}")

        
    def _chunk_kv(
            self, 
            kv_tensors: KVCache,
            fmt: str,
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
        return self._slice_kv_at(0, kv_tensors, fmt)

    def _make_chunks_skip_exsiting(
            self, 
            tokens: torch.Tensor,
            kv_tensors: torch.Tensor,
            fmt: str,
        ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Skip the existing chunks and return the rest of the chunks
        """
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens))#, device))
        num_tokens = self._num_tokens_in_kv(kv_tensors, fmt)

        start_token_idx = None
        start_chunk_idx = 0
        for chunk_hash, idx in zip(chunk_hashes, range(0, num_tokens, self.chunk_size)):
            if not self.engine_.contains(self._make_key(chunk_hash, fmt)):
                start_token_idx = idx
                break
            start_chunk_idx += 1

        if start_token_idx is None:
            return zip([], [])
        chunk_kvs = self._slice_kv_at(start_token_idx, kv_tensors, fmt)#, device)
        chunk_hashes = chunk_hashes[start_chunk_idx:]
        return zip(chunk_hashes, chunk_kvs)

    def _make_chunks(
            self, 
            tokens: torch.Tensor,
            kv_tensors: torch.Tensor,
            fmt: str,
            skip_existing = True,
        ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns a generator of zipped (chunk_hash, chunk_kv) tuples
        """
        if skip_existing:
            return self._make_chunks_skip_exsiting(tokens, kv_tensors, fmt)
        else:
            return zip(self._prefix_hash(self._chunk_tokens(tokens)), self._chunk_kv(kv_tensors, fmt))
    
    @torch.no_grad()
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


        kv_tensors = self._tuple_kv_to_blob(kv_tensors)


        ''' chunk the tokens and the kv caches '''
        chunk_hashes_and_kvs = self._make_chunks(tokens, kv_tensors, fmt, skip_existing=skip_existing)
        if not blocking:
            chunk_hashes_and_kvs = list(chunk_hashes_and_kvs)
        end_make_chunks = time.perf_counter()

        ''' store them into the dictionary '''
        n_chunks = self.engine_.batched_put(
                ((
                    self._make_key(chunk_hash, fmt), 
                    kv_chunk
                ) for chunk_hash, kv_chunk in chunk_hashes_and_kvs), 
                blocking=blocking
            )

        end_time = time.perf_counter()
        logger.info(f"Stored/updated {n_chunks} chunks, total time {end_time - start_time:.2f}s, make chunks time {end_make_chunks - start_time:.2f}s")

    @_lmcache_nvtx_annotate
    @torch.no_grad()
    def retrive(self,
                tokens: torch.Tensor,
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
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens))

        retrival_iterator = self.engine_.batched_get(
                (self._make_key(chunk_hash, fmt) for chunk_hash in chunk_hashes),
            )
        
        retrived_kv_chunks = []
        for chunk in retrival_iterator:
            if chunk is None:
                break
            retrived_kv_chunks.append(chunk)#.to(device))

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
        ret = self._blob_to_tuple_kv(torch.cat(retrived_kv_chunks, dim=dim + 2))
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

    @classmethod
    def destroy(
            cls,
            instance_id: str
        ) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)

