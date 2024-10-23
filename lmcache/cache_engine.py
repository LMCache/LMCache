import hashlib
import logging
import time
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.storage_backend import CreateStorageBackend
from lmcache.utils import CacheEngineKey, KVCache, _lmcache_nvtx_annotate

logger = init_logger(__name__)


class LMCacheEngine:

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ):
        """
        Throws:
            RuntimeError if the loaded configuration does not match the current
            configuration
        """

        self.config = config
        self.metadata = metadata
        self.chunk_size = config.chunk_size
        self.save_decode_cache = config.save_decode_cache

        self.engine_ = CreateStorageBackend(config, metadata)
        logger.debug(f"Current storage backend type {type(self.engine_)}")

    def _make_key(self, chunk_hash: str, fmt: str) -> CacheEngineKey:
        return CacheEngineKey(
            fmt,
            self.metadata.model_name,
            self.metadata.world_size,
            self.metadata.worker_id,
            chunk_hash,
        )

    def _num_tokens_in_kv(self, kv_tensors: Union[KVCache, torch.Tensor],
                          fmt: str) -> int:
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
        return hashlib.sha256(
            prefix_hash.encode("ascii") +
            tokens.cpu().numpy().tobytes()).hexdigest()

    def _chunk_tokens(
        self,
        tokens: torch.Tensor,
    ) -> Iterable[torch.Tensor]:
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
            yield tokens[i:i + self.chunk_size]

    def _prefix_hash(
        self,
        token_chunks: Iterable[torch.Tensor],
        num_skip_chunk: Optional[int] = 0,
    ) -> List[str]:
        prefix_hash = self._get_init_hash()
        prefix_hashes = []
        for token_chunk in token_chunks:
            prefix_hash = self._hash(token_chunk, prefix_hash)
            prefix_hashes.append(prefix_hash)
        return prefix_hashes[num_skip_chunk:]

    def _tuple_kv_to_blob(
        self,
        kv_tensors: KVCache,
    ) -> torch.Tensor:
        """
        Convert the nested tuple of kv tensors to a single big tensor with 2
        extra dimensions
        """
        k_temp = []
        v_temp = []
        for kv_layer in kv_tensors:
            k_temp.append(kv_layer[0])
            v_temp.append(kv_layer[1])
        k_tensor_blob = torch.stack(k_temp)
        v_tensor_blob = torch.stack(v_temp)

        # kv_tensors: [num_layer, 2, num_tok, num_kv_head, head_size]
        kv_tensors_flatten = torch.stack((k_tensor_blob, v_tensor_blob))
        kv_tensors_flatten = kv_tensors_flatten.permute([1, 0, 2, 3, 4])

        return kv_tensors_flatten

    def _blob_to_tuple_kv(
        self,
        blob: torch.Tensor,
    ) -> KVCache:
        """
        Convert a single big tensor to the nested tuple of kv tensors
        """
        outer_unbound = torch.unbind(blob, dim=0)
        return tuple((inner_tensor[0], inner_tensor[1])
                     for inner_tensor in outer_unbound)

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
                return [
                    x.contiguous() for x in list(
                        torch.split(
                            kv_tensors[:, :, start_idx:, ...],
                            self.chunk_size,
                            dim=2,
                        ))
                ]
            case "huggingface":
                return [
                    x.contiguous() for x in list(
                        torch.split(
                            kv_tensors[:, :, :, start_idx:, ...],
                            self.chunk_size,
                            dim=3,
                        ))
                ]
            case _:
                raise ValueError(f"Invalid format: {fmt}")

    def _chunk_kv(
        self,
        kv_tensors: torch.Tensor,
        fmt: str,
    ) -> Iterable[torch.Tensor]:
        """
        Chunk the kv cache into chunks of size self.chunk_size.

        Input:
            tokens: the input tokens, with shape [seq_len]
            kv_tensors: the kv cache of the tokens, in the format of nested 
            tuples
            fmt: either 'huggingface' or 'vllm'

        Output:
            a generator of tuples, each tuple is a chunk of tokens and the 
            corresponding kv cache.
        """
        return self._slice_kv_at(0, kv_tensors, fmt)

    def _make_chunks_skip_existing(
        self,
        tokens: torch.Tensor,
        kv_tensors: torch.Tensor,
        fmt: str,
        num_skip_chunk=0,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Skip the existing chunks and return the rest of the chunks
        """
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens),
                                         num_skip_chunk)
        # With num_skip_chunks, the following is relative to
        # the new start after skip.
        num_tokens: int = self._num_tokens_in_kv(kv_tensors, fmt)

        start_token_idx = None
        start_chunk_idx = 0
        for chunk_hash, idx in zip(chunk_hashes,
                                   range(0, num_tokens, self.chunk_size)):
            if not self.engine_.contains(self._make_key(chunk_hash, fmt)):
                start_token_idx = idx
                break
            start_chunk_idx += 1

        if start_token_idx is None:
            return zip([], [])
        chunk_kvs = self._slice_kv_at(start_token_idx, kv_tensors, fmt)
        chunk_hashes = chunk_hashes[start_chunk_idx:]
        return zip(chunk_hashes, chunk_kvs)

    def _make_chunks(
        self,
        tokens: torch.Tensor,
        kv_tensors: torch.Tensor,
        fmt: str,
        num_skip_chunk=0,
        skip_existing=True,
    ) -> Iterable[Tuple[str, torch.Tensor]]:
        """
        Returns a generator of zipped (chunk_hash, chunk_kv) tuples
        """
        if skip_existing:
            return self._make_chunks_skip_existing(tokens, kv_tensors, fmt,
                                                   num_skip_chunk)
        else:
            assert num_skip_chunk == 0
            return zip(
                self._prefix_hash(self._chunk_tokens(tokens)),
                self._chunk_kv(kv_tensors, fmt),
            )

    @_lmcache_nvtx_annotate
    @torch.no_grad()
    def store(
        self,
        tokens: torch.Tensor,
        kv_tensors_raw: KVCache,
        kv_tensors_mask: torch.Tensor,
        skip_existing=True,
        blocking=True,
    ) -> None:
        """
        Store the KV cache of the tokens into the cache engine.

        Input:
            tokens: the input tokens, with shape [seq_len]
            kv_tensors_raw: the kv cache of the tokens, in the format of nested 
            tuples
            kv_tensors_mask: a boolean mask of tokens indicating which tokens'
            KV Cache should be stored. Currently, only support suffix mask.
            
            format: either 'huggingface' or 'vllm'
                For huggingface, it should have the shape of 
                [num_heads, num_tokens, head_size]

                For vllm, it should have the shape of 
                [num_tokens, num_heads, head_size]

        Returns:
            None

        Note:
            The KV cache should NOT have the "batch" dimension.
        """
        start_time = time.perf_counter()
        fmt = self.metadata.fmt

        assert (len(
            tokens.shape) == 1), f"Invalid shape of tokens: {tokens.shape}"
        assert (len(
            kv_tensors_mask.shape) == 1), \
        f"Invalid shape of mask: {kv_tensors_mask.shape}"
        assert len(tokens) == len(kv_tensors_mask), \
            "token length does not match mask length"
        num_skip_chunk = 0
        num_skip_tok = 0
        if kv_tensors_mask is not None:
            # NOTE(Sixian): Now kv_tensors_mask always a suffix mask.
            num_skip_tok = (len(kv_tensors_mask) - torch.sum(kv_tensors_mask))
            num_skip_chunk = num_skip_tok // self.chunk_size
        assert num_skip_tok == num_skip_chunk * self.chunk_size, \
            "Only support chunk skip in store."
        assert num_skip_tok < len(tokens), "No tokens to store"
        assert len(kv_tensors_raw) > 0, "Empty kv_tensors"
        assert len(tokens) == self._num_tokens_in_kv(
            kv_tensors_raw, fmt
        ) + num_skip_tok, \
            "Number of tokens in the kv cache does not match the input tokens"
        kv_tensors = self._tuple_kv_to_blob(kv_tensors_raw)
        """ chunk the tokens and the kv caches """
        chunk_hashes_and_kvs = self._make_chunks(tokens,
                                                 kv_tensors,
                                                 fmt,
                                                 num_skip_chunk,
                                                 skip_existing=skip_existing)
        if not blocking:
            chunk_hashes_and_kvs = list(chunk_hashes_and_kvs)
        end_make_chunks = time.perf_counter()
        """ store them into the dictionary """
        n_chunks = self.engine_.batched_put(
            ((self._make_key(chunk_hash, fmt), kv_chunk)
             for chunk_hash, kv_chunk in chunk_hashes_and_kvs),
            blocking=blocking,
        )

        end_time = time.perf_counter()
        logger.info(f"Stored/updated {n_chunks} chunks, total time "
                    f"{end_time - start_time:.2f}s, make chunks time "
                    f"{end_make_chunks - start_time:.2f}s")

    # prefix caching only needs a mask_len
    # but non-prefix might need an roi
    @_lmcache_nvtx_annotate
    @torch.no_grad()
    def retrieve(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[KVCache, torch.Tensor]:
        """
        Retrieve the KV cache of the tokens from the cache engine. The 
        retrieved KV cache should be a prefix of the input tokens.

        Input:
            tokens: the input tokens, with shape [seq_len]

            format: either 'huggingface' or 'vllm'
                For huggingface, it should have the shape of 
                [num_heads, num_tokens, head_size]
                
                For vllm, it should have the shape of 
                [num_tokens, num_heads, head_size]
                
            mask: a boolean mask of tokens indicating which tokens'
            KV Cache should be retrieved. Currently, only support
            suffix mask.

        Output:
            kv_tensors: the kv cache of the tokens, in the format of nested 
                        tuples. Will be an empty tuple if no kv cache is 
                        retrieved.

            ret_mask: indicate which tokens are retrieved
        """
        num_skip_chunk = 0
        num_skip_tok = 0
        ret_mask = torch.ones_like(tokens, dtype=torch.bool)
        if mask is not None:
            num_skip_tok = (len(mask) - torch.sum(mask))
            num_skip_chunk = num_skip_tok // self.chunk_size
        ret_mask[:num_skip_tok] = False

        st = time.perf_counter()
        fmt = self.metadata.fmt
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens),
                                         num_skip_chunk)

        retrival_iterator = self.engine_.batched_get(
            (self._make_key(chunk_hash, fmt) for chunk_hash in chunk_hashes), )

        retrieved_kv_chunks = []
        for chunk in retrival_iterator:
            if chunk is None:
                break
            retrieved_kv_chunks.append(chunk)
        """ concatenate the kv cache """
        dim = None
        match fmt:
            case "huggingface":
                dim = 1
            case "vllm":
                dim = 0
            case _:
                raise ValueError(f"Invalid format: {fmt}")

        if len(retrieved_kv_chunks) == 0:
            logging.info("Retrieved 0 chunks")
            ret_mask[:] = False
            return (), ret_mask

        st2 = time.perf_counter()

        # drop extra tokens in the first chunk
        extra_token_len = num_skip_tok - num_skip_chunk * self.chunk_size
        retrieved_kv_chunks[0] = self._slice_kv_at(extra_token_len,
                                                   retrieved_kv_chunks[0],
                                                   fmt)[0]

        ret = self._blob_to_tuple_kv(
            torch.cat(retrieved_kv_chunks, dim=dim + 2))
        ed2 = time.perf_counter()
        logger.info(
            f"Concatenated {len(retrieved_kv_chunks)} chunks -- elapsed time"
            f"{ed2 - st2}")
        retrieved_token_count = 0 if len(ret) == 0 else ret[0][0].shape[dim]
        ed = time.perf_counter()
        logger.info(f"Retrieved {len(retrieved_kv_chunks)} chunks "
                    f"({retrieved_token_count} tokens in total) --"
                    f"elapsed time {ed - st}")

        ret_mask[num_skip_tok + retrieved_token_count:] = False

        return ret, ret_mask

    @_lmcache_nvtx_annotate
    @torch.no_grad()
    def lookup(
        self,
        tokens: torch.Tensor,
        align_to_chunk: bool,
    ) -> torch.Tensor:
        """
        Checks the existence of KV cache of the tokens from the cache engine.
        The return should be a boolean tensor indicating the existence of 
        the KV cache of each token.

        Input:
            tokens: the input tokens, with shape [seq_len]
            align_to_chunk: whether to align the output to the chunk size, 
            this truncates available tokens to the last full chunk.

        Output:
            a boolean tensor indicating the existence of the KV cache of 
            each token
        """
        # NOTE(Sixian): Now this is a prefix lookup.
        fmt = self.metadata.fmt
        total_token_cnt = len(tokens)
        ret = torch.zeros(total_token_cnt, dtype=torch.bool)
        current_token_idx = 0
        full_chunk_cnt = total_token_cnt // self.chunk_size
        next_token_idx = total_token_cnt if full_chunk_cnt == 0 \
            else self.chunk_size
        chunk_hashes = self._prefix_hash(self._chunk_tokens(tokens), 0)
        for chunk_hash in chunk_hashes:
            interval_len = next_token_idx - current_token_idx
            if interval_len < self.chunk_size and align_to_chunk:
                break
            if not self.engine_.contains(self._make_key(chunk_hash, fmt)):
                break
            else:
                ret[current_token_idx:next_token_idx] = True
            current_token_idx = next_token_idx
            next_token_idx = min(next_token_idx + self.chunk_size,
                                 total_token_cnt)
        return ret

    def close(self):
        self.engine_.close()


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheEngineMetadata] = {}

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheEngineMetadata,
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        Raises:
            ValueError if the instance already exists with a different
            configuration.
        """
        if instance_id not in cls._instances:
            engine = LMCacheEngine(config, metadata)
            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            return engine
        else:
            if (cls._cfgs[instance_id] != config
                    or cls._metadatas[instance_id] != metadata):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata.")
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID, 
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        if instance_id in cls._instances:
            engine = cls._instances[instance_id]
            engine.close()
            cls._instances.pop(instance_id, None)
            cls._cfgs.pop(instance_id, None)
            cls._metadatas.pop(instance_id, None)