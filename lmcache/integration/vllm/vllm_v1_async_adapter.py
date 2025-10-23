# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from typing import TYPE_CHECKING
import concurrent.futures
import threading
from collections import defaultdict

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.sched.output import NewRequestData
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorV1Impl,
    RequestTracker,
    ReqMeta,
    LMCacheConnectorMetadata,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks

logger = init_logger(__name__)


class LMCacheConnectorV1AsyncImpl(LMCacheConnectorV1Impl):
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        super().__init__(vllm_config, role, parent)
        self._new_async_load_requests: list[Request] = []
        # Initialize thread pool for async KV loading
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.loading_lock = threading.Lock()
        # Track request IDs with their corresponding futures
        self.request_futures: dict[str, concurrent.futures.Future] = {}
        self._connector_role = role

        # Multi-worker communication setup (similar to NixlConnectorWorker) if role is worker
        if self._connector_role == KVConnectorRole.WORKER:
            self.tp_rank = get_tensor_model_parallel_rank()
            self.world_size = get_tensor_model_parallel_world_size()
            self.tp_group = get_tp_group() if self.world_size > 1 else None
        else:
            self._req_to_block_ids: dict[str, list[int]] = {}

        # Complete transfer tracker for multi-worker synchronization
        # Used by rank 0 to track finished requests on ranks 1 to N-1 and their number of tokens
        self._done_recving_tracker: defaultdict[str, dict[int, int]] = defaultdict(dict)

    ####################
    # Worker side APIs
    ####################

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache asynchronously using a thread pool.

        This async version submits KV retrieval operations to a thread pool
        instead of blocking the main thread.
        """
        super().start_load_kv(forward_context, **kwargs)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        for idx, request in enumerate(metadata.async_requests):
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            slot_mapping = request.slot_mapping.cuda()
            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens

            assert len(tokens[:lmcache_cached_tokens]) == len(slot_mapping), f"shape(tokens): {tokens.shape}, shape(slot_mapping): {slot_mapping.shape}"

            token_mask = torch.ones_like(tokens, dtype=torch.bool)

            if self.tp_rank == 0:
                masked_token_count = (
                    request.load_spec.vllm_cached_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                token_mask[:masked_token_count] = False

                # We always load async requests in a non-layerwise manner
                future = self.thread_pool.submit(
                    self._retrieve_kv_async,
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches,
                    slot_mapping[:lmcache_cached_tokens],
                    request.load_spec.vllm_cached_tokens,
                    lmcache_cached_tokens,
                    request.req_id,
                )
                with self.loading_lock:
                    self.request_futures[request.req_id] = future

    def _retrieve_kv_async(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        kvcaches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
        vllm_cached_tokens: int,
        lmcache_cached_tokens: int,
        req_id: str,
    ) -> int:
        """Helper method to run retrieve operation in thread pool."""
        masked_token_count = (
            vllm_cached_tokens // self._lmcache_chunk_size * self._lmcache_chunk_size
        )
        try:
            ret_token_mask, reordered_chunks, slot_mapping = self.lmcache_engine.retrieve(
                tokens,
                token_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                skip_broadcast=True,
            )

            # Check the result
            num_retrieved_tokens = ret_token_mask.sum().item()
            num_expected_tokens = lmcache_cached_tokens - vllm_cached_tokens
            if num_retrieved_tokens < num_expected_tokens:
                logger.error(
                    "The number of retrieved tokens is less than the "
                    "expected number of tokens! This should not happen!"
                )
                logger.error(
                    "Num retrieved tokens: %d, num expected tokens: %d, for request %s",
                    num_retrieved_tokens,
                    num_expected_tokens,
                    req_id,
                )
            else:
                logger.info(
                    "Successfully retrieved %d tokens for request %s",
                    num_retrieved_tokens,
                    req_id,
                )
            total_computed_tokens = max(
                masked_token_count + num_retrieved_tokens, vllm_cached_tokens
            )

        except Exception as e:
            logger.exception(f"Error in async KV retrieval for request {req_id}: {e}")
            total_computed_tokens = vllm_cached_tokens

        return total_computed_tokens, ret_token_mask, reordered_chunks, slot_mapping

    def get_finished_loading(self, scheduler_output: SchedulerOutput) -> dict[str, int]:
        """
        Gets the actual number of tokens loaded for requests that have
        completed the asynchronous loading process from the remote KV cache.

        Returns:
            A dictionary where the keys are request IDs and the values are the
            corresponding number of tokens that have been successfully loaded
            for each request.
        """
        assert self._connector_role == KVConnectorRole.WORKER, (
            "This method should only be called by worker"
        )
        # Get local finished receiving requests
        done_recv_req_num_tokens = dict()
        done_recv_req_info = dict()
        if self.tp_rank == 0:
            with self.loading_lock:
                # Check which requests have finished loading locally
                remaining_request_futures = dict()
                for req_id, future in self.request_futures.items():
                    if future.done():
                        try:
                            num_computed_tokens, ret_token_mask, reordered_chunks, slot_mapping = (
                                future.result()
                            )  # This will raise if there was an exception
                            done_recv_req_num_tokens[req_id] = num_computed_tokens
                            done_recv_req_info[req_id] = (ret_token_mask, reordered_chunks, slot_mapping)
                            logger.debug(
                                f"Async loading completed for request {req_id} on rank {self.tp_rank}"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Error in async loading for request {req_id} on rank {self.tp_rank}: {e}"
                            )
                            done_recv_req_num_tokens[req_id] = 0
                    else:
                        remaining_request_futures[req_id] = future

                self.request_futures = remaining_request_futures

        
        self.lmcache_engine.broadcast_at_finish(done_recv_req_info, list(self.kv_caches.values()))

        return done_recv_req_num_tokens

    ####################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        need_to_allocate = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if need_to_allocate > 0:
            load_kv_async = True
            # remove this request from the _lookup_requests_in_step
            # and add it to the _new_async_load_requests
            # self._lookup_requests_in_step.remove(request.request_id)
            self._new_async_load_requests.append(request)
        else:
            load_kv_async = False
        return need_to_allocate, load_kv_async

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        super().update_state_after_alloc(request, num_external_tokens)
        if request in self._new_async_load_requests:
            self._req_to_block_ids[request.request_id] = blocks.get_block_ids()
            logger.warning(
                f"Request {request.request_id} has been allocated number of blocks: {len(self._req_to_block_ids[request.request_id][0])}"
            )

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> LMCacheConnectorMetadata:
        meta: LMCacheConnectorMetadata = super().build_connector_meta(scheduler_output)

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        unallocated_async_load_requests = []

        for request in self._new_async_load_requests:
            if request.request_id not in self._req_to_block_ids:
                # logger.warning(f"Request {request.request_id} has NOT been allocated blocks")
                unallocated_async_load_requests.append(request)
                continue

            # Right now, we only load KV for new requests
            load_spec = self.load_specs.pop(request.request_id, None)
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            num_tokens_to_compute = lmcache_cached_tokens
            new_request_data = NewRequestData.from_request(
                request, self._req_to_block_ids[request.request_id]
            )
            request_tracker = RequestTracker.from_new_request(
                self.config,
                new_request_data,
                num_tokens_to_compute,
                lmcache_cached_tokens,
            )
            self._request_trackers[request.request_id] = request_tracker

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks,
            )
            if req_meta is not None:
                meta.add_async_request(req_meta)

        self._new_async_load_requests = unallocated_async_load_requests

        return meta
