# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union
import json
import threading

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.lookup_client.abstract_client import LookupClientInterface
from lmcache.v1.lookup_client.semantic_provider import (
    SemanticLookupProvider,
    SemanticLookupResult,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.rpc.transport import (
    RpcClientTransport,
    RpcServerTransport,
)

logger = init_logger(__name__)


class LMCacheLookupClient(LookupClientInterface):
    """
    Lookup client that communicates with a lookup server
    via an injected RpcClientTransport.

    The client is decoupled from the underlying communication
    mechanism. The transport layer handles connection management,
    retries, and error recovery.

    Related extra_config:
    - lookup_server_worker_ids:
        is a config to control create lookup server on some
        workers.
        if mla is not enabled, default is [];
        if mla is enabled, default is [0];
        - if lookup_server_worker_ids is [], start lookup
          server on all workers
        - if lookup_server_worker_ids is [0], start lookup
          server on worker0
        - if lookup_server_worker_ids is [0, 3, 6], start
          lookup server on worker0, worker3 and worker6
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        transport: RpcClientTransport,
    ):
        self.config = config
        self.transport = transport

        # NOTE: map from lookup_id (i.e., req_id) to
        # req's status.
        # int indicates number of hit tokens.
        # The assumption here is that once a request is
        # looked up, the following lookups of the same
        # request must have the same result.
        self.reqs_status: dict[str, int] = {}

        # First Party
        from lmcache.v1.token_database import (
            ChunkedTokenDatabase,
            SegmentTokenDatabase,
            TokenDatabase,
        )

        self.enable_blending = config.enable_blending
        self.token_database: TokenDatabase
        if self.enable_blending:
            self.token_database = SegmentTokenDatabase(config, metadata)
        else:
            self.token_database = ChunkedTokenDatabase(config, metadata)

        # Semantic fallback provider (optional)
        self._semantic_provider: Optional[SemanticLookupProvider] = None
        # Pending donor substitutions keyed by lookup_id
        self._pending_substitutions: dict[str, SemanticLookupResult] = {}

    def lookup_cache(self, lookup_id: str) -> Optional[int]:
        """
        "-1 means not found;
        None means ongoing; (not supported in sync client)
        int >= 0 means number of hit tokens
        """
        return self.reqs_status.get(lookup_id, -1)

    def set_semantic_provider(self, provider: SemanticLookupProvider) -> None:
        """Register a SemanticLookupProvider for approximate KV cache matching.

        Args:
            provider: An instance of a SemanticLookupProvider subclass.
        """
        self._semantic_provider = provider
        logger.info(
            "SemanticLookupProvider registered in LMCacheLookupClient: %s",
            type(provider).__name__,
        )

    def pop_pending_substitution(
        self, lookup_id: str
    ) -> Optional[SemanticLookupResult]:
        """Pop and return a pending semantic substitution result, if any.

        Args:
            lookup_id: The request ID to check.

        Returns:
            SemanticLookupResult if a substitution is pending, else None.
        """
        return self._pending_substitutions.pop(lookup_id, None)

    def notify_request_finished(
        self,
        request_id: str,
        token_ids: list[int],
        num_prompt_tokens: int,
    ) -> None:
        """Notify the semantic provider that a request has finished.

        Also cleans up any leftover pending substitution state for the request.

        Args:
            request_id: vLLM request ID of the finished request.
            token_ids: Full prompt token IDs of the finished request.
            num_prompt_tokens: Number of prompt tokens in the request.
        """
        # Clean up any leftover substitution state
        self._pending_substitutions.pop(request_id, None)

        if self._semantic_provider is not None:
            try:
                self._semantic_provider.on_request_finished(
                    request_id, token_ids, num_prompt_tokens
                )
            except Exception:
                logger.warning(
                    "SemanticLookupProvider.on_request_finished raised for req %s",
                    request_id,
                    exc_info=True,
                )

    def _do_transport_lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs_str: str,
    ) -> int:
        """Send a lookup request via transport and return the hit token count.

        Returns 0 on transport failure or empty response.
        """
        if not self.enable_blending:
            hashes = []
            offsets = []

            for (
                start,
                end,
                key,
            ) in self.token_database.process_tokens(token_ids, make_key=False):
                hashes.append(key)
                offsets.append(end - start)

            # if the token database returns no hashes, return 0
            if not hashes:
                return 0

            msg_buf = [
                hashes,
                offsets,
                lookup_id,
                request_configs_str,
            ]
        else:
            msg_buf = [
                token_ids,
                lookup_id,
                request_configs_str,
            ]

        responses = self.transport.send_and_recv_all(msg_buf)

        # Transport returns empty list on failure
        if not responses:
            return 0

        results = [int.from_bytes(resp, "big") for resp in responses]

        assert len(results) == self.transport.world_size
        if len(set(results)) > 1:
            logger.warning(
                "Lookup results (number of hit tokens) "
                "differ across (TP and PP) ranks: %s.",
                results,
            )
        # NOTE: it is possible that the number of hit
        # tokens is different across (TP and PP) ranks,
        # so we can use the minimum value.
        return min(results)

    def lookup(
        self,
        token_ids: Union[torch.Tensor, list[int]],
        lookup_id: str,
        request_configs: Optional[dict] = None,
        num_computed_tokens: int = 0,
    ) -> Optional[int]:
        request_configs_str = ""
        if request_configs is not None and len(request_configs) != 0:
            request_configs_str = json.dumps(request_configs)

        num_hit_toks = self._do_transport_lookup(
            token_ids, lookup_id, request_configs_str
        )
        self.reqs_status[lookup_id] = num_hit_toks

        # Semantic fallback: only in non-blending path, only on zero hit
        if (
            num_hit_toks == 0
            and not self.enable_blending
            and self._semantic_provider is not None
        ):
            result = None
            try:
                result = self._semantic_provider.on_lookup_miss(
                    lookup_id, list(token_ids), num_computed_tokens
                )
            except Exception:
                logger.warning(
                    "SemanticLookupProvider.on_lookup_miss raised for req %s",
                    lookup_id,
                    exc_info=True,
                )

            if result is not None:
                # Clear the cached zero-hit so the re-lookup is accepted
                self.reqs_status.pop(lookup_id, None)
                donor_hit = self._do_transport_lookup(
                    result.alternate_token_ids, lookup_id, request_configs_str
                )
                if donor_hit > num_computed_tokens:
                    self.reqs_status[lookup_id] = donor_hit
                    self._pending_substitutions[lookup_id] = result
                    return donor_hit
                else:
                    # Donor not in store or too few tokens — cold prefill
                    self.reqs_status[lookup_id] = 0
                    return 0

        return num_hit_toks

    def clear_lookup_status(self, lookup_id: str) -> None:
        self.reqs_status.pop(lookup_id, None)
        self._pending_substitutions.pop(lookup_id, None)

    def supports_producer_reuse(self) -> bool:
        """Return True as LMCacheLookupClient supports
        producer kvcache reuse"""
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        if self._semantic_provider is not None:
            try:
                self._semantic_provider.on_shutdown()
            except Exception:
                logger.warning(
                    "SemanticLookupProvider.on_shutdown raised",
                    exc_info=True,
                )
        self.transport.close()


class LMCacheLookupServer:
    """Lookup server that handles lookup requests using
    LMCacheEngine, with an injected RpcServerTransport."""

    def __init__(
        self,
        lmcache_engine: LMCacheEngine,
        metadata: LMCacheMetadata,
        transport: RpcServerTransport,
    ):
        self.transport = transport
        self.lmcache_engine = lmcache_engine
        self.running = True
        self.enable_blending = lmcache_engine.config.enable_blending

        def process_request():
            while self.running:
                try:
                    result = self.transport.recv_request()
                    if result is None:
                        continue

                    identity, data_frames = result

                    # Validate frame structure
                    if len(data_frames) < 3:
                        logger.warning("Malformed request received: not enough frames.")
                        continue

                    # Validate and decode lookup_id
                    lookup_id_bytes = data_frames[-2]
                    request_configs_bytes = data_frames[-1]

                    if not isinstance(lookup_id_bytes, (bytes, str)):
                        logger.warning(
                            "Malformed request received: lookup_id is not bytes or str."
                        )
                        continue

                    if not isinstance(request_configs_bytes, (bytes, str)):
                        logger.warning(
                            "Malformed request received: "
                            "request_configs is not bytes or str."
                        )
                        continue

                    # Decode to strings
                    if isinstance(lookup_id_bytes, bytes):
                        lookup_id = lookup_id_bytes.decode("utf-8")
                    else:
                        lookup_id = lookup_id_bytes

                    if isinstance(request_configs_bytes, bytes):
                        request_configs_str = request_configs_bytes.decode("utf-8")
                    else:
                        request_configs_str = request_configs_bytes

                    request_configs = (
                        json.loads(request_configs_str) if request_configs_str else None
                    )

                    if not self.enable_blending:
                        hashes = data_frames[0]
                        offsets = data_frames[1]
                        lookup_result = self.lmcache_engine.lookup(
                            hashes=hashes,
                            offsets=offsets,
                            lookup_id=lookup_id,
                            pin=True,
                            request_configs=request_configs,
                        )
                    else:
                        tokens = data_frames[0]
                        lookup_result = self.lmcache_engine.lookup(
                            tokens=tokens,
                            lookup_id=lookup_id,
                            pin=True,
                            request_configs=request_configs,
                        )
                    response = lookup_result.to_bytes(4, "big")
                    self.transport.send_response(identity, response)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON in lookup request: {e}")
                except UnicodeDecodeError as e:
                    logger.error(f"Error decoding UTF-8 in lookup request: {e}")
                except Exception as e:
                    logger.error(f"Error processing lookup request: {e}")

        logger.info("lmcache lookup server started")
        self.thread = threading.Thread(
            target=process_request,
            daemon=True,
            name="lookup-server-thread",
        )
        self.thread.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def close(self):
        # Stop the processing thread first
        self.running = False

        # Wait for thread to finish with timeout
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Lookup server thread did not terminate gracefully")

        # Close transport after thread is stopped
        self.transport.close()
