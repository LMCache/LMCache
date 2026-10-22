# SPDX-License-Identifier: Apache-2.0
"""Answer how much committed L1 a set of chunks holds.

A client, such as a scheduler measuring how much of L1 vLLM's GPU prefix
cache also holds, sends chunk hashes by cache salt. The module expands them
into exact object keys and answers the committed L1 usage
(:class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`) of those
keys and of the whole model, with the server's probe hash so the client can
check that both sides hash tokens alike.

The module lives outside the built-in ``modules`` package, so a server pays
nothing unless it measures. Start the MP server with::

    --server-module '{"module_path": "lmcache.v1.multiprocess.l1_residency"}'

Design: ``docs/design/v1/multiprocess/l1_residency.md``.
"""

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import server_module_handler
from lmcache.v1.multiprocess.protocols.l1_residency import (
    L1_RESIDENCY_QUERY_METHOD,
    L1ResidencyQuery,
    L1ResidencyReply,
    compute_probe_hash,
)

# With object groups merged, every chunk is one object per kv rank in group 0.
_OBJECT_GROUP_IDS = [0]


def _query_keys(request: L1ResidencyQuery) -> list[ObjectKey]:
    """Expand a query's chunks into their object keys, as a lookup would.

    Args:
        request: The decoded query.

    Returns:
        One key per (chunk, kv rank), in object group 0, salted per chunk.

    Raises:
        ValueError: If a cache salt violates the ``ObjectKey`` invariants.
    """
    keys: list[ObjectKey] = []
    for cache_salt, chunk_hashes in request.chunk_hashes_by_salt.items():
        # Only model, world size and salt select the keys; worker_id None
        # fans out to every kv rank, as for a lookup.
        ipc_key = IPCCacheServerKey(
            model_name=request.model_name,
            world_size=request.world_size,
            worker_id=None,
            token_ids=(),
            start=0,
            end=0,
            request_id="",
            cache_salt=cache_salt,
        )
        keys.extend(ipc_key_to_object_keys(ipc_key, chunk_hashes, _OBJECT_GROUP_IDS)[0])
    return keys


class L1ResidencyModule(DiscoverableModule):
    """Serve ``l1_residency.query`` from the storage manager's L1.

    Args:
        ctx: Engine context whose storage manager is read.

    Raises:
        ValueError: If the server separates object groups: a chunk is then
            several objects, and the query counts object group 0 only.
    """

    module_name = "l1_residency"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the module over ``build_ctx``'s engine context.

        Args:
            build_ctx: The server's module build context.

        Returns:
            The module; never ``None``, since it needs no optional resource.
        """
        return cls(build_ctx.engine_context)

    def __init__(self, ctx: MPCacheServerContext) -> None:
        if ctx.separate_object_groups:
            raise ValueError(
                "lmcache.v1.multiprocess.l1_residency requires "
                "separate_object_groups=False: it counts each chunk as one "
                "object per kv rank in object group 0"
            )
        self._ctx = ctx

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    def report_status(self) -> dict:
        """Return no status; the module keeps no state."""
        return {}

    def close(self) -> None:
        """Nothing to release; the module keeps no state."""

    @server_module_handler(L1_RESIDENCY_QUERY_METHOD)
    def query(self, payload: bytes) -> bytes:
        """Answer an encoded :class:`L1ResidencyQuery`.

        Args:
            payload: The msgpack-encoded query.

        Returns:
            The msgpack-encoded :class:`L1ResidencyReply`.

        Raises:
            msgspec.DecodeError: If the payload is not an encoded query
                (its subclass ``msgspec.ValidationError`` when well-formed
                msgpack has the wrong shape).
            ValueError: If a cache salt violates the ``ObjectKey`` invariants.
        """
        request = msgspec.msgpack.decode(payload, type=L1ResidencyQuery)
        usage = self._ctx.storage_manager.get_l1_committed_usage(
            _query_keys(request), request.model_name
        )
        return msgspec.msgpack.encode(
            L1ResidencyReply(
                probe_hash=compute_probe_hash(self._ctx.token_hasher),
                matched_objects=usage.matched_objects,
                matched_bytes=usage.matched_bytes,
                model_objects=usage.model_objects,
                model_bytes=usage.model_bytes,
            )
        )
