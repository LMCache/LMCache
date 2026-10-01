# SPDX-License-Identifier: Apache-2.0
"""Transport contract for the layer-wise retrieve path.

Kept in its own module so the shared
:mod:`lmcache.v1.multiprocess.transport.base` stays untouched by this
feature; nothing on the per-chunk path imports it.

The two operations are deliberately not declared on
:class:`~lmcache.v1.multiprocess.transport.base.RequestClient`. Every RPC
declared there must also have a generated gRPC method, and gRPC has no
streaming handler yet, so the layer-wise path is served over ZMQ only. The
wire contract below is written in the same form as the shared one, so
moving it onto ``RequestClient`` once gRPC can stream responses is a
relocation rather than a redesign.
"""

# Standard
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Protocol

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.rpc import RpcOperation, RpcSpec, _build_rpc_spec
from lmcache.v1.multiprocess.transport.base import RequestClient


class LayerwiseRequestClient(RequestClient, Protocol):
    """A request client that also speaks the layer-wise retrieve protocol.

    Deliberately separate from :class:`RequestClient`. Declaring these two
    methods on the base would hand every transport an inherited ``...`` body,
    so a transport that never implemented the layer-wise path would return
    ``None`` from the middle of a retrieve instead of failing. Keeping them
    here lets the layer-wise worker reject such a client up front, while
    transports that do not serve this path stay unaware of it.

    Deliberately not ``runtime_checkable``: since Python 3.12 ``isinstance``
    against a runtime protocol resolves members with
    ``inspect.getattr_static``, so duck-typed stand-ins that serve attributes
    from ``__getattr__`` (test doubles, proxies) would be rejected. Callers
    that need to check a client probe for the methods themselves.
    """

    def register_layerwise_ipc_event_pool(
        self, instance_id: int
    ) -> MessagingFuture[Any]:
        """Import the server's per-layer IPC event pool."""
        ...

    def retrieve_layerwise(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event_ipc_handle: Any,
        skip_first_n_tokens: int,
        future: Any,
    ) -> MessagingFuture[Any]:
        """Retrieve one chunk as a stream of per-layer-batch frames."""
        ...


class _LayerwiseWireContract(Protocol):
    """Payloads and response of each layer-wise operation, as sent on the wire.

    Separate from :class:`LayerwiseRequestClient` because the client-side
    ``retrieve_layerwise`` also takes the future to complete, which is not a
    payload. Only this class feeds the RPC specs.
    """

    def register_layerwise_ipc_event_pool(
        self, instance_id: int
    ) -> MessagingFuture[tuple[int, list[bytes]]]:
        """Fetch the per-layer IPC event pool of a registered instance.

        Issued once, right after ``register_kv_cache``, so that call keeps
        its ``None`` response for every non-layer-wise deployment.

        Args:
            instance_id: The GPU instance ID (such as PID).

        Returns:
            ``(layerwise_batch, pool handles)``. The handle list is empty
            when layer-wise mode is disabled.
        """
        ...

    def retrieve_layerwise(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int,
    ) -> MessagingFuture[tuple[bytes, bool, bool]]:
        """Layer-wise variant of ``retrieve``, with the same payloads.

        The server signals a pooled IPC event as each layer batch lands on
        the device and answers with one frame per batch, so the worker can
        start attention on layer 0 while later layers are still loading.

        Args:
            key: Cache key; ``worker_id`` must not be None.
            instance_id: The GPU instance ID (such as PID).
            gpu_block_ids: Destination blocks per KV group.
            event_ipc_handle: Producer event handle to order against.
            skip_first_n_tokens: Tokens to skip at the range start.

        Returns:
            One ``(payload, is_final, succeeded)`` per frame. Intermediate
            frames carry a packed ``(first_layer, count, pool_index)`` batch
            descriptor with ``is_final=False``; the closing frame carries the
            packed pool indices (empty if already reported) with
            ``is_final=True``. Completion is decided by the future, not by
            the transport.
        """
        ...


# Operations answered with more than one frame under the same request uid.
STREAMING_OPERATIONS: frozenset[RpcOperation] = frozenset({"retrieve_layerwise"})


@lru_cache(maxsize=1)
def get_layerwise_rpc_specs() -> Mapping[RpcOperation, RpcSpec]:
    """Return the RPC specs of the layer-wise operations.

    Built with the same rules as the shared contract: every payload typed,
    and the response declared as ``MessagingFuture[T]``.
    """
    specs = {
        name: _build_rpc_spec(name, method)
        for name, method in _LayerwiseWireContract.__dict__.items()
        if not name.startswith("_") and callable(method)
    }
    return MappingProxyType(specs)


def get_layerwise_rpc_spec(operation: RpcOperation) -> RpcSpec:
    """Return the layer-wise RPC spec for ``operation`` or raise ``KeyError``."""
    return get_layerwise_rpc_specs()[operation]
