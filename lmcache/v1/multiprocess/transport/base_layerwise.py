# SPDX-License-Identifier: Apache-2.0
"""Transport contract for the layer-wise retrieve path.

Kept in its own module so the shared
:mod:`lmcache.v1.multiprocess.transport.base` stays untouched by this
feature; nothing on the per-chunk path imports it.
"""

# Standard
from typing import Any, Protocol

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
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
