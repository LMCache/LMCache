# SPDX-License-Identifier: Apache-2.0
"""Method-oriented ZMQ client for the multiprocess server."""

# Standard
from inspect import Parameter
from typing import Any, Callable

# First Party
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.rpc import RpcSpec, get_rpc_specs
from lmcache.v1.multiprocess.transport.base import RequestClient
from lmcache.v1.multiprocess.transport.zmq_impl.mq import MessageQueueClient

ClientRpcCallable = Callable[..., MessagingFuture[Any]]


class ZmqMultiprocessClient(RequestClient):
    """Expose the typed request-client contract over ZMQ.

    RPC methods are installed from the shared request contract, so adding an
    operation does not require another transport-specific wrapper method.

    Args:
        message_queue_client: Existing ZMQ message queue client to wrap.
    """

    def __init__(self, message_queue_client: MessageQueueClient) -> None:
        self._message_queue_client = message_queue_client

    def cb_register_rope_v3(self, *args: Any, **kwargs: Any) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``cb_register_rope``."""
        return self.cb_register_rope(*args, **kwargs)

    def cb_unregister_rope_v3(self, *args: Any, **kwargs: Any) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``cb_unregister_rope``."""
        return self.cb_unregister_rope(*args, **kwargs)

    def cb_retrieve_pre_computed_v3(
        self, *args: Any, **kwargs: Any
    ) -> MessagingFuture[Any]:
        """Call the compatibility alias for ``cb_retrieve_pre_computed``."""
        return self.cb_retrieve_pre_computed(*args, **kwargs)

    def close(self) -> None:
        """Close the wrapped ZMQ client."""
        self._message_queue_client.close()

    def _call(
        self,
        rpc_spec: RpcSpec,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> MessagingFuture[Any]:
        payloads = rpc_spec.bind_payloads(args, kwargs)
        return self._message_queue_client.submit_request(
            rpc_spec.operation,
            list(payloads),
        )


def _make_client_rpc_method(rpc_spec: RpcSpec) -> ClientRpcCallable:
    def invoke(
        self: ZmqMultiprocessClient,
        *args: Any,
        **kwargs: Any,
    ) -> MessagingFuture[Any]:
        return self._call(rpc_spec, args, kwargs)

    invoke.__name__ = rpc_spec.operation
    invoke.__qualname__ = f"ZmqMultiprocessClient.{rpc_spec.operation}"
    invoke.__signature__ = rpc_spec.signature.replace(  # type: ignore[attr-defined]
        parameters=(
            Parameter("self", Parameter.POSITIONAL_OR_KEYWORD),
            *rpc_spec.signature.parameters.values(),
        )
    )
    return invoke


def _install_client_rpc_methods() -> None:
    """Install contract-derived RPC methods on the concrete client class."""
    for rpc_spec in get_rpc_specs().values():
        if rpc_spec.operation in ZmqMultiprocessClient.__dict__:
            raise RuntimeError(
                f"ZMQ client method conflicts with {rpc_spec.operation!r}"
            )
        setattr(
            ZmqMultiprocessClient,
            rpc_spec.operation,
            _make_client_rpc_method(rpc_spec),
        )


_install_client_rpc_methods()
