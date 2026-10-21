# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and service registration shared by the gRPC request servers."""

# Standard
from abc import abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import threading

# First Party
from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool
from lmcache.v1.multiprocess.ext_server_module import register_grpc_services
from lmcache.v1.multiprocess.transport.base import RequestServer
from lmcache.v1.multiprocess.transport.grpc_impl.client import parse_grpc_target
from lmcache.v1.multiprocess.transport.grpc_impl.descriptors import (
    ServiceBinding,
    get_service_bindings,
)
from lmcache.v1.multiprocess.transport.grpc_impl.server_handlers import (
    GrpcRequestHandler,
    bind_module_handlers,
    build_service_handlers,
)


class GrpcMultiprocessServerBase(RequestServer):
    """Serve transport-neutral modules as the generated gRPC services.

    Owns what the sync and aio servers share: the affinity pool, the gRPC
    server executor, binding the port, registering every generated service,
    and the close sequence. Subclasses provide the gRPC server itself, a
    servicer for each service, how normal handlers are bounded by
    ``max_cpu_workers``, and how the server is started and stopped.
    """

    def __init__(
        self,
        bind_url: str,
        max_cpu_workers: int,
        max_gpu_workers: int,
        grpc_server_workers: int,
    ) -> None:
        if max_cpu_workers <= 0:
            raise ValueError("max_cpu_workers must be greater than 0")
        self._bind_url = bind_url
        self._affinity_pool = AffinityThreadPool(
            max_workers=max_gpu_workers,
            thread_name_prefix="grpc-affinity",
        )
        # The sync server runs every call here; grpc.aio only runs plain
        # functions that out-of-tree services register.
        self._executor = ThreadPoolExecutor(
            max_workers=grpc_server_workers,
            thread_name_prefix="grpc-server",
        )
        self._closed = threading.Event()
        self._bound_port = 0

    @property
    def bound_port(self) -> int:
        """Return the TCP port selected by gRPC, including for port zero."""
        return self._bound_port

    def add_modules(self, modules: Sequence[object]) -> None:
        """Register decorated module methods as generated gRPC services.

        Args:
            modules: Ordered business modules. A later module overrides an
                earlier handler for the same request type.

        Raises:
            TypeError: If a module handler does not match its protocol types.
            ValueError: If a module exposes invalid handler metadata.
        """
        handlers_by_operation = bind_module_handlers(modules)
        server = self._grpc_server()
        for binding in get_service_bindings().values():
            add_servicer = getattr(
                binding.grpc_module,
                f"add_{binding.descriptor.name}Servicer_to_server",
            )
            service_handlers = build_service_handlers(binding, handlers_by_operation)
            add_servicer(self._make_servicer(binding, service_handlers), server)
        register_grpc_services(modules, server)

    def close(self) -> None:
        """Stop the gRPC server and release its threads. Safe to call twice."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._stop_server()
        self._release_threads(wait=False)

    def _bind(self, server: Any) -> None:
        self._bound_port = server.add_insecure_port(parse_grpc_target(self._bind_url))
        if self._bound_port == 0:
            raise RuntimeError(
                f"Failed to bind gRPC multiprocess server: {self._bind_url}"
            )

    def _release_threads(self, wait: bool) -> None:
        self._affinity_pool.shutdown(wait=wait)
        self._executor.shutdown(wait=wait)

    @abstractmethod
    def start(self) -> None:
        """Start accepting gRPC requests."""

    @abstractmethod
    def _grpc_server(self) -> Any:
        """Return the grpc.Server or grpc.aio.Server services are added to."""

    @abstractmethod
    def _make_servicer(
        self,
        binding: ServiceBinding,
        handlers: dict[str, GrpcRequestHandler],
    ) -> object:
        """Build the servicer that dispatches one generated service."""

    @abstractmethod
    def _stop_server(self) -> None:
        """Stop the gRPC server without waiting for in-flight calls."""
