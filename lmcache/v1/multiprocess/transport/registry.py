# SPDX-License-Identifier: Apache-2.0
"""URL-scheme based registry for transport plug-ins.

Built-in plug-ins register themselves at import time in
:mod:`lmcache.v1.multiprocess.transport` (see ``__init__``).  Third-party
plug-ins can register via ``setuptools`` entry points at package install
time — the mechanism is intentionally simple: a dict keyed by URL scheme.

Example:
    >>> from lmcache.v1.multiprocess.transport import (
    ...     register_client, create_client_transport,
    ... )
    >>> @register_client("myscheme")
    ... def _make():
    ...     return MyClientTransport()
    >>> t = create_client_transport("myscheme://host:port")
"""

# Standard
from typing import Callable

# First Party
from lmcache.v1.multiprocess.transport.base import (
    ClientTransport,
    ServerTransport,
)

_CLIENT_FACTORIES: dict[str, Callable[..., ClientTransport]] = {}
_SERVER_FACTORIES: dict[str, Callable[..., ServerTransport]] = {}


def _scheme_of(url: str) -> str:
    if "://" not in url:
        raise ValueError(
            "transport URL must be of the form 'scheme://...', got: " + url
        )
    return url.split("://", 1)[0]


def register_client(scheme: str):
    """Register a client-side transport factory for ``scheme``.

    The decorated callable is invoked lazily by
    :func:`create_client_transport` and receives no positional arguments;
    factories that need extra state should close over it or accept
    keyword arguments.
    """

    def deco(
        factory: Callable[..., ClientTransport],
    ) -> Callable[..., ClientTransport]:
        _CLIENT_FACTORIES[scheme] = factory
        return factory

    return deco


def register_server(scheme: str):
    """Register a server-side transport factory for ``scheme``."""

    def deco(
        factory: Callable[..., ServerTransport],
    ) -> Callable[..., ServerTransport]:
        _SERVER_FACTORIES[scheme] = factory
        return factory

    return deco


def create_client_transport(url: str, **kwargs) -> ClientTransport:
    """Create a client transport whose scheme matches ``url``.

    Extra ``kwargs`` are forwarded to the factory so callers can pass
    transport-specific state (e.g. a shared ``zmq.Context``).
    """
    scheme = _scheme_of(url)
    factory = _CLIENT_FACTORIES.get(scheme)
    if factory is None:
        raise ValueError(
            "no client transport registered for scheme '"
            + scheme
            + "'; registered: "
            + ", ".join(sorted(_CLIENT_FACTORIES.keys()))
        )
    return factory(**kwargs)


def create_server_transport(url: str, **kwargs) -> ServerTransport:
    """Create a server transport whose scheme matches ``url``."""
    scheme = _scheme_of(url)
    factory = _SERVER_FACTORIES.get(scheme)
    if factory is None:
        raise ValueError(
            "no server transport registered for scheme '"
            + scheme
            + "'; registered: "
            + ", ".join(sorted(_SERVER_FACTORIES.keys()))
        )
    return factory(**kwargs)


def available_client_schemes() -> list[str]:
    return sorted(_CLIENT_FACTORIES.keys())


def available_server_schemes() -> list[str]:
    return sorted(_SERVER_FACTORIES.keys())
