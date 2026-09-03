# SPDX-License-Identifier: Apache-2.0
"""Explicit registration errors for unsupported MP transfer modes."""

# Standard
from typing import NoReturn

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    KVCache,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import HandlerSpec, ThreadPoolType
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)

_TRANSFER_MODES = {"auto", "engine_driven", "lmcache_driven"}


class TransferModeGuardModule:
    """Reject registration requests disabled by the server configuration.

    A mode-specific transfer module owns each supported registration request.
    This guard owns only the complementary request types, so mismatched clients
    receive a deterministic error instead of falling through as an unregistered
    request. ``auto`` builds both transfer modules and therefore needs no guard
    handlers.
    """

    def __init__(self, ctx: MPCacheServerContext, supported_transfer_mode: str) -> None:
        if supported_transfer_mode not in _TRANSFER_MODES:
            raise ValueError(
                f"Unsupported supported_transfer_mode '{supported_transfer_mode}'"
            )
        self._ctx = ctx
        self._supported_transfer_mode = supported_transfer_mode

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    def get_handlers(self) -> list[HandlerSpec]:
        """Return handlers for registration modes this server rejects."""
        if self._supported_transfer_mode == "auto":
            return []
        if self._supported_transfer_mode == "engine_driven":
            return [
                HandlerSpec(
                    RequestType.REGISTER_KV_CACHE,
                    self._reject_lmcache_driven,
                    ThreadPoolType.SYNC,
                ),
                HandlerSpec(
                    RequestType.REGISTER_Q_CACHE,
                    self._reject_lmcache_driven,
                    ThreadPoolType.SYNC,
                ),
            ]
        return [
            HandlerSpec(
                RequestType.REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT,
                self._reject_engine_driven,
                ThreadPoolType.SYNC,
            )
        ]

    def report_status(self) -> dict:
        """Expose the configured transfer mode in server status."""
        return {"supported_transfer_mode": self._supported_transfer_mode}

    def close(self) -> None:
        """The guard owns no resources."""

    def _reject_lmcache_driven(
        self,
        _instance_id: int,
        _kv_cache: KVCache,
        _model_name: str,
        _world_size: int,
        _engine_type: EngineType,
        _layout_hints: LayoutHints,
        _engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        self._raise_mismatch("lmcache_driven")

    def _reject_engine_driven(
        self, _payload: RegisterEngineDrivenContextPayload
    ) -> RegisterEngineDrivenContextResponse:
        self._raise_mismatch("engine_driven")

    def _raise_mismatch(self, requested: str) -> NoReturn:
        raise ValueError(
            f"Client requested transfer mode '{requested}', but the server "
            "was started with "
            f"supported_transfer_mode='{self._supported_transfer_mode}'. "
            "Use the same transfer mode on the client and server, or start "
            "the server with supported_transfer_mode='auto'."
        )
