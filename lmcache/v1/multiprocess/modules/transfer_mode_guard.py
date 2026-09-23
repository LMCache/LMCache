# SPDX-License-Identifier: Apache-2.0
"""Report unsupported multiprocess transfer modes to registration clients."""

from typing import NoReturn

from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    KVCache,
    RegisterEngineDrivenContextPayload,
)
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocols.engine import (
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.request_handler import request_handler


class TransferModeGuardModule:
    def __init__(self, ctx: MPCacheServerContext, supported_transfer_mode: str) -> None:
        if supported_transfer_mode not in {"auto", "engine_driven", "lmcache_driven"}:
            raise ValueError(
                f"Unsupported supported_transfer_mode '{supported_transfer_mode}'"
            )
        self._ctx = ctx
        self._supported_transfer_mode = supported_transfer_mode

    @property
    def context(self) -> MPCacheServerContext:
        return self._ctx

    def report_status(self) -> dict:
        return {"supported_transfer_mode": self._supported_transfer_mode}

    def close(self) -> None:
        pass

    def _raise_mismatch(self, requested: str) -> NoReturn:
        raise ValueError(
            f"Client requested transfer mode '{requested}', but the server "
            f"was started with supported_transfer_mode='{self._supported_transfer_mode}'. "
            "Use the same transfer mode on client and server, or start the server "
            "with supported_transfer_mode='auto'."
        )


class EngineDrivenTransferModeGuardModule(TransferModeGuardModule):
    @request_handler(operation="register_kv_cache")
    def reject_register_kv_cache(
        self,
        instance_id: int,
        kv_cache: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        self._raise_mismatch("lmcache_driven")

    @request_handler(operation="register_q_cache")
    def reject_register_q_cache(
        self,
        instance_id: int,
        q_cache: KVCache,
        model_name: str,
        world_size: int,
        engine_type: EngineType,
        layout_hints: LayoutHints,
        engine_group_infos: list[EngineGroupInfo],
    ) -> None:
        self._raise_mismatch("lmcache_driven")


class LMCacheDrivenTransferModeGuardModule(TransferModeGuardModule):
    @request_handler(operation="register_kv_cache_engine_driven_context")
    def reject_register_kv_cache_engine_driven_context(
        self, payload: RegisterEngineDrivenContextPayload
    ) -> RegisterEngineDrivenContextResponse:
        self._raise_mismatch("engine_driven")


def create_transfer_mode_guard(
    ctx: MPCacheServerContext, supported_transfer_mode: str
) -> TransferModeGuardModule:
    guard_types = {
        "auto": TransferModeGuardModule,
        "engine_driven": EngineDrivenTransferModeGuardModule,
        "lmcache_driven": LMCacheDrivenTransferModeGuardModule,
    }
    if supported_transfer_mode not in guard_types:
        raise ValueError(
            f"Unsupported supported_transfer_mode '{supported_transfer_mode}'"
        )
    return guard_types[supported_transfer_mode](ctx, supported_transfer_mode)
