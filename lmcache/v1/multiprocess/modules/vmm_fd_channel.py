# SPDX-License-Identifier: Apache-2.0
"""MP-server owner of the VMM fd channel (``--enable-vmm-fd-channel``)."""

# Standard
import sys

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.vmm_fd import VmmFdResolver, fd_channel_path
from lmcache.v1.platform.devices.cuda.ipc_wrapper import set_vmm_fd_resolver

logger = init_logger(__name__)


class VmmFdChannelModule(DiscoverableModule):
    """Receive POSIX fds of VMM KV caches and resolve them on import.

    Built only with ``--enable-vmm-fd-channel`` on a Linux CUDA server whose
    LMCache-driven path is enabled. It installs its resolver's ``resolve`` as the
    process's VMM fd resolver hook, so ``VmmCudaIPCWrapper.to_tensor`` inside the
    LMCache-driven register handler finds each fd by export id.

    Args:
        ctx: The shared engine context.
        resolver: The running fd resolver this module owns.
    """

    module_name = "vmm_fd_channel"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build when the flag is set and the server can use the channel."""
        mp_config = build_ctx.mp_config
        if (
            not mp_config.enable_vmm_fd_channel
            or mp_config.supported_transfer_mode == "engine_driven"
        ):
            return None
        if sys.platform != "linux" or torch_device_type != "cuda":
            logger.warning(
                "--enable-vmm-fd-channel ignored: needs a Linux CUDA server "
                "(platform=%s, device=%s)",
                sys.platform,
                torch_device_type,
            )
            return None
        # Workers send their fds before REGISTER, so the resolver must
        # already listen when the first one arrives.
        path = fd_channel_path(mp_config.port)
        try:
            resolver = VmmFdResolver(path)
        except OSError:
            # Only POSIX-fd VMM registrations need the channel, and they fail
            # loudly on their own; the rest of the server keeps working.
            logger.warning(
                "VMM fd channel unavailable at %s; POSIX-fd VMM KV-cache "
                "registration will fail on this server.",
                path,
                exc_info=True,
            )
            return None
        return cls(build_ctx.engine_context, resolver)

    def __init__(self, ctx: MPCacheServerContext, resolver: VmmFdResolver) -> None:
        self._ctx = ctx
        self._resolver = resolver
        set_vmm_fd_resolver(resolver.resolve)

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context. Exposed for testing only."""
        return self._ctx

    def report_status(self) -> dict[str, str]:
        """Return module-specific status information.

        Returns:
            ``{"path": <fd channel socket path>}``.
        """
        return {"path": self._resolver.path}

    def close(self) -> None:
        """Uninstall the resolver hook and close the fd resolver."""
        set_vmm_fd_resolver(None)
        self._resolver.close()
