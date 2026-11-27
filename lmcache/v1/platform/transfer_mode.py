# SPDX-License-Identifier: Apache-2.0
"""MP transfer mode vocabulary shared by the platform and multiprocess layers.

``MPTransferMode`` lives in this dependency-free leaf module so that
:class:`lmcache.v1.platform.base.device_spec.DeviceSpec` can reference it for
``default_mp_transfer_mode`` without an import cycle: the multiprocess
transfer-context code imports the platform package at module level.
"""

# Standard
from enum import Enum


class MPTransferMode(str, Enum):
    """Routing mode used by :func:`create_transfer_context`.

    * ``AUTO``: defer to the device spec's declared default
      (:meth:`DeviceSpec.default_mp_transfer_mode`, derived from
      :meth:`DeviceSpec.is_lmcache_driven_available`); CUDA and NPU get
      lmcache-driven, every other device engine-driven.
    * ``ENGINE_DRIVEN``: force :class:`EngineDrivenTransferContext`
      (worker-side gather / scatter copy path).
    * ``LMCACHE_DRIVEN``: force :class:`LMCacheDrivenTransferContext`
      (IPC / SHM zero-copy path). Requires a registered KV-wrapper factory
      for the device.
    """

    AUTO = "auto"
    ENGINE_DRIVEN = "engine_driven"
    LMCACHE_DRIVEN = "lmcache_driven"
