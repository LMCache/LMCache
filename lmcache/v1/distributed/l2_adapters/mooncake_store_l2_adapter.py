# SPDX-License-Identifier: Apache-2.0
"""
Mooncake Store native L2 adapter config and factory.
"""

# Future
from __future__ import annotations

# Standard
from typing import (
    TYPE_CHECKING,
    Dict,
    Optional,
)

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)

# Keys consumed only by LMCache (never sent to mooncake).
_LMCACHE_ONLY_KEYS = {"type", "num_workers", "eviction", "preregister_l1_memory"}


class MooncakeStoreL2AdapterConfig(L2AdapterConfigBase):
    """Config for an L2 adapter backed by the native
    C++ Mooncake Store connector.

    ``setup_config`` is a string-to-string dict that is
    forwarded **as-is** to mooncake's
    ``RealClient::setup_internal(ConfigDict)``.
    LMCache does NOT interpret, validate, or fill in
    defaults for any mooncake keys — that is mooncake's
    responsibility.

    ``num_workers`` and ``preregister_l1_memory`` are
    LMCache-specific knobs.
    """

    def __init__(
        self,
        setup_config: Dict[str, str],
        num_workers: int = 4,
        preregister_l1_memory: bool = False,
    ):
        super().__init__()
        self.setup_config: Dict[str, str] = dict(setup_config)
        self.num_workers = num_workers
        self.preregister_l1_memory = preregister_l1_memory

    @classmethod
    def from_dict(cls, d: dict) -> "MooncakeStoreL2AdapterConfig":
        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")
        preregister_l1_memory = d.get("preregister_l1_memory", False)
        if not isinstance(preregister_l1_memory, bool):
            raise ValueError("preregister_l1_memory must be a boolean")

        # Everything except LMCache-only keys is
        # forwarded to mooncake as str values.
        setup: Dict[str, str] = {}
        for k, v in d.items():
            if k in _LMCACHE_ONLY_KEYS:
                continue
            if v is not None:
                setup[k] = str(v)

        return cls(
            setup_config=setup,
            num_workers=num_workers,
            preregister_l1_memory=preregister_l1_memory,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Mooncake Store L2 adapter config.\n"
            "All keys except LMCache-only keys are "
            "forwarded as-is to mooncake's "
            "setup_internal(ConfigDict).\n"
            "Refer to mooncake documentation for "
            "available setup keys.\n"
            "- num_workers (int): C++ worker threads "
            "(default 4, >0)\n"
            "- preregister_l1_memory (bool): "
            "pre-register the provided L1 memory descriptor with Mooncake "
            "(default False)"
        )


def _create_mooncake_store_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ Mooncake Store connector."""
    try:
        # First Party
        from lmcache.lmcache_mooncake import (
            LMCacheMooncakeClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Mooncake Store L2 adapter requires the "
            "C++ Mooncake extension. Build with: "
            "MOONCAKE_INCLUDE_DIR=/path/to/mooncake-"
            "store/include pip install -e ."
        ) from e

    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, MooncakeStoreL2AdapterConfig)
    preregister_base = 0
    preregister_size = 0
    if config.preregister_l1_memory:
        if l1_memory_desc is None:
            logger.warning(
                "preregister_l1_memory is enabled, but no L1 memory descriptor "
                "was provided; falling back to lazy per-object registration."
            )
        elif l1_memory_desc.ptr == 0 or l1_memory_desc.size <= 0:
            logger.warning(
                "preregister_l1_memory is enabled, but the L1 memory descriptor "
                "is invalid (ptr=%d, size=%d); falling back to lazy registration.",
                l1_memory_desc.ptr,
                l1_memory_desc.size,
            )
        else:
            preregister_base = l1_memory_desc.ptr
            preregister_size = l1_memory_desc.size

    native_client = LMCacheMooncakeClient(
        config=config.setup_config,
        num_workers=config.num_workers,
        preregister_l1_base=preregister_base,
        preregister_l1_size=preregister_size,
    )
    logger.info(
        "Created Mooncake Store L2 adapter (workers=%d, preregister_l1_memory=%s)",
        config.num_workers,
        config.preregister_l1_memory and preregister_size > 0,
    )
    return NativeConnectorL2Adapter(native_client)


# Self-register config type and adapter factory
register_l2_adapter_type("mooncake_store", MooncakeStoreL2AdapterConfig)
register_l2_adapter_factory("mooncake_store", _create_mooncake_store_l2_adapter)
