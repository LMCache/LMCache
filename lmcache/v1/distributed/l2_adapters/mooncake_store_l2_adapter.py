# SPDX-License-Identifier: Apache-2.0
"""
Mooncake Store native L2 adapter config and factory.
"""

# Future
from __future__ import annotations

# Standard
from typing import (
    TYPE_CHECKING,
    cast,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.distributed.l2_adapters.base import (
        L2AdapterInterface,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.mooncake_process_adapter import (
    MooncakeProcessConfig,
    create_process_adapter,
    parse_process_options,
)
from lmcache.v1.distributed.l2_adapters.mooncake_resilience import GuardConfig

logger = init_logger(__name__)

# Keys consumed only by LMCache (never sent to mooncake).
_LMCACHE_ONLY_KEYS = {
    "type",
    "num_workers",
    "eviction",
    "per_op_workers",
    "process",
    "serde",
}


class MooncakeStoreL2AdapterConfig(L2AdapterConfigBase):
    """Config for an L2 adapter backed by the native
    C++ Mooncake Store connector.

    ``setup_config`` is a string-to-string dict forwarded
    **as-is** to mooncake's
    ``RealClient::setup_internal(ConfigDict)``.
    LMCache does NOT interpret, validate, or fill in
    defaults for any mooncake keys — that is mooncake's
    responsibility.

    Fields:
        setup_config: Mooncake SDK configuration forwarded
            as-is to ``RealClient::setup_internal()``.
        num_workers: Shared worker thread count (default 4,
            must be > 0).  Used for any op whose lane key
            is not present in ``per_op_workers``.
        per_op_workers: Optional dict mapping lane keys
            (``"lookup"``, ``"retrieve"``, ``"store"``,
            ``"delete"``) to dedicated worker counts.  Ops
            not mentioned use the shared ``num_workers`` pool.
        process_config: Opt-in isolated SDK process policy. Requires a named
            shared L1 arena and non-expiring controller I/O pins.
        guard_config: Admission and recovery policy used in process mode.
    """

    def __init__(
        self,
        setup_config: dict[str, str],
        num_workers: int = 4,
        per_op_workers: dict[str, int] | None = None,
        process_config: MooncakeProcessConfig | None = None,
        guard_config: GuardConfig | None = None,
    ):
        super().__init__()
        self.num_workers = L2AdapterConfigBase._validate_num_workers(num_workers)
        self.per_op_workers = L2AdapterConfigBase._validate_per_op_workers(
            per_op_workers
        )
        self.setup_config: dict[str, str] = dict(setup_config)
        self.process_config = process_config
        self.guard_config = guard_config or GuardConfig()

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "MooncakeStoreL2AdapterConfig":
        """Construct a config from a raw configuration dict.

        LMCache-only keys (``type``, ``num_workers``, ``eviction``,
        ``per_op_workers``, ``process``, ``serde``) are consumed locally. Other keys are
        forwarded to mooncake as string values.

        Args:
            d: Raw configuration dict (typically from JSON/CLI).

        Returns:
            A validated ``MooncakeStoreL2AdapterConfig``.

        Raises:
            ValueError: If ``num_workers`` or ``per_op_workers`` values
                are invalid.
        """
        num_workers = cast(int, d.get("num_workers", 4))  # validated in __init__

        per_op_workers = L2AdapterConfigBase._parse_per_op_workers_from_dict(d)
        process_config = None
        guard_config = None
        if "process" in d:
            if d.get("serde") is not None:
                raise ValueError("Process Mooncake does not support serde buffers")
            process_config, guard_config = parse_process_options(d["process"])
        # Everything except LMCache-only keys is
        # forwarded to mooncake as str values.
        setup: dict[str, str] = {}
        for k, v in d.items():
            if k in _LMCACHE_ONLY_KEYS:
                continue
            if v is not None:
                setup[k] = str(v)

        return cls(
            setup_config=setup,
            num_workers=num_workers,
            per_op_workers=per_op_workers,
            process_config=process_config,
            guard_config=guard_config,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "Mooncake Store L2 adapter config.\n"
            "All keys except LMCache-only keys are "
            "forwarded as-is to mooncake's "
            "setup_internal(ConfigDict).\n"
            "When protocol=rdma, LMCache must provide "
            "a valid L1 memory descriptor for "
            "preregistration.\n"
            "Refer to mooncake documentation for "
            "available setup keys.\n"
            "- num_workers (int): C++ worker threads "
            "(default 4, >0). Used for ops not in "
            "per_op_workers.\n"
            "- per_op_workers (dict[str, int]): Optional "
            "dict mapping lane keys to dedicated worker "
            "counts. Valid keys: lookup, retrieve, "
            "store, delete. Ops not mentioned use the "
            "shared num_workers pool.\n"
            "- process (dict, optional): Enable isolated SDK workers and the "
            "admission guard. Requires Linux, named shared L1, no lazy allocator "
            "and no serde. Nested guard config controls probe/admission policy."
        )


def _create_mooncake_store_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "L1MemoryDesc | None" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    C++ Mooncake Store connector.

    When ``config.setup_config["protocol"] == "rdma"``,
    a valid ``l1_memory_desc`` must be provided so the
    native Mooncake client can preregister the L1 memory
    region for RDMA access.

    Raises:
        RuntimeError: If the native C++ Mooncake extension
            is unavailable.
        ValueError: If RDMA protocol is requested but
            ``l1_memory_desc`` is missing or invalid.
    """
    if not isinstance(config, MooncakeStoreL2AdapterConfig):
        raise ValueError(f"Expected MooncakeStoreL2AdapterConfig, got {type(config)}")
    if config.process_config is not None:
        if config.serde_config is not None:
            raise ValueError("Process Mooncake does not support serde buffers")
        return create_process_adapter(
            config.setup_config,
            config.num_workers,
            config.per_op_workers,
            l1_memory_desc,
            config.process_config,
            config.guard_config,
        )
    try:
        # First Party
        from lmcache.lmcache_mooncake import (
            L1RegistrationConfig,
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

    if not isinstance(config, MooncakeStoreL2AdapterConfig):
        raise ValueError(f"Expected MooncakeStoreL2AdapterConfig, got {type(config)}")
    l1_registration = L1RegistrationConfig()
    if config.setup_config.get("protocol") == "rdma":
        if l1_memory_desc is None:
            raise ValueError(
                "RDMA protocol is enabled, but no L1 memory descriptor "
                "was provided; cannot create Mooncake Store L2 adapter."
            )
        elif l1_memory_desc.ptr == 0 or l1_memory_desc.size <= 0:
            raise ValueError(
                "RDMA protocol is enabled, but the L1 memory descriptor "
                "is invalid (ptr=%d, size=%d); cannot create Mooncake Store L2 adapter."
                % (l1_memory_desc.ptr, l1_memory_desc.size)
            )
        else:
            l1_registration.enabled = True
            l1_registration.base = l1_memory_desc.ptr
            l1_registration.size = l1_memory_desc.size

    native_client = LMCacheMooncakeClient(
        config=config.setup_config,
        num_workers=config.num_workers,
        l1_registration=l1_registration,
        per_op_workers=config.per_op_workers,
    )
    logger.info(
        "Created Mooncake Store L2 adapter "
        "(workers=%d, per_op_workers=%s, preregister_l1_memory=%s)",
        config.num_workers,
        config.per_op_workers,
        l1_registration.enabled and l1_registration.size > 0,
    )
    return NativeConnectorL2Adapter(
        native_client, MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE)
    )


# Self-register config type and adapter factory
register_l2_adapter_type("mooncake_store", MooncakeStoreL2AdapterConfig)
register_l2_adapter_factory("mooncake_store", _create_mooncake_store_l2_adapter)
