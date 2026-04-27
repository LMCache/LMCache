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
_LMCACHE_ONLY_KEYS = {
    "type",
    "num_workers",
    "eviction",
    "lookup_workers",
    "retrieve_workers",
    "store_workers",
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
            must be > 0).  Ignored when per-operation
            worker counts are set.
        lookup_workers: Optional dedicated worker count
            for EXISTS operations.  Must be set together
            with ``retrieve_workers`` and
            ``store_workers``.
        retrieve_workers: Optional dedicated worker count
            for GET/load operations.  Must be set together
            with ``lookup_workers`` and
            ``store_workers``.
        store_workers: Optional dedicated worker count for
            SET/put operations.  Must be set together with
            ``lookup_workers`` and ``retrieve_workers``.
    """

    def __init__(
        self,
        setup_config: Dict[str, str],
        num_workers: int = 4,
        lookup_workers: Optional[int] = None,
        retrieve_workers: Optional[int] = None,
        store_workers: Optional[int] = None,
    ):
        super().__init__()
        self.num_workers = self._validate_num_workers(num_workers)
        self._validate_per_op_worker_counts(
            lookup_workers,
            retrieve_workers,
            store_workers,
        )
        self.setup_config: Dict[str, str] = dict(setup_config)
        self.lookup_workers = lookup_workers
        self.retrieve_workers = retrieve_workers
        self.store_workers = store_workers

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "MooncakeStoreL2AdapterConfig":
        num_workers: int = 4
        if "num_workers" in d:
            num_workers = cls._validate_num_workers(d["num_workers"])
        lookup_workers = cls._parse_optional_worker_count(d, "lookup_workers")
        retrieve_workers = cls._parse_optional_worker_count(d, "retrieve_workers")
        store_workers = cls._parse_optional_worker_count(d, "store_workers")
        cls._validate_per_op_worker_counts(
            lookup_workers,
            retrieve_workers,
            store_workers,
        )

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
            lookup_workers=lookup_workers,
            retrieve_workers=retrieve_workers,
            store_workers=store_workers,
        )

    @staticmethod
    def _parse_optional_worker_count(d: dict[str, object], key: str) -> Optional[int]:
        value = d.get(key)
        if value is None:
            return None
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
        return value

    @staticmethod
    def _validate_num_workers(raw: object) -> int:
        if not isinstance(raw, int) or raw <= 0:
            raise ValueError("num_workers must be a positive integer")
        return raw

    @staticmethod
    def _validate_per_op_worker_counts(
        lookup_workers: Optional[int],
        retrieve_workers: Optional[int],
        store_workers: Optional[int],
    ) -> None:
        values = {
            "lookup_workers": lookup_workers,
            "retrieve_workers": retrieve_workers,
            "store_workers": store_workers,
        }
        specified = [name for name, value in values.items() if value is not None]
        if not specified:
            return
        if len(specified) != len(values):
            raise ValueError(
                "lookup_workers, retrieve_workers, and store_workers must "
                "all be set together"
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
            "(default 4, >0)\n"
            "- lookup_workers (int): EXISTS worker threads (>0); must be set "
            "together with retrieve_workers and store_workers\n"
            "- retrieve_workers (int): GET/load worker threads (>0); must be "
            "set together with lookup_workers and store_workers\n"
            "- store_workers (int): SET/put worker threads (>0); must be set "
            "together with lookup_workers and retrieve_workers"
        )


def _create_mooncake_store_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
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

    native_client_kwargs = {
        "config": config.setup_config,
        "num_workers": config.num_workers,
        "l1_registration": l1_registration,
    }
    if (
        config.lookup_workers is not None
        or config.retrieve_workers is not None
        or config.store_workers is not None
    ):
        native_client_kwargs.update(
            {
                "lookup_workers": config.lookup_workers,
                "retrieve_workers": config.retrieve_workers,
                "store_workers": config.store_workers,
            }
        )

    native_client = LMCacheMooncakeClient(**native_client_kwargs)
    logger.info(
        "Created Mooncake Store L2 adapter "
        "(workers=%d, lookup_workers=%s, retrieve_workers=%s, "
        "store_workers=%s, preregister_l1_memory=%s)",
        config.num_workers,
        config.lookup_workers,
        config.retrieve_workers,
        config.store_workers,
        l1_registration.enabled and l1_registration.size > 0,
    )
    return NativeConnectorL2Adapter(native_client)


# Self-register config type and adapter factory
register_l2_adapter_type("mooncake_store", MooncakeStoreL2AdapterConfig)
register_l2_adapter_factory("mooncake_store", _create_mooncake_store_l2_adapter)
