# SPDX-License-Identifier: Apache-2.0
"""Factory registration for accelerated KV compression serde adapters.

Registers the "accel_kv_compress" serde type with the LMCache serde factory.
This module is imported by the distributed package __init__ to ensure
registration happens at startup.

Integration via SerdeL2AdapterWrapper: registered as serde type
"accel_kv_compress", creates AsyncSerdeProcessor for serialize/deserialize.
The wrapper can be paired with any L2AdapterInterface (DramL2Adapter for
in-DRAM compression, SSD adapter for offload, etc.).

Config example:
    {
        "serde": {
            "type": "accel_kv_compress",
            "backend": "qat",
            "lib_path": "/path/to/libkvclip_qzip.so",
            "byte_reorder": true,
            "truncate_bits": 2,
            "element_size": 2,
            "max_workers": 4
        }
    }
"""

from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend
from lmcache.v1.distributed.compress_adapters.serde import (
    AccelCompressDeserializer,
    AccelCompressSerializer,
)
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import SerdeProcessor
from lmcache.v1.distributed.serde.factory import register_serde_factory


def _create_backend(kwargs: dict[str, object]) -> AccelCompressBackend:
    """Instantiate the compression backend from config kwargs."""
    backend_name = str(kwargs.get("backend", "qat")).lower()

    if backend_name == "qat":
        from lmcache.v1.distributed.compress_adapters.qat_backend import (
            QatBackend,
        )

        lib_path = kwargs.get("lib_path")
        return QatBackend(lib_path=lib_path)
    else:
        raise ValueError(
            f"Unknown accel_kv_compress backend: {backend_name!r}. "
            f"Supported: 'qat'"
        )


def _create_accel_kv_compress(kwargs: dict[str, object]) -> SerdeProcessor:
    """Factory function for the 'accel_kv_compress' serde type."""
    backend = _create_backend(kwargs)

    byte_reorder = bool(kwargs.get("byte_reorder", False))
    truncate_bits = int(kwargs.get("truncate_bits", 0))
    element_size = int(kwargs.get("element_size", 2))
    max_workers = int(kwargs.get("max_workers", 1))

    serializer = AccelCompressSerializer(
        backend=backend,
        byte_reorder=byte_reorder,
        truncate_bits=truncate_bits,
        element_size=element_size,
    )
    deserializer = AccelCompressDeserializer(
        backend=backend,
        byte_reorder=byte_reorder,
        element_size=element_size,
    )

    return AsyncSerdeProcessor(
        serializer=serializer,
        deserializer=deserializer,
        max_workers=max_workers,
    )


# Register serde factory on import
register_serde_factory("accel_kv_compress", _create_accel_kv_compress)
