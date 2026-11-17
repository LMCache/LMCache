# SPDX-License-Identifier: Apache-2.0
"""Factory registration for accelerated KV compression serde adapters.

Registers the "accel_kv_compress" serde type with the LMCache serde factory.
This module is imported by the distributed package __init__ to ensure
registration happens at startup.

Integration via SerdeL2AdapterWrapper: registered as serde type
"accel_kv_compress", creates AccelCompressSerdeProcessor for
serialize/deserialize.
The wrapper can be paired with any L2AdapterInterface (DramL2Adapter for
in-DRAM compression, SSD adapter for offload, etc.).

Config example:
    {
        "serde": {
            "type": "accel_kv_compress",
            "backend": "iaxl",
            "dtype": "bfloat16",
            "num_groups": 1,
            "max_workers": 4,
            "pipelined_load": false
        }
    }

The ``iaxl`` backend (intel-accel-for-llm) compresses on QAT, IAA or CPU
workers. ``dtype`` names the KV element type; iaxl's ``IAXL_*`` environment
variables select data shuffling and lossy truncation.

``num_groups`` splits each object into that many independently decodable
groups. Under a layer-major layout, setting it to the number of layer groups
makes one group one N-layer slice, so a consumer can decompress and use layer
groups one at a time instead of waiting for the whole chunk.

``pipelined_load`` completes a load once the compressed bytes are in memory
and decompresses group by group in the background, so a layer-wise retrieve
can overlap decompression with its copies and the forward pass.
"""

# Third Party
import torch

# First Party
from lmcache.v1.distributed.compress_adapters.backend import AccelCompressBackend
from lmcache.v1.distributed.compress_adapters.serde import (
    AccelCompressDeserializer,
    AccelCompressSerdeProcessor,
    AccelCompressSerializer,
)
from lmcache.v1.distributed.serde.base import SerdeProcessor
from lmcache.v1.distributed.serde.factory import register_serde_factory


def _create_backend(kwargs: dict[str, object]) -> AccelCompressBackend:
    """Instantiate the compression backend from config kwargs."""
    backend_name = str(kwargs.get("backend", "iaxl")).lower()

    if backend_name == "iaxl":
        # First Party
        from lmcache.v1.distributed.compress_adapters.iaxl_backend import (
            IaxlBackend,
        )

        dtype = getattr(torch, str(kwargs.get("dtype", "bfloat16")), None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown dtype for backend 'iaxl': {kwargs['dtype']!r}")
        return IaxlBackend(dtype=dtype)
    else:
        raise ValueError(
            f"Unknown accel_kv_compress backend: {backend_name!r}. Supported: 'iaxl'"
        )


def _create_accel_kv_compress(kwargs: dict[str, object]) -> SerdeProcessor:
    """Factory function for the 'accel_kv_compress' serde type."""
    backend = _create_backend(kwargs)

    max_workers = int(kwargs.get("max_workers", 1))  # type: ignore[call-overload]
    num_groups = int(kwargs.get("num_groups", 1))  # type: ignore[call-overload]
    pipelined_load = kwargs.get("pipelined_load", False)
    if not isinstance(pipelined_load, bool):
        raise ValueError(f"pipelined_load must be a boolean, got {pipelined_load!r}")

    serializer = AccelCompressSerializer(backend=backend, num_groups=num_groups)
    deserializer = AccelCompressDeserializer(backend=backend)

    return AccelCompressSerdeProcessor(
        serializer=serializer,
        deserializer=deserializer,
        max_workers=max_workers,
        pipelined_load=pipelined_load,
    )


# Register serde factory on import
register_serde_factory("accel_kv_compress", _create_accel_kv_compress)
