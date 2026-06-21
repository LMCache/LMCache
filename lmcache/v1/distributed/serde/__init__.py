# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.distributed.serde.async_processor import AsyncSerdeProcessor
from lmcache.v1.distributed.serde.base import (
    Deserializer,
    SerdeConfig,
    SerdeProcessor,
    SerdeTaskId,
    Serializer,
)
from lmcache.v1.distributed.serde.factory import (
    create_serde_processor,
    get_registered_serde_types,
    register_serde_factory,
)
from lmcache.v1.distributed.serde.fp8 import (
    Fp8QuantizationDeserializer,
    Fp8QuantizationSerializer,
)
from lmcache.v1.distributed.serde.multi import (
    LayoutDescGroup,
    MemoryObjGroup,
    MultiDeserializer,
    MultiSerializer,
    single_to_multi_deserializer,
    single_to_multi_serializer,
    validate_group_size,
)
from lmcache.v1.distributed.serde.utils import (
    make_temp_key,
    serialized_layout_desc,
)


def _create_cachegen_serde(kwargs: dict[str, object]) -> SerdeProcessor:
    # First Party
    from lmcache.v1.distributed.serde.cachegen import (  # noqa: PLC0415
        _create_cachegen_serde as create_cachegen_serde,
    )

    return create_cachegen_serde(kwargs)


register_serde_factory("cachegen", _create_cachegen_serde)


def __getattr__(name: str) -> object:
    if name in {"CacheGenMpDeserializer", "CacheGenMpSerializer"}:
        # First Party
        from lmcache.v1.distributed.serde.cachegen import (  # noqa: PLC0415
            CacheGenMpDeserializer,
            CacheGenMpSerializer,
        )

        globals()["CacheGenMpDeserializer"] = CacheGenMpDeserializer
        globals()["CacheGenMpSerializer"] = CacheGenMpSerializer
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AsyncSerdeProcessor",
    "CacheGenMpDeserializer",
    "CacheGenMpSerializer",
    "Deserializer",
    "Fp8QuantizationDeserializer",
    "Fp8QuantizationSerializer",
    "LayoutDescGroup",
    "MemoryObjGroup",
    "MultiDeserializer",
    "MultiSerializer",
    "SerdeConfig",
    "SerdeProcessor",
    "SerdeTaskId",
    "Serializer",
    "create_serde_processor",
    "get_registered_serde_types",
    "make_temp_key",
    "register_serde_factory",
    "serialized_layout_desc",
    "single_to_multi_deserializer",
    "single_to_multi_serializer",
    "validate_group_size",
]
