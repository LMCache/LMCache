# SPDX-License-Identifier: Apache-2.0
"""Python value codecs for the multiprocess gRPC protocol.

The protobuf descriptor is the source of truth for services, methods, and wire
messages.  This module only records how those protobuf messages map to LMCache's
Python domain objects and how each method should be scheduled on the server.
"""

# Standard
from dataclasses import dataclass, fields, is_dataclass
from typing import (
    Any,
    Callable,
    Union,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)
import enum
import pickle
import types

# Third Party
import msgspec
import torch

# First Party
from lmcache.utils import EngineType
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    IPCCacheServerKey,
    KVCache,
    RegisterEngineDrivenContextPayload,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.protocol import (
    RPC_METHODS,
    HandlerType,
    RpcMethod,
    coerce_rpc_method,
)
from lmcache.v1.multiprocess.protocols.engine import (
    PrepareRetrieveResponse,
    PrepareStoreResponse,
    RegisterEngineDrivenContextResponse,
)
from lmcache.v1.multiprocess.transport.grpc_impl._proto_gen import (
    lmcache_mq_pb2 as _pb2_typed,
)

# Generated protobuf classes are dynamic and opaque to static analysis.
lmcache_mq_pb2: Any = _pb2_typed

_NONE_TYPE = type(None)


@dataclass(frozen=True)
class _PythonRpcContract:
    """Python-side value and scheduling contract for one protobuf method."""

    payload_types: tuple[Any, ...]
    response_type: Any
    handler_type: HandlerType
    requires_client_affinity: bool = False


_PYTHON_RPC_CONTRACTS: dict[str, _PythonRpcContract] = {
    "RegisterKvCache": _PythonRpcContract(
        (
            int,
            KVCache,
            str,
            int,
            EngineType,
            LayoutHints,
            list[EngineGroupInfo],
        ),
        None,
        HandlerType.SYNC,
    ),
    "RegisterQCache": _PythonRpcContract(
        (
            int,
            KVCache,
            str,
            int,
            EngineType,
            LayoutHints,
            list[EngineGroupInfo],
        ),
        None,
        HandlerType.SYNC,
    ),
    "UnregisterKvCache": _PythonRpcContract((int,), None, HandlerType.SYNC),
    "UnregisterQCache": _PythonRpcContract((int,), None, HandlerType.SYNC),
    "StoreQ": _PythonRpcContract(
        (IPCCacheServerKey, int, list[list[int]], bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "Store": _PythonRpcContract(
        (IPCCacheServerKey, int, list[list[int]], bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "Retrieve": _PythonRpcContract(
        (IPCCacheServerKey, int, list[list[int]], bytes, int),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "Lookup": _PythonRpcContract(
        (IPCCacheServerKey, int),
        None,
        HandlerType.BLOCKING,
    ),
    "QueryPrefetchStatus": _PythonRpcContract(
        (str,),
        int | None,
        HandlerType.BLOCKING,
    ),
    "WaitPrefetchStatus": _PythonRpcContract(
        (str, float),
        int | None,
        HandlerType.BLOCKING,
    ),
    "QueryPrefetchLookupHits": _PythonRpcContract(
        (str,),
        int | None,
        HandlerType.BLOCKING,
    ),
    "FreeLookupLocks": _PythonRpcContract(
        (IPCCacheServerKey, int),
        None,
        HandlerType.BLOCKING,
    ),
    "EndSession": _PythonRpcContract((str,), None, HandlerType.BLOCKING),
    "RegisterKvCacheEngineDrivenContext": _PythonRpcContract(
        (RegisterEngineDrivenContextPayload,),
        RegisterEngineDrivenContextResponse,
        HandlerType.SYNC,
    ),
    "UnregisterKvCacheEngineDrivenContext": _PythonRpcContract(
        (int,),
        None,
        HandlerType.SYNC,
    ),
    "PrepareStore": _PythonRpcContract(
        (IPCCacheServerKey, int),
        PrepareStoreResponse,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CommitStore": _PythonRpcContract(
        (IPCCacheServerKey, int, bytes),
        bool,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "PrepareRetrieve": _PythonRpcContract(
        (IPCCacheServerKey, int),
        PrepareRetrieveResponse,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CommitRetrieve": _PythonRpcContract(
        (IPCCacheServerKey, int),
        bool,
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "Clear": _PythonRpcContract((), None, HandlerType.BLOCKING),
    "GetChunkSize": _PythonRpcContract((), int, HandlerType.SYNC),
    "GetExperimental": _PythonRpcContract((), list[str], HandlerType.SYNC),
    "Ping": _PythonRpcContract((int | None,), bool, HandlerType.BLOCKING),
    "ReportBlockAllocation": _PythonRpcContract(
        (int, str, list[BlockAllocationRecord]),
        None,
        HandlerType.BLOCKING,
    ),
    "Noop": _PythonRpcContract((), str, HandlerType.SYNC),
    "CbRegisterKvCache": _PythonRpcContract(
        (int, KVCache, str, int),
        None,
        HandlerType.SYNC,
    ),
    "CbUnregisterKvCache": _PythonRpcContract((int,), None, HandlerType.SYNC),
    "CbStorePreComputed": _PythonRpcContract(
        (IPCCacheServerKey, int, int, bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CbLookupPreComputed": _PythonRpcContract(
        (IPCCacheServerKey,),
        list[tuple[int, int]],
        HandlerType.BLOCKING,
    ),
    "CbRetrievePreComputed": _PythonRpcContract(
        (IPCCacheServerKey, list[tuple[int, int]], int, int, bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CbStoreFinal": _PythonRpcContract(
        (IPCCacheServerKey, int, int, bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CbLookupPreComputedV2": _PythonRpcContract(
        (IPCCacheServerKey,),
        list[CBMatchResult],
        HandlerType.BLOCKING,
    ),
    "CbRetrievePreComputedV2": _PythonRpcContract(
        (IPCCacheServerKey, list[CBMatchResult], int, int, bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CbRegisterRopeV3": _PythonRpcContract(
        (int, list[DeviceIPCWrapper], int, bool, list[int], list[list[int]]),
        None,
        HandlerType.SYNC,
    ),
    "CbUnregisterRopeV3": _PythonRpcContract((int,), None, HandlerType.SYNC),
    "CbRetrievePreComputedV3": _PythonRpcContract(
        (IPCCacheServerKey, list[CBMatchResult], list[list[int]], int, bytes),
        tuple[bytes, bool],
        HandlerType.BLOCKING,
        requires_client_affinity=True,
    ),
    "CbUnifiedLookup": _PythonRpcContract(
        (IPCCacheServerKey, int),
        CBUnifiedLookupResult | None,
        HandlerType.BLOCKING,
    ),
    "P2PLookupAndLock": _PythonRpcContract(
        (list[ObjectKey], dict[int, MemoryLayoutDesc]),
        int,
        HandlerType.BLOCKING,
    ),
    "P2PQueryLookupResults": _PythonRpcContract(
        (int,),
        list[TransferChannelAddress] | None,
        HandlerType.BLOCKING,
    ),
    "P2PUnlockObjects": _PythonRpcContract(
        (list[ObjectKey],),
        None,
        HandlerType.BLOCKING,
    ),
}


def request_type_to_method_name(request_type: RpcMethod | str) -> str:
    """Return the protobuf service method name for a request type.

    Args:
        request_type: Multiprocess protocol operation.

    Returns:
        The CamelCase method name used by ``MessageQueue``.
    """
    return str(coerce_rpc_method(request_type))


def _unwrap_optional(py_type: Any) -> tuple[Any, bool]:
    origin = get_origin(py_type)
    if origin not in (Union, types.UnionType):
        return py_type, False
    args = get_args(py_type)
    non_none = tuple(arg for arg in args if arg is not _NONE_TYPE)
    if len(non_none) == len(args):
        return py_type, False
    if len(non_none) != 1:
        raise TypeError(f"unsupported optional union {py_type!r}")
    return non_none[0], True


def _is_enum_type(py_type: Any) -> bool:
    return isinstance(py_type, type) and issubclass(py_type, enum.Enum)


def _is_device_wrapper_type(py_type: Any) -> bool:
    return isinstance(py_type, type) and issubclass(py_type, DeviceIPCWrapper)


def _sequence_type(py_type: Any) -> tuple[Any, bool] | None:
    origin = get_origin(py_type)
    args = get_args(py_type)
    if origin is list:
        return (args[0] if args else Any), False
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        return args[0], True
    return None


def _fixed_tuple_types(py_type: Any) -> tuple[Any, ...] | None:
    if get_origin(py_type) is not tuple:
        return None
    args = get_args(py_type)
    if len(args) == 2 and args[1] is Ellipsis:
        return None
    return args


def _structured_fields(py_type: Any) -> tuple[tuple[str, Any], ...] | None:
    py_type, _ = _unwrap_optional(py_type)
    if not isinstance(py_type, type) or is_typeddict(py_type):
        return None

    hints = get_type_hints(py_type)
    if is_dataclass(py_type):
        return tuple(
            (field.name, hints.get(field.name, Any))
            for field in fields(py_type)
            if field.init
        )
    if issubclass(py_type, msgspec.Struct):
        return tuple((name, hints.get(name, Any)) for name in py_type.__struct_fields__)
    return None


def _is_map_field(field: Any) -> bool:
    return bool(
        field.is_repeated
        and field.message_type is not None
        and field.message_type.GetOptions().map_entry
    )


def _pickle_mapping(value: Any) -> bytes:
    if not value:
        return b""
    return pickle.dumps(dict(value))


def _unpickle_mapping(value: bytes) -> dict[Any, Any]:
    if not value:
        return {}
    decoded = pickle.loads(value)
    if not isinstance(decoded, dict):
        raise TypeError(f"expected a pickled dict, got {type(decoded)!r}")
    return decoded


ValueEncoder = Callable[[Any], Any]
ValueDecoder = Callable[[Any], Any]
FieldWriter = Callable[[Any, Any], None]
FieldReader = Callable[[Any], Any]
MessageWriter = Callable[[Any, Any], None]
MessageReader = Callable[[Any], Any]
RequestEncoder = Callable[..., Any]
RequestDecoder = Callable[[Any], tuple[Any, ...]]
ResponseEncoder = Callable[[Any], Any]
ResponseDecoder = Callable[[Any], Any]


def _identity(value: Any) -> Any:
    return value


def _compile_scalar_codec(
    field: Any, py_type: Any
) -> tuple[ValueEncoder, ValueDecoder]:
    if field.type == field.TYPE_BYTES:
        if py_type is bytes:
            return bytes, bytes
        if py_type is dict or get_origin(py_type) is dict or is_typeddict(py_type):
            return _pickle_mapping, _unpickle_mapping
    if field.type == field.TYPE_STRING:
        if _is_enum_type(py_type):

            def encode_enum(value: Any) -> str:
                return str(value.value)

            return encode_enum, py_type
        if py_type is torch.dtype:

            def encode_dtype(value: torch.dtype) -> str:
                return str(value).removeprefix("torch.")

            def decode_dtype(value: str) -> torch.dtype:
                dtype = getattr(torch, value, None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError(f"unknown torch dtype name: {value!r}")
                return dtype

            return encode_dtype, decode_dtype
    if py_type in (bool, int, float, str, bytes):
        return py_type, py_type
    return _identity, _identity


def _compile_map_field(field: Any, py_type: Any) -> tuple[FieldWriter, FieldReader]:
    args = get_args(py_type)
    key_type, value_type = args if len(args) == 2 else (Any, Any)
    key_field = field.message_type.fields_by_name["key"]
    value_field = field.message_type.fields_by_name["value"]
    encode_key, decode_key = _compile_scalar_codec(key_field, key_type)
    field_name = field.name

    if value_field.message_type is None:
        encode_value, decode_value = _compile_scalar_codec(value_field, value_type)

        def write_map(message: Any, value: Any) -> None:
            container = getattr(message, field_name)
            for key, item in value.items():
                container[encode_key(key)] = encode_value(item)

        def read_map(message: Any) -> dict[Any, Any]:
            return {
                decode_key(key): decode_value(item)
                for key, item in getattr(message, field_name).items()
            }

        return write_map, read_map

    write_value, read_value = _compile_message_codec(
        value_field.message_type, value_type
    )

    def write_message_map(message: Any, value: Any) -> None:
        container = getattr(message, field_name)
        for key, item in value.items():
            write_value(container[encode_key(key)], item)

    def read_message_map(message: Any) -> dict[Any, Any]:
        return {
            decode_key(key): read_value(item)
            for key, item in getattr(message, field_name).items()
        }

    return write_message_map, read_message_map


def _compile_field_codec(field: Any, py_type: Any) -> tuple[FieldWriter, FieldReader]:
    py_type, optional = _unwrap_optional(py_type)
    if optional and (field.is_repeated or not field.has_presence):
        raise TypeError(
            f"field {field.full_name} cannot represent Python None; "
            "declare it optional in the proto"
        )

    if _is_map_field(field):
        writer, reader = _compile_map_field(field, py_type)
    elif field.is_repeated:
        sequence = _sequence_type(py_type)
        if sequence is None:
            raise TypeError(
                f"field {field.full_name} is repeated but {py_type!r} is not"
            )
        item_type, as_tuple = sequence
        field_name = field.name
        if field.message_type is None:
            encode_item, decode_item = _compile_scalar_codec(field, item_type)

            def write_repeated(message: Any, value: Any) -> None:
                getattr(message, field_name).extend(encode_item(item) for item in value)

            def read_repeated(message: Any) -> Any:
                decoded = [decode_item(item) for item in getattr(message, field_name)]
                return tuple(decoded) if as_tuple else decoded

            writer, reader = write_repeated, read_repeated
        else:
            write_item, read_item = _compile_message_codec(
                field.message_type, item_type
            )

            def write_repeated_message(message: Any, value: Any) -> None:
                container = getattr(message, field_name)
                for item in value:
                    write_item(container.add(), item)

            def read_repeated_message(message: Any) -> Any:
                decoded = [read_item(item) for item in getattr(message, field_name)]
                return tuple(decoded) if as_tuple else decoded

            writer, reader = write_repeated_message, read_repeated_message
    elif field.message_type is not None:
        write_child, read_child = _compile_message_codec(field.message_type, py_type)
        field_name = field.name

        def write_message(message: Any, value: Any) -> None:
            child = getattr(message, field_name)
            write_child(child, value)
            child.SetInParent()

        def read_message(message: Any) -> Any:
            return read_child(getattr(message, field_name))

        writer, reader = write_message, read_message
    else:
        encode_value, decode_value = _compile_scalar_codec(field, py_type)
        field_name = field.name

        def write_scalar(message: Any, value: Any) -> None:
            setattr(message, field_name, encode_value(value))

        def read_scalar(message: Any) -> Any:
            return decode_value(getattr(message, field_name))

        writer, reader = write_scalar, read_scalar

    if not optional:
        return writer, reader

    field_name = field.name

    def write_optional(message: Any, value: Any) -> None:
        if value is not None:
            writer(message, value)

    def read_optional(message: Any) -> Any:
        if not message.HasField(field_name):
            return None
        return reader(message)

    return write_optional, read_optional


def _compile_message_codec(
    descriptor: Any, py_type: Any
) -> tuple[MessageWriter, MessageReader]:
    py_type, _ = _unwrap_optional(py_type)
    proto_fields = tuple(descriptor.fields)

    if _is_device_wrapper_type(py_type):

        def write_wrapper(message: Any, value: DeviceIPCWrapper) -> None:
            message.pickled_payload = DeviceIPCWrapper.Serialize(value)

        def read_wrapper(message: Any) -> DeviceIPCWrapper:
            return DeviceIPCWrapper.Deserialize(message.pickled_payload)

        return write_wrapper, read_wrapper

    if py_type is torch.Size:

        def write_size(message: Any, value: torch.Size) -> None:
            message.dims.extend(value)

        def read_size(message: Any) -> torch.Size:
            return torch.Size(message.dims)

        return write_size, read_size

    sequence = _sequence_type(py_type)
    if sequence is not None and len(proto_fields) == 1:
        return _compile_field_codec(proto_fields[0], py_type)

    tuple_types = _fixed_tuple_types(py_type)
    if tuple_types is not None:
        if len(tuple_types) != len(proto_fields):
            raise TypeError(
                f"{py_type!r} has {len(tuple_types)} values but "
                f"{descriptor.full_name} has {len(proto_fields)} fields"
            )
        tuple_codecs = tuple(
            _compile_field_codec(field, item_type)
            for field, item_type in zip(proto_fields, tuple_types, strict=True)
        )

        def write_tuple(message: Any, value: Any) -> None:
            for item, (writer, _) in zip(value, tuple_codecs, strict=True):
                writer(message, item)

        def read_tuple(message: Any) -> tuple[Any, ...]:
            return tuple(reader(message) for _, reader in tuple_codecs)

        return write_tuple, read_tuple

    py_fields = _structured_fields(py_type)
    if py_fields is None or len(py_fields) != len(proto_fields):
        raise TypeError(
            f"no structural codec from {py_type!r} to {descriptor.full_name}"
        )

    struct_codecs: list[tuple[str, FieldWriter, FieldReader]] = []
    for (name, field_type), field in zip(py_fields, proto_fields, strict=True):
        if field.name not in (name, f"pickled_{name}"):
            raise TypeError(
                f"{py_type.__name__}.{name} does not match "
                f"{descriptor.full_name}.{field.name}"
            )
        writer, reader = _compile_field_codec(field, field_type)
        struct_codecs.append((name, writer, reader))

    def write_struct(message: Any, value: Any) -> None:
        for name, writer, _ in struct_codecs:
            writer(message, getattr(value, name))

    def read_struct(message: Any) -> Any:
        return py_type(**{name: reader(message) for name, _, reader in struct_codecs})

    return write_struct, read_struct


def _compile_scalar_message_codec(
    message_cls: Any, field: Any, py_type: Any
) -> tuple[ResponseEncoder, ValueDecoder]:
    py_type, optional = _unwrap_optional(py_type)
    if optional and not field.has_presence:
        raise TypeError(
            f"field {field.full_name} cannot represent Python None; "
            "declare it optional in the proto"
        )
    encode_value, decode_value = _compile_scalar_codec(field, py_type)
    field_name = field.name

    if optional:

        def encode_optional(value: Any) -> Any:
            message = message_cls()
            if value is not None:
                setattr(message, field_name, encode_value(value))
            return message

        def decode_optional(message: Any) -> Any:
            if not message.HasField(field_name):
                return None
            return decode_value(getattr(message, field_name))

        return encode_optional, decode_optional

    def encode_scalar(value: Any) -> Any:
        message = message_cls()
        setattr(message, field_name, encode_value(value))
        return message

    def decode_scalar(message: Any) -> Any:
        return decode_value(getattr(message, field_name))

    return encode_scalar, decode_scalar


def _compile_request_codec(
    message_cls: Any, payload_types: tuple[Any, ...]
) -> tuple[RequestEncoder, RequestDecoder]:
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)
    if (
        len(payload_types) == 1
        and len(proto_fields) == 1
        and not proto_fields[0].is_repeated
        and proto_fields[0].message_type is None
    ):
        encoder, reader = _compile_scalar_message_codec(
            message_cls, proto_fields[0], payload_types[0]
        )

        def decode_scalar_request(message: Any) -> tuple[Any, ...]:
            return (reader(message),)

        return encoder, decode_scalar_request

    if len(payload_types) == len(proto_fields):
        codecs = tuple(
            _compile_field_codec(field, py_type)
            for field, py_type in zip(proto_fields, payload_types, strict=True)
        )
        if not codecs:
            return (lambda: message_cls()), (lambda message: ())
        if len(codecs) == 1:
            writer, reader = codecs[0]

            def encode_one(value: Any) -> Any:
                message = message_cls()
                writer(message, value)
                return message

            def decode_one(message: Any) -> tuple[Any, ...]:
                return (reader(message),)

            return encode_one, decode_one

        def encode_many(*payloads: Any) -> Any:
            if len(payloads) != len(codecs):
                raise TypeError(
                    f"{message_cls.DESCRIPTOR.full_name} expects "
                    f"{len(codecs)} payloads, got {len(payloads)}"
                )
            message = message_cls()
            for value, (writer, _) in zip(payloads, codecs, strict=True):
                writer(message, value)
            return message

        def decode_many(message: Any) -> tuple[Any, ...]:
            return tuple(reader(message) for _, reader in codecs)

        return encode_many, decode_many

    if len(payload_types) == 1:
        write_message, read_message = _compile_message_codec(
            message_cls.DESCRIPTOR, payload_types[0]
        )

        def encode_flat(value: Any) -> Any:
            message = message_cls()
            write_message(message, value)
            return message

        def decode_flat(message: Any) -> tuple[Any, ...]:
            return (read_message(message),)

        return encode_flat, decode_flat

    raise TypeError(
        f"protocol has {len(payload_types)} payloads but "
        f"{message_cls.DESCRIPTOR.full_name} has {len(proto_fields)} fields"
    )


def _compile_response_codec(
    message_cls: Any, response_type: Any
) -> tuple[ResponseEncoder, ResponseDecoder]:
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)
    if response_type is None:
        if proto_fields:
            raise TypeError(
                f"{message_cls.DESCRIPTOR.full_name} must be an empty response"
            )
        return (lambda result: message_cls()), (lambda response: None)

    py_fields = _structured_fields(response_type)
    if py_fields is not None and len(py_fields) == len(proto_fields):
        writer, reader = _compile_message_codec(message_cls.DESCRIPTOR, response_type)

        def encode_struct(result: Any) -> Any:
            message = message_cls()
            writer(message, result)
            return message

        return encode_struct, reader

    if (
        len(proto_fields) == 1
        and not proto_fields[0].is_repeated
        and proto_fields[0].message_type is None
    ):
        return _compile_scalar_message_codec(
            message_cls, proto_fields[0], response_type
        )

    if len(proto_fields) != 1:
        raise TypeError(
            f"response type {response_type!r} does not match "
            f"{message_cls.DESCRIPTOR.full_name}"
        )
    writer, reader = _compile_field_codec(proto_fields[0], response_type)

    def encode_one(result: Any) -> Any:
        message = message_cls()
        writer(message, result)
        return message

    return encode_one, reader


@dataclass(frozen=True)
class TypedRpcSpec:
    """A descriptor-derived binding between one RPC and its Python contract."""

    rpc_method: RpcMethod
    service_name: str
    method_name: str
    request_message: Any
    response_message: Any
    payload_types: tuple[Any, ...]
    response_type: Any
    handler_type: HandlerType
    requires_client_affinity: bool
    python_to_request: RequestEncoder
    request_to_python: RequestDecoder
    python_to_response: ResponseEncoder
    response_to_python: ResponseDecoder

    @property
    def request_type(self) -> RpcMethod:
        """Backward-compatible alias for older tests and helper code."""
        return self.rpc_method


def _build_typed_rpcs() -> dict[RpcMethod, TypedRpcSpec]:
    expected_methods = {str(item) for item in RPC_METHODS}
    service_methods: dict[str, Any] = {}
    for service in lmcache_mq_pb2.DESCRIPTOR.services_by_name.values():
        for method in service.methods:
            if method.name in service_methods:
                raise RuntimeError(f"Duplicate gRPC method name: {method.name}")
            service_methods[method.name] = method

    actual_methods = set(service_methods)
    contract_methods = set(_PYTHON_RPC_CONTRACTS)
    if actual_methods != expected_methods or actual_methods != contract_methods:
        missing = sorted(expected_methods - actual_methods)
        extra = sorted(actual_methods - expected_methods)
        missing_contracts = sorted(actual_methods - contract_methods)
        extra_contracts = sorted(contract_methods - actual_methods)
        raise RuntimeError(
            "gRPC service/RpcMethod/codec mismatch: "
            f"missing={missing}, extra={extra}, "
            f"missing_contracts={missing_contracts}, extra_contracts={extra_contracts}"
        )

    specs: dict[RpcMethod, TypedRpcSpec] = {}
    for rpc_method in RPC_METHODS:
        method_name = str(rpc_method)
        method = service_methods[method_name]
        request_message = getattr(lmcache_mq_pb2, method.input_type.name)
        response_message = getattr(lmcache_mq_pb2, method.output_type.name)
        contract = _PYTHON_RPC_CONTRACTS[method_name]
        request_encoder, request_decoder = _compile_request_codec(
            request_message, contract.payload_types
        )
        response_encoder, response_decoder = _compile_response_codec(
            response_message, contract.response_type
        )
        specs[rpc_method] = TypedRpcSpec(
            rpc_method=rpc_method,
            service_name=rpc_method.service_name,
            method_name=method_name,
            request_message=request_message,
            response_message=response_message,
            payload_types=contract.payload_types,
            response_type=contract.response_type,
            handler_type=contract.handler_type,
            requires_client_affinity=contract.requires_client_affinity,
            python_to_request=request_encoder,
            request_to_python=request_decoder,
            python_to_response=response_encoder,
            response_to_python=response_decoder,
        )
    return specs


# Generated once from protobuf descriptors plus Python value contracts.
TYPED_RPCS = _build_typed_rpcs()


# These serializers remain part of ``multiprocess.mq``'s compatibility API.
_SPECIAL_ENCODER_DECODERS = {
    DeviceIPCWrapper: (
        get_customized_encoder(DeviceIPCWrapper),
        get_customized_decoder(DeviceIPCWrapper),
    ),
    list[DeviceIPCWrapper]: (
        get_customized_encoder(list[DeviceIPCWrapper]),
        get_customized_decoder(list[DeviceIPCWrapper]),
    ),
    MemoryLayoutDesc: (
        get_customized_encoder(MemoryLayoutDesc),
        get_customized_decoder(MemoryLayoutDesc),
    ),
    dict[int, MemoryLayoutDesc]: (
        get_customized_encoder(dict[int, MemoryLayoutDesc]),
        get_customized_decoder(dict[int, MemoryLayoutDesc]),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    """Encode a value with the public msgspec utility serializers."""
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    """Decode a value with the public msgspec utility serializers."""
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)


__all__ = [
    "TYPED_RPCS",
    "TypedRpcSpec",
    "msgspec_decode",
    "msgspec_encode",
    "request_type_to_method_name",
]
