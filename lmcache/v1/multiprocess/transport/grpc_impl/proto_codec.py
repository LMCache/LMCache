# SPDX-License-Identifier: Apache-2.0
"""Descriptor-driven protobuf helpers for the multiprocess gRPC transport.

This module intentionally has no per-RPC registry.  Request/response message
classes come from the generated protobuf descriptor, and Python value
conversion is derived from handler annotations plus protobuf field names.
Adding an RPC should not require editing this file.
"""

# Standard
from dataclasses import fields, is_dataclass
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
import inspect
import types

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.transfer_channel.api import TransferChannelAddress
from lmcache.v1.multiprocess.custom_types import (
    CBMatchResult,
    CBUnifiedLookupResult,
    DeviceIPCWrapper,
    get_customized_decoder,
    get_customized_encoder,
)
from lmcache.v1.multiprocess.protocol import (
    RPC_METHODS,
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

ValueEncoder = Callable[[Any], Any]
ValueDecoder = Callable[[Any], Any]
FieldWriter = Callable[[Any, Any], None]
FieldReader = Callable[[Any], Any]
RequestEncoder = Callable[..., Any]
RequestDecoder = Callable[[Any], tuple[Any, ...]]
ResponseEncoder = Callable[[Any], Any]
ResponseDecoder = Callable[[Any], Any]


def _build_service_methods() -> dict[str, tuple[str, Any]]:
    methods: dict[str, tuple[str, Any]] = {}
    for service in lmcache_mq_pb2.DESCRIPTOR.services_by_name.values():
        for method in service.methods:
            if method.name in methods:
                raise RuntimeError(f"Duplicate gRPC method name: {method.name}")
            methods[method.name] = (service.name, method)
    return methods


_SERVICE_METHODS = _build_service_methods()


def request_type_to_method_name(request_type: RpcMethod | str) -> str:
    """Return the protobuf service method name for an RPC method.

    Args:
        request_type: Multiprocess protocol operation.

    Returns:
        The CamelCase protobuf method name.
    """
    return str(coerce_rpc_method(request_type))


def get_request_message_class(request_type: RpcMethod | str) -> Any:
    """Return the generated protobuf request class for an RPC method."""
    method = _SERVICE_METHODS[str(coerce_rpc_method(request_type))][1]
    return getattr(lmcache_mq_pb2, method.input_type.name)


def get_response_message_class(request_type: RpcMethod | str) -> Any:
    """Return the generated protobuf response class for an RPC method."""
    method = _SERVICE_METHODS[str(coerce_rpc_method(request_type))][1]
    return getattr(lmcache_mq_pb2, method.output_type.name)


def get_service_name(request_type: RpcMethod | str) -> str:
    """Return the protobuf service name for an RPC method."""
    return _SERVICE_METHODS[str(coerce_rpc_method(request_type))][0]


def get_service_names() -> set[str]:
    """Return all generated protobuf service names."""
    return {
        service.name for service in lmcache_mq_pb2.DESCRIPTOR.services_by_name.values()
    }


def validate_protocol_descriptor() -> None:
    """Verify the descriptor-derived protocol method set is self-consistent."""
    descriptor_methods = set(_SERVICE_METHODS)
    protocol_methods = {str(method) for method in RPC_METHODS}
    if descriptor_methods != protocol_methods:
        missing = sorted(protocol_methods - descriptor_methods)
        extra = sorted(descriptor_methods - protocol_methods)
        raise RuntimeError(
            f"gRPC service/RpcMethod mismatch: missing={missing}, extra={extra}"
        )


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


def _encode_mapping(value: Any) -> bytes:
    if not value:
        return b""
    return msgspec.msgpack.encode(dict(value))


def _decode_mapping(value: bytes) -> dict[Any, Any]:
    if not value:
        return {}
    decoded = msgspec.msgpack.decode(value)
    if not isinstance(decoded, dict):
        raise TypeError(f"expected a msgpack dict, got {type(decoded)!r}")
    return decoded


def _identity(value: Any) -> Any:
    return value


def _compile_scalar_codec(
    field: Any, py_type: Any
) -> tuple[ValueEncoder, ValueDecoder]:
    if field.type == field.TYPE_BYTES:
        if py_type is bytes:
            return bytes, bytes
        if py_type is dict or get_origin(py_type) is dict or is_typeddict(py_type):
            return _encode_mapping, _decode_mapping
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
) -> tuple[FieldWriter, FieldReader]:
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
        if field.name not in (name, f"encoded_{name}"):
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


def _proto_has_same_descriptor(value: Any, descriptor: Any) -> bool:
    return hasattr(value, "DESCRIPTOR") and value.DESCRIPTOR is descriptor


def _field_source_name(field_name: str, value: Any) -> str:
    if hasattr(value, field_name):
        return field_name
    if field_name.startswith("encoded_") and hasattr(value, field_name[8:]):
        return field_name[8:]
    return field_name


def _runtime_py_type_for_value(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.__class__
    if isinstance(value, torch.dtype):
        return torch.dtype
    return type(value)


def _write_field_from_runtime_value(message: Any, field: Any, value: Any) -> None:
    if value is None:
        return
    if _is_map_field(field):
        key_field = field.message_type.fields_by_name["key"]
        value_field = field.message_type.fields_by_name["value"]
        container = getattr(message, field.name)
        for key, item in value.items():
            encoded_key, _ = _compile_scalar_codec(
                key_field, _runtime_py_type_for_value(key)
            )
            if value_field.message_type is None:
                encoded_value, _ = _compile_scalar_codec(
                    value_field, _runtime_py_type_for_value(item)
                )
                container[encoded_key(key)] = encoded_value(item)
            else:
                _write_message_from_runtime_value(
                    container[encoded_key(key)], value_field.message_type, item
                )
        return

    if field.is_repeated:
        container = getattr(message, field.name)
        if field.message_type is None:
            for item in value:
                encode_item, _ = _compile_scalar_codec(
                    field, _runtime_py_type_for_value(item)
                )
                container.append(encode_item(item))
        else:
            for item in value:
                _write_message_from_runtime_value(
                    container.add(), field.message_type, item
                )
        return

    if field.message_type is not None:
        child = getattr(message, field.name)
        _write_message_from_runtime_value(child, field.message_type, value)
        child.SetInParent()
        return

    encode_value, _ = _compile_scalar_codec(field, _runtime_py_type_for_value(value))
    setattr(message, field.name, encode_value(value))


def _write_message_from_runtime_value(
    message: Any,
    descriptor: Any,
    value: Any,
) -> None:
    if _proto_has_same_descriptor(value, descriptor):
        message.CopyFrom(value)
        return
    if descriptor.name == "DeviceIpcWrapper":
        message.pickled_payload = DeviceIPCWrapper.Serialize(value)
        return
    if descriptor.name == "TensorShape":
        message.dims.extend(value)
        return

    proto_fields = tuple(descriptor.fields)
    if isinstance(value, tuple) and len(value) == len(proto_fields):
        for item, field in zip(value, proto_fields, strict=True):
            _write_field_from_runtime_value(message, field, item)
        return
    if (
        len(proto_fields) == 1
        and not is_dataclass(value)
        and not isinstance(value, msgspec.Struct)
    ):
        _write_field_from_runtime_value(message, proto_fields[0], value)
        return

    for field in proto_fields:
        source_name = _field_source_name(field.name, value)
        if isinstance(value, dict):
            field_value = value.get(source_name)
        else:
            field_value = getattr(value, source_name)
        _write_field_from_runtime_value(message, field, field_value)


def _is_structured_runtime_value(value: Any, descriptor: Any) -> bool:
    return (
        _proto_has_same_descriptor(value, descriptor)
        or is_dataclass(value)
        or isinstance(value, (dict, msgspec.Struct))
        or (isinstance(value, tuple) and len(value) == len(tuple(descriptor.fields)))
    )


def _compile_request_codec(
    message_cls: Any,
    payload_types: tuple[Any, ...],
) -> tuple[RequestEncoder, RequestDecoder]:
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)
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
        f"handler has {len(payload_types)} payloads but "
        f"{message_cls.DESCRIPTOR.full_name} has {len(proto_fields)} fields"
    )


def _handler_params_and_payload_types(
    handler: Callable[..., Any],
) -> tuple[list[inspect.Parameter], tuple[Any, ...]]:
    sig = inspect.signature(handler)
    hints = get_type_hints(handler)
    params = [
        p
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    payload_types = tuple(hints.get(param.name, param.annotation) for param in params)
    return (
        params,
        tuple(
            Any if item is inspect.Signature.empty else item for item in payload_types
        ),
    )


def compile_request_decoder(
    message_cls: Any, handler: Callable[..., Any]
) -> tuple[RequestDecoder, tuple[Any, ...]]:
    """Compile a protobuf request decoder from a handler's annotations.

    Args:
        message_cls: Generated protobuf request class.
        handler: Bound Python RPC implementation method.

    Returns:
        Callable that decodes one protobuf request into handler positional args,
        plus the payload types inferred from the handler.
    """
    params, payload_types = _handler_params_and_payload_types(handler)
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)

    if not params:
        return (lambda _message: ()), payload_types

    if len(params) == len(proto_fields) or len(params) == 1:
        _encoder, decoder = _compile_request_codec(message_cls, payload_types)
        return decoder, payload_types

    fields_by_name = message_cls.DESCRIPTOR.fields_by_name
    selected_codecs: list[tuple[Any, FieldReader]] = []
    for param, py_type in zip(params, payload_types, strict=True):
        field = fields_by_name.get(param.name)
        if field is None:
            field = fields_by_name.get(f"encoded_{param.name}")
        if field is None:
            raise TypeError(
                f"{message_cls.DESCRIPTOR.full_name} has no field matching "
                f"handler parameter {param.name!r}"
            )
        _writer, reader = _compile_field_codec(field, py_type)
        selected_codecs.append((field, reader))

    def decode_subset(message: Any) -> tuple[Any, ...]:
        return tuple(reader(message) for _field, reader in selected_codecs)

    return decode_subset, payload_types


def compile_response_encoder(
    message_cls: Any,
    handler: Callable[..., Any],
) -> tuple[ResponseEncoder, Any]:
    """Compile a protobuf response encoder from a handler return annotation."""
    sig = inspect.signature(handler)
    hints = get_type_hints(handler)
    response_type = hints.get("return", sig.return_annotation)
    if response_type is inspect.Signature.empty:
        response_type = Any
    encoder = compile_response_encoder_for_type(message_cls, response_type)
    return encoder, response_type


def compile_response_encoder_for_type(
    message_cls: Any,
    response_type: Any,
) -> ResponseEncoder:
    """Compile a protobuf response encoder for a Python return type."""
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)
    response_type, optional = _unwrap_optional(response_type)

    if optional:

        def encode_optional(result: Any) -> Any:
            message = message_cls()
            if result is not None:
                _write_response_value(message, proto_fields, response_type, result)
            return message

        return encode_optional

    def encode_response(result: Any) -> Any:
        message = message_cls()
        if response_type is None or response_type is type(None):
            if proto_fields:
                raise TypeError(
                    f"{message_cls.DESCRIPTOR.full_name} must be an empty response"
                )
            return message
        if result is None:
            if proto_fields:
                raise TypeError(
                    f"{message_cls.DESCRIPTOR.full_name} got None for "
                    "non-empty response"
                )
            return message
        _write_response_value(message, proto_fields, response_type, result)
        return message

    return encode_response


def _write_response_value(
    message: Any,
    proto_fields: tuple[Any, ...],
    response_type: Any,
    result: Any,
) -> None:
    py_fields = _structured_fields(response_type)
    if py_fields is not None and len(py_fields) == len(proto_fields):
        writer, _reader = _compile_message_codec(message.DESCRIPTOR, response_type)
        writer(message, result)
        return

    tuple_types = _fixed_tuple_types(response_type)
    if tuple_types is not None and len(tuple_types) == len(proto_fields):
        for field, item_type, item in zip(
            proto_fields, tuple_types, result, strict=True
        ):
            writer, _reader = _compile_field_codec(field, item_type)
            writer(message, item)
        return

    if len(proto_fields) == 1:
        writer, _reader = _compile_field_codec(proto_fields[0], response_type)
        writer(message, result)
        return

    _write_message_from_runtime_value(message, message.DESCRIPTOR, result)


def encode_request_from_call(
    message_cls: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Build a protobuf request from a client method call."""
    if (
        len(args) == 1
        and not kwargs
        and _proto_has_same_descriptor(args[0], message_cls.DESCRIPTOR)
    ):
        return args[0]
    if args and kwargs:
        raise TypeError("RPC call accepts either positional args or keyword fields")

    message = message_cls()
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)
    if kwargs:
        fields_by_name = message_cls.DESCRIPTOR.fields_by_name
        for name, value in kwargs.items():
            field = fields_by_name.get(name)
            if field is None:
                raise TypeError(
                    f"{message_cls.DESCRIPTOR.full_name} has no field {name!r}"
                )
            _write_field_from_runtime_value(message, field, value)
        return message

    if (
        len(args) == 1
        and len(proto_fields) != 1
        and _is_structured_runtime_value(args[0], message_cls.DESCRIPTOR)
    ):
        _write_message_from_runtime_value(message, message_cls.DESCRIPTOR, args[0])
        return message

    if len(args) > len(proto_fields):
        raise TypeError(
            f"{message_cls.DESCRIPTOR.full_name} accepts at most "
            f"{len(proto_fields)} positional values, got {len(args)}"
        )
    for field, value in zip(proto_fields, args, strict=False):
        _write_field_from_runtime_value(message, field, value)
    return message


def decode_response_to_python(response: Any) -> Any:
    """Decode a protobuf response into LMCache's historical Python shape."""
    descriptor = response.DESCRIPTOR
    proto_fields = tuple(descriptor.fields)
    if not proto_fields:
        return None

    if descriptor.name == "RegisterKvCacheEngineDrivenContextResponse":
        return RegisterEngineDrivenContextResponse(
            shm_name=response.shm_name,
            pool_size=response.pool_size,
        )
    if descriptor.name == "PrepareStoreResponse":
        return PrepareStoreResponse(context=_decode_mapping(response.encoded_context))
    if descriptor.name == "PrepareRetrieveResponse":
        return PrepareRetrieveResponse(
            success=response.success,
            data=response.data,
            context=_decode_mapping(response.encoded_context),
        )
    if descriptor.name == "CbUnifiedLookupResponse":
        if not response.HasField("payload"):
            return None
        return _read_cb_unified_lookup_payload(response.payload)
    if descriptor.name == "P2pQueryLookupResultsResponse":
        if not response.HasField("addresses"):
            return None
        return [
            _read_transfer_channel_address(item)
            for item in response.addresses.addresses
        ]

    if len(proto_fields) == 1:
        return _read_response_field(response, proto_fields[0])
    return tuple(_read_response_field(response, field) for field in proto_fields)


def _read_response_field(message: Any, field: Any) -> Any:
    if field.has_presence and not message.HasField(field.name):
        return None
    value = getattr(message, field.name)
    if field.message_type is not None:
        return _read_response_message(value)
    if field.is_repeated:
        return list(value)
    return value


def _read_response_message(message: Any) -> Any:
    descriptor = message.DESCRIPTOR
    if descriptor.name == "EventIpcHandleResult":
        return (message.event_ipc_handle, message.success)
    if descriptor.name == "TransferChannelAddress":
        return _read_transfer_channel_address(message)
    if descriptor.name == "TransferChannelAddressList":
        return [_read_transfer_channel_address(item) for item in message.addresses]
    if descriptor.name == "CBUnifiedLookupPayload":
        return _read_cb_unified_lookup_payload(message)
    if len(descriptor.fields) == 1:
        return _read_response_field(message, descriptor.fields[0])
    return tuple(_read_response_field(message, field) for field in descriptor.fields)


def _read_cb_match_result(message: Any) -> CBMatchResult:
    return CBMatchResult(
        old_st=message.old_st,
        old_ed=message.old_ed,
        cur_st=message.cur_st,
        cur_ed=message.cur_ed,
        hash=message.hash,
    )


def _read_cb_unified_lookup_payload(message: Any) -> CBUnifiedLookupResult:
    return CBUnifiedLookupResult(
        prefix_coverage_tokens=message.prefix_coverage_tokens,
        non_prefix_segments=[
            _read_cb_match_result(item) for item in message.non_prefix_segments
        ],
        segmented_prefix_segments=[
            _read_cb_match_result(item) for item in message.segmented_prefix_segments
        ],
    )


def _read_transfer_channel_address(message: Any) -> TransferChannelAddress:
    return TransferChannelAddress(offset=message.offset, size=message.size)


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
    "compile_request_decoder",
    "compile_response_encoder",
    "decode_response_to_python",
    "encode_request_from_call",
    "get_request_message_class",
    "get_response_message_class",
    "get_service_name",
    "get_service_names",
    "msgspec_decode",
    "msgspec_encode",
    "request_type_to_method_name",
    "validate_protocol_descriptor",
]
