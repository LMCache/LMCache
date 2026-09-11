# SPDX-License-Identifier: Apache-2.0
"""Automatically map transport-neutral Python messages to protobuf messages.

The same structural mapper is used for requests and responses. RPC ownership
and message pairs are supplied by the method registry, while non-structural
leaf conversions are supplied by message adapter modules.
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
import types

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.message_adapters import (
    get_message_adapter_registry,
)

_NONE_TYPE = type(None)

ValueEncoder = Callable[[Any], Any]
ValueDecoder = Callable[[Any], Any]
FieldWriter = Callable[[Any, Any], None]
FieldReader = Callable[[Any], Any]
MessageToProto = Callable[[Any], Any]
ProtoToMessage = Callable[[Any], Any]


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


def _build_scalar_adapter(
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
    encode_key, decode_key = _build_scalar_adapter(key_field, key_type)
    field_name = field.name

    if value_field.message_type is None:
        encode_value, decode_value = _build_scalar_adapter(value_field, value_type)

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

    write_value, read_value = _build_message_adapter(
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


def _build_field_adapter(field: Any, py_type: Any) -> tuple[FieldWriter, FieldReader]:
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
            encode_item, decode_item = _build_scalar_adapter(field, item_type)

            def write_repeated(message: Any, value: Any) -> None:
                getattr(message, field_name).extend(encode_item(item) for item in value)

            def read_repeated(message: Any) -> Any:
                decoded = [decode_item(item) for item in getattr(message, field_name)]
                return tuple(decoded) if as_tuple else decoded

            writer, reader = write_repeated, read_repeated
        else:
            write_item, read_item = _build_message_adapter(
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
        write_child, read_child = _build_message_adapter(field.message_type, py_type)
        field_name = field.name

        def write_message(message: Any, value: Any) -> None:
            child = getattr(message, field_name)
            write_child(child, value)
            child.SetInParent()

        def read_message(message: Any) -> Any:
            return read_child(getattr(message, field_name))

        writer, reader = write_message, read_message
    else:
        encode_value, decode_value = _build_scalar_adapter(field, py_type)
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


def _build_message_adapter(
    descriptor: Any, py_type: Any
) -> tuple[FieldWriter, FieldReader]:
    py_type, _ = _unwrap_optional(py_type)
    proto_fields = tuple(descriptor.fields)

    registered = get_message_adapter_registry().find(descriptor, py_type)
    if registered is not None:
        return registered.writer, registered.reader

    sequence = _sequence_type(py_type)
    if sequence is not None and len(proto_fields) == 1:
        return _build_field_adapter(proto_fields[0], py_type)

    tuple_types = _fixed_tuple_types(py_type)
    if tuple_types is not None:
        if len(tuple_types) != len(proto_fields):
            raise TypeError(
                f"{py_type!r} has {len(tuple_types)} values but "
                f"{descriptor.full_name} has {len(proto_fields)} fields"
            )
        tuple_adapters = tuple(
            _build_field_adapter(field, item_type)
            for field, item_type in zip(proto_fields, tuple_types, strict=True)
        )

        def write_tuple(message: Any, value: Any) -> None:
            for item, (writer, _) in zip(value, tuple_adapters, strict=True):
                writer(message, item)

        def read_tuple(message: Any) -> tuple[Any, ...]:
            return tuple(reader(message) for _, reader in tuple_adapters)

        return write_tuple, read_tuple

    py_fields = _structured_fields(py_type)
    if py_fields is None or len(py_fields) != len(proto_fields):
        raise TypeError(
            f"no structural adapter from {py_type!r} to {descriptor.full_name}"
        )

    struct_adapters: list[tuple[str, FieldWriter, FieldReader]] = []
    for (name, field_type), field in zip(py_fields, proto_fields, strict=True):
        if field.name not in (name, f"encoded_{name}"):
            raise TypeError(
                f"{py_type.__name__}.{name} does not match "
                f"{descriptor.full_name}.{field.name}"
            )
        writer, reader = _build_field_adapter(field, field_type)
        struct_adapters.append((name, writer, reader))

    def write_struct(message: Any, value: Any) -> None:
        for name, writer, _ in struct_adapters:
            writer(message, getattr(value, name))

    def read_struct(message: Any) -> Any:
        return py_type(**{name: reader(message) for name, _, reader in struct_adapters})

    return write_struct, read_struct


def build_message_conversion(
    protobuf_class: type[Any],
    python_class: type[Any],
) -> tuple[MessageToProto, ProtoToMessage]:
    """Build symmetric converters for one complete RPC message pair.

    Field mappings are derived recursively from the protobuf descriptor and
    Python type hints. No request/response or per-RPC conversion definition is
    needed.
    """
    write_message, read_message = _build_message_adapter(
        protobuf_class.DESCRIPTOR,
        python_class,
    )

    def to_proto(value: Any) -> Any:
        if not isinstance(value, python_class):
            raise TypeError(
                f"expected {python_class.__name__}, got {type(value).__name__}"
            )
        protobuf_message = protobuf_class()
        write_message(protobuf_message, value)
        return protobuf_message

    def from_proto(protobuf_message: Any) -> Any:
        if not isinstance(protobuf_message, protobuf_class):
            raise TypeError(
                f"expected {protobuf_class.__name__}, "
                f"got {type(protobuf_message).__name__}"
            )
        return read_message(protobuf_message)

    return to_proto, from_proto


__all__ = ["MessageToProto", "ProtoToMessage", "build_message_conversion"]
