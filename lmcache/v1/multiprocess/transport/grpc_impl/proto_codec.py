# SPDX-License-Identifier: Apache-2.0
"""Compile protobuf/Python value converters for the gRPC transport.

This module contains only structural conversion rules. RPC ownership and
Python request/response types are supplied by the method codec registry, while
non-structural message conversions are supplied by service codec modules.
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
from lmcache.v1.multiprocess.transport.grpc_impl.codecs import (
    get_message_codec_registry,
)

_NONE_TYPE = type(None)

ValueEncoder = Callable[[Any], Any]
ValueDecoder = Callable[[Any], Any]
FieldWriter = Callable[[Any, Any], None]
FieldReader = Callable[[Any], Any]
RequestEncoder = Callable[[tuple[Any, ...], dict[str, Any]], Any]
RequestDecoder = Callable[[Any], tuple[Any, ...]]
ResponseEncoder = Callable[[Any], Any]
ResponseDecoder = Callable[[Any], Any]


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

    registered = get_message_codec_registry().find(descriptor, py_type)
    if registered is not None:
        return registered.writer, registered.reader

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


def compile_request_codec_for_types(
    message_cls: Any,
    payload_types: tuple[Any, ...],
) -> tuple[RequestEncoder, RequestDecoder]:
    """Compile one request codec from gRPC service payload types.

    Args:
        message_cls: Generated protobuf request class.
        payload_types: Python payload types declared by the gRPC service.

    Returns:
        A call encoder accepting ``(args, kwargs)`` and a protobuf decoder.

    Raises:
        TypeError: If the protobuf request cannot represent the payload types.
    """
    proto_fields = tuple(message_cls.DESCRIPTOR.fields)
    if len(payload_types) == len(proto_fields):
        codecs = tuple(
            _compile_field_codec(field, py_type)
            for field, py_type in zip(proto_fields, payload_types, strict=True)
        )
        codecs_by_name = {
            field.name: codec for field, codec in zip(proto_fields, codecs, strict=True)
        }

        def encode_fields(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if (
                len(args) == 1
                and not kwargs
                and _proto_has_same_descriptor(args[0], message_cls.DESCRIPTOR)
            ):
                return args[0]
            if args and kwargs:
                raise TypeError(
                    "RPC call accepts either positional args or keyword fields"
                )
            message = message_cls()
            if kwargs:
                for name, value in kwargs.items():
                    codec = codecs_by_name.get(name)
                    if codec is None:
                        raise TypeError(
                            f"{message_cls.DESCRIPTOR.full_name} has no field {name!r}"
                        )
                    codec[0](message, value)
                return message
            if len(args) > len(codecs):
                raise TypeError(
                    f"{message_cls.DESCRIPTOR.full_name} accepts at most "
                    f"{len(codecs)} positional values, got {len(args)}"
                )
            for value, (writer, _) in zip(args, codecs, strict=False):
                writer(message, value)
            return message

        def decode_fields(message: Any) -> tuple[Any, ...]:
            return tuple(reader(message) for _, reader in codecs)

        return encode_fields, decode_fields

    if len(payload_types) == 1:
        write_message, read_message = _compile_message_codec(
            message_cls.DESCRIPTOR, payload_types[0]
        )
        py_fields = _structured_fields(payload_types[0])
        if py_fields is None or len(py_fields) != len(proto_fields):
            raise TypeError(
                f"no keyword request codec from {payload_types[0]!r} to "
                f"{message_cls.DESCRIPTOR.full_name}"
            )
        keyword_codecs = {
            field.name: _compile_field_codec(field, field_type)
            for (_name, field_type), field in zip(py_fields, proto_fields, strict=True)
        }

        def encode_struct(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
            if (
                len(args) == 1
                and not kwargs
                and _proto_has_same_descriptor(args[0], message_cls.DESCRIPTOR)
            ):
                return args[0]
            if args and kwargs:
                raise TypeError(
                    "RPC call accepts either positional args or keyword fields"
                )
            message = message_cls()
            if kwargs:
                for name, value in kwargs.items():
                    codec = keyword_codecs.get(name)
                    if codec is None:
                        raise TypeError(
                            f"{message_cls.DESCRIPTOR.full_name} has no field {name!r}"
                        )
                    codec[0](message, value)
                return message
            if len(args) > 1:
                raise TypeError(
                    f"{message_cls.DESCRIPTOR.full_name} accepts at most one "
                    f"structured payload, got {len(args)}"
                )
            if args:
                write_message(message, args[0])
            return message

        def decode_struct(message: Any) -> tuple[Any, ...]:
            return (read_message(message),)

        return encode_struct, decode_struct

    raise TypeError(
        f"gRPC contract has {len(payload_types)} payloads but "
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
    if params and params[0].name in ("self", "cls"):
        params = params[1:]
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
        _encoder, decoder = compile_request_codec_for_types(message_cls, payload_types)
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
    if _proto_has_same_descriptor(result, message.DESCRIPTOR):
        message.CopyFrom(result)
        return

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

    raise TypeError(
        f"no response codec from {response_type!r} to {message.DESCRIPTOR.full_name}"
    )


def compile_response_decoder_for_type(
    message_cls: Any, response_type: Any
) -> ResponseDecoder:
    """Compile a protobuf response decoder for a gRPC response type.

    Args:
        message_cls: Generated protobuf response class.
        response_type: Python response type declared by the gRPC service.

    Returns:
        A callable that decodes the generated response message.

    Raises:
        TypeError: If the protobuf response cannot represent the Python type.
    """
    fields = tuple(message_cls.DESCRIPTOR.fields)
    if response_type is None or response_type is type(None):
        if fields:
            raise TypeError(
                f"{message_cls.DESCRIPTOR.full_name} must be an empty response"
            )
        return lambda _message: None

    response_type, optional = _unwrap_optional(response_type)
    if optional:
        if len(fields) != 1 or not fields[0].has_presence:
            raise TypeError(
                f"{message_cls.DESCRIPTOR.full_name} cannot represent an "
                "optional response"
            )
        _writer, reader = _compile_field_codec(fields[0], response_type)

        def decode_optional(message: Any) -> Any:
            if not message.HasField(fields[0].name):
                return None
            return reader(message)

        return decode_optional

    py_fields = _structured_fields(response_type)
    if py_fields is not None and len(py_fields) == len(fields):
        _writer, reader = _compile_message_codec(message_cls.DESCRIPTOR, response_type)
        return reader

    if len(fields) == 1:
        _writer, reader = _compile_field_codec(fields[0], response_type)
        return reader

    _writer, reader = _compile_message_codec(message_cls.DESCRIPTOR, response_type)
    return reader


def decode_response_to_type(response: Any, response_type: Any) -> Any:
    """Decode one protobuf response to its declared Python type.

    Args:
        response: Generated protobuf response instance.
        response_type: Python response type declared by the gRPC service.

    Returns:
        The decoded Python value.
    """
    decoder = compile_response_decoder_for_type(type(response), response_type)
    return decoder(response)
