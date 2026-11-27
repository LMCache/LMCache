# SPDX-License-Identifier: Apache-2.0
"""MessagePack codec for the ZMQ request transport."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable

# Third Party
import msgspec
import torch

# First Party
from lmcache.v1.platform.base.ipc_wrapper import DeviceIPCWrapper
from lmcache.v1.transport.qstore import QRingInfo


@dataclass
class CustomizedSerdeConfig:
    serializer: Callable[[Any], bytes]
    deserializer: Callable[[bytes], Any]
    code: int


_CUSTOMERIZED_SERIALIZERS = {
    DeviceIPCWrapper: CustomizedSerdeConfig(
        serializer=DeviceIPCWrapper.Serialize,
        deserializer=DeviceIPCWrapper.Deserialize,
        code=1,
    ),
}


def get_customized_encoder(type: Any) -> msgspec.msgpack.Encoder:
    # TODO: `type` is not used here
    def enc_hook(obj: Any) -> Any:
        for supported_type, cfg in _CUSTOMERIZED_SERIALIZERS.items():
            if isinstance(obj, supported_type):
                data = cfg.serializer(obj)
                return msgspec.msgpack.Ext(cfg.code, data)
        if isinstance(obj, torch.dtype):
            return str(obj).removeprefix("torch.")
        if isinstance(obj, torch.Size):
            return list(obj)
        raise TypeError(f"Unsupported type for serialization: {type(obj)}")

    return msgspec.msgpack.Encoder(enc_hook=enc_hook)


def get_customized_decoder(type: Any) -> msgspec.msgpack.Decoder:
    def ext_hook(code: int, data: bytes) -> Any:
        for cfg in _CUSTOMERIZED_SERIALIZERS.values():
            if cfg.code == code:
                return cfg.deserializer(data)
        raise TypeError(f"Unsupported ext code for deserialization: {code}")

    def dec_hook(expected_type: type, obj: Any) -> Any:
        if expected_type is torch.dtype:
            return getattr(torch, obj)
        if expected_type is torch.Size:
            return torch.Size(obj)
        if isinstance(obj, expected_type):
            return obj
        raise NotImplementedError(
            f"Unsupported type for deserialization: {expected_type}"
        )

    return msgspec.msgpack.Decoder(ext_hook=ext_hook, dec_hook=dec_hook, type=type)


_SPECIAL_ENCODER_DECODERS = {
    DeviceIPCWrapper: (
        get_customized_encoder(DeviceIPCWrapper),
        get_customized_decoder(DeviceIPCWrapper),
    ),
    list[DeviceIPCWrapper]: (
        get_customized_encoder(list[DeviceIPCWrapper]),
        get_customized_decoder(list[DeviceIPCWrapper]),
    ),
    list[QRingInfo]: (
        get_customized_encoder(list[QRingInfo]),
        get_customized_decoder(list[QRingInfo]),
    ),
}


def msgspec_encode(obj: Any, cls: Any) -> bytes:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        encoder, _ = _SPECIAL_ENCODER_DECODERS[cls]
        return encoder.encode(obj)
    # Defensive guard: coerce obj to the declared cls so that
    # e.g. a bool passed as int (or vice-versa) is encoded in the
    # wire format that msgspec_decode expects for that cls.
    if cls in (bool, int):
        obj = cls(obj)
    return msgspec.msgpack.encode(obj)


def msgspec_decode(b_obj: bytes, cls: Any) -> Any:
    # Handle special cases
    if cls in _SPECIAL_ENCODER_DECODERS:
        _, decoder = _SPECIAL_ENCODER_DECODERS[cls]
        return decoder.decode(b_obj)
    # Defensive guard: msgspec strict-validates wire format
    # (bool ≠ int in msgpack), but runtime type may not match
    # declared cls. Decode untyped, then coerce.
    if cls in (bool, int):
        return cls(msgspec.msgpack.decode(b_obj))
    return msgspec.msgpack.decode(b_obj, type=cls)
