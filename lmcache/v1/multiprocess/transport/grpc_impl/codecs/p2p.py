# SPDX-License-Identifier: Apache-2.0
"""Custom protobuf codecs owned by the P2P gRPC service."""

# Standard
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.codecs.base import (
    RegisteredMessageCodec,
)


def _write_tensor_shape(message: Any, value: torch.Size) -> None:
    message.dims.extend(value)


def _read_tensor_shape(message: Any) -> torch.Size:
    return torch.Size(message.dims)


def get_message_codecs() -> tuple[RegisteredMessageCodec, ...]:
    """Return custom codecs used by the P2P protobuf service.

    Returns:
        Explicit registrations for P2P non-structural message types.
    """
    return (
        RegisteredMessageCodec(
            protobuf_type="lmcache.mp.TensorShape",
            python_type=torch.Size,
            writer=_write_tensor_shape,
            reader=_read_tensor_shape,
        ),
    )
