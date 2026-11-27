# SPDX-License-Identifier: Apache-2.0
"""MUSA IPC wire descriptor."""

# Standard
from typing import cast

# First Party
from lmcache.v1.platform.base.ipc_wire import IPCData


def validate_musa_handle(value: object) -> None:
    """Reject opaque objects and extensions in a MUSA handle.

    Args:
        value: A handle composed of scalars, lists and string-keyed dictionaries.

    Raises:
        ValueError: If the handle would require Python object reconstruction.
    """
    if type(value) in (type(None), bool, int, float, str, bytes):
        return
    if type(value) is list:
        for item in cast(list[object], value):
            validate_musa_handle(item)
        return
    if type(value) is dict and all(
        type(key) is str for key in cast(dict[object, object], value)
    ):
        for item in cast(dict[str, object], value).values():
            validate_musa_handle(item)
        return
    raise ValueError("MUSA IPC handles must contain only MessagePack primitive values")


class MusaIPCData(IPCData, tag="musa"):
    """MUSA handle restricted to MessagePack primitive values."""

    handle: object

    def __post_init__(self) -> None:
        validate_musa_handle(self.handle)
