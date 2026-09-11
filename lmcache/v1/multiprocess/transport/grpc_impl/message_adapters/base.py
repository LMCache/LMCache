# SPDX-License-Identifier: Apache-2.0
"""Registration primitives for protobuf/Python message adapters."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar

# Third Party
from google.protobuf.message import Message

ProtoMessageT = TypeVar("ProtoMessageT", bound=Message)
PythonValueT = TypeVar("PythonValueT")

MessageWriter = Callable[[ProtoMessageT, PythonValueT], None]
MessageReader = Callable[[ProtoMessageT], PythonValueT]


@dataclass(frozen=True)
class RegisteredMessageAdapter(Generic[ProtoMessageT, PythonValueT]):
    """Convert one Python type to and from one protobuf message type.

    Args:
        protobuf_type: Fully-qualified protobuf message name.
        python_type: Python type handled by this adapter.
        writer: Function that writes a Python value into a protobuf message.
        reader: Function that reads a Python value from a protobuf message.
        include_subclasses: Whether subclasses of ``python_type`` also match.
    """

    protobuf_type: str
    python_type: type[PythonValueT]
    writer: MessageWriter[ProtoMessageT, PythonValueT]
    reader: MessageReader[ProtoMessageT, PythonValueT]
    include_subclasses: bool = False

    def matches(self, descriptor: Any, python_type: Any) -> bool:
        """Return whether this adapter handles a descriptor/type pair.

        Args:
            descriptor: Protobuf message descriptor to inspect.
            python_type: Python type requested by the structural compiler.

        Returns:
            Whether this registration handles both values.
        """
        if descriptor.full_name != self.protobuf_type:
            return False
        if python_type is self.python_type:
            return True
        return bool(
            self.include_subclasses
            and isinstance(python_type, type)
            and issubclass(python_type, self.python_type)
        )


class MessageAdapterRegistry:
    """Immutable collection of explicitly registered message adapters."""

    def __init__(
        self, adapters: tuple[RegisteredMessageAdapter[Any, Any], ...]
    ) -> None:
        keys: set[tuple[str, type[Any]]] = set()
        for adapter in adapters:
            key = (adapter.protobuf_type, adapter.python_type)
            if key in keys:
                raise ValueError(
                    "Duplicate protobuf/Python message adapter registration: "
                    f"{adapter.protobuf_type} and {adapter.python_type!r}"
                )
            keys.add(key)
        self._adapters = adapters

    def find(
        self, descriptor: Any, python_type: Any
    ) -> RegisteredMessageAdapter[Any, Any] | None:
        """Return the unique adapter for a descriptor/type pair.

        Args:
            descriptor: Protobuf message descriptor to inspect.
            python_type: Python type requested by the structural compiler.

        Returns:
            The matching registration, or ``None`` when structural conversion
            should be used.

        Raises:
            TypeError: If multiple registrations match the same pair.
        """
        matches = tuple(
            adapter
            for adapter in self._adapters
            if adapter.matches(descriptor, python_type)
        )
        if len(matches) > 1:
            raise TypeError(
                "Ambiguous protobuf/Python message adapter registration for "
                f"{descriptor.full_name} and {python_type!r}"
            )
        return matches[0] if matches else None
