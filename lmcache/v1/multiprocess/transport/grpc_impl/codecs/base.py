# SPDX-License-Identifier: Apache-2.0
"""Registration primitives for protobuf/Python message codecs."""

# Standard
from dataclasses import dataclass
from typing import Any, Callable

MessageWriter = Callable[[Any, Any], None]
MessageReader = Callable[[Any], Any]


@dataclass(frozen=True)
class RegisteredMessageCodec:
    """Convert one Python type to and from one protobuf message type.

    Args:
        protobuf_type: Fully-qualified protobuf message name.
        python_type: Python type handled by this codec.
        writer: Function that writes a Python value into a protobuf message.
        reader: Function that reads a Python value from a protobuf message.
        include_subclasses: Whether subclasses of ``python_type`` also match.
    """

    protobuf_type: str
    python_type: type[Any]
    writer: MessageWriter
    reader: MessageReader
    include_subclasses: bool = False

    def matches(self, descriptor: Any, python_type: Any) -> bool:
        """Return whether this codec handles a descriptor/type pair.

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


class MessageCodecRegistry:
    """Immutable collection of explicitly registered message codecs."""

    def __init__(self, codecs: tuple[RegisteredMessageCodec, ...]) -> None:
        keys: set[tuple[str, type[Any]]] = set()
        for codec in codecs:
            key = (codec.protobuf_type, codec.python_type)
            if key in keys:
                raise ValueError(
                    "Duplicate protobuf/Python message codec registration: "
                    f"{codec.protobuf_type} and {codec.python_type!r}"
                )
            keys.add(key)
        self._codecs = codecs

    def find(self, descriptor: Any, python_type: Any) -> RegisteredMessageCodec | None:
        """Return the unique codec for a descriptor/type pair.

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
            codec for codec in self._codecs if codec.matches(descriptor, python_type)
        )
        if len(matches) > 1:
            raise TypeError(
                "Ambiguous protobuf/Python message codec registration for "
                f"{descriptor.full_name} and {python_type!r}"
            )
        return matches[0] if matches else None
