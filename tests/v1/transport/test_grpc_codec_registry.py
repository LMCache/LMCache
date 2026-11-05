# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.transport.grpc_impl.codecs import base as legacy
from lmcache.v1.transport.grpc.codec_registry import (
    MessageCodecRegistry,
    RegisteredMessageCodec,
)


class Parent:
    pass


class Child(Parent):
    pass


def _writer(message: object, value: object) -> None:
    pass


def _reader(message: object) -> object:
    return message


def _codec(
    python_type: type[object] = Parent, *, include_subclasses: bool = False
) -> RegisteredMessageCodec:
    return RegisteredMessageCodec(
        protobuf_type="lmcache.test.Message",
        python_type=python_type,
        writer=_writer,
        reader=_reader,
        include_subclasses=include_subclasses,
    )


def test_legacy_imports_are_canonical_objects() -> None:
    assert legacy.RegisteredMessageCodec is RegisteredMessageCodec
    assert legacy.MessageCodecRegistry is MessageCodecRegistry


def test_registry_exact_and_subclass_matching() -> None:
    descriptor = SimpleNamespace(full_name="lmcache.test.Message")
    exact = _codec()
    assert MessageCodecRegistry((exact,)).find(descriptor, Parent) is exact
    assert MessageCodecRegistry((exact,)).find(descriptor, Child) is None

    inherited = _codec(include_subclasses=True)
    assert MessageCodecRegistry((inherited,)).find(descriptor, Child) is inherited


def test_registry_rejects_duplicate_and_ambiguous_matches() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        MessageCodecRegistry((_codec(), _codec()))

    descriptor = SimpleNamespace(full_name="lmcache.test.Message")
    registry = MessageCodecRegistry(
        (_codec(Parent, include_subclasses=True), _codec(Child))
    )
    with pytest.raises(TypeError, match="Ambiguous"):
        registry.find(descriptor, Child)
