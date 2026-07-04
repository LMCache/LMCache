# SPDX-License-Identifier: Apache-2.0
"""Stable mechanism tests for the universal 3-D platform registry."""

# Standard
from typing import Any, ClassVar, Generator
import abc

# Third Party
import pytest

# First Party
from lmcache.v1.platform._registry import (
    _register_impl,
    get_impl,
    reset_registry_for_tests,
    resolve_impl,
    restore_registry,
    snapshot_registry,
)
import lmcache.v1.platform._registry as _reg_module


@pytest.fixture(autouse=True)
def isolated_registry() -> Generator[None, None, None]:
    """Isolate universal registry state between tests."""
    state = snapshot_registry()
    reset_registry_for_tests()
    try:
        yield
    finally:
        restore_registry(state)


class OptionalBase(abc.ABC):  # noqa: B024
    """Synthetic optional capability base (concrete despite ABC)."""

    def run(self) -> str:
        return "base"


class RequiredBase(abc.ABC):
    """Synthetic required capability base."""

    @abc.abstractmethod
    def run(self) -> str:
        raise NotImplementedError


class OptionalDefault(OptionalBase):
    device_type: ClassVar[str] = "cuda"
    impl_key: ClassVar[str] = "default"

    def run(self) -> str:
        return "optional-default"


class OptionalVariant(OptionalBase):
    device_type: ClassVar[str] = "cuda"
    impl_key: ClassVar[str] = "variant"

    def run(self) -> str:
        return "optional-variant"


class RequiredDefault(RequiredBase):
    device_type: ClassVar[str] = "cuda"
    impl_key: ClassVar[str] = "default"

    def run(self) -> str:
        return "required-default"


class TestGetImpl:
    """Strict lookup behavior."""

    def test_get_impl_strict_success_and_default_key(self) -> None:
        _register_impl(OptionalBase, OptionalDefault)
        assert get_impl(OptionalBase, "cuda") is OptionalDefault
        assert get_impl(OptionalBase, "cuda", "default") is OptionalDefault

    def test_get_impl_3d_lookup_and_multiple_impl_keys(self) -> None:
        _register_impl(OptionalBase, OptionalDefault)
        _register_impl(OptionalBase, OptionalVariant)
        assert get_impl(OptionalBase, "cuda", "default") is OptionalDefault
        assert get_impl(OptionalBase, "cuda", "variant") is OptionalVariant

    def test_get_impl_strict_failures(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            get_impl(OptionalBase, "cuda", "default")

        _register_impl(OptionalBase, OptionalDefault)
        with pytest.raises(ValueError, match="device_type"):
            get_impl(OptionalBase, "cpu", "default")
        with pytest.raises(ValueError, match="impl_key"):
            get_impl(OptionalBase, "cuda", "missing")

    def test_duplicate_registration_keeps_first_and_warns(self, monkeypatch) -> None:
        class OptionalDuplicate(OptionalBase):
            device_type: ClassVar[str] = "cuda"
            impl_key: ClassVar[str] = "default"

            def run(self) -> str:
                return "optional-duplicate"

        messages: list[str] = []

        def _record_warning(message: str, *args: Any, **kwargs: Any) -> None:
            messages.append(message % args if args else message)

        monkeypatch.setattr(_reg_module.logger, "warning", _record_warning)
        _register_impl(OptionalBase, OptionalDefault)
        _register_impl(OptionalBase, OptionalDuplicate)

        assert get_impl(OptionalBase, "cuda", "default") is OptionalDefault
        assert any("keeping the first" in message for message in messages)

    def test_missing_or_empty_device_type_is_skipped_and_warns(
        self, monkeypatch
    ) -> None:
        class MissingDeviceType(OptionalBase):
            def run(self) -> str:
                return "missing-device-type"

        class EmptyDeviceType(OptionalBase):
            device_type: ClassVar[str] = ""

            def run(self) -> str:
                return "empty-device-type"

        messages: list[str] = []

        def _record_warning(message: str, *args: Any, **kwargs: Any) -> None:
            messages.append(message % args if args else message)

        monkeypatch.setattr(_reg_module.logger, "warning", _record_warning)
        _register_impl(OptionalBase, MissingDeviceType)
        _register_impl(OptionalBase, EmptyDeviceType)

        with pytest.raises(ValueError):
            get_impl(OptionalBase, "", "default")
        assert any("empty device_type" in message for message in messages)


class TestResolveImpl:
    """Policy-aware lookup behavior."""

    def test_resolve_impl_returns_registered_concrete_subclass(self) -> None:
        _register_impl(OptionalBase, OptionalDefault)
        assert resolve_impl(OptionalBase, "cuda", "default") is OptionalDefault

    def test_resolve_impl_falls_back_for_concrete_base(self) -> None:
        _register_impl(OptionalBase, OptionalDefault)
        assert resolve_impl(OptionalBase, "missing_device", "default") is OptionalBase

    def test_resolve_impl_reraises_for_abstract_base(self) -> None:
        _register_impl(RequiredBase, RequiredDefault)
        with pytest.raises(ValueError):
            resolve_impl(RequiredBase, "missing_device", "default")
