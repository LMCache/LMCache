# SPDX-License-Identifier: Apache-2.0
"""Tests for the lmcache-client extension build boundary."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch.utils

# First Party
from setup_extensions import common_cpp, policy


def test_client_common_extensions_exclude_storage_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client build keeps lmcache_native but omits server storage bindings."""
    extension_names: list[str] = []

    class FakeCppExtension:
        def __init__(self, name: str, **_: object) -> None:
            extension_names.append(name)

    fake_cpp = SimpleNamespace(
        CppExtension=FakeCppExtension,
        BuildExtension=object(),
    )
    monkeypatch.setattr(torch.utils, "cpp_extension", fake_cpp, raising=False)

    common_cpp.build_common_cpp(client_only=True)

    assert extension_names == ["lmcache.lmcache_native"]


def test_client_policy_skips_optional_storage_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Client policy does not discover or build optional server backends."""
    monkeypatch.setattr(policy.BuildProfile, "is_building_sdist", lambda: False)
    monkeypatch.setattr(policy.BuildProfile, "is_native_ext_disabled", lambda: False)
    monkeypatch.setattr(policy, "build_common_cpp", lambda *_args, **_kwargs: ([], {}))

    def fail_storage_discovery(_flags: list[str]) -> list[object]:
        raise AssertionError("storage backend discovery must not run")

    monkeypatch.setattr(
        policy.BuildPolicy,
        "collect_storage_backends",
        fail_storage_discovery,
    )

    extensions, commands, requirements = policy.BuildPolicy.collect_extensions(
        None,
        client_only=True,
    )

    assert extensions == []
    assert commands == {}
    assert requirements is None
