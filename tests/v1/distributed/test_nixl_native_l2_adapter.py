# SPDX-License-Identifier: Apache-2.0
"""Public configuration and factory tests for ``nixl_native``."""

# Standard
from types import ModuleType
import os
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import EvictionConfig
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (
    NativeConnectorL2Adapter,
)
from lmcache.v1.distributed.l2_adapters.nixl_native_l2_adapter import (
    NixlNativeL2AdapterConfig,
)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "posix", "uppercase NIXL plugin"),
        ("backend", "BAD-NAME", "uppercase NIXL plugin"),
        ("num_workers", 0, "positive integer"),
        ("num_workers", True, "positive integer"),
        ("max_capacity_gb", -1, "non-negative"),
        ("max_capacity_gb", True, "non-negative"),
    ],
)
def test_invalid_generic_config_fields(field: str, value: object, message: str) -> None:
    """Invalid generic fields fail before native construction."""
    raw: dict[str, object] = {
        "backend": "POSIX",
        "backend_params": {"file_path": "/tmp/nixl"},
    }
    raw[field] = value
    with pytest.raises(ValueError, match=message):
        NixlNativeL2AdapterConfig.from_dict(raw)


@pytest.mark.parametrize(
    "backend_params",
    [None, [], {"file_path": 123}, {"file_path": "/tmp", "x": 1}],
)
def test_backend_params_are_strings(backend_params: object) -> None:
    """Backend parameters remain an opaque string-to-string map."""
    with pytest.raises(ValueError, match="dict of string key-value pairs"):
        NixlNativeL2AdapterConfig.from_dict(
            {
                "backend": "POSIX",
                "backend_params": backend_params,
            }
        )


def test_storage_type_setting_is_rejected() -> None:
    """The backend capability, rather than configuration, selects storage."""
    with pytest.raises(ValueError, match="storage_type is not configurable"):
        NixlNativeL2AdapterConfig.from_dict(
            {
                "backend": "POSIX",
                "storage_type": "FILE",
                "backend_params": {"file_path": "/tmp/nixl"},
            }
        )


@pytest.mark.parametrize("value", ["yes", 1, 0])
def test_pad_buffers_to_alignment_field_is_gone(value: object) -> None:
    """pad_buffers_to_alignment is not a config field; it is derived from
    use_direct_io. Unknown fields are ignored by from_dict."""
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "POSIX",
            "backend_params": {"file_path": "/tmp/nixl"},
            "pad_buffers_to_alignment": value,
        }
    )
    assert not hasattr(config, "pad_buffers_to_alignment")


def test_factory_requires_l1_memory_desc() -> None:
    """The dedicated factory rejects a missing L1 arena."""
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "OBJ",
            "backend_params": {},
        }
    )
    with pytest.raises(ValueError, match="L1MemoryDesc"):
        create_l2_adapter(config)


def test_factory_missing_extension_has_build_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selecting an unbuilt connector reports actionable build settings."""
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "OBJ",
            "backend_params": {},
        }
    )
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_nixl", None)
    with pytest.raises(RuntimeError, match="BUILD_WITH_NIXL=1"):
        create_l2_adapter(config, L1MemoryDesc(4096, 8192, 4096))


def test_factory_forwards_l1_arena_and_safe_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory forwards the L1 descriptor unchanged and hides secrets."""
    captured: dict[str, object] = {}

    class FakeNixlClient:
        """Minimal native-client contract used by the public wrapper."""

        storage_type = "OBJECT"
        supports_query = True
        supports_delete = False
        supports_direct_io = False
        atomic_publication = False

        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.read_fd, self.write_fd = os.pipe()

        def event_fd(self) -> int:
            """Return the pollable completion descriptor."""
            return self.read_fd

        def close(self) -> None:
            """Close the fake completion descriptor."""
            os.close(self.read_fd)
            os.close(self.write_fd)

    fake_module = ModuleType("lmcache.lmcache_nixl")
    fake_module.LMCacheNixlClient = FakeNixlClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_nixl", fake_module)

    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "OBJ",
            "backend_params": {
                "bucket": "test-bucket",
                "secret_access_key": "must-not-leak",
            },
            "num_workers": 3,
            "max_capacity_gb": 2,
        }
    )
    l1_desc = L1MemoryDesc(ptr=0x12340000, size=0x400000, align_bytes=0x1000)
    adapter = create_l2_adapter(config, l1_desc)
    try:
        assert isinstance(adapter, NativeConnectorL2Adapter)
        assert captured == {
            "backend": "OBJ",
            "backend_params": {
                "bucket": "test-bucket",
                "secret_access_key": "must-not-leak",
            },
            "num_workers": 3,
            "l1_base": l1_desc.ptr,
            "l1_size": l1_desc.size,
            "l1_alignment": l1_desc.align_bytes,
        }
        status = adapter.report_status()
        assert status["type"] == "nixl_native"
        assert status["backend"] == "OBJ"
        assert status["storage_type"] == "OBJECT"
        assert status["supports_delete"] is False
        assert "backend_params" not in status
        assert "must-not-leak" not in repr(status)
    finally:
        adapter.close()


def test_factory_rejects_eviction_for_inferred_object_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction validation uses the native strategy's delete capability."""
    closed = False

    class FakeNixlClient:
        """Native client reporting inferred OBJECT capabilities."""

        storage_type = "OBJECT"
        supports_query = True
        supports_delete = False
        supports_direct_io = False
        atomic_publication = False

        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def close(self) -> None:
            """Record cleanup after capability validation fails."""
            nonlocal closed
            closed = True

    fake_module = ModuleType("lmcache.lmcache_nixl")
    fake_module.LMCacheNixlClient = FakeNixlClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_nixl", fake_module)

    config = NixlNativeL2AdapterConfig.from_dict(
        {"backend": "OBJ", "backend_params": {}}
    )
    config.eviction_config = EvictionConfig(eviction_policy="LRU")
    with pytest.raises(ValueError, match="OBJECT storage does not support eviction"):
        create_l2_adapter(config, L1MemoryDesc(4096, 8192, 4096))
    assert closed


class _FakeNixlClientForPadding:
    """Native-client stub with a selectable storage type.

    Mirrors the real connector's capability semantics: ``supports_direct_io``
    is true only for FILE storage with ``use_direct_io: "true"`` (see
    ``NixlFileStorage::capabilities()``). Provides the event fd required by
    NativeConnectorL2Adapter's demux thread so the factory can run end to end
    without a NIXL build.
    """

    supports_query = True
    supports_delete = False
    atomic_publication = False

    def __init__(self, storage_type: str, **kwargs: object) -> None:
        backend_params = kwargs.get("backend_params", {})
        assert isinstance(backend_params, dict)
        self.storage_type = storage_type
        self.supports_direct_io = (
            storage_type == "FILE"
            and backend_params.get("use_direct_io", "false") == "true"
        )
        self.read_fd, self.write_fd = os.pipe()

    def event_fd(self) -> int:
        """Return the pollable completion descriptor."""
        return self.read_fd

    def close(self) -> None:
        """Close the fake completion descriptor."""
        os.close(self.read_fd)
        os.close(self.write_fd)


def _create_adapter_with_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    storage_type: str,
    config_dict: dict[str, object],
) -> L2AdapterInterface:
    """Run the nixl_native factory with a stubbed native client module."""

    def fake_client_factory(**kwargs: object) -> _FakeNixlClientForPadding:
        return _FakeNixlClientForPadding(storage_type, **kwargs)

    fake_module = ModuleType("lmcache.lmcache_nixl")
    fake_module.LMCacheNixlClient = fake_client_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_nixl", fake_module)

    config = NixlNativeL2AdapterConfig.from_dict(config_dict)
    return create_l2_adapter(
        config, L1MemoryDesc(ptr=0x12340000, size=0x400000, align_bytes=0x1000)
    )


@pytest.mark.parametrize(
    ("config_dict", "storage_type", "expected"),
    [
        # Padding follows direct I/O: on only for FILE + use_direct_io=true.
        (
            {
                "backend": "POSIX",
                "backend_params": {"file_path": "/tmp/x", "use_direct_io": "true"},
            },
            "FILE",
            True,
        ),
        (
            {
                "backend": "POSIX",
                "backend_params": {"file_path": "/tmp/x", "use_direct_io": "false"},
            },
            "FILE",
            False,
        ),
        # Absent use_direct_io must not raise (KeyError regression) and
        # defaults to no padding.
        (
            {"backend": "POSIX", "backend_params": {"file_path": "/tmp/x"}},
            "FILE",
            False,
        ),
        # use_direct_io is meaningless for OBJECT storage: never pad.
        (
            {"backend": "OBJ", "backend_params": {}},
            "OBJECT",
            False,
        ),
        (
            {"backend": "OBJ", "backend_params": {"use_direct_io": "true"}},
            "OBJECT",
            False,
        ),
    ],
)
def test_pad_buffers_to_alignment_resolution(
    monkeypatch: pytest.MonkeyPatch,
    config_dict: dict[str, object],
    storage_type: str,
    expected: bool,
) -> None:
    """Padding is derived from direct I/O being in effect, not configured."""
    adapter = _create_adapter_with_fake_client(monkeypatch, storage_type, config_dict)
    try:
        assert isinstance(adapter, NativeConnectorL2Adapter)
        assert adapter.report_status()["pad_buffers_to_alignment"] is expected
    finally:
        adapter.close()
