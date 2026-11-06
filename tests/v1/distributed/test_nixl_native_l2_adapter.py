# SPDX-License-Identifier: Apache-2.0
"""Public configuration and factory tests for ``nixl_native``."""

# Standard
from types import ModuleType
import importlib.abc
import importlib.util
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
import lmcache


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


def test_factory_missing_extension_has_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without either connector module, a release names the matching
    lmcache[nixl] install; every build names the source-build settings."""
    config = NixlNativeL2AdapterConfig.from_dict(
        {
            "backend": "OBJ",
            "backend_params": {},
        }
    )
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_nixl", None)
    monkeypatch.setitem(sys.modules, "lmcache_nixl", None)
    monkeypatch.setattr(lmcache, "__version__", "0.5.8")
    with pytest.raises(RuntimeError) as excinfo:
        create_l2_adapter(config, L1MemoryDesc(4096, 8192, 4096))
    assert 'pip install "lmcache[nixl]==0.5.8"' in str(excinfo.value)
    assert "BUILD_WITH_NIXL=1" in str(excinfo.value)

    monkeypatch.setattr(lmcache, "__version__", "0.5.9.dev3")
    with pytest.raises(RuntimeError) as excinfo:
        create_l2_adapter(config, L1MemoryDesc(4096, 8192, 4096))
    assert "pip install" not in str(excinfo.value)
    assert "BUILD_WITH_NIXL=1" in str(excinfo.value)


class _FakeNixlClient:
    """Minimal native-client contract for the import-order tests."""

    storage_type = "OBJECT"
    supports_query = True
    supports_delete = False
    supports_direct_io = False
    atomic_publication = False

    def __init__(self, **kwargs: object) -> None:
        self.read_fd, self.write_fd = os.pipe()

    def event_fd(self) -> int:
        """Return the pollable completion descriptor."""
        return self.read_fd

    def close(self) -> None:
        """Close the fake completion descriptor."""
        os.close(self.read_fd)
        os.close(self.write_fd)


class _ImportScenario(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Simulate how the two connector modules and the nixl wheel load.

    ``lmcache.lmcache_nixl`` (source build) loads only when ``source_build``;
    ``lmcache_nixl`` (lmcache-nixl wheel) loads only once ``nixl`` has been
    imported, as its libnixl comes from there; importing ``nixl`` raises
    ``nixl_error`` when set. Records every import attempt in ``attempts``.
    """

    _NAMES = ("lmcache.lmcache_nixl", "lmcache_nixl", "nixl")

    def __init__(self, source_build: bool, nixl_error: str = "") -> None:
        self.source_build = source_build
        self.nixl_error = nixl_error
        self.attempts: list[str] = []

    def find_spec(self, fullname, path, target=None):
        """Claim the connector modules and nixl; leave other imports alone."""
        if fullname not in self._NAMES:
            return None
        self.attempts.append(fullname)
        if fullname == "nixl" and self.nixl_error:
            raise ImportError(self.nixl_error)
        if fullname == "lmcache.lmcache_nixl" and not self.source_build:
            raise ImportError("No module named 'lmcache.lmcache_nixl'")
        if fullname == "lmcache_nixl" and "nixl" not in sys.modules:
            raise ImportError("libnixl.so: cannot open shared object file")
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        """Use the default module creation."""
        return None

    def exec_module(self, module: ModuleType) -> None:
        """Populate the fake connector modules with the client class."""
        if module.__name__ != "nixl":
            module.LMCacheNixlClient = _FakeNixlClient  # type: ignore[attr-defined]

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Route the connector and nixl imports through this scenario."""
        for name in self._NAMES:
            # setitem records the real entry (or its absence), so the fake
            # modules imported during the test are dropped on teardown.
            monkeypatch.setitem(sys.modules, name, None)
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "meta_path", [self, *sys.meta_path])


def _obj_config() -> NixlNativeL2AdapterConfig:
    return NixlNativeL2AdapterConfig.from_dict({"backend": "OBJ", "backend_params": {}})


def test_factory_source_build_does_not_import_nixl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source build's connector, which finds its NIXL SDK through RUNPATH,
    is used without importing the nixl wheel, which would load its libnixl."""
    scenario = _ImportScenario(source_build=True)
    scenario.install(monkeypatch)

    adapter = create_l2_adapter(_obj_config(), L1MemoryDesc(4096, 8192, 4096))
    adapter.close()

    assert scenario.attempts == ["lmcache.lmcache_nixl"]


def test_factory_wheel_imports_nixl_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lmcache-nixl wheel's connector loads after the nixl wheel."""
    scenario = _ImportScenario(source_build=False)
    scenario.install(monkeypatch)

    adapter = create_l2_adapter(_obj_config(), L1MemoryDesc(4096, 8192, 4096))
    adapter.close()

    assert scenario.attempts == ["lmcache.lmcache_nixl", "nixl", "lmcache_nixl"]


def test_factory_reports_nixl_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing nixl import (e.g. missing OpenSSL 3) is reported."""
    scenario = _ImportScenario(
        source_build=False,
        nixl_error="libssl.so.3: cannot open shared object file",
    )
    scenario.install(monkeypatch)

    with pytest.raises(RuntimeError, match="libssl.so.3"):
        create_l2_adapter(_obj_config(), L1MemoryDesc(4096, 8192, 4096))


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
