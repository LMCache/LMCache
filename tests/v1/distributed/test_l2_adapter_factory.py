# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for the L2 adapter factory registry and
PluginL2AdapterConfig.
"""

# Standard
import types

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
    register_l2_adapter_factory,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.l2_adapters.plugin_l2_adapter import PluginL2AdapterConfig

# =========================================================
# Helpers
# =========================================================


class _FakeL2Adapter:
    """Minimal stub that passes issubclass check."""

    def __init__(self, params=None, **kwargs):
        self.params = params
        self.kwargs = kwargs


class _NotAnAdapter:
    """Class that does NOT subclass L2AdapterInterface."""

    pass


def _make_fake_module(
    adapter_cls: type,
    class_name: str = "FakeL2Adapter",
) -> types.ModuleType:
    """Create a fake Python module containing *adapter_cls*."""
    mod = types.ModuleType("fake_l2_module")
    setattr(mod, class_name, adapter_cls)
    return mod


# =========================================================
# Tests: factory registry basics
# =========================================================


class TestFactoryRegistry:
    """Tests for register/create via factory registry."""

    def test_mock_factory_is_registered(self):
        """Mock adapter factory should be auto-registered
        on import."""
        config = MockL2AdapterConfig(
            max_size_gb=0.001,
            mock_bandwidth_gb=10.0,
        )
        name = get_type_name_for_config(config)
        assert name == "mock"

    def test_create_mock_via_registry(self):
        """create_l2_adapter_from_registry creates a
        MockL2Adapter."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
            MockL2Adapter,
        )

        config = MockL2AdapterConfig(
            max_size_gb=0.001,
            mock_bandwidth_gb=10.0,
        )
        adapter = create_l2_adapter_from_registry(config)
        assert isinstance(adapter, MockL2Adapter)
        adapter.close()

    def test_duplicate_factory_raises(self):
        """Registering the same factory name twice should
        raise ValueError."""
        with pytest.raises(ValueError, match="already registered"):
            register_l2_adapter_factory("mock", lambda c, **kw: None)

    def test_unknown_config_raises(self):
        """Config class not in the registry should fail."""

        class _UnknownConfig:
            pass

        with pytest.raises(ValueError, match="Unregistered"):
            get_type_name_for_config(_UnknownConfig())


# =========================================================
# Tests: PluginL2AdapterConfig parsing
# =========================================================


class TestPluginL2AdapterConfig:
    """Tests for PluginL2AdapterConfig.from_dict."""

    def test_valid_config(self):
        d = {
            "type": "plugin",
            "module_path": "my.module",
            "class_name": "MyAdapter",
            "adapter_params": {"host": "localhost"},
        }
        cfg = PluginL2AdapterConfig.from_dict(d)
        assert cfg.module_path == "my.module"
        assert cfg.class_name == "MyAdapter"
        assert cfg.adapter_params == {"host": "localhost"}

    def test_default_adapter_params(self):
        d = {
            "type": "plugin",
            "module_path": "my.module",
            "class_name": "MyAdapter",
        }
        cfg = PluginL2AdapterConfig.from_dict(d)
        assert cfg.adapter_params == {}

    def test_missing_module_path_raises(self):
        d = {"type": "plugin", "class_name": "X"}
        with pytest.raises(ValueError, match="module_path"):
            PluginL2AdapterConfig.from_dict(d)

    def test_missing_class_name_raises(self):
        d = {"type": "plugin", "module_path": "x"}
        with pytest.raises(ValueError, match="class_name"):
            PluginL2AdapterConfig.from_dict(d)

    def test_invalid_adapter_params_raises(self):
        d = {
            "type": "plugin",
            "module_path": "x",
            "class_name": "X",
            "adapter_params": "not_a_dict",
        }
        with pytest.raises(ValueError, match="adapter_params"):
            PluginL2AdapterConfig.from_dict(d)


# =========================================================
# Tests: plugin adapter factory dynamic loading
# =========================================================


class TestPluginAdapterFactory:
    """Tests for the plugin adapter factory using monkeypatch
    to mock importlib.import_module."""

    def test_load_external_adapter(self, monkeypatch):
        """Successfully load an external adapter class."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import plugin_l2_adapter as plugin_mod

        # Make _FakeL2Adapter pass issubclass check
        monkeypatch.setattr(
            plugin_mod,
            "_L2AI",
            _FakeL2Adapter,
        )

        fake_mod = _make_fake_module(_FakeL2Adapter)
        monkeypatch.setattr(
            "importlib.import_module",
            lambda path: fake_mod,
        )

        config = PluginL2AdapterConfig(
            module_path="fake_l2_module",
            class_name="FakeL2Adapter",
            adapter_params={"host": "localhost"},
        )
        adapter = create_l2_adapter_from_registry(config)
        assert isinstance(adapter, _FakeL2Adapter)
        assert adapter.params["host"] == "localhost"

    def test_import_error_raises(self, monkeypatch):
        """ImportError propagates when module not found."""
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: (_ for _ in ()).throw(ImportError("no such module")),
        )
        config = PluginL2AdapterConfig(
            module_path="nonexistent",
            class_name="X",
        )
        with pytest.raises(ImportError, match="nonexistent"):
            create_l2_adapter_from_registry(config)

    def test_missing_class_raises(self, monkeypatch):
        """AttributeError when class not in module."""
        fake_mod = types.ModuleType("empty_mod")
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )
        config = PluginL2AdapterConfig(
            module_path="empty_mod",
            class_name="NoSuchClass",
        )
        with pytest.raises(AttributeError, match="NoSuchClass"):
            create_l2_adapter_from_registry(config)

    def test_not_subclass_raises(self, monkeypatch):
        """TypeError when class is not an L2AdapterInterface
        subclass."""
        fake_mod = _make_fake_module(_NotAnAdapter, "BadAdapter")
        monkeypatch.setattr(
            "importlib.import_module",
            lambda p: fake_mod,
        )
        config = PluginL2AdapterConfig(
            module_path="fake_mod",
            class_name="BadAdapter",
        )
        with pytest.raises(TypeError, match="not a subclass"):
            create_l2_adapter_from_registry(config)


# =========================================================
# Tests: plugin registration and initialization
# =========================================================


class _FakeL2AdapterWithDesc:
    """Stub that records l1_memory_desc."""

    def __init__(self, config, **kwargs):
        self.config = config
        self.l1_memory_desc = kwargs.get("l1_memory_desc")


class _FakeConfig(L2AdapterConfigBase):
    """Minimal config subclass for discovery tests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @classmethod
    def from_dict(cls, d: dict) -> "_FakeConfig":
        return cls(**d)

    @classmethod
    def help(cls) -> str:
        return "fake"


class TestPluginRegistration:
    """Verify plugin self-registration on import."""

    def test_plugin_type_registered(self):
        """'plugin' config type should be registered."""
        cfg = PluginL2AdapterConfig(module_path="x", class_name="X")
        name = get_type_name_for_config(cfg)
        assert name == "plugin"

    def test_plugin_factory_registered(self):
        """'plugin' factory should be callable via
        create_l2_adapter_from_registry (smoke test
        with ImportError)."""
        cfg = PluginL2AdapterConfig(
            module_path="nonexistent.module",
            class_name="X",
        )
        with pytest.raises(ImportError):
            create_l2_adapter_from_registry(cfg)


class TestPluginInitialization:
    """Verify config-class discovery and l1_memory_desc
    forwarding during plugin initialization."""

    def _patch_base(self, monkeypatch, adapter_cls):
        """Make *adapter_cls* pass issubclass check."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import plugin_l2_adapter as plugin_mod

        monkeypatch.setattr(
            plugin_mod,
            "_L2AI",
            adapter_cls,
        )

    # -- l1_memory_desc forwarding --

    def test_l1_memory_desc_forwarded(self, monkeypatch):
        """l1_memory_desc should reach the adapter
        constructor when provided."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            adapter_params={"k": "v"},
        )
        sentinel = object()
        adapter = create_l2_adapter_from_registry(cfg, l1_memory_desc=sentinel)
        assert adapter.l1_memory_desc is sentinel

    def test_l1_memory_desc_omitted(self, monkeypatch):
        """l1_memory_desc should be None when not
        provided."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert adapter.l1_memory_desc is None

    # -- config class discovery --

    def test_init_with_explicit_config_class(self, monkeypatch):
        """Explicit config_class_name should be used."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        fake_mod.MyCfg = _FakeConfig  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            config_class_name="MyCfg",
            adapter_params={"x": 1},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter.config, _FakeConfig)
        assert adapter.config.kwargs == {"x": 1}

    def test_init_with_convention_config_class(self, monkeypatch):
        """Config class discovered via ClassName+'Config'
        convention."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        # Convention: "Adapter" + "Config"
        fake_mod.AdapterConfig = _FakeConfig  # type: ignore[attr-defined]
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            adapter_params={"y": 2},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter.config, _FakeConfig)

    def test_init_fallback_raw_dict(self, monkeypatch):
        """When no config class is found, adapter receives
        a raw dict."""
        self._patch_base(monkeypatch, _FakeL2AdapterWithDesc)
        fake_mod = _make_fake_module(
            _FakeL2AdapterWithDesc,
            "Adapter",
        )
        monkeypatch.setattr(
            "importlib.import_module",
            lambda _: fake_mod,
        )
        cfg = PluginL2AdapterConfig(
            module_path="m",
            class_name="Adapter",
            adapter_params={"z": 3},
        )
        adapter = create_l2_adapter_from_registry(cfg)
        assert isinstance(adapter.config, dict)
        assert adapter.config == {"z": 3}


# =========================================================
# Tests: create_l2_adapter public API
# =========================================================


class TestCreateL2Adapter:
    """Tests for the public create_l2_adapter function."""

    def test_create_mock_adapter(self):
        """create_l2_adapter dispatches to MockL2Adapter."""
        # First Party
        from lmcache.v1.distributed.l2_adapters import (
            create_l2_adapter,
        )
        from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import (
            MockL2Adapter,
        )

        config = MockL2AdapterConfig(
            max_size_gb=0.001,
            mock_bandwidth_gb=10.0,
        )
        adapter = create_l2_adapter(config)
        assert isinstance(adapter, MockL2Adapter)
        adapter.close()
