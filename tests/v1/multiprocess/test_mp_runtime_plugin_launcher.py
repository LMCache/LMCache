# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MPRuntimePluginLauncher and helper functions.
"""

# Standard
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch
import json

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.mp_runtime_plugin_launcher import (
    MPRuntimePluginLauncher,
    _make_json_safe,
    _MPPluginConfig,
    _safe_asdict,
)

# ---------------------------------------------------------------------------
# Test dataclasses used as config fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeServerConfig:
    host: str = "localhost"
    port: int = 8080
    chunk_size: int = 256


@dataclass
class _FakeStorageConfig:
    backend: str = "local"
    capacity: int = 1024


@dataclass
class _FakeConfigWithPath:
    """Config containing a non-JSON-serializable field."""

    name: str = "test"
    path: Path = Path("/tmp/test")


# ---------------------------------------------------------------------------
# _make_json_safe
# ---------------------------------------------------------------------------


class TestMakeJsonSafe:
    def test_primitive_types(self):
        assert _make_json_safe("hello") == "hello"
        assert _make_json_safe(42) == 42
        assert _make_json_safe(3.14) == 3.14
        assert _make_json_safe(True) is True
        assert _make_json_safe(None) is None

    def test_dict(self):
        result = _make_json_safe({"a": 1, "b": "two"})
        assert result == {"a": 1, "b": "two"}

    def test_list_and_tuple(self):
        assert _make_json_safe([1, "x", None]) == [1, "x", None]
        assert _make_json_safe((1, 2)) == [1, 2]

    def test_nested_structure(self):
        data = {"a": [1, {"b": (True, None)}]}
        result = _make_json_safe(data)
        assert result == {"a": [1, {"b": [True, None]}]}

    def test_non_serializable_fallback(self):
        p = Path("/some/path")
        result = _make_json_safe(p)
        assert result == str(p)

    def test_nested_non_serializable(self):
        data = {"path": Path("/tmp"), "items": [Path("/a")]}
        result = _make_json_safe(data)
        assert result == {
            "path": "/tmp",
            "items": ["/a"],
        }


# ---------------------------------------------------------------------------
# _safe_asdict
# ---------------------------------------------------------------------------


class TestSafeAsdict:
    def test_simple_dataclass(self):
        cfg = _FakeServerConfig()
        result = _safe_asdict(cfg)
        assert result == {
            "host": "localhost",
            "port": 8080,
            "chunk_size": 256,
        }

    def test_non_serializable_field(self):
        cfg = _FakeConfigWithPath(name="x", path=Path("/foo"))
        result = _safe_asdict(cfg)
        assert result["name"] == "x"
        assert result["path"] == "/foo"
        # Ensure the result is JSON-serializable
        json.dumps(result)

    def test_rejects_non_dataclass(self):
        with pytest.raises(TypeError, match="Expected a dataclass"):
            _safe_asdict({"not": "a dataclass"})

    def test_rejects_plain_object(self):
        with pytest.raises(TypeError, match="Expected a dataclass"):
            _safe_asdict(object())


# ---------------------------------------------------------------------------
# _MPPluginConfig
# ---------------------------------------------------------------------------


class TestMPPluginConfig:
    def test_to_json(self):
        cfg = _MPPluginConfig(
            runtime_plugin_locations=["/a", "/b"],
            configs_dict={"server": {"host": "localhost"}},
        )
        parsed = json.loads(cfg.to_json())
        assert parsed == {"server": {"host": "localhost"}}

    def test_runtime_plugin_locations(self):
        cfg = _MPPluginConfig(
            runtime_plugin_locations=["/x"],
            configs_dict={},
        )
        assert cfg.runtime_plugin_locations == ["/x"]


# ---------------------------------------------------------------------------
# MPRuntimePluginLauncher
# ---------------------------------------------------------------------------


class TestMPRuntimePluginLauncher:
    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_init_aggregates_configs(self, mock_rpl_cls):
        """Configs are aggregated into a single JSON dict."""
        srv = _FakeServerConfig(host="0.0.0.0", port=9090)
        stg = _FakeStorageConfig(backend="redis", capacity=2048)

        MPRuntimePluginLauncher(
            runtime_plugin_locations=["/plugins"],
            server=srv,
            storage=stg,
        )

        mock_rpl_cls.assert_called_once()
        call_kwargs = mock_rpl_cls.call_args[1]
        wrapper = call_kwargs["config"]

        assert wrapper.runtime_plugin_locations == ["/plugins"]
        parsed = json.loads(wrapper.to_json())
        assert parsed["server"]["host"] == "0.0.0.0"
        assert parsed["server"]["port"] == 9090
        assert parsed["storage"]["backend"] == "redis"
        assert parsed["storage"]["capacity"] == 2048

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_init_with_non_serializable_fields(self, mock_rpl_cls):
        """Non-serializable fields are converted to str."""
        cfg = _FakeConfigWithPath(name="test", path=Path("/data"))

        MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
            my_config=cfg,
        )

        wrapper = mock_rpl_cls.call_args[1]["config"]
        parsed = json.loads(wrapper.to_json())
        assert parsed["my_config"]["path"] == "/data"

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_init_no_configs(self, mock_rpl_cls):
        """Launcher works with zero extra configs."""
        MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
        )

        wrapper = mock_rpl_cls.call_args[1]["config"]
        assert json.loads(wrapper.to_json()) == {}

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_init_passes_server_role(self, mock_rpl_cls):
        """Inner launcher is created with role=SERVER."""
        MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
        )

        call_kwargs = mock_rpl_cls.call_args[1]
        assert call_kwargs["role"] == "SERVER"
        assert call_kwargs["worker_count"] == 1
        assert call_kwargs["worker_id"] == 0

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_launch_plugins_delegates(self, mock_rpl_cls):
        """launch_plugins delegates to inner launcher."""
        launcher = MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
        )
        launcher.launch_plugins()
        mock_rpl_cls.return_value.launch_plugins.assert_called_once()

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_stop_plugins_delegates(self, mock_rpl_cls):
        """stop_plugins delegates to inner launcher."""
        launcher = MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
        )
        launcher.stop_plugins()
        mock_rpl_cls.return_value.stop_plugins.assert_called_once()
