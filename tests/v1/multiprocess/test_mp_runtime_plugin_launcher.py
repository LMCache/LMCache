# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for MPRuntimePluginLauncher.
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
    def test_init_rejects_non_dataclass(self, mock_rpl_cls):
        """Passing a non-dataclass config raises TypeError."""
        with pytest.raises(TypeError, match="Expected a dataclass"):
            MPRuntimePluginLauncher(
                runtime_plugin_locations=["/p"],
                bad_config={"not": "a dataclass"},
            )

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_init_no_configs(self, mock_rpl_cls):
        """Launcher works with zero extra configs."""
        MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
        )

        wrapper = mock_rpl_cls.call_args[1]["config"]
        assert json.loads(wrapper.to_json()) == {}

    @patch("lmcache.v1.multiprocess.mp_runtime_plugin_launcher.RuntimePluginLauncher")
    def test_init_passes_no_role(self, mock_rpl_cls):
        """MP mode has no role; inner launcher gets role=None."""
        MPRuntimePluginLauncher(
            runtime_plugin_locations=["/p"],
        )

        call_kwargs = mock_rpl_cls.call_args[1]
        assert call_kwargs["role"] is None

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
