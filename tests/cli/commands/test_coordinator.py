# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``lmcache coordinator`` CLI command."""

# Standard
from unittest.mock import MagicMock, patch
import argparse

# Third Party
import pytest

# First Party
from lmcache.cli.commands.coordinator import (
    CoordinatorCommand,
    _parse_extra_config,
)


@pytest.fixture
def cmd():
    return CoordinatorCommand()


@pytest.fixture
def parser(cmd):
    """An ArgumentParser with CoordinatorCommand's arguments registered."""
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    cmd.register(sub)
    return p


class TestCoordinatorCommandMetadata:
    def test_name(self, cmd):
        assert cmd.name() == "coordinator"

    def test_help(self, cmd):
        assert "coordinator" in cmd.help().lower()


class TestCoordinatorCommandArguments:
    def test_all_flags_registered(self, parser):
        """Every MPCoordinatorConfig field is settable via a CLI flag."""
        args = parser.parse_args(
            [
                "coordinator",
                "--host",
                "127.0.0.1",
                "--port",
                "9999",
                "--instance-timeout",
                "15",
                "--health-check-interval",
                "7",
                "--eviction-check-interval",
                "3",
                "--eviction-ratio",
                "0.5",
                "--trigger-watermark",
                "0.9",
                "--chunk-size",
                "512",
                "--hash-algorithm",
                "sha256",
                "--blend-probe-stride",
                "2",
                "--checkpoint-path",
                "/tmp/checkpoint",
                "--checkpoint-interval",
                "30",
                "--metadata-path",
                "/tmp/metadata.json",
                "--extra-config",
                '{"my_view.window": 8}',
                "--timeout-keep-alive",
                "15",
                "--disable-metrics",
                "--otlp-endpoint",
                "http://collector:4317",
            ]
        )
        assert args.host == "127.0.0.1"
        assert args.port == 9999
        assert args.chunk_size == 512
        assert args.hash_algorithm == "sha256"
        assert args.blend_probe_stride == 2
        assert args.checkpoint_path == "/tmp/checkpoint"
        assert args.checkpoint_interval == 30.0
        assert args.metadata_path == "/tmp/metadata.json"
        assert args.extra_config == '{"my_view.window": 8}'
        assert args.timeout_keep_alive == 15
        assert args.disable_metrics is True
        assert args.otlp_endpoint == "http://collector:4317"

    def test_enable_blend_lookup_flag(self, parser):
        """The blend-lookup switch parses as True when passed."""
        args = parser.parse_args(["coordinator", "--enable-blend-lookup"])
        assert args.enable_blend_lookup is True

    def test_flags_default_to_none(self, parser):
        """Unset flags default to None so the config defaults win."""
        args = parser.parse_args(["coordinator"])
        assert args.chunk_size is None
        assert args.hash_algorithm is None
        assert args.enable_blend_lookup is None
        assert args.blend_probe_stride is None
        assert args.checkpoint_path is None
        assert args.checkpoint_interval is None
        assert args.metadata_path is None
        assert args.extra_config is None
        assert args.timeout_keep_alive is None
        assert args.disable_metrics is None
        assert args.otlp_endpoint is None


class TestCoordinatorCommandExecute:
    def test_overrides_applied(self, cmd):
        """chunk_size/hash_algorithm/blend flags override the config."""
        # First Party
        from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

        args = argparse.Namespace(
            host=None,
            port=None,
            instance_timeout=None,
            health_check_interval=None,
            eviction_check_interval=None,
            eviction_ratio=None,
            trigger_watermark=None,
            chunk_size=512,
            hash_algorithm="sha256",
            enable_blend_lookup=True,
            blend_probe_stride=2,
            checkpoint_path="/var/lib/lmcache/checkpoint",
            checkpoint_interval=30.0,
            metadata_path="/var/lib/lmcache/metadata.json",
            extra_config='{"my_view.window": 8}',
            timeout_keep_alive=None,
            disable_metrics=True,
            otlp_endpoint="http://collector:4317",
        )

        captured = {}

        def fake_create_app(config: MPCoordinatorConfig):
            captured["config"] = config
            return MagicMock()

        with (
            patch("uvicorn.run"),
            patch(
                "lmcache.v1.mp_coordinator.observability.init_coordinator_metrics"
            ) as mock_init_metrics,
            patch(
                "lmcache.v1.mp_coordinator.app.create_app",
                side_effect=fake_create_app,
            ),
        ):
            cmd.execute(args)

        assert captured["config"].chunk_size == 512
        assert captured["config"].hash_algorithm == "sha256"
        assert captured["config"].enable_blend_lookup is True
        assert captured["config"].blend_probe_stride == 2
        assert captured["config"].checkpoint_path == "/var/lib/lmcache/checkpoint"
        assert captured["config"].checkpoint_interval == 30.0
        assert captured["config"].metadata_path == "/var/lib/lmcache/metadata.json"
        assert captured["config"].extra_config == {"my_view.window": 8}
        assert captured["config"].metrics_enabled is False
        assert captured["config"].otlp_endpoint == "http://collector:4317"
        # Unset flags keep the config defaults.
        assert captured["config"].host == MPCoordinatorConfig.host
        assert captured["config"].port == MPCoordinatorConfig.port
        mock_init_metrics.assert_called_once_with(captured["config"])

    def test_env_vars_ignored(self, cmd, monkeypatch):
        """Config is CLI-only: LMCACHE_MP_COORDINATOR_* no longer has an effect."""
        # First Party
        from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig

        monkeypatch.setenv("LMCACHE_MP_COORDINATOR_PORT", "7777")
        monkeypatch.setenv("LMCACHE_MP_COORDINATOR_HOST", "10.0.0.1")
        monkeypatch.setenv("LMCACHE_MP_COORDINATOR_OTLP_ENDPOINT", "http://x:4317")

        args = argparse.Namespace(
            host=None,
            port=None,
            instance_timeout=None,
            health_check_interval=None,
            eviction_check_interval=None,
            eviction_ratio=None,
            trigger_watermark=None,
            chunk_size=None,
            hash_algorithm=None,
            enable_blend_lookup=None,
            blend_probe_stride=None,
            checkpoint_path=None,
            checkpoint_interval=None,
            metadata_path=None,
            extra_config=None,
            timeout_keep_alive=None,
            disable_metrics=None,
            otlp_endpoint=None,
        )

        captured = {}

        def fake_create_app(config: MPCoordinatorConfig):
            captured["config"] = config
            return MagicMock()

        with (
            patch("uvicorn.run"),
            patch("lmcache.v1.mp_coordinator.observability.init_coordinator_metrics"),
            patch(
                "lmcache.v1.mp_coordinator.app.create_app",
                side_effect=fake_create_app,
            ),
        ):
            cmd.execute(args)

        assert captured["config"] == MPCoordinatorConfig()


class TestExtraConfig:
    """``--extra-config`` is how a discovered view or controller gets a
    setting without this class, the CLI and the docs all having to learn
    its name."""

    def test_an_unset_flag_leaves_the_default_alone(self):
        assert _parse_extra_config(None) is None

    @pytest.mark.parametrize(
        ("raw", "reason"),
        [
            ("[1, 2]", "a list"),
            ('"text"', "a bare string"),
            ("{not json", "unparsable"),
        ],
    )
    def test_a_value_that_is_not_an_object_is_refused(self, raw: str, reason: str):
        """Left to the config, this would surface far from here -- on the
        first lookup by whichever component reads it."""
        with pytest.raises(ValueError, match="--extra-config"):
            _parse_extra_config(raw)
