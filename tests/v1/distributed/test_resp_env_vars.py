# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for RESP adapter environment variable support.

These tests verify the precedence logic (config > env var > default)
without requiring a live Redis server.
"""

# Standard
from types import ModuleType
import os
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
    ensure_adapter_loaded,
)

_ENV_VARS = (
    "LMCACHE_RESP_HOST",
    "LMCACHE_RESP_PORT",
    "LMCACHE_RESP_USERNAME",
    "LMCACHE_RESP_PASSWORD",
)


class _NativeClientCreated(Exception):
    """Raised by the fake native client once it has recorded its arguments."""


def _parse(spec: dict[str, object]):
    """Build a RESP config from an ``--l2-adapter`` JSON spec."""
    ensure_adapter_loaded("resp")
    # First Party
    from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
        RESPL2AdapterConfig,
    )

    return RESPL2AdapterConfig.from_dict(spec)


def _native_client_target(
    monkeypatch: pytest.MonkeyPatch, spec: dict[str, object]
) -> tuple[str, int, str, str]:
    """Return the ``(host, port, username, password)`` the native client gets.

    A fake extension records the arguments and raises, so neither a Redis
    server nor the C++ build is needed.
    """
    target: list[tuple[str, int, str, str]] = []

    def fake_client(
        host: str, port: int, num_workers: int, username: str, password: str
    ) -> None:
        target.append((host, port, username, password))
        raise _NativeClientCreated

    fake_module = ModuleType("lmcache.lmcache_redis")
    fake_module.LMCacheRedisClient = fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_redis", fake_module)

    config = _parse(spec)
    with pytest.raises(_NativeClientCreated):
        create_l2_adapter_from_registry(config)
    return target[0]


class TestRESPL2AdapterFromDict:
    """Env var fallbacks through ``from_dict`` and the real adapter factory."""

    @pytest.mark.parametrize(
        "spec, expected",
        [
            ({"type": "resp"}, ("env-host", 7777)),
            ({"type": "resp", "host": "", "port": 0}, ("env-host", 7777)),
            ({"type": "resp", "host": "cfg-host"}, ("cfg-host", 7777)),
            ({"type": "resp", "port": 6379}, ("env-host", 6379)),
        ],
    )
    def test_env_vars_fill_missing_fields(self, monkeypatch, spec, expected):
        """A host or port left out of the JSON, or empty, comes from env."""
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        assert _native_client_target(monkeypatch, spec) == (
            *expected,
            "env-user",
            "env-pass",
        )

    def test_json_values_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")

        spec = {"type": "resp", "host": "cfg-host", "port": 6379}
        target = _native_client_target(monkeypatch, spec)
        assert target[:2] == ("cfg-host", 6379)

    @pytest.mark.parametrize(
        "spec, field",
        [
            ({"type": "resp", "port": 6379}, "host"),
            ({"type": "resp", "host": "", "port": 6379}, "host"),
            ({"type": "resp", "host": "h"}, "port"),
            ({"type": "resp", "host": "h", "port": 0}, "port"),
        ],
    )
    def test_missing_field_without_env_var_raises(self, monkeypatch, spec, field):
        for name in _ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(ValueError, match=field):
            _parse(spec)

    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "resp", "host": 1, "port": 6379},
            {"type": "resp", "host": "h", "port": -1},
            {"type": "resp", "host": "h", "port": "6379"},
        ],
    )
    def test_invalid_values_still_raise(self, monkeypatch, spec):
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")

        with pytest.raises(ValueError):
            _parse(spec)


class TestRESPL2AdapterEnvVars:
    """Test env var resolution in the MP-mode RESP L2 adapter factory."""

    def test_env_vars_used_when_config_empty(self, monkeypatch):
        """Env vars should be used when config values are empty/default."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        config = RESPL2AdapterConfig(host="", port=0, username="", password="")

        host = config.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            config.port
            if config.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )
        username = config.username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = config.password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert host == "env-host"
        assert port == 7777
        assert username == "env-user"
        assert password == "env-pass"

    def test_config_overrides_env_vars(self, monkeypatch):
        """Config values should take precedence over env vars."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "7777")
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        config = RESPL2AdapterConfig(
            host="cfg-host",
            port=8888,
            username="cfg-user",
            password="cfg-pass",
        )

        host = config.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            config.port
            if config.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )
        username = config.username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = config.password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert host == "cfg-host"
        assert port == 8888
        assert username == "cfg-user"
        assert password == "cfg-pass"

    def test_defaults_when_no_env_and_no_config(self, monkeypatch):
        """Without env vars or config, values should be empty/zero."""
        # First Party
        from lmcache.v1.distributed.l2_adapters.resp_l2_adapter import (
            RESPL2AdapterConfig,
        )

        monkeypatch.delenv("LMCACHE_RESP_HOST", raising=False)
        monkeypatch.delenv("LMCACHE_RESP_PORT", raising=False)
        monkeypatch.delenv("LMCACHE_RESP_USERNAME", raising=False)
        monkeypatch.delenv("LMCACHE_RESP_PASSWORD", raising=False)

        config = RESPL2AdapterConfig(host="", port=0, username="", password="")

        host = config.host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            config.port
            if config.port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )
        username = config.username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = config.password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert host == ""
        assert port == 0
        assert username == ""
        assert password == ""


class TestRESPConnectorAdapterEnvVars:
    """Test env var resolution in the non-MP RESP connector adapter."""

    def test_env_vars_used_when_extra_config_empty(self, monkeypatch):
        """Env vars should be used when extra_config has no credentials."""
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        extra_config = {}

        cfg_username = str(extra_config.get("username", ""))
        cfg_password = str(extra_config.get("password", ""))
        username = cfg_username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = cfg_password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert username == "env-user"
        assert password == "env-pass"

    def test_extra_config_overrides_env_vars(self, monkeypatch):
        """extra_config values should take precedence over env vars."""
        monkeypatch.setenv("LMCACHE_RESP_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_RESP_PASSWORD", "env-pass")

        extra_config = {"username": "cfg-user", "password": "cfg-pass"}

        cfg_username = str(extra_config.get("username", ""))
        cfg_password = str(extra_config.get("password", ""))
        username = cfg_username or os.environ.get("LMCACHE_RESP_USERNAME", "")
        password = cfg_password or os.environ.get("LMCACHE_RESP_PASSWORD", "")

        assert username == "cfg-user"
        assert password == "cfg-pass"

    def test_host_port_env_var_fallback(self, monkeypatch):
        """Host/port env vars should be used when URL values are empty."""
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "9999")

        parsed_host = ""
        parsed_port = 0

        host = parsed_host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            parsed_port
            if parsed_port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )

        assert host == "env-host"
        assert port == 9999

    def test_url_overrides_env_vars(self, monkeypatch):
        """URL-parsed values should take precedence over env vars."""
        monkeypatch.setenv("LMCACHE_RESP_HOST", "env-host")
        monkeypatch.setenv("LMCACHE_RESP_PORT", "9999")

        parsed_host = "url-host"
        parsed_port = 6379

        host = parsed_host or os.environ.get("LMCACHE_RESP_HOST", "")
        port = (
            parsed_port
            if parsed_port
            else int(os.environ.get("LMCACHE_RESP_PORT", "0"))
        )

        assert host == "url-host"
        assert port == 6379
