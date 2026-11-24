# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Valkey adapter environment variable support.

These tests verify the precedence logic (JSON config > env var > default)
for LMCACHE_VALKEY_USERNAME and LMCACHE_VALKEY_PASSWORD without requiring
a live Valkey server.

Unlike the RESP adapter, the Valkey adapter resolves env vars inside
``from_dict`` / ``_kwargs_from_dict``, so the resulting config object
already holds the final credentials. No factory-level mocking is needed.
"""

# First Party
from lmcache.v1.distributed.l2_adapters.factory import ensure_adapter_loaded

_ENV_VARS = (
    "LMCACHE_VALKEY_USERNAME",
    "LMCACHE_VALKEY_PASSWORD",
)

_BASE = {"type": "valkey", "startup_nodes": "127.0.0.1:6379"}


def _parse(spec: dict) -> object:
    ensure_adapter_loaded("valkey")
    from lmcache.v1.distributed.l2_adapters.valkey_l2_adapter import (
        ValkeyL2AdapterConfig,
    )

    return ValkeyL2AdapterConfig.from_dict(spec)


class TestValkeyL2AdapterEnvVars:
    """Env var fallbacks through ``from_dict``; no live server required."""

    def test_env_vars_used_when_config_empty(self, monkeypatch):
        monkeypatch.setenv("LMCACHE_VALKEY_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_VALKEY_PASSWORD", "env-pass")

        cfg = _parse({**_BASE})
        assert cfg.username == "env-user"
        assert cfg.password == "env-pass"

    def test_json_values_override_env_vars(self, monkeypatch):
        monkeypatch.setenv("LMCACHE_VALKEY_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_VALKEY_PASSWORD", "env-pass")

        cfg = _parse({**_BASE, "username": "cfg-user", "password": "cfg-pass"})
        assert cfg.username == "cfg-user"
        assert cfg.password == "cfg-pass"

    def test_partial_override(self, monkeypatch):
        """JSON username wins; env var fills missing password."""
        monkeypatch.setenv("LMCACHE_VALKEY_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_VALKEY_PASSWORD", "env-pass")

        cfg = _parse({**_BASE, "username": "cfg-user"})
        assert cfg.username == "cfg-user"
        assert cfg.password == "env-pass"

    def test_defaults_when_no_env_and_no_json(self, monkeypatch):
        for name in _ENV_VARS:
            monkeypatch.delenv(name, raising=False)

        cfg = _parse({**_BASE})
        assert cfg.username == ""
        assert cfg.password == ""

    def test_empty_string_in_json_falls_back_to_env(self, monkeypatch):
        """An explicit empty string in JSON is treated as not set."""
        monkeypatch.setenv("LMCACHE_VALKEY_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_VALKEY_PASSWORD", "env-pass")

        cfg = _parse({**_BASE, "username": "", "password": ""})
        assert cfg.username == "env-user"
        assert cfg.password == "env-pass"
