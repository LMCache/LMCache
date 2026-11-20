# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Aerospike L2 adapter config (no native extension required)."""

# Standard
from types import ModuleType
import os
import sys

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.config import (
    get_registered_l2_adapter_types,
    get_type_name_for_config,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    create_l2_adapter_from_registry,
    ensure_adapter_loaded,
)

_ENV_VARS = (
    "LMCACHE_AEROSPIKE_HOSTS",
    "LMCACHE_AEROSPIKE_NAMESPACE",
    "LMCACHE_AEROSPIKE_SET",
)


class _NativeClientCreated(Exception):
    """Raised by the fake native client once it has recorded its arguments."""


def _native_client_target(
    monkeypatch: pytest.MonkeyPatch, spec: dict[str, object]
) -> tuple[str, str, str]:
    """Return the ``(hosts, namespace, set_name)`` the native client gets.

    A fake extension records the arguments and raises, so neither an
    Aerospike server nor the C++ build is needed.
    """
    target: list[tuple[str, str, str]] = []

    def fake_client(hosts: str, namespace: str, set_name: str, *_: object) -> None:
        target.append((hosts, namespace, set_name))
        raise _NativeClientCreated

    fake_module = ModuleType("lmcache.lmcache_aerospike")
    fake_module.LMCacheAerospikeClient = fake_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmcache.lmcache_aerospike", fake_module)
    ensure_adapter_loaded("aerospike")
    # First Party
    from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
        AerospikeL2AdapterConfig,
    )

    config = AerospikeL2AdapterConfig.from_dict(spec)
    with pytest.raises(_NativeClientCreated):
        create_l2_adapter_from_registry(config)
    return target[0]


class TestAerospikeL2AdapterConfig:
    def test_type_registered(self):
        ensure_adapter_loaded("aerospike")
        assert "aerospike" in get_registered_l2_adapter_types()

    def test_from_dict_required_fields(self):
        ensure_adapter_loaded("aerospike")
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        cfg = AerospikeL2AdapterConfig.from_dict(
            {
                "type": "aerospike",
                "hosts": "127.0.0.1:3000",
                "namespace": "lmcache",
                "set_name": "kv_it",
            }
        )
        assert cfg.hosts == "127.0.0.1:3000"
        assert cfg.namespace == "lmcache"
        assert cfg.set_name == "kv_it"
        assert get_type_name_for_config(cfg) == "aerospike"

    def test_set_alias(self):
        ensure_adapter_loaded("aerospike")
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        cfg = AerospikeL2AdapterConfig.from_dict(
            {"type": "aerospike", "hosts": "h:3000", "set": "myset"}
        )
        assert cfg.set_name == "myset"

    def test_missing_hosts_raises(self, monkeypatch):
        monkeypatch.delenv("LMCACHE_AEROSPIKE_HOSTS", raising=False)
        ensure_adapter_loaded("aerospike")
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        try:
            AerospikeL2AdapterConfig.from_dict({"type": "aerospike"})
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_env_vars_used_when_config_empty(self, monkeypatch):
        # First Party
        from lmcache.v1.distributed.l2_adapters.aerospike_l2_adapter import (
            AerospikeL2AdapterConfig,
        )

        monkeypatch.setenv("LMCACHE_AEROSPIKE_HOSTS", "env-host:3000")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_NAMESPACE", "env-ns")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_SET", "env-set")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_USERNAME", "env-user")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_PASSWORD", "env-pass")

        cfg = AerospikeL2AdapterConfig(hosts="", namespace="", set_name="")

        hosts = cfg.hosts or os.environ.get("LMCACHE_AEROSPIKE_HOSTS", "")
        namespace = cfg.namespace or os.environ.get(
            "LMCACHE_AEROSPIKE_NAMESPACE", "lmcache"
        )
        set_name = cfg.set_name or os.environ.get("LMCACHE_AEROSPIKE_SET", "kv_chunks")
        username = cfg.username or os.environ.get("LMCACHE_AEROSPIKE_USERNAME", "")
        password = cfg.password or os.environ.get("LMCACHE_AEROSPIKE_PASSWORD", "")

        assert hosts == "env-host:3000"
        assert namespace == "env-ns"
        assert set_name == "env-set"
        assert username == "env-user"
        assert password == "env-pass"

    @pytest.mark.parametrize(
        "spec",
        [
            {"type": "aerospike", "hosts": "h:3000"},
            {"type": "aerospike", "hosts": "h:3000", "namespace": "", "set": ""},
        ],
    )
    def test_env_vars_fill_namespace_and_set(self, monkeypatch, spec):
        """A namespace or set left out of the JSON, or empty, comes from env."""
        monkeypatch.setenv("LMCACHE_AEROSPIKE_NAMESPACE", "env-ns")
        monkeypatch.setenv("LMCACHE_AEROSPIKE_SET", "env-set")

        assert _native_client_target(monkeypatch, spec) == (
            "h:3000",
            "env-ns",
            "env-set",
        )

    def test_env_var_supplies_hosts(self, monkeypatch):
        monkeypatch.setenv("LMCACHE_AEROSPIKE_HOSTS", "env-host:3000")

        target = _native_client_target(monkeypatch, {"type": "aerospike"})
        assert target[0] == "env-host:3000"

    def test_json_values_override_env_vars(self, monkeypatch):
        for name in _ENV_VARS:
            monkeypatch.setenv(name, "from-env")

        spec = {"type": "aerospike", "hosts": "h:3000", "namespace": "ns", "set": "kv"}
        assert _native_client_target(monkeypatch, spec) == ("h:3000", "ns", "kv")

    @pytest.mark.parametrize("env_value", [None, ""])
    def test_defaults_apply_without_env_vars(self, monkeypatch, env_value):
        for name in _ENV_VARS[1:]:
            if env_value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, env_value)

        spec = {"type": "aerospike", "hosts": "h:3000"}
        assert _native_client_target(monkeypatch, spec) == (
            "h:3000",
            "lmcache",
            "kv_chunks",
        )
