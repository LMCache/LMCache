# SPDX-License-Identifier: Apache-2.0
"""Shared CLI validation and supported server modes."""

# Standard
from dataclasses import replace
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    DevDaxL1ManagerConfig,
    DRAML1ManagerConfig,
    parse_args,
)
from lmcache.v1.multiprocess.config import (
    MPServerConfig,
    P2PConfig,
    RuntimePluginConfig,
)
from lmcache.v1.multiprocess.server import prepare_shared_l1


@pytest.fixture
def spec() -> dict[str, object]:
    return {
        "type": "DRAM",
        "tag": "pool",
        "size_gb": 1,
        "path": "/unused",
        "shared": {"orchestrator": "h:1", "region_id": "region", "client_id": "host"},
    }


def args(specs: list[dict[str, object]]) -> list[str]:
    return [
        "--eviction-policy",
        "LRU",
        *[arg for spec in specs for arg in ("--l1-manager", json.dumps(spec))],
    ]


@pytest.mark.parametrize(
    "kind,cls", [("DRAM", DRAML1ManagerConfig), ("DEVDAX", DevDaxL1ManagerConfig)]
)
def test_shared_config_and_affinity(
    spec: dict[str, object], kind: str, cls: type
) -> None:
    spec["type"] = kind
    private = {"type": "DRAM", "tag": "_default", "size_gb": 1}
    flags = args([spec, private])
    config = parse_args(flags)
    shared = config.l1_manager_configs[0]
    assert isinstance(shared, cls) and shared.shared is not None
    assert (shared.shared.client_id, shared.shared.path) == ("host", "/unused")
    assert shared.eviction is not None and shared.eviction.eviction_policy == "noop"
    prepare_shared_l1(MPServerConfig(supported_transfer_mode="lmcache_driven"), config)
    adapter = {"type": "fs", "base_path": "/unused", "affinity_tag": "_default"}
    parse_args(flags + ["--l2-adapter", json.dumps(adapter)])
    adapter["affinity_tag"] = "pool"
    with pytest.raises(ValueError, match="cannot bind to the shared"):
        parse_args(flags + ["--l2-adapter", json.dumps(adapter)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("orchestrator", ""),
        ("region_id", None),
        ("client_id", None),
        ("rpc_timeout_seconds", 0),
        ("unknown", 1),
    ],
)
def test_malformed_shared_fields(
    spec: dict[str, object], field: str, value: object
) -> None:
    shared = spec["shared"]
    assert isinstance(shared, dict)
    shared[field] = value
    with pytest.raises(ValueError):
        parse_args(args([spec]))


@pytest.mark.parametrize(
    "change",
    [
        {"shared": "h:1"},
        {"type": "GDS"},
        {"eviction": {"eviction_policy": "LRU"}},
        {"path": None},
        {"shared": {"orchestrator": "h:1", "region_id": "r"}},
    ],
)
def test_invalid_shared_settings(
    spec: dict[str, object], change: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        parse_args(args([{**spec, **change}]))


def test_only_one_shared_region(spec: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Only one shared"):
        parse_args(args([spec, {**spec, "tag": "peer"}]))


BASE_MP = MPServerConfig(supported_transfer_mode="lmcache_driven")


@pytest.mark.parametrize(
    "mp_config",
    [
        replace(BASE_MP, supported_transfer_mode="auto"),
        replace(BASE_MP, engine_type="blend"),
        replace(BASE_MP, p2p_config=P2PConfig(advertise_url="h:9000")),
        replace(
            BASE_MP, runtime_plugin_config=RuntimePluginConfig(locations=["plugin.py"])
        ),
    ],
)
def test_unsupported_server_modes(
    spec: dict[str, object], mp_config: MPServerConfig
) -> None:
    with pytest.raises(ValueError):
        prepare_shared_l1(mp_config, parse_args(args([spec])))
