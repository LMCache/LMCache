# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for the opt-in shared Device-DAX L1 path."""

# Standard
import argparse
import json

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
    validate_storage_manager_config,
)
from lmcache.v1.multiprocess.config import MPServerConfig, P2PConfig, add_mp_server_args
from lmcache.v1.multiprocess.server import _validate_shared_l1_runtime_config


def _shared_args() -> list[str]:
    return [
        "--l1-size-gb",
        "2",
        "--l1-devdax-path",
        "/dev/dax1.0",
        "--no-l1-use-lazy",
        "--shm-name",
        "",
        "--l1-align-bytes",
        "64",
        "--eviction-policy",
        "noop",
        "--shared-l1-coordinator",
        "coordinator.shared-dax-e2e.svc:9301",
        "--shared-l1-authkey-file",
        "/var/run/secrets/lmcache/shared-l1-authkey",
        "--shared-l1-region-id",
        "cxl-window-0",
        "--shared-l1-layout-id",
        "scratch-v1",
        "--shared-l1-mapping-offset",
        "4096",
        "--shared-l1-visibility-library-path",
        "/opt/lmcache/lib/liblmcache_shared_l1_visibility.so",
    ]


def _parse_mp_args(args: list[str]) -> StorageManagerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    return parse_args_to_config(parser.parse_args(args))


def test_shared_l1_flags_build_expected_config() -> None:
    config = _parse_mp_args(_shared_args())
    validate_storage_manager_config(config)

    shared = config.l1_manager_config.shared_l1_config
    assert shared is not None
    assert (shared.coordinator_host, shared.coordinator_port) == (
        "coordinator.shared-dax-e2e.svc",
        9301,
    )
    assert shared.region_id == "cxl-window-0"
    assert shared.layout_id == "scratch-v1"
    assert shared.mapping_offset == 4096
    assert shared.authkey_file == "/var/run/secrets/lmcache/shared-l1-authkey"


def test_shared_l1_requires_device_dax() -> None:
    args = _shared_args()
    path_index = args.index("--l1-devdax-path")
    del args[path_index : path_index + 2]

    with pytest.raises(ValueError, match="requires l1-devdax-path"):
        _parse_mp_args(args)


def test_shared_l1_rejects_malformed_coordinator_address() -> None:
    args = _shared_args()
    args[args.index("--shared-l1-coordinator") + 1] = "coordinator"

    with pytest.raises(ValueError, match="HOST:PORT"):
        _parse_mp_args(args)


def test_shared_l1_rejects_eviction_and_matching_dax_l2() -> None:
    eviction_args = _shared_args()
    eviction_args[eviction_args.index("noop")] = "LRU"
    with pytest.raises(ValueError, match="eviction_policy='noop'"):
        _parse_mp_args(eviction_args)

    dax_l2_args = _shared_args() + [
        "--l2-adapter",
        json.dumps(
            {
                "type": "dax",
                "slot_bytes": 196608,
                "device_path": "/dev/dax1.0",
                "max_dax_size_gb": 2,
            }
        ),
    ]
    with pytest.raises(ValueError, match="cannot be combined with L2"):
        _parse_mp_args(dax_l2_args)


def test_shared_l1_runtime_requires_lmcache_driven_without_p2p() -> None:
    storage_config = _parse_mp_args(_shared_args())
    _validate_shared_l1_runtime_config(
        MPServerConfig(supported_transfer_mode="lmcache_driven"),
        storage_config,
    )

    with pytest.raises(ValueError, match="supported_transfer_mode"):
        _validate_shared_l1_runtime_config(MPServerConfig(), storage_config)
    with pytest.raises(ValueError, match="P2P"):
        _validate_shared_l1_runtime_config(
            MPServerConfig(
                supported_transfer_mode="lmcache_driven",
                p2p_config=P2PConfig(advertise_url="10.0.0.1:9000"),
            ),
            storage_config,
        )
