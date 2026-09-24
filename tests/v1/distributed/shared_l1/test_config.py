# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for the opt-in shared Device-DAX L1 path."""

# Standard
import argparse
import json
import shlex

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc
from lmcache.v1.distributed.config import (
    StorageManagerConfig,
    add_storage_manager_args,
    parse_args_to_config,
    validate_storage_manager_config,
)
from lmcache.v1.distributed.shared_l1.layouts import (
    dtype_to_wire,
    layout_to_wire,
    wire_to_dtype,
    wire_to_layout,
)
from lmcache.v1.multiprocess.config import MPServerConfig, P2PConfig, add_mp_server_args
from lmcache.v1.multiprocess.server import _validate_shared_l1_runtime_config


def _shared_args() -> list[str]:
    return shlex.split("""
        --l1-size-gb 2 --l1-devdax-path /dev/dax1.0
        --no-l1-use-lazy --shm-name '' --l1-align-bytes 64 --eviction-policy noop
        --l1-coordinator-endpoint http://memory-coordinator.shared-dax-e2e.svc:9400
        --l1-coordinator-token-file /var/run/secrets/lmcache/memory-coordinator-token
        --l1-shared-region-id cxl-window-0 --l1-shared-layout-id scratch-v1
        --l1-shared-mapping-offset-bytes 4096
        --l1-shared-visibility-library-path
        /opt/lmcache/lib/liblmcache_shared_l1_visibility.so
    """)


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
    assert shared.coordinator_endpoint == (
        "http://memory-coordinator.shared-dax-e2e.svc:9400"
    )
    assert shared.coordinator_token_file == (
        "/var/run/secrets/lmcache/memory-coordinator-token"
    )
    assert shared.region_id == "cxl-window-0"
    assert shared.layout_id == "scratch-v1"
    assert shared.mapping_offset_bytes == 4096
    assert shared.visibility_library_path == (
        "/opt/lmcache/lib/liblmcache_shared_l1_visibility.so"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--l1-devdax-path",
        "--l1-coordinator-token-file",
        "--l1-shared-region-id",
        "--l1-shared-layout-id",
        "--l1-shared-visibility-library-path",
    ],
)
def test_shared_l1_requires_every_companion_flag(flag: str) -> None:
    args = _shared_args()
    index = args.index(flag)
    del args[index : index + 2]
    with pytest.raises(ValueError, match=flag.lstrip("-")):
        _parse_mp_args(args)


def test_shared_l1_rejects_malformed_endpoint() -> None:
    args = _shared_args()
    args[args.index("--l1-coordinator-endpoint") + 1] = "memory-coordinator:9400"

    with pytest.raises(ValueError, match="http"):
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
        _validate_shared_l1_runtime_config(
            MPServerConfig(supported_transfer_mode="engine_driven"),
            storage_config,
        )
    with pytest.raises(ValueError, match="P2P"):
        _validate_shared_l1_runtime_config(
            MPServerConfig(
                supported_transfer_mode="lmcache_driven",
                p2p_config=P2PConfig(advertise_url="10.0.0.1:9000"),
            ),
            storage_config,
        )
    with pytest.raises(ValueError, match="optional MP modules"):
        _validate_shared_l1_runtime_config(
            MPServerConfig(enable=["transfer_query"]),
            storage_config,
        )


def test_old_options_alone_leave_shared_l1_disabled() -> None:
    args = shlex.split("""
        --l1-size-gb 2 --l1-devdax-path /dev/dax1.0
        --no-l1-use-lazy --shm-name '' --eviction-policy noop
    """)
    config = _parse_mp_args(args)
    assert config.l1_manager_config.shared_l1_config is None
    memory_config = config.l1_manager_config.memory_config
    assert memory_config.devdax_path == "/dev/dax1.0"
    assert memory_config.size_in_bytes == 2 * (1 << 30)
    assert memory_config.use_lazy is False
    assert memory_config.shm_name == ""


def test_layout_wire_roundtrip_and_unknown_dtype_fails_closed() -> None:
    layout = layout_to_wire(MemoryLayoutDesc([torch.Size([2, 3])], [torch.bfloat16]))
    assert layout.shapes == ((2, 3),)
    assert layout.dtypes == ("bfloat16",)
    decoded = wire_to_layout(layout)
    assert decoded.shapes == [torch.Size([2, 3])]
    assert decoded.dtypes == [torch.bfloat16]

    assert dtype_to_wire(torch.float16) == "float16"
    assert wire_to_dtype("float16") == torch.float16
    with pytest.raises(ValueError, match="unknown wire dtype"):
        dtype_to_wire(torch.complex64)
    with pytest.raises(ValueError, match="unknown wire dtype"):
        wire_to_dtype("object")
