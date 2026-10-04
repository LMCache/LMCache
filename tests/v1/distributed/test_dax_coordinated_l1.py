# SPDX-License-Identifier: Apache-2.0
"""DAX configuration, optional build, mapped-region and Python client contracts.

Native client cases skip individually when the DAX extension is unavailable.
Device identity and CUDA registration are mocked; no real DAX/GPU is required.
"""

# Standard
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, PropertyMock
import argparse
import gc
import json
import mmap
import os
import stat
import struct
import threading
import traceback
import weakref

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    GroupedObjectKeys,
    L1BackendType,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    DaxCoordinatedL1Config,
    EvictionConfig,
    GdsL1Config,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
    add_storage_manager_args,
    get_configured_capacity_bytes,
    parse_args_to_config,
)
from lmcache.v1.distributed.dax_coordinated_l1 import (
    devdax_client,
    devdax_region,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_client import (
    DaxCoordinatedL1Client,
    DevDaxBucketIndexCore,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_l1_backend import (
    DaxCoordinatedL1Backend,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_layout import (
    DaxCoordinatedL1RankPlacementConfig,
    DevDaxTPRegion,
    resolve_payload_geometry,
    resolve_payload_mappings,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_model_profile import (
    resolve_devdax_model_profile,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_region import (
    DaxCoordinatedL1Region,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_types import (
    DaxCoordinatedL1RawResult as Raw,
)
from lmcache.v1.distributed.dax_coordinated_l1.devdax_types import (
    DevDaxReservationResult,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.l2_adapters.dax_l2_adapter import (
    DaxDeviceConfig,
    DaxL2AdapterConfig,
)
from lmcache.v1.distributed.l2_adapters.mock_l2_adapter import MockL2AdapterConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import get_event_bus
from lmcache.v1.multiprocess.config import add_mp_server_args
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from setup_extensions import policy as build_policy_module
from setup_extensions.common_cpp import COMMON_EXTENSIONS
from setup_extensions.policy import BuildPolicy, discover_subclasses
from setup_extensions.storage_backend_profiles import StorageBackendProfile
from setup_extensions.storage_backend_profiles.dax_coordinated_l1 import (
    DaxCoordinatedL1BuildProfile,
    is_dax_coordinated_l1_build_host_supported,
)
import lmcache.v1.distributed.l1_manager as l1_manager_module
import lmcache.v1.distributed.storage_manager as storage_manager_module

_MODEL = "Qwen/Qwen3-8B"
_GIB = 1 << 30


def _config_values(**overrides: object) -> dict[str, object]:
    region: dict[str, object] = {
        "devdax_path": "/dev/dax0.0",
        "metadata_offset_bytes": 0,
        "metadata_reservation_bytes": _GIB,
        "payload_offset_bytes": 0x7F80000000,
        "payload_size_GiB": 512,
    }
    for name in region:
        if name in overrides:
            region[name] = overrides.pop(name)
    values: dict[str, object] = {
        "rank_placement": {"tp_size": 1, "regions": [region]},
        "region_id": "qualification-range",
        "region_epoch": 1,
        "participant_id": 0,
        "hardware_qualification_digest": "ab" * 32,
    }
    values.update(overrides)
    return values


def _base_config(**overrides: object) -> DaxCoordinatedL1Config:
    return DaxCoordinatedL1Config(**_config_values(**overrides))  # type: ignore[arg-type]


def _l1_config(path: str = "/dev/dax0.0") -> L1ManagerConfig:
    return L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(
            size_in_bytes=64 << 20,
            use_lazy=False,
            shm_name="",
            devdax_path=path,
        ),
        dax_coordinated_l1_config=_base_config(),
    )


def _model_layout(
    shape: tuple[int, ...] = (2, 32, 256, 1024),
    dtype: torch.dtype = torch.bfloat16,
) -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size(shape)], [dtype])


def _parse_cli(args: list[str]) -> StorageManagerConfig:
    parser = argparse.ArgumentParser()
    add_mp_server_args(parser)
    add_storage_manager_args(parser)
    return parse_args_to_config(parser.parse_args(args))


def _cli_base() -> list[str]:
    return [
        "--l1-size-gb",
        "1",
        "--eviction-policy",
        "LRU",
        "--no-l1-use-lazy",
        "--shm-name",
        "",
    ]


def _configs(payload_offset: int = 4096, payload_gib: int = 1) -> tuple:
    return (
        DaxCoordinatedL1Config(
            rank_placement=DaxCoordinatedL1RankPlacementConfig(
                1,
                (
                    DevDaxTPRegion(
                        "/dev/null",
                        payload_offset,
                        payload_gib,
                        metadata_offset_bytes=0,
                        metadata_reservation_bytes=4096,
                    ),
                ),
            ),
            region_id="region-mock-test",
            region_epoch=1,
            participant_id=0,
            hardware_qualification_digest="ab" * 32,
        ),
        L1MemoryManagerConfig(
            devdax_path="/dev/null",
            size_in_bytes=4096,
            use_lazy=False,
            align_bytes=4096,
            shm_name="",
        ),
    )


def _emulate_mapping(monkeypatch: pytest.MonkeyPatch, size: int) -> None:
    real_mmap = mmap.mmap
    monkeypatch.setattr(
        devdax_region,
        "resolve_payload_mappings",
        lambda config: [
            devdax_region.DevDaxPayloadMapping(
                config.devdax_path,
                int(config.rank_placement.regions[0].payload_offset_bytes),
                size,
            )
        ],
    )
    monkeypatch.setattr(devdax_region, "_read_devdax_size", lambda _: 4096 + size)
    monkeypatch.setattr(devdax_region, "_read_devdax_alignment", lambda _: 4096)
    monkeypatch.setattr(
        devdax_region.mmap,
        "mmap",
        lambda _fd, length, **_kwargs: real_mmap(-1, length),
    )
    monkeypatch.setattr(devdax_region.torch_dev, "is_available", lambda: False)


def _registration(
    monkeypatch: pytest.MonkeyPatch,
    pinned: list[tuple[int, int]],
    unpinned: list[int],
    accept: Callable[[int], bool] = lambda _: True,
) -> None:
    def pin(pointer: int, size: int) -> bool:
        pinned.append((pointer, size))
        return accept(len(pinned))

    monkeypatch.setattr(
        devdax_region,
        "current_device_spec",
        SimpleNamespace(
            is_pin_supported=True,
            pin_memory=pin,
            unpin_memory=unpinned.append,
        ),
    )


def _assert_success(results: Mapping[ObjectKey, Raw | DevDaxReservationResult]) -> None:
    """Require nonempty successful results for either completion or reservation."""
    try:
        assert results
        assert all(
            (item.result if isinstance(item, DevDaxReservationResult) else item)
            == Raw.SUCCESS
            for item in results.values()
        )
    finally:
        # Failed assertions must not retain mapped views through traceback frames.
        results = {}


def _payload_values(
    results: dict[ObjectKey, DevDaxReservationResult],
    fill: dict[ObjectKey, int] | None = None,
) -> dict[ObjectKey, torch.Tensor]:
    """Fill reserved tensors or copy read values, retaining no mapped views."""
    values = {}
    item = memory_obj = tensor = None
    try:
        for key, item in results.items():
            assert item.result == Raw.SUCCESS
            memory_obj = item.memory_obj
            assert memory_obj is not None
            tensor = memory_obj.tensor
            assert tensor is not None
            if fill is not None:
                tensor.fill_(fill[key])
            else:
                values[key] = tensor.view(torch.bfloat16).clone()
        return values
    finally:
        item = memory_obj = tensor = None
        results = {}


def _rank_placement(tp_size: int = 4) -> DaxCoordinatedL1RankPlacementConfig:
    return DaxCoordinatedL1RankPlacementConfig(
        tp_size,
        (
            DevDaxTPRegion(
                "/dev/dax0.0",
                4 * _GIB,
                2,
                metadata_offset_bytes=0,
                metadata_reservation_bytes=_GIB,
            ),
            DevDaxTPRegion("/dev/dax1.0", 4 * _GIB, 2),
        )[: min(tp_size, 2)],
    )


def _with_metadata_reservation(
    placement: DaxCoordinatedL1RankPlacementConfig, size: int
) -> DaxCoordinatedL1RankPlacementConfig:
    return replace(
        placement,
        regions=(
            replace(placement.regions[0], metadata_reservation_bytes=size),
            *placement.regions[1:],
        ),
    )


def _config(
    rank_placement: DaxCoordinatedL1RankPlacementConfig,
    participant: int = 0,
    **kwargs: Any,
) -> DaxCoordinatedL1Config:
    return DaxCoordinatedL1Config(
        region_id="rank_placement-test",
        region_epoch=1,
        participant_id=participant,
        hardware_qualification_digest="ab" * 32,
        buckets_per_level=[17, 19, 23],
        rank_placement=rank_placement,
        **kwargs,
    )


def _client(config: DaxCoordinatedL1Config) -> DaxCoordinatedL1Client:
    return DaxCoordinatedL1Client(
        config,
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            shm_name="",
            devdax_path=config.devdax_path,
        ),
    )


def _layout() -> MemoryLayoutDesc:
    return MemoryLayoutDesc([torch.Size([2, 1, 256, 1024])], [torch.bfloat16])


def _key(rank: int, world: int = 4, chunk: int = 1) -> ObjectKey:
    return ObjectKey(
        chunk_hash=bytes([chunk]) * 32,
        model_name=_MODEL,
        kv_rank=ObjectKey.ComputeKVRank(world, rank, world, rank),
    )


@pytest.fixture
def emulated_devices(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[DaxCoordinatedL1RankPlacementConfig, list[tuple[int, int, int]], list[int]]:
    """Only OS device identity/pinning are emulated; mmap and native cores are real."""
    pytest.importorskip("lmcache.lmcache_dax_coordinated_l1")
    paths = [tmp_path / f"dax-{i}" for i in range(2)]
    for path in paths:
        with path.open("wb") as stream:
            stream.truncate(8 * _GIB)
    identities = {str(path): i for i, path in enumerate(paths)}
    monkeypatch.setattr(
        devdax_region,
        "os",
        SimpleNamespace(
            stat=lambda path: SimpleNamespace(
                st_mode=stat.S_IFCHR, st_rdev=identities[path]
            ),
            open=os.open,
            close=os.close,
            O_RDWR=os.O_RDWR,
        ),
    )
    monkeypatch.setattr(devdax_region, "_read_devdax_size", lambda _: 8 * _GIB)
    monkeypatch.setattr(devdax_region, "_read_devdax_alignment", lambda _: 2 << 20)
    pinned: list[tuple[int, int, int]] = []
    unpinned: list[int] = []

    def pin(pointer: int, size: int, flags: int = 0) -> bool:
        pinned.append((pointer, size, flags))
        return True

    monkeypatch.setattr(
        devdax_region,
        "current_device_spec",
        SimpleNamespace(
            is_pin_supported=True,
            pin_memory=pin,
            unpin_memory=unpinned.append,
        ),
    )
    monkeypatch.setattr(devdax_region.torch_dev, "is_available", lambda: False)
    return (
        replace(
            _rank_placement(),
            regions=tuple(
                replace(region, devdax_path=str(path))
                for region, path in zip(_rank_placement().regions, paths, strict=True)
            ),
        ),
        pinned,
        unpinned,
    )


def _storage(config: DaxCoordinatedL1Config) -> StorageManager:
    """Create the production storage stack with file-backed test arenas."""
    return StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=4096,
                    use_lazy=False,
                    shm_name="",
                    devdax_path=config.devdax_path,
                ),
                dax_coordinated_l1_config=config,
            ),
            eviction_config=EvictionConfig("noop", trigger_watermark=2.0),
        )
    )


def test_default_build_neither_compiles_nor_auto_detects_dax_coordinated_l1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ordinary LMCache build must remain independent of Device-DAX."""
    profile = DaxCoordinatedL1BuildProfile()
    monkeypatch.delenv(profile.env_var, raising=False)

    assert not profile.is_explicitly_requested()
    assert not profile.detect()
    assert all(
        "dax_coordinated_l1" not in source
        for spec in COMMON_EXTENSIONS
        for source in spec.sources
    )


def test_exact_build_flag_selects_the_additive_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing build policy must select only an exact opt-in value."""
    profile = DaxCoordinatedL1BuildProfile()
    monkeypatch.setattr(
        build_policy_module,
        "_discover_storage_backends",
        lambda: [profile],
    )
    monkeypatch.setattr(profile, "build", lambda flags: [("devdax", flags)])

    for value in (None, "0", "true"):
        if value is None:
            monkeypatch.delenv(profile.env_var, raising=False)
        else:
            monkeypatch.setenv(profile.env_var, value)
        assert BuildPolicy.collect_storage_backends([]) == []

    monkeypatch.setenv(profile.env_var, "1")
    assert BuildPolicy.collect_storage_backends(["-DABI=1"]) == [
        ("devdax", ["-DABI=1"])
    ]


def test_build_discovery_finds_the_opt_in_profile_once() -> None:
    """Filesystem discovery exposes the backend under its public build name."""
    profiles = list(
        discover_subclasses(
            "setup_extensions.storage_backend_profiles",
            StorageBackendProfile,  # type: ignore[type-abstract]
        )
    )
    matches = [p for p in profiles if p.name == "dax_coordinated_l1"]
    assert matches == [DaxCoordinatedL1BuildProfile]
    assert matches[0].env_var == "BUILD_WITH_DAX_COORDINATED_L1"


def test_x86_visibility_build_fails_closed_on_unsupported_hosts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit request must not silently omit unsupported native code."""
    assert is_dax_coordinated_l1_build_host_supported(system="Linux", machine="x86_64")
    assert not is_dax_coordinated_l1_build_host_supported(
        system="Linux", machine="aarch64"
    )

    # First Party
    from setup_extensions.storage_backend_profiles import dax_coordinated_l1

    monkeypatch.setattr(
        dax_coordinated_l1,
        "is_dax_coordinated_l1_build_host_supported",
        lambda: False,
    )
    monkeypatch.setattr(dax_coordinated_l1.platform, "system", lambda: "Linux")
    monkeypatch.setattr(dax_coordinated_l1.platform, "machine", lambda: "aarch64")

    with pytest.raises(RuntimeError, match="requires a Linux x86-64 build host"):
        DaxCoordinatedL1BuildProfile().build([])


@pytest.mark.parametrize(
    "count,pid", [(0, 0), (3, 0), (8, 0), (True, 0), (4, 4), (4, -1), (4, True)]
)
def test_invalid_participant_topology_is_rejected(count: int, pid: int) -> None:
    with pytest.raises(ValueError, match="participant"):
        _base_config(participant_count=count, participant_id=pid)


def test_json_opt_in_preserves_configuration_defaults() -> None:
    """JSON enables DAX with two participants by default; four remain opt-in."""
    assert _parse_cli(_cli_base()).l1_manager_config.dax_coordinated_l1_config is None
    empty = _parse_cli(_cli_base() + ["--dax-coordinated-l1-config-json", ""])
    assert empty.l1_manager_config.dax_coordinated_l1_config is None

    config = DaxCoordinatedL1Config.from_json(json.dumps(_config_values()))
    assert config is not None
    assert config.buckets_per_level == list(
        DaxCoordinatedL1Config.DEFAULT_BUCKETS_PER_LEVEL
    )
    assert config.ownership_mode == "equal"
    assert config.participant_count == 2
    assert config.read_view_cache_max_entries == 8192
    assert not config.skip_payload_flush
    region = config.rank_placement.regions[0]
    assert region.metadata_offset_bytes == 0
    payload_offset_bytes = region.payload_offset_bytes
    assert isinstance(payload_offset_bytes, int)
    assert payload_offset_bytes == 0x7F80000000
    assert config.payload_size_bytes == 512 << 30
    assert payload_offset_bytes + config.payload_size_bytes == 0xFF80000000


@pytest.mark.parametrize(
    ("raw_config", "error_type", "message"),
    [
        ("null", ValueError, "must be an object"),
        ("[]", ValueError, "must be an object"),
        ("{}", ValueError, "invalid DAX-Coordinated L1 config"),
        (
            json.dumps(_config_values(unknown_option=True)),
            ValueError,
            "invalid DAX-Coordinated L1 config",
        ),
        ("{", json.JSONDecodeError, None),
    ],
)
def test_from_json_rejects_invalid_configuration(
    raw_config: str, error_type: type[ValueError], message: str | None
) -> None:
    """Malformed JSON and invalid objects preserve their public error types."""
    with pytest.raises(error_type, match=message):
        DaxCoordinatedL1Config.from_json(raw_config)


@pytest.mark.parametrize("offset", [4096, "0x1000"])
def test_offset_inputs_are_normalized_for_direct_field_access(
    offset: int | str,
) -> None:
    """Integer and hexadecimal JSON offsets expose the same integer fields."""
    config = DaxCoordinatedL1Config.from_json(
        json.dumps(
            _config_values(metadata_offset_bytes=offset, payload_offset_bytes=4 * _GIB)
        )
    )
    assert config is not None
    region = config.rank_placement.regions[0]
    assert type(region.metadata_offset_bytes) is int
    assert type(region.payload_offset_bytes) is int
    assert region.metadata_offset_bytes == 4096
    assert region.payload_offset_bytes == 4 * _GIB


@pytest.mark.parametrize(
    "field_name", ["metadata_offset_bytes", "payload_offset_bytes"]
)
@pytest.mark.parametrize("offset", [None, True, 1.5, -4096, 1, 1 << 64, "invalid"])
def test_invalid_offsets_are_rejected_at_configuration_creation(
    field_name: str,
    offset: object,
) -> None:
    """Both mappings require a valid aligned 64-bit offset before use."""
    with pytest.raises(ValueError, match=field_name):
        DaxCoordinatedL1Config.from_json(
            json.dumps(_config_values(**{field_name: offset}))
        )


@pytest.mark.parametrize("field_name", ["skip_payload_flush", "memcheck_on_attach"])
def test_host_local_policies_are_explicit_opt_in_booleans(field_name: str) -> None:
    """Payload flush bypass and attach diagnostics require explicit opt-in."""
    default = _base_config()
    assert getattr(default, field_name) is False
    enabled = DaxCoordinatedL1Config.from_json(
        json.dumps(_config_values(**{field_name: True}))
    )
    assert getattr(enabled, field_name) is True
    with pytest.raises(ValueError, match="must be a boolean"):
        _base_config(**{field_name: 1})


def test_read_view_cache_limit_accepts_json_overrides_and_rejects_invalid_values() -> (
    None
):
    """The host-local descriptor limit accepts non-negative integers only."""
    for limit in (0, 2):
        config = DaxCoordinatedL1Config.from_json(
            json.dumps(_config_values(read_view_cache_max_entries=limit))
        )
        assert config.read_view_cache_max_entries == limit
    for invalid_limit in (-1, True, 1.5, "8192", None):
        with pytest.raises(ValueError, match="read_view_cache_max_entries"):
            DaxCoordinatedL1Config.from_json(
                json.dumps(_config_values(read_view_cache_max_entries=invalid_limit))
            )


def test_cli_requires_the_same_explicit_l1_devdax_device() -> None:
    """DAX-Coordinated L1 JSON supplements rather than replaces ``--l1-devdax-path``."""
    raw = json.dumps(_config_values(devdax_path="/dev/dax0.1"))
    parsed = _parse_cli(
        [
            *_cli_base(),
            "--l1-devdax-path",
            "/dev/dax0.1",
            "--dax-coordinated-l1-config-json",
            raw,
        ]
    )
    assert parsed.l1_manager_config.memory_config.devdax_path == "/dev/dax0.1"
    assert parsed.l1_manager_config.dax_coordinated_l1_config == _base_config(
        devdax_path="/dev/dax0.1"
    )

    with pytest.raises(ValueError, match="requires --l1-devdax-path"):
        _parse_cli([*_cli_base(), "--dax-coordinated-l1-config-json", raw])
    with pytest.raises(ValueError, match=r"regions\[0\]\.devdax_path must match"):
        _parse_cli(
            [
                *_cli_base(),
                "--l1-devdax-path",
                "/dev/dax0.0",
                "--dax-coordinated-l1-config-json",
                raw,
            ]
        )


def test_current_vllm_packed_kv_layout_keeps_the_same_slot_bytes() -> None:
    """Current NHD vLLM folds K/V into hidden width with kv_size one."""
    separate = resolve_devdax_model_profile(
        "meta-llama/Meta-Llama-3-8B-Instruct", 1, 256, [_model_layout()]
    )
    packed = resolve_devdax_model_profile(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        1,
        256,
        [_model_layout((1, 32, 256, 2048))],
    )

    assert packed.payload_bytes == separate.payload_bytes == 32 << 20
    assert packed.layout_digest != separate.layout_digest


@pytest.mark.parametrize("payload_bytes", [(1 << 32) - 1, 1 << 32, (1 << 32) + 1])
def test_model_payload_respects_native_length_limit(payload_bytes: int) -> None:
    """Validate the uint32 boundary without allocating a payload tensor."""
    layout = _model_layout((1, 1, 1, payload_bytes), torch.uint8)
    if payload_bytes < 1 << 32:
        profile = resolve_devdax_model_profile("Qwen/Qwen3-8B", 1, 1, [layout])
        assert profile.payload_bytes == payload_bytes
    else:
        with pytest.raises(ValueError, match="uint32 byte length"):
            resolve_devdax_model_profile("Qwen/Qwen3-8B", 1, 1, [layout])


def test_oversized_model_is_rejected_before_metadata_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup payload registration does not permit oversized native layouts."""
    # First Party
    from lmcache.v1.distributed.dax_coordinated_l1.devdax_client import (
        DaxCoordinatedL1Client,
    )

    region = Mock()
    monkeypatch.setattr(
        "lmcache.v1.distributed.dax_coordinated_l1.devdax_client.DaxCoordinatedL1Region",
        Mock(return_value=region),
    )
    client = DaxCoordinatedL1Client(
        _base_config(devdax_path="/nonexistent/lmcache-test-dax"),
        L1MemoryManagerConfig(
            devdax_path="/nonexistent/lmcache-test-dax",
            size_in_bytes=1 << 30,
            use_lazy=False,
            shm_name="",
        ),
    )
    with pytest.raises(ValueError, match="uint32 byte length"):
        client.initialize_model_layouts(
            "Qwen/Qwen3-8B", 1, 8192, [_model_layout((2, 128, 8192, 1024))]
        )
    assert not client.initialized
    region.map_metadata.assert_not_called()
    client.close()
    region.close.assert_called_once()


def test_storage_manager_exposes_dax_backend_without_binding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DAX client owns layout binding, independent of manager construction."""
    backend = Mock()
    backend.client.get_l1_memory_desc.return_value = L1MemoryDesc(0, 0, 4096)
    monkeypatch.setattr(
        l1_manager_module,
        "DaxCoordinatedL1Backend",
        Mock(return_value=backend),
    )
    # Keep background controllers from introducing unrelated events or reads.
    for controller in (
        "L1EvictionController",
        "L2EvictionController",
        "StoreController",
        "PrefetchController",
    ):
        monkeypatch.setattr(storage_manager_module, controller, Mock())
    bus = Mock()
    monkeypatch.setattr(storage_manager_module, "get_event_bus", Mock(return_value=bus))
    config = StorageManagerConfig(
        l1_manager_config=_l1_config(),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )
    with closing(StorageManager(config)) as manager:
        assert manager.uses_dax_coordinated_l1
        assert manager.dax_coordinated_l1_backend is backend
        backend.client.initialize_model_layouts.assert_not_called()


@pytest.mark.parametrize("medium", ["dram", "devdax", "gds"])
def test_other_l1_backends_construct_without_dax_or_usage_queries(
    monkeypatch: pytest.MonkeyPatch, medium: str
) -> None:
    """Ordinary StorageManager construction needs no DAX access or usage query."""
    l1 = Mock()
    l1.get_l1_memory_desc.return_value = None
    l1.get_memory_usage.side_effect = AssertionError("unexpected capacity query")
    backend_access = PropertyMock(side_effect=AssertionError("unexpected DAX access"))
    type(l1).dax_coordinated_l1_backend = backend_access
    monkeypatch.setattr(storage_manager_module, "L1Manager", Mock(return_value=l1))
    # Keep the periodic eviction thread from independently sampling usage.
    monkeypatch.setattr(storage_manager_module, "L1EvictionController", Mock())
    memory = L1MemoryManagerConfig(
        size_in_bytes=4096,
        use_lazy=False,
        shm_name="",
        devdax_path="/dev/dax0.0" if medium == "devdax" else None,
    )
    config = StorageManagerConfig(
        l1_manager_config=L1ManagerConfig(
            memory_config=memory,
            gds_l1_config=(
                GdsL1Config(size_in_bytes=4096, file_location="/unused-gds")
                if medium == "gds"
                else None
            ),
        ),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )
    with closing(StorageManager(config)) as manager:
        assert not manager.uses_dax_coordinated_l1
        l1.get_memory_usage.assert_not_called()
        backend_access.assert_not_called()


def test_runtime_l2_add_is_rejected_before_adapter_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DAX-Coordinated L1 rejects L2 additions before registering an adapter."""
    backend = Mock()
    backend.client.get_memory_usage.return_value = (0, 64 << 20)
    backend.client.get_l1_memory_desc.return_value = L1MemoryDesc(0, 0, 4096)
    backend.client.report_status.return_value = {"is_healthy": True}
    monkeypatch.setattr(
        l1_manager_module, "DaxCoordinatedL1Backend", Mock(return_value=backend)
    )
    factory = Mock(side_effect=AssertionError("L2 factory must not be called"))
    monkeypatch.setattr(storage_manager_module, "create_l2_adapter", factory)
    config = StorageManagerConfig(
        l1_manager_config=_l1_config(),
        eviction_config=EvictionConfig(eviction_policy="noop"),
    )
    manager = StorageManager(config)
    with closing(manager):
        with pytest.raises(ValueError, match="cannot be combined with L2 adapters"):
            manager.add_l2_adapter(
                MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1)
            )
        factory.assert_not_called()
        status = manager.report_status()
        assert status["num_l2_adapters"] == 0
        assert status["store_controller"]["num_active_adapters"] == 0
        assert status["prefetch_controller"]["num_active_adapters"] == 0


def test_unqualified_l2_adapter_and_conflicting_device_are_rejected() -> None:
    """The initial DAX-Coordinated L1 path must not compose with unqualified storage."""
    l1_config = _l1_config()
    with pytest.raises(ValueError, match="initially rejects L2 adapters"):
        StorageManagerConfig(
            l1_manager_config=l1_config,
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=L2AdaptersConfig(
                [MockL2AdapterConfig(max_size_gb=1, mock_bandwidth_gb=1)]
            ),
        )

    with pytest.raises(ValueError, match="must match"):
        StorageManagerConfig(
            l1_manager_config=_l1_config("/dev/dax0.1"),
            eviction_config=EvictionConfig(eviction_policy="LRU"),
        )


@pytest.mark.parametrize("infer_from_l2", [False, True])
def test_rejects_explicit_and_inferred_hybrid_overflow(infer_from_l2: bool) -> None:
    """DAX-Coordinated L1 rejects explicit and DAX-L2-inferred overflow."""
    l1_config = _l1_config()
    l2_config = L2AdaptersConfig([])
    if infer_from_l2:
        l2_config.adapters.append(
            DaxL2AdapterConfig(
                slot_bytes=4096,
                devices=[DaxDeviceConfig("/dev/dax0.0", max_dax_size_gb=2)],
            )
        )
    else:
        l1_config.memory_config.devdax_size_in_bytes = 2 << 30

    with pytest.raises(ValueError, match="does not support hybrid DRAM overflow"):
        StorageManagerConfig(
            l1_manager_config=l1_config,
            eviction_config=EvictionConfig(eviction_policy="LRU"),
            l2_adapter_config=l2_config,
        )


def test_device_dax_payload_range_overflow_is_rejected_before_mmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload range past Device-DAX capacity must fail before mmap."""
    mmap_calls: list[int] = []
    dax_coordinated_l1_config, memory_config = _configs(0xF000000000, 128)
    monkeypatch.setattr(devdax_region, "_read_devdax_size", lambda _path: 1 << 40)
    monkeypatch.setattr(devdax_region, "_read_devdax_alignment", lambda _path: 4096)
    monkeypatch.setattr(
        devdax_region.mmap,
        "mmap",
        lambda *_args, **_kwargs: mmap_calls.append(1),
    )

    with pytest.raises(ValueError, match="payload offset range exceeds"):
        DaxCoordinatedL1Region(dax_coordinated_l1_config, memory_config)
    assert mmap_calls == []


def test_gpu_dma_registration_fails_closed_without_pageable_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public constructor must reject an unregistered DAX payload."""
    calls: list[tuple[int, int]] = []
    dax_coordinated_l1_config, memory_config = _configs()
    _emulate_mapping(monkeypatch, 4096)

    _registration(monkeypatch, calls, [], lambda _: False)
    with pytest.raises(RuntimeError, match="refusing pageable or CPU-staged"):
        DaxCoordinatedL1Region(dax_coordinated_l1_config, memory_config)
    assert len(calls) == 1
    assert calls[0][0] != 0
    assert calls[0][1] == 4096

    unpinned: list[int] = []

    _registration(monkeypatch, calls, unpinned)
    region = DaxCoordinatedL1Region(dax_coordinated_l1_config, memory_config)
    with closing(region):
        assert region.cuda_registered
        pointer, size, alignment = region.get_memory_desc()
        assert calls[-1] == (pointer, size)
        assert size == 4096
        assert alignment == 4096
        # A caller's tensor outlives the MemoryObj and blocks premature unmapping.
        view = region.make_memory_obj(0, 64).tensor
        assert view is not None
        try:
            with pytest.raises(BufferError):
                region.close()
            view.fill_(7)
        finally:
            del view
    assert unpinned == [pointer]


def test_gpu_dma_registration_is_segmented_and_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large DAX mappings use bounded registrations and undo partial success."""
    dax_coordinated_l1_config, memory_config = _configs()
    payload_size = 10 * 4096
    segment_size = 4 * 4096
    _emulate_mapping(monkeypatch, payload_size)
    monkeypatch.setattr(
        devdax_region,
        "_CUDA_REGISTRATION_SEGMENT_BYTES",
        segment_size,
    )
    pinned: list[tuple[int, int]] = []
    unpinned: list[int] = []

    _registration(monkeypatch, pinned, unpinned)
    region = DaxCoordinatedL1Region(dax_coordinated_l1_config, memory_config)
    with closing(region):
        assert region.cuda_registered
        base, size, _alignment = region.get_memory_desc()
        assert size == payload_size
        assert pinned == [
            (base, segment_size),
            (base + segment_size, segment_size),
            (base + 2 * segment_size, payload_size - 2 * segment_size),
        ]
    assert unpinned == [pointer for pointer, _size in reversed(pinned)]

    pinned.clear()
    unpinned.clear()

    _registration(monkeypatch, pinned, unpinned, lambda n: n != 2)
    with pytest.raises(RuntimeError, match="segment offset"):
        DaxCoordinatedL1Region(dax_coordinated_l1_config, memory_config)
    assert unpinned == [pinned[0][0]]


def test_metadata_binding_reuses_registered_payload_and_rejects_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Late metadata mapping preserves payload pointers and validates its extent."""
    config, memory = _configs()
    _emulate_mapping(monkeypatch, 4096)
    pinned: list[tuple[int, int]] = []
    unpinned: list[int] = []
    _registration(monkeypatch, pinned, unpinned)
    with closing(DaxCoordinatedL1Region(config, memory)) as region:
        descriptor = region.get_memory_desc()
        assert region.size == 0
        with pytest.raises(ValueError, match="exceeds metadata_reservation_bytes"):
            region.map_metadata(8192)
        assert region.cuda_registered
        assert unpinned == []
        region.map_metadata(4096)
        address = region.base_address
        region.map_metadata(4096)
        assert region.base_address == address
        assert region.get_memory_desc() == descriptor
        assert len(pinned) == 1
    assert unpinned == [descriptor[0]]


@pytest.mark.parametrize("tp_size", [1, 3, 8])
@pytest.mark.parametrize("participants", [2, 4])
@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
def test_slot_ranges_partition_rank_mappings(
    tp_size: int, participants: int, ownership: str
) -> None:
    """All MPs agree on disjoint physical ranges and owner-contiguous slot IDs."""
    rank_placement = DaxCoordinatedL1RankPlacementConfig(
        tp_size,
        (
            DevDaxTPRegion(
                "/dev/dax0.0",
                4 * _GIB,
                5,
                metadata_offset_bytes=0,
                metadata_reservation_bytes=_GIB,
            ),
            DevDaxTPRegion("/dev/dax1.0", 4 * _GIB, 7),
        )[: min(tp_size, 2)],
    )
    configs = [
        _config(
            rank_placement,
            pid,
            participant_count=participants,
            ownership_mode=ownership,
        )
        for pid in range(participants)
    ]
    geometries = [resolve_payload_geometry(config, 192) for config in configs]
    geometry = geometries[0]
    mappings = resolve_payload_mappings(configs[0])
    owners = participants if ownership == "equal" else 1
    owner_slots = geometry.payload_slot_count // owners
    assert geometry.payload_slot_count % owners == 0
    assert [g.owner_slot_begin for g in geometries] == [
        min(pid, owners) * owner_slots for pid in range(participants)
    ]
    assert [g.owner_slot_count for g in geometries] == [
        owner_slots if pid < owners else 0 for pid in range(participants)
    ]
    assert [config.owner_payload_size_bytes for config in configs] == [
        sum(mapping.size_bytes for mapping in mappings) // owners if pid < owners else 0
        for pid in range(participants)
    ]
    assert len(mappings) == tp_size
    assert all(g.slot_ranges == geometry.slot_ranges for g in geometries)
    assert geometry.participant_0_slot_count == geometry.owner_slot_count
    assert sum(g.owner_slot_count for g in geometries) == geometry.payload_slot_count
    cursor = 0
    physical_ends = [0] * tp_size
    for slots in geometry.slot_ranges:
        assert slots.first_slot == cursor
        assert slots.mapping_offset_bytes == physical_ends[slots.rank]
        cursor += slots.slot_count
        physical_ends[slots.rank] += slots.slot_count * geometry.payload_slot_bytes
        assert physical_ends[slots.rank] <= mappings[slots.rank].size_bytes
        owner = geometries[slots.owner]
        assert owner.owner_slot_begin <= slots.first_slot
        assert cursor <= owner.owner_slot_begin + owner.owner_slot_count
        assert slots.slot_count == owner.owner_rank_slot_counts[slots.rank]
    assert cursor == geometry.payload_slot_count
    assert sum(physical_ends) == geometry.payload_bytes_used
    for pid, owner in enumerate(geometries):
        assert sum(owner.owner_rank_slot_counts) == owner.owner_slot_count
        if ownership == "participant_0_all" and pid:
            assert owner.owner_slot_count == 0
            assert not any(slots.owner == pid for slots in geometry.slot_ranges)


def test_tagged_writer_cannot_publish_another_writers_payload(
    emulated_devices: tuple,
) -> None:
    """Mismatched completion and unsupported abort never publish a slot."""
    rank_placement, _, _ = emulated_devices
    config = _config(rank_placement)
    backend = DaxCoordinatedL1Backend(
        config,
        L1MemoryManagerConfig(
            size_in_bytes=4096,
            use_lazy=False,
            shm_name="",
            devdax_path=config.devdax_path,
        ),
        [],
        get_event_bus(),
    )
    key = _key(0)
    with closing(backend.client):
        assert backend.client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        error, obj = backend.reserve_write([key], [False], _layout(), tag="writer-A")[
            key
        ]
        assert error == L1Error.SUCCESS and obj is not None
        assert backend.get_staging_memory_usage() == obj.get_size()
        assert backend.finish_write([key], tag="writer-B")[key] == L1Error.KEY_NOT_EXIST
        assert backend.finish_write_and_reserve_read([key], 1, tag="writer-B")[key] == (
            L1Error.KEY_NOT_EXIST,
            None,
        )
        assert (
            backend.finish_write_and_delete([key], tag="writer-A")[key]
            == L1Error.KEY_NOT_WRITABLE
        )
        assert backend.reserve_read([key], 1)[key][0] != L1Error.SUCCESS
        assert backend.finish_write([key], tag="writer-A")[key] == L1Error.SUCCESS
        assert backend.get_staging_memory_usage() == 0
        assert backend.reserve_write([key], [False], _layout(), tag="writer-B")[
            key
        ] == (
            L1Error.KEY_NOT_WRITABLE,
            None,
        )
        assert backend.reserve_read([key], 1)[key][0] == L1Error.SUCCESS
        assert backend.finish_read([key], 1)[key] == L1Error.SUCCESS
        del obj


@pytest.mark.parametrize(
    "world, regions, offsets, sizes",
    [
        (1, [0], [4], [2]),
        (2, [0, 1], [4, 4], [2, 2]),
        (4, [0, 1, 0, 1], [4, 4, 5, 5], [1, 1, 1, 1]),
        (3, [0, 1, 0], [4, 4, 5], [1, 2, 1]),
    ],
)
def test_each_rank_gets_one_disjoint_contiguous_slice(
    world: int, regions: list[int], offsets: list[int], sizes: list[int]
) -> None:
    """TP rank placement matches the public modulo and contiguous-slice contract."""
    placements = _rank_placement(world).placements()
    assert [p.region_index for p in placements] == regions
    assert [p.payload_offset_bytes // _GIB for p in placements] == offsets
    assert [p.payload_size_GiB for p in placements] == sizes


def test_metadata_reservation_config_preserves_existing_arena_digest() -> None:
    """The renamed JSON key keeps the reservation and shared v1 identity."""
    values = asdict(_rank_placement())
    values["regions"][0]["metadata_reservation_bytes"] = "0x40000000"
    config_values = asdict(_config(_rank_placement()))
    config_values["rank_placement"] = values
    config = DaxCoordinatedL1Config.from_json(json.dumps(config_values))
    rank_placement = config.rank_placement
    assert rank_placement is not None
    assert rank_placement.regions[0].metadata_reservation_bytes == _GIB
    # Digest from the same placement before the configuration field was renamed.
    assert rank_placement.layout_digest().hex() == (
        "e82fc8b6a757dc03df20e56e2b66dbb4886c224efc2df3bb1eac8582b0d56efa"
    )
    assert _with_metadata_reservation(rank_placement, 2 * _GIB).layout_digest() != (
        rank_placement.layout_digest()
    )
    with pytest.raises(ValueError, match="overlap"):
        _with_metadata_reservation(rank_placement, 5 * _GIB)


@pytest.mark.parametrize("region_count", [1, 2])
def test_only_first_region_needs_metadata_offset(region_count: int) -> None:
    """Payload-only regions omit the offset; zero is an explicit valid first offset."""
    values = asdict(_rank_placement())
    values["regions"] = list(values["regions"][:region_count])
    values["regions"][0]["payload_size_GiB"] = 4
    for region in values["regions"][1:]:
        region.pop("metadata_offset_bytes")
    values["regions"][0]["metadata_offset_bytes"] = "0x0"
    config = DaxCoordinatedL1Config.from_json(
        json.dumps({**asdict(_config(_rank_placement())), "rank_placement": values})
    )
    assert config.rank_placement is not None
    assert config.rank_placement.regions[0].metadata_offset_bytes == 0
    assert all(
        region.metadata_offset_bytes is None
        for region in config.rank_placement.regions[1:]
    )


@pytest.mark.parametrize(
    "field_name", ["metadata_offset_bytes", "metadata_reservation_bytes"]
)
@pytest.mark.parametrize("omit", [True, False])
def test_first_region_requires_explicit_metadata_settings(
    field_name: str, omit: bool
) -> None:
    values = asdict(_rank_placement())
    if omit:
        values["regions"][0].pop(field_name)
    else:
        values["regions"][0][field_name] = None
    with pytest.raises(ValueError, match=rf"regions\[0\] requires {field_name}"):
        DaxCoordinatedL1Config.from_json(
            json.dumps({**asdict(_config(_rank_placement())), "rank_placement": values})
        )


@pytest.mark.parametrize("offset,region_index", [(0, 1), ("0x0", 2), (4096, 1)])
def test_later_region_rejects_metadata_offset(
    offset: int | str, region_index: int
) -> None:
    values = asdict(_rank_placement())
    values["regions"] = list(values["regions"])
    values["regions"].append(asdict(DevDaxTPRegion("/dev/dax2.0", 4 * _GIB, 2)))
    values["regions"][region_index]["metadata_offset_bytes"] = offset
    if offset == 4096:
        # Reusing the same path still cannot declare a second metadata arena.
        values["regions"][region_index]["devdax_path"] = values["regions"][0][
            "devdax_path"
        ]
    with pytest.raises(
        ValueError, match=rf"regions\[{region_index}\] must omit metadata_offset_bytes"
    ):
        DaxCoordinatedL1Config.from_json(
            json.dumps({**asdict(_config(_rank_placement())), "rank_placement": values})
        )


@pytest.mark.parametrize("reservation", [True, 0, 4097, 1 << 64])
def test_invalid_metadata_reservation_is_rejected(reservation: int) -> None:
    with pytest.raises(ValueError, match="metadata_reservation_bytes"):
        _with_metadata_reservation(_rank_placement(), reservation)


def test_later_region_rejects_metadata_reservation_without_offset() -> None:
    """A payload-only region cannot independently reserve a metadata arena."""
    placement = _rank_placement()
    with pytest.raises(
        ValueError, match=r"regions\[1\] must omit metadata_reservation_bytes"
    ):
        replace(
            placement,
            regions=(
                placement.regions[0],
                replace(placement.regions[1], metadata_reservation_bytes=_GIB),
            ),
        )


@pytest.mark.parametrize(
    "field_name",
    [
        "devdax_path",
        "metadata_offset_bytes",
        "payload_offset_bytes",
        "payload_size_GiB",
    ],
)
def test_removed_top_level_placement_fields_are_rejected(field_name: str) -> None:
    """Legacy fields must fail instead of silently shadowing the region settings."""
    values = _config_values()
    values[field_name] = "/dev/dax0.0" if field_name == "devdax_path" else 0
    with pytest.raises(ValueError, match=field_name):
        DaxCoordinatedL1Config.from_json(json.dumps(values))


def test_documented_configs_parse_and_match_l1_device() -> None:
    """Both complete JSON examples remain valid public CLI configurations."""
    doc = Path(__file__).resolve().parents[3] / (
        "docs/design/v1/distributed/dax_coordinated_l1/dax-coordinated-l1.md"
    )
    worlds = []
    for block in doc.read_text().split("```json\n")[1:]:
        raw = block.split("```", 1)[0].replace("<64 hex characters>", "ab" * 32)
        config = DaxCoordinatedL1Config.from_json(raw)
        parsed = _parse_cli(
            [
                *_cli_base(),
                "--l1-devdax-path",
                config.devdax_path,
                "--dax-coordinated-l1-config-json",
                raw,
            ]
        )
        assert parsed.l1_manager_config.dax_coordinated_l1_config == config
        worlds.append(config.rank_placement.tp_size)
    assert worlds == [1, 2]


def test_rank_placement_is_required_and_owns_no_metadata_reservation() -> None:
    values = _config_values()
    values.pop("rank_placement")
    with pytest.raises(ValueError, match="rank_placement"):
        DaxCoordinatedL1Config.from_json(json.dumps(values))
    with pytest.raises(ValueError, match="rank_placement"):
        DaxCoordinatedL1Config.from_json(
            json.dumps(_config_values(rank_placement=None))
        )
    placement = asdict(_rank_placement())
    placement["metadata_reservation_bytes"] = _GIB
    with pytest.raises(ValueError, match="metadata_reservation_bytes"):
        DaxCoordinatedL1Config.from_json(
            json.dumps(_config_values(rank_placement=placement))
        )


@pytest.mark.parametrize(
    "failure", ["alias", "capacity", "alignment", "regular", "metadata_reservation"]
)
def test_preflight_rejects_entire_plan_before_mapping(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Device aliases, bounds, alignment and file type fail before mmap."""
    mapped = []
    monkeypatch.setattr(
        devdax_region,
        "os",
        SimpleNamespace(
            stat=lambda path: SimpleNamespace(
                st_mode=stat.S_IFREG if failure == "regular" else stat.S_IFCHR,
                st_rdev=1 if failure == "alias" else (1 if path.endswith("0.0") else 2),
            )
        ),
    )
    monkeypatch.setattr(
        devdax_region,
        "_read_devdax_size",
        lambda path: (
            5 * _GIB if failure == "capacity" and path.endswith("1.0") else 8 * _GIB
        ),
    )
    monkeypatch.setattr(
        devdax_region,
        "_read_devdax_alignment",
        lambda _: 2 * _GIB if failure == "alignment" else 2 << 20,
    )
    monkeypatch.setattr(
        devdax_region.mmap, "mmap", lambda *args, **kwargs: mapped.append(args)
    )
    rank_placement = _rank_placement()
    if failure == "metadata_reservation":
        rank_placement = _with_metadata_reservation(rank_placement, 4096)
    with pytest.raises(ValueError):
        _client(_config(rank_placement))
    assert mapped == []


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
def test_tp_native_cross_participant_payload_routing_and_lifecycle(
    emulated_devices: tuple, world: int, ownership: str
) -> None:
    """Both hosts use rank-specific bytes and preserve native lifecycle rules."""
    rank_placement, pinned, unpinned = emulated_devices
    rank_placement = replace(rank_placement, tp_size=world)
    owner = _client(_config(rank_placement, ownership_mode=ownership))
    reader = _client(_config(rank_placement, 1, ownership_mode=ownership))
    keys = [_key(r, world) for r in range(world)]
    expected = {key: rank + 10 for rank, key in enumerate(keys)}
    with closing(owner), closing(reader):
        startup_pins = list(pinned)
        assert len(startup_pins) == 2 * world
        assert reader.report_status()["cuda_registered"]
        for _ in range(2):
            assert not reader.initialize_model_layouts(_MODEL, world, 256, [_layout()])
        assert pinned == startup_pins
        assert unpinned == []
        assert owner.initialize_model_layouts(_MODEL, world, 256, [_layout()])
        # Read retries attach independently after participant 0 formats.
        assert all(
            item.result == Raw.NOT_FOUND for item in reader.reserve_read(keys).values()
        )
        assert reader.initialized
        assert pinned == startup_pins
        assert unpinned == []
        _payload_values(owner.reserve_write(keys, _layout()), expected)
        _assert_success(owner.finish_write(keys))
        status = owner.report_status()
        assert status["skip_payload_flush"] is False
        assert status["memcheck_on_attach"] is False
        assert reader.report_status()["memcheck_on_attach"] is False
        assert status["payload_put_flush_enabled"] is True
        assert status["payload_get_refresh_enabled"] is True
        assert status["payload_slot_used"] == world
        free_slots = status["payload_slot_free"]
        used_slots = status["payload_slot_used"]
        assert isinstance(free_slots, int) and isinstance(used_slots, int)
        assert free_slots + used_slots == status["owner_slot_count"]
        # Only the first region's common index is formatted.
        for path, offset in [
            (rank_placement.regions[1].devdax_path, 0),
            (rank_placement.regions[0].devdax_path, _GIB),
        ]:
            with open(path, "rb") as stream:
                stream.seek(offset)
                assert stream.read(64) == bytes(64)
        # Verify physical file offsets independently of read routing.
        for rank, placement in enumerate(rank_placement.placements()):
            with open(placement.devdax_path, "rb") as stream:
                stream.seek(placement.payload_offset_bytes)
                assert (
                    stream.read(2)
                    == torch.tensor([rank + 10], dtype=torch.bfloat16)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        observed = _payload_values(reader.reserve_read(list(reversed(keys)), 2))
        assert all(torch.all(observed[key] == value) for key, value in expected.items())
        _assert_success(reader.unsafe_read(keys))
        _assert_success(reader.finish_read(keys, read_locks=2))
        assert owner.report_status()["payload_slot_used"] == world
        assert owner.memcheck() and reader.memcheck()
        if ownership == "participant_0_all":
            assert all(
                item.result == Raw.NO_LOCAL_PAYLOAD_SLOT
                for item in reader.reserve_write(
                    [_key(0, world, 2)], _layout()
                ).values()
            )
        else:
            p1_key = _key(0, world, 2)
            _assert_success(reader.reserve_write([p1_key], _layout()))
            _assert_success(reader.finish_write([p1_key]))
            _assert_success(reader.delete([p1_key]))
        _assert_success(owner.delete(keys))
        owner.reserve_write(keys, _layout())
        _assert_success(owner.finish_write_and_reserve_read(keys))
        _assert_success(owner.finish_read(keys))
    assert all(flags == 1 for _, _, flags in pinned)
    assert sorted(unpinned) == sorted(pointer for pointer, _, _ in pinned)


def test_deferred_attach_and_reregistration_preserve_read_reservation(
    emulated_devices: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent registration must not replace the core holding a peer read."""
    placement, _, _ = emulated_devices
    placement = replace(placement, tp_size=2)
    with (
        closing(_client(_config(placement))) as owner,
        closing(_client(_config(placement, 1))) as reader,
    ):
        assert not reader.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
        assert owner.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
        key = _key(0, 2)
        _payload_values(owner.reserve_write([key], _layout()), {key: 7})
        _assert_success(owner.finish_write([key]))

        entered = threading.Event()
        release = threading.Event()
        read_started = threading.Event()
        read_done = threading.Event()
        is_formatted = devdax_client.dax_coordinated_l1_is_formatted

        def paused_format_check(address: int) -> bool:
            if not entered.is_set():
                entered.set()
                assert release.wait(5), "registration was not released"
            return is_formatted(address)

        def reserve_peer_read() -> dict:
            read_started.set()
            result = reader.reserve_read([key])
            read_done.set()
            return result

        monkeypatch.setattr(
            devdax_client, "dax_coordinated_l1_is_formatted", paused_format_check
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            registration = pool.submit(
                reader.initialize_model_layouts, _MODEL, 2, 256, [_layout()]
            )
            try:
                assert entered.wait(5)
                read = pool.submit(reserve_peer_read)
                assert read_started.wait(5)
                # An unprotected attach completes the read before registration
                # resumes, then replaces its core. A serialized attach waits.
                read_done.wait(1)
            finally:
                release.set()
            assert registration.result(timeout=5)
            result = read.result(timeout=5)
        _assert_success(result)
        assert torch.all(_payload_values(result)[key] == 7)
        result.clear()
        assert owner.delete([key])[key][0] == Raw.ACTIVE_READER
        _assert_success(reader.finish_read([key]))
        _assert_success(owner.delete([key]))


@pytest.mark.parametrize("memcheck_on_attach", [False, True])
def test_client_attach_honors_memcheck_policy(
    emulated_devices: tuple, memcheck_on_attach: bool
) -> None:
    """The client must enable or bypass the native scan of a damaged index."""
    placement, _, _ = emulated_devices
    placement = replace(placement, tp_size=2)
    config = _config(placement)
    with closing(_client(config)) as owner:
        assert owner.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
        assert owner.memcheck()

    # Inject corruption while all participants are closed. In format v2,
    # bucket_metadata_offsets starts at superblock byte 160; a bucket's
    # control state is byte 20 of its second 64-byte cache line.
    region = placement.regions[0]
    assert region.metadata_offset_bytes is not None
    metadata_offset = int(region.metadata_offset_bytes)
    with open(region.devdax_path, "r+b", buffering=0) as stream:
        stream.seek(metadata_offset + 8)
        assert struct.unpack("<II", stream.read(8)) == (2, 384)
        stream.seek(metadata_offset + 160)
        bucket_offset = struct.unpack("<Q", stream.read(8))[0]
        state_offset = metadata_offset + bucket_offset + 64 + 20
        stream.seek(state_offset)
        original = stream.read(1)
        stream.seek(state_offset)
        stream.write(b"\xff")
        try:
            peer_config = replace(
                config, participant_id=1, memcheck_on_attach=memcheck_on_attach
            )
            with closing(_client(peer_config)) as peer:
                if memcheck_on_attach:
                    with pytest.raises(RuntimeError, match="requires recovery"):
                        peer.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
                    assert not peer.initialized
                else:
                    assert peer.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
                    assert not peer.memcheck()
        finally:
            stream.seek(state_offset)
            stream.write(original)


@pytest.mark.parametrize("close_reader", [False, True])
def test_finished_peer_reads_bound_views_and_release_deleted_slots(
    emulated_devices: tuple, close_reader: bool
) -> None:
    """Cached descriptors stay bounded without retaining native read ownership."""
    rank_placement, _, _ = emulated_devices
    rank_placement = replace(rank_placement, tp_size=2)
    owner = _client(_config(rank_placement))
    reader = _client(_config(rank_placement, 1, read_view_cache_max_entries=8))
    view_refs = []
    key_refs = []
    try:
        for client in (owner, reader):
            assert client.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
        for chunk in range(64):
            key = _key(0, 2, chunk)
            key_refs.append(weakref.ref(key))
            write = owner.reserve_write([key], _layout())[key]
            assert write.result == Raw.SUCCESS and write.memory_obj is not None
            owner_view = weakref.ref(write.memory_obj)
            del write
            _assert_success(owner.finish_write([key]))

            read = reader.reserve_read([key], read_locks=2)[key]
            assert read.result == Raw.SUCCESS and read.memory_obj is not None
            view = weakref.ref(read.memory_obj)
            view_refs.append(view)
            del read
            assert reader.finish_read([key])[key] == Raw.SUCCESS
            assert view() is not None  # The remaining reader still owns it.
            assert reader.unsafe_read([key])[key].memory_obj is view()
            if close_reader:
                reader.close()
                reader = _client(
                    _config(rank_placement, 1, read_view_cache_max_entries=8)
                )
                assert reader.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
            else:
                assert reader.finish_read([key])[key] == Raw.SUCCESS

            # Completed writes retain no Python view; delete obtains the size
            # from the shared index even without a local reservation history.
            assert owner_view() is None
            result, deleted_view = owner.delete([key])[key]
            assert result == Raw.SUCCESS and deleted_view is not None
            assert deleted_view.get_size() == 1 << 20
            del deleted_view, key
        gc.collect()
        retained = 0 if close_reader else 8
        assert sum(ref() is not None for ref in view_refs) == retained
        assert sum(ref() is not None for ref in key_refs) == retained
        assert reader.report_status()["active_read_reservations"] == 0
        reader.close()
        assert all(ref() is None for ref in view_refs)
        assert all(ref() is None for ref in key_refs)
        assert owner.report_status()["payload_slot_used"] == 0
    finally:
        reader.close()
        owner.close()


@pytest.mark.parametrize("reserve_read", [False, True])
def test_peer_cannot_overwrite_published_owner_payload(
    emulated_devices: tuple, reserve_read: bool
) -> None:
    """Both idle and actively read objects remain immutable across participants."""
    rank_placement, _, _ = emulated_devices
    rank_placement = replace(rank_placement, tp_size=2)
    owner = _client(_config(rank_placement))
    peer = _client(_config(rank_placement, 1))
    key = _key(0, 2)
    with closing(owner), closing(peer):
        for client in (owner, peer):
            assert client.initialize_model_layouts(_MODEL, 2, 256, [_layout()])
        _payload_values(owner.reserve_write([key], _layout()), {key: 7})
        _assert_success(owner.finish_write([key]))
        if reserve_read:
            _assert_success(peer.reserve_read([key]))
        rejected = peer.reserve_write([key], _layout())[key]
        assert rejected.result == Raw.WRITER_BUSY and rejected.memory_obj is None
        observed = _payload_values(peer.reserve_read([key]))
        assert torch.all(observed[key] == 7)
        assert peer.finish_read([key])[key] == Raw.SUCCESS
        if reserve_read:
            assert owner.delete([key])[key][0] == Raw.ACTIVE_READER
            assert peer.finish_read([key])[key] == Raw.SUCCESS
        _assert_success(owner.delete([key]))


@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
def test_payload_usage_tracks_reservations_and_preserves_slot_ownership(
    emulated_devices: tuple, ownership: str
) -> None:
    """Usage counts occupied owner slots, including writes not yet committed."""
    rank_placement, _, _ = emulated_devices
    owner = _client(_config(rank_placement, ownership_mode=ownership))
    peer = _client(_config(rank_placement, 1, ownership_mode=ownership))
    key, pending = _key(0), _key(1, chunk=2)
    slot_bytes = 1 << 20
    owner_capacity = 2 * _GIB if ownership == "equal" else 4 * _GIB
    peer_capacity = 2 * _GIB if ownership == "equal" else 0
    try:
        for client in (owner, peer):
            assert client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        assert owner.get_memory_usage() == (0, owner_capacity)
        assert peer.get_memory_usage() == (0, peer_capacity)
        _assert_success(owner.reserve_write([key], _layout()))
        assert owner.get_memory_usage() == (slot_bytes, owner_capacity)
        _assert_success(owner.finish_write([key]))
        assert owner.get_memory_usage() == (slot_bytes, owner_capacity)

        assert peer.reserve_write([key], _layout())[key].result == Raw.WRITER_BUSY
        assert peer.get_memory_usage() == (0, peer_capacity)
        assert owner.get_memory_usage() == (slot_bytes, owner_capacity)
        _assert_success(peer.reserve_read([key]))
        assert peer.get_memory_usage() == (0, peer_capacity)
        assert peer.finish_read([key])[key] == Raw.SUCCESS

        _assert_success(owner.reserve_write([pending], _layout()))
        assert owner.get_memory_usage() == (2 * slot_bytes, owner_capacity)
        owner.close()
        owner = _client(_config(rank_placement, ownership_mode=ownership))
        assert owner.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        assert owner.get_memory_usage() == (slot_bytes, owner_capacity)
        _assert_success(owner.delete([key]))
        assert owner.get_memory_usage() == (0, owner_capacity)
        status = owner.report_status()
        assert status["payload_slot_used"] == 0
        assert status["payload_slot_free"] == status["owner_slot_count"]
    finally:
        peer.close()
        owner.close()


@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
@pytest.mark.parametrize("participant", [0, 1])
def test_dax_coordinated_capacity_uses_owner_budget_and_tp_partition_tails(
    emulated_devices: tuple, ownership: str, participant: int
) -> None:
    """Configured capacity excludes peer ownership and unassigned TP tails."""
    placement, _, _ = emulated_devices
    rank_placement = replace(
        placement,
        tp_size=6,
        regions=tuple(
            replace(r, payload_size_GiB=4, payload_offset_bytes=4 * _GIB)
            for r in placement.regions
        ),
    )
    dax_coordinated_l1_config = _config(
        rank_placement, participant, ownership_mode=ownership
    )
    config = L1ManagerConfig(
        memory_config=L1MemoryManagerConfig(size_in_bytes=4096, use_lazy=False),
        dax_coordinated_l1_config=dax_coordinated_l1_config,
    )
    expected = (
        3 * _GIB if ownership == "equal" else (6 * _GIB if participant == 0 else 0)
    )
    assert get_configured_capacity_bytes(config) == {L1BackendType.DEVDAX: expected}
    client = _client(dax_coordinated_l1_config)
    with closing(client):
        assert client.get_memory_usage() == (0, expected)


@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
@pytest.mark.parametrize("participant", [0, 1])
def test_capacity_publication_stays_configured_after_layout_binding(
    emulated_devices: tuple,
    monkeypatch: pytest.MonkeyPatch,
    ownership: str,
    participant: int,
) -> None:
    """Declarations stay configured while runtime usage follows rounded slots."""
    rank_placement, _, _ = emulated_devices
    dax_coordinated_l1_config = _config(
        rank_placement, participant, ownership_mode=ownership
    )
    events: list[Event] = []
    monkeypatch.setattr(get_event_bus(), "publish", events.append)
    storage = _storage(dax_coordinated_l1_config)
    # A 3 MiB object leaves a tail in each 1 GiB rank arena.
    layout = MemoryLayoutDesc([torch.Size([2, 1, 256, 3072])], [torch.bfloat16])
    slot_bytes = 3 << 20
    slot_count = _GIB // slot_bytes
    owner_slots = (
        slot_count // 2
        if ownership == "equal"
        else (slot_count if participant == 0 else 0)
    )
    expected = 4 * owner_slots * slot_bytes

    def declarations() -> list:
        return [
            e.metadata["snapshot"]
            for e in events
            if e.event_type == EventType.SM_CAPACITY_CHANGED
        ]

    with closing(storage):
        storage.publish_capacity()
        assert (
            declarations()[-1].modules[0].capacity_bytes
            == dax_coordinated_l1_config.owner_payload_size_bytes
        )
        before = len(declarations())
        assert storage.dax_coordinated_l1_backend is not None
        storage.dax_coordinated_l1_backend.client.initialize_model_layouts(
            _MODEL, 4, 256, [layout]
        )
        assert len(declarations()) == before
        storage.publish_capacity()
        storage.publish_capacity()
        for snapshot in declarations()[-2:]:
            assert len(snapshot.modules) == 1
            module = snapshot.modules[0]
            assert module.backend == "devdax"
            assert (
                module.capacity_bytes
                == dax_coordinated_l1_config.owner_payload_size_bytes
            )
            assert not module.shared
        assert storage.get_l1_usage() == (0, expected)
        assert (
            storage.report_status()["l1_manager"]["memory_configured_bytes"]
            == dax_coordinated_l1_config.owner_payload_size_bytes
        )


@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
def test_configured_capacity_does_not_allow_allocation_past_real_slots(
    emulated_devices: tuple,
    monkeypatch: pytest.MonkeyPatch,
    ownership: str,
) -> None:
    """A larger declared budget cannot bypass slot exhaustion or prevent reuse."""
    rank_placement, _, _ = emulated_devices
    rank_placement = replace(rank_placement, tp_size=2)
    config = _config(
        rank_placement,
        ownership_mode=ownership,
        skip_payload_flush=True,
    )
    events: list[Event] = []
    monkeypatch.setattr(get_event_bus(), "publish", events.append)
    # Each rank has 2 GiB; 768 MiB slots leave a significant unused tail.
    # Reserve views without touching GiBs of emulated payload pages.
    slot_bytes = 768 << 20
    layout = MemoryLayoutDesc([torch.Size([1, 1, 1, slot_bytes])], [torch.uint8])
    per_rank = 1 if ownership == "equal" else 2
    keys = [_key(rank, 2, i + 1) for rank in range(2) for i in range(per_rank)]
    extra = _key(0, 2, 20)
    with closing(_storage(config)) as storage:
        assert storage.dax_coordinated_l1_backend is not None
        assert storage.dax_coordinated_l1_backend.client.initialize_model_layouts(
            _MODEL, 2, 1, [layout]
        )
        storage.publish_capacity()
        declared = [
            e.metadata["snapshot"].modules[0].capacity_bytes
            for e in events
            if e.event_type == EventType.SM_CAPACITY_CHANGED
        ][-1]
        actual = len(keys) * slot_bytes
        assert declared == config.owner_payload_size_bytes > actual
        assert storage.get_l1_usage() == (0, actual)
        allocated = storage.reserve_write(keys, layout)
        assert set(allocated) == set(keys)
        allocated.clear()
        assert storage.get_l1_usage() == (actual, actual)
        storage.publish_capacity()
        assert storage.reserve_write([extra], layout) == {}
        assert storage.get_l1_usage() == (actual, actual)
        assert any(e.event_type == EventType.L1_ALLOCATION_FAILED for e in events)
        storage.finish_write(keys)
        storage.delete_l1_keys([keys[0]])
        assert storage.get_l1_usage() == (actual - slot_bytes, actual)
        allocated = storage.reserve_write([extra], layout)
        assert set(allocated) == {extra}
        allocated.clear()
        assert storage.get_l1_usage() == (actual, actual)


@pytest.mark.parametrize(
    "completion", ["retrieve", "cancel", "retrieve_failure", "missing_registration"]
)
def test_request_completion_preserves_local_and_peer_readers(
    emulated_devices: tuple,
    completion: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Real storage/lookup cleanup releases one request without clearing peer flags."""
    rank_placement, _, _ = emulated_devices
    owner = _client(_config(rank_placement))
    storage = _storage(_config(rank_placement, 1))
    hasher = TokenHasher(256)
    sessions = SessionManager(hasher, cleanup_interval=None)
    ipc_key = IPCCacheServerKey(
        model_name=_MODEL,
        world_size=4,
        worker_id=None,
        token_ids=tuple(range(256)),
        start=0,
        end=256,
        request_id="completed-request",
        num_kv_readers=1,
    )
    hashes = hasher.compute_chunk_hashes(list(ipc_key.token_ids))
    keys = ipc_key_to_object_keys(ipc_key, hashes, [0])[0]
    attn = AttnWindowDesc([-1], world_size=4)
    context = SimpleNamespace(
        chunk_size=256,
        token_hasher=hasher,
        session_manager=sessions,
        storage_manager=storage,
        event_bus=get_event_bus(),
        layout_desc_registry=SimpleNamespace(find_attn_desc=lambda *_: attn),
    )
    lookup = LookupModule(cast(MPCacheServerContext, context))
    transfer = None
    objects = None
    try:
        assert owner.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        assert storage.dax_coordinated_l1_backend is not None
        assert storage.dax_coordinated_l1_backend.client.initialize_model_layouts(
            _MODEL, 4, 256, [_layout()]
        )
        owner.reserve_write(keys, _layout())
        _assert_success(owner.finish_write(keys))
        # The owner host also reads these objects throughout remote cleanup.
        _assert_success(owner.reserve_read(keys))
        spec = PrefetchTaskSpec(
            [
                GroupedObjectKeys(
                    [key for key in keys if key.kv_rank == rank], 0, _layout()
                )
                for rank in sorted({key.kv_rank for key in keys})
            ]
        )
        for request_id in ["surviving-request", ipc_key.request_id]:
            handle = storage.submit_prefetch_task(spec, request_id)
            assert storage.wait_prefetch_status(handle, timeout=5)
            result = storage.query_prefetch_status(handle)
            assert (
                result is not None
                and sum(row.popcount() for row in result.hit_cells) == 4
            )
        session = sessions.get_or_create(ipc_key.request_id)
        session.begin_lookup(ipc_key, (-1,))
        session.record_prefetch_result(1, (0,))

        if completion == "missing_registration":
            transfer = LMCacheDrivenTransferModule(cast(MPCacheServerContext, context))
            for rank in range(4):
                worker_key = replace(ipc_key, worker_id=rank)
                args = (worker_key, 100 + rank, [[0]], b"unused-producer-event")
                assert transfer.retrieve(*args) == (b"", False)
                # A duplicate failed RETRIEVE must not consume a second request's lock.
                assert transfer.retrieve(*args) == (b"", False)
        elif completion == "cancel":
            lookup.free_lookup_locks(ipc_key, 4)
        elif completion == "retrieve_failure":
            with pytest.raises(RuntimeError, match="simulated transfer failure"):
                with storage.read_prefetched_results(keys) as objects:
                    assert objects is not None
                    raise RuntimeError("simulated transfer failure")
        else:
            with storage.read_prefetched_results(keys) as objects:
                assert objects is not None
            # Production GPU completion calls this after the transfer stream completes.
            storage.finish_read_prefetched(keys)
        del session
        lookup.end_session(ipc_key.request_id)
        lookup.end_session(ipc_key.request_id)
        assert sessions.get(ipc_key.request_id) is None
        # END_SESSION must not release the other local request's locks.
        assert storage.report_status()["l1_manager"]["active_read_reservations"] == 4
        assert all(result != Raw.SUCCESS for result, _ in owner.delete(keys).values())
        _assert_success(owner.finish_read(keys))
        # The surviving remote reader alone still prevents deletion.
        assert all(result != Raw.SUCCESS for result, _ in owner.delete(keys).values())
        storage.finish_read_prefetched(keys)
        assert storage.report_status()["l1_manager"]["active_read_reservations"] == 0
        _assert_success(owner.delete(keys))
    finally:
        objects = None
        if transfer is not None:
            transfer.close()
        # Pytest's captured exception keeps generator locals (mapped tensors)
        # alive; release those traceback references before unmapping test files.
        for record in caplog.records:
            if record.exc_info is not None and record.exc_info[2] is not None:
                traceback.clear_frames(record.exc_info[2])
        caplog.clear()
        gc.collect()
        storage.close()
        owner.close()
        sessions.close()


def test_read_view_cache_reuses_views_without_owning_read_reservations(
    emulated_devices: tuple,
) -> None:
    """LRU eviction preserves active reads; cached views never pin shared slots."""
    placement, _, _ = emulated_devices
    owner = _client(_config(placement))
    reader = _client(_config(placement, 1, read_view_cache_max_entries=2))
    a, b, c = keys = [_key(0), _key(1), _key(0, chunk=2)]
    with closing(owner), closing(reader):
        for client in (owner, reader):
            assert client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        _payload_values(owner.reserve_write(keys, _layout()), dict.fromkeys(keys, 7))
        _assert_success(owner.finish_write(keys))
        first = reader.reserve_read([a], 2)[a].memory_obj
        assert first is not None
        assert reader.reserve_read([a])[a].memory_obj is first
        assert reader.report_status()["active_read_reservations"] == 2
        # Evict the cached reference to a while its two native tokens stay live.
        _assert_success(reader.reserve_read([b, c]))
        assert reader.report_status()["read_view_cache_limit"] == 2
        assert reader.report_status()["read_view_cache_entries"] == 2
        assert reader.report_status()["read_view_cache_hits"] == 1
        assert reader.unsafe_read([a])[a].memory_obj is first
        assert first.tensor is not None
        assert torch.all(first.tensor.view(torch.bfloat16) == 7)
        assert owner.delete([a])[a].result != Raw.SUCCESS
        _assert_success(reader.finish_read([a], 2))
        assert owner.delete([a])[a].result != Raw.SUCCESS
        _assert_success(reader.finish_read([a]))
        # The cached b view survives completion but does not prevent deletion.
        _assert_success(reader.finish_read([b, c]))
        _assert_success(owner.delete([a, b]))
        assert reader.reserve_read([b])[b].result == Raw.NOT_FOUND
        assert reader.report_status()["active_read_reservations"] == 0
        held = reader.reserve_read([c])[c].memory_obj
        assert held is not None
        reference = weakref.ref(held)
        del held, first
        reader.close()
        assert reader.report_status()["read_view_cache_entries"] == 0
        assert reference() is None
        _assert_success(owner.delete([c]))


def test_read_view_cache_rejects_replaced_generation_at_same_address(
    emulated_devices: tuple,
) -> None:
    """Slot generations distinguish reuse even when bucket generations coincide."""
    placement, _, _ = emulated_devices
    placement = replace(placement, tp_size=2)
    config = replace(
        _config(placement, skip_payload_flush=True), buckets_per_level=[1, 1, 1]
    )
    owner = _client(config)
    reader = _client(replace(config, participant_id=1))
    layout = MemoryLayoutDesc([torch.Size([1, 1, 1, _GIB])], [torch.uint8])
    key = _key(0, 2)
    with closing(owner), closing(reader):
        for client in (owner, reader):
            assert client.initialize_model_layouts(_MODEL, 2, 1, [layout])
        write = owner.reserve_write([key], layout)[key].memory_obj
        assert write is not None and write.tensor is not None
        write.tensor.flatten()[:8].fill_(17)
        _assert_success(owner.finish_write([key]))
        old = reader.reserve_read([key])[key].memory_obj
        assert old is not None and old.tensor is not None
        address = old.tensor.data_ptr()
        _assert_success(reader.finish_read([key]))
        _assert_success(owner.delete([key]))
        # Occupy the first bucket using a different rank's slot. The original
        # key now moves to a fresh bucket (generation 1 again), reusing its old
        # payload address. Only the slot generation distinguishes this reuse.
        other = _key(1, 2, 2)
        _assert_success(owner.reserve_write([other], layout))
        _assert_success(owner.finish_write([other]))
        write = owner.reserve_write([key], layout)[key].memory_obj
        assert write is not None and write.tensor is not None
        # An old cache entry cannot turn an unpublished write into a read hit.
        assert reader.reserve_read([key])[key].result != Raw.SUCCESS
        write.tensor.flatten()[:8].fill_(29)
        _assert_success(owner.finish_write([key]))
        current = reader.reserve_read([key])[key].memory_obj
        assert current is not None and current is not old
        assert current.tensor is not None
        assert current.tensor.data_ptr() == address
        assert torch.all(current.tensor.flatten()[:8] == 29)
        assert reader.report_status()["read_view_cache_hits"] == 0
        assert reader.report_status()["read_view_cache_misses"] == 2
        _assert_success(reader.finish_read([key]))
        assert owner.memcheck() and reader.memcheck()
        del write, old, current


def test_read_view_cache_rebuilds_invalidated_descriptor(
    emulated_devices: tuple,
) -> None:
    """A consumer-invalidated descriptor must not be returned by a later read."""
    placement, _, _ = emulated_devices
    key = _key(0)
    with closing(_client(_config(placement))) as client:
        assert client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        _assert_success(client.reserve_write([key], _layout()))
        _assert_success(client.finish_write([key]))
        old = client.reserve_read([key])[key].memory_obj
        assert old is not None
        _assert_success(client.finish_read([key]))
        old.invalidate()
        current = client.reserve_read([key])[key].memory_obj
        assert current is not None and current is not old and current.is_valid()
        assert current.get_size() == 1 << 20
        _assert_success(client.finish_read([key]))
        del old, current


def test_read_lock_totals_allow_partial_and_interleaved_returns(
    emulated_devices: tuple,
) -> None:
    """Per-call totals add/subtract locally, including split worker completion."""
    rank_placement, _, _ = emulated_devices
    owner = _client(_config(rank_placement))
    reader = _client(_config(rank_placement, 1))
    keys = [_key(0)]
    with closing(owner), closing(reader):
        assert owner.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        assert reader.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        owner.reserve_write(keys, _layout())
        _assert_success(owner.finish_write_and_reserve_read(keys, read_locks=2))
        _assert_success(reader.reserve_read(keys, read_locks=2))
        _assert_success(reader.reserve_read(keys, read_locks=1))
        assert reader.finish_read(keys, read_locks=2)[keys[0]] == Raw.SUCCESS
        assert reader.finish_read(keys, read_locks=2)[keys[0]] == Raw.INVALID_STATE
        assert owner.finish_read(keys, read_locks=1)[keys[0]] == Raw.SUCCESS
        assert owner.delete(keys)[keys[0]][0] != Raw.SUCCESS
        assert owner.finish_read(keys, read_locks=1)[keys[0]] == Raw.SUCCESS
        assert owner.delete(keys)[keys[0]][0] != Raw.SUCCESS
        assert reader.finish_read(keys, read_locks=1)[keys[0]] == Raw.SUCCESS
        assert reader.finish_read(keys, read_locks=1)[keys[0]] == Raw.INVALID_STATE
        _assert_success(owner.delete(keys))
        owner.reserve_write(keys, _layout())
        _assert_success(owner.finish_write_and_reserve_read(keys))
        owner.close()
        _assert_success(reader.reserve_read(keys))
        _assert_success(reader.finish_read(keys))


@pytest.mark.parametrize("world", [1, 3])
def test_common_client_reopens_persisted_rank_payloads(
    emulated_devices: tuple, world: int
) -> None:
    """TP=1 and uneven TP use the same lifecycle and survive a fresh attachment."""
    rank_placement, _, _ = emulated_devices
    config = _config(
        replace(
            rank_placement,
            tp_size=world,
            regions=rank_placement.regions[: min(world, 2)],
        )
    )
    keys = [_key(rank, world) for rank in range(world)]
    writer = _client(config)
    expected = {key: rank + 21 for rank, key in enumerate(keys)}
    with closing(writer):
        descriptor = writer.get_l1_memory_desc()
        if world == 1:
            assert descriptor.ptr != 0
            assert descriptor.size == config.payload_size_bytes
        assert writer.initialize_model_layouts(_MODEL, world, 256, [_layout()])
        assert writer.get_l1_memory_desc() == descriptor
        _payload_values(writer.reserve_write(keys, _layout()), expected)
        _assert_success(writer.finish_write(keys))
    reader = _client(replace(config, participant_id=1))
    reopened = _client(config)
    with closing(reader), closing(reopened):
        assert reader.initialize_model_layouts(_MODEL, world, 256, [_layout()])
        assert reopened.initialize_model_layouts(_MODEL, world, 256, [_layout()])
        observed = _payload_values(reader.reserve_read(keys))
        assert all(torch.all(observed[key] == value) for key, value in expected.items())
        _assert_success(reader.finish_read(keys))
        deleted = reopened.delete(keys)
        try:
            _assert_success(deleted)
            assert all(
                view is not None and view.get_size() == 1 << 20
                for _, view in deleted.values()
            )
        finally:
            deleted.clear()
        # Rebuilt free lists and reclaimed slots still accept each rank's writes.
        _assert_success(reopened.reserve_write(keys, _layout()))
        reopened.close()


@pytest.mark.parametrize("ownership", ["equal", "participant_0_all"])
def test_full_rank_does_not_borrow_other_ranks_slots(
    emulated_devices: tuple, ownership: str
) -> None:
    """One exhausted rank cannot spill; delete and close return its slots."""
    rank_placement, _, _ = emulated_devices
    rank_placement = replace(rank_placement, tp_size=2)
    client = _client(
        _config(rank_placement, ownership_mode=ownership, skip_payload_flush=True)
    )
    # Views reserve capacity without touching GiBs of emulated payload pages.
    layout = MemoryLayoutDesc([torch.Size([1, 1, 1, _GIB])], [torch.uint8])
    count = 1 if ownership == "equal" else 2
    keys = [_key(0, 2, chunk) for chunk in range(count)]
    extra, other = _key(0, 2, 10), _key(1, 2, 10)
    try:
        assert client.initialize_model_layouts(_MODEL, 2, 1, [layout])
        _assert_success(client.reserve_write(keys, layout))
        assert (
            client.reserve_write([extra], layout)[extra].result
            == Raw.NO_LOCAL_PAYLOAD_SLOT
        )
        _assert_success(client.reserve_write([other], layout))
        status = client.report_status()
        assert status["skip_payload_flush"] is True
        assert status["payload_put_flush_enabled"] is False
        assert status["payload_get_refresh_enabled"] is False
        assert status["rank_0_payload_slot_free"] == 0
        assert status["rank_0_payload_slot_used"] == count
        assert status["rank_1_payload_slot_used"] == 1
        _assert_success(client.finish_write([other]))
        _assert_success(client.delete([other]))
        assert (
            client.reserve_write([extra], layout)[extra].result
            == Raw.NO_LOCAL_PAYLOAD_SLOT
        )
        _assert_success(client.finish_write(keys))
        _assert_success(client.delete([keys[0]]))
        _assert_success(client.reserve_write([extra], layout))
        client.close()
        client = _client(
            _config(rank_placement, ownership_mode=ownership, skip_payload_flush=True)
        )
        assert client.initialize_model_layouts(_MODEL, 2, 1, [layout])
        _assert_success(client.reserve_write([extra], layout))
    finally:
        client.close()


def test_invalid_mixed_rank_batch_has_no_partial_reservations(
    emulated_devices: tuple,
) -> None:
    """Validate all rank identities before either writes or reads mutate the index."""
    rank_placement, _, _ = emulated_devices
    client = _client(_config(rank_placement))
    valid, invalid = _key(0), _key(0, world=2)
    with closing(client):
        assert client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        with pytest.raises(ValueError, match="kv_rank"):
            client.reserve_write([valid, invalid], _layout())
        assert client.report_status()["active_write_reservations"] == 0
        client.reserve_write([valid], _layout())
        client.finish_write([valid])
        with pytest.raises(ValueError, match="kv_rank"):
            client.reserve_read([valid, invalid])
        assert client.report_status()["active_read_reservations"] == 0
        _assert_success(client.delete([valid]))


def test_later_rank_registration_failure_unpins_without_formatting(
    emulated_devices: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed rank rolls back earlier registrations without formatting metadata."""
    rank_placement, pinned, unpinned = emulated_devices

    def fail_third(pointer: int, size: int, flags: int = 0) -> bool:
        pinned.append((pointer, size, flags))
        return len(pinned) != 3

    monkeypatch.setattr(devdax_region.current_device_spec, "pin_memory", fail_third)
    with pytest.raises(RuntimeError, match="cudaHostRegister failed"):
        _client(_config(rank_placement))
    assert unpinned == [pointer for pointer, _, _ in reversed(pinned[:2])]
    with open(rank_placement.regions[0].devdax_path, "rb") as stream:
        assert stream.read(64) == bytes(64)


@pytest.mark.parametrize("tokens", [64, 256])
def test_reopened_owner_publishes_deleted_object_size(
    emulated_devices: tuple,
    monkeypatch: pytest.MonkeyPatch,
    tokens: int,
) -> None:
    """Eviction events use persisted object bytes without retaining Python views."""
    rank_placement, _, _ = emulated_devices
    config = _config(rank_placement)
    key = _key(0)
    writer = _client(config)
    with closing(writer):
        assert writer.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        layout = MemoryLayoutDesc([torch.Size([2, 1, tokens, 1024])], [torch.bfloat16])
        _assert_success(writer.reserve_write([key], layout))
        _assert_success(writer.finish_write([key]))
    storage = _storage(config)
    events: list[Event] = []
    monkeypatch.setattr(get_event_bus(), "publish", events.append)
    with closing(storage):
        assert storage.dax_coordinated_l1_backend is not None
        assert storage.dax_coordinated_l1_backend.client.initialize_model_layouts(
            _MODEL, 4, 256, [_layout()]
        )
        assert storage.delete_l1_keys([key]) == (1, 0)
        evictions = [
            event for event in events if event.event_type == EventType.L1_KEYS_EVICTED
        ]
        assert len(evictions) == 1
        assert evictions[0].metadata["keys"] == [key]
        metadata = evictions[0].metadata["meta"]
        assert len(metadata) == 1
        assert metadata[0].size_bytes == 2 * tokens * 1024 * 2
        assert metadata[0].backend == L1BackendType.DEVDAX


def test_write_completion_returns_native_failures(
    emulated_devices: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed native commit must return its error for every requested key."""
    rank_placement, _, _ = emulated_devices
    monkeypatch.setattr(
        DevDaxBucketIndexCore,
        "finish_writes",
        lambda self, tokens: [Raw.GENERATION_MISMATCH] * len(tokens),
    )
    with closing(_client(_config(rank_placement))) as client:
        assert client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        key = _key(0)
        _assert_success(client.reserve_write([key], _layout()))
        assert client.finish_write([key]) == {
            key: DevDaxReservationResult(Raw.GENERATION_MISMATCH)
        }


@pytest.mark.parametrize("bind_layout", [False, True])
def test_close_before_metadata_ready_releases_startup_registrations(
    emulated_devices: tuple,
    bind_layout: bool,
) -> None:
    """A waiting or unused participant releases mappings and pinning on close."""
    placement, pinned, unpinned = emulated_devices
    client = _client(_config(placement, 1))
    assert len(pinned) == placement.tp_size
    assert client.report_status()["cuda_registered"]
    if bind_layout:
        assert not client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
    assert not client.initialized
    assert unpinned == []
    client.close()
    client.close()
    assert unpinned == [pointer for pointer, _, _ in reversed(pinned)]
    assert not client.report_status()["cuda_registered"]


@pytest.mark.parametrize("participant", [0, 1])
def test_existing_magic_requires_matching_epoch_before_attach(
    emulated_devices: tuple,
    participant: int,
) -> None:
    """Ready metadata with an incompatible epoch is rejected without reformatting."""
    placement, pinned, unpinned = emulated_devices
    config = _config(placement)
    with closing(_client(config)) as owner:
        assert owner.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        with open(placement.regions[0].devdax_path, "rb") as stream:
            header = stream.read(4096)
        mismatched = replace(
            config, participant_id=participant, region_epoch=config.region_epoch + 1
        )
        with closing(_client(mismatched)) as client:
            startup_pins = list(pinned)
            for _ in range(2):
                with pytest.raises(RuntimeError, match="contract mismatch"):
                    client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
            assert not client.initialized
            assert client.report_status()["cuda_registered"]
            assert pinned == startup_pins
            assert unpinned == []
            with open(placement.regions[0].devdax_path, "rb") as stream:
                assert stream.read(4096) == header
        assert owner.memcheck()
    assert sorted(unpinned) == sorted(pointer for pointer, _, _ in pinned)


def test_zero_read_view_cache_limit_preserves_reads_without_retaining_views(
    emulated_devices: tuple,
) -> None:
    """Disabling descriptor retention still reserves and reads peer objects."""
    placement, _, _ = emulated_devices
    with (
        closing(_client(_config(placement))) as owner,
        closing(
            _client(_config(placement, 1, read_view_cache_max_entries=0))
        ) as reader,
    ):
        for client in (owner, reader):
            assert client.initialize_model_layouts(_MODEL, 4, 256, [_layout()])
        key = _key(0)
        _payload_values(owner.reserve_write([key], _layout()), {key: 7})
        _assert_success(owner.finish_write([key]))
        first = reader.reserve_read([key])[key].memory_obj
        second = reader.reserve_read([key])[key].memory_obj
        assert first is not None and second is not None and first is not second
        assert second.tensor is not None
        assert torch.all(second.tensor.view(torch.bfloat16) == 7)
        status = reader.report_status()
        assert status["read_view_cache_limit"] == 0
        assert status["read_view_cache_entries"] == 0
        assert status["read_view_cache_hits"] == 0
        assert status["read_view_cache_misses"] == 2
        _assert_success(reader.finish_read([key], 2))
        assert reader.report_status()["active_read_reservations"] == 0
        del first, second
        _assert_success(owner.delete([key]))
