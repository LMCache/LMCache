# SPDX-License-Identifier: Apache-2.0
"""
3FS native L2 adapter config and factory.

Backed by the native C++ HF3fsConnector wrapped with
``NativeConnectorL2Adapter``.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING, Optional
import json
import os

if TYPE_CHECKING:
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import MemoryResidency, MemorySegment, MemoryTags
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)

logger = init_logger(__name__)


# Default path to the 3FS cluster config file (JSON), looked up in the current
# working directory. Overridable via the LMCACHE_3FS_CONFIG env var or the
# `config_file` l2-adapter field.
DEFAULT_CONFIG_FILE = os.environ.get(
    "LMCACHE_3FS_CONFIG", os.path.join(os.getcwd(), "lmcache_3fs.json")
)

# Category-A cluster params that may come from the config file / explicit JSON.
_CLUSTER_KEYS = (
    "meta_addr",
    "mgmtd_addr",
    "cluster_id",
    "chain_table_id",
    "base_dir",
    "force_use_tcp",
    "dir_permission",
    "file_permission",
)


def _parse_permission(value):
    """Accept an octal permission as int (e.g. 493) or string (e.g. '0755')."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s.startswith("0o"):
            s = s[2:]
        elif s.startswith("0") and len(s) > 1:
            s = s.lstrip("0") or "0"
        try:
            return int(s, 8)
        except ValueError:
            return None
    return None


def _load_cluster_config(path):
    """Load the 3FS cluster config JSON file. Returns {} on any failure."""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - config loading must never crash
        logger.warning("Failed to parse 3FS cluster config %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_cluster_params(d):
    """Merge config-file values (defaults) with explicit --l2-adapter JSON.

    Precedence (highest wins): explicit l2-adapter JSON key > config file >
    hardcoded C++ defaults.
    """
    config_file = d.get("config_file") or DEFAULT_CONFIG_FILE
    file_cfg = _load_cluster_config(config_file)

    params = {"config_file": config_file}
    for key in _CLUSTER_KEYS:
        if key in d:
            params[key] = d[key]
        elif key in file_cfg:
            params[key] = file_cfg[key]
    return params


class RawHf3fsL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for an L2 adapter backed by the native C++
    HF3fsConnector (3FS MetaClient-based storage).

    Fields:
    - num_workers: Number of C++ worker threads for I/O (default 4).
    - chunk_size: Chunk size in bytes (default 4MB). Should be <= the
      typical key size and divide it evenly (a key spans ceil(len/cs)
      chunks, each on a different chain).
    - meta_addr: Meta server address (default RDMA://192.168.1.100:8000).
    - mgmtd_addr: Mgmtd server address (default RDMA://192.168.1.100:8000).
    - max_capacity_gb: Max L2 capacity in GB for usage tracking /
      eviction (default 0 = disabled).
    - transport_pool_max_connections: Max connections per IO worker
      (default 8).
    - max_batch_bytes: Max bytes per StorageClient batchRead/batchWrite
      RPC (default 32MB).
    - max_concurrent_requests: Max concurrent StorageClient requests (default 256).
    - max_concurrent_requests_per_server: Max concurrent requests per storage
      node (default 64; 0 = 3FS default of 8). Raising this is the key lever
      that lets large single-chunk (32MB) reads/writes saturate RDMA bandwidth.
    - per_op_workers: Map from op lane (lookup/retrieve/store/delete) to
      dedicated C++ worker count. Lanes not listed share the num_workers pool.
      This isolates store (write) worker threads from load/lookup (read) so a
      slow, high-volume store stream does not starve concurrent loads.
      Default {} (all ops share one pool).
    - config_file: Path to the 3FS cluster config JSON (default
      $LMCACHE_3FS_CONFIG or ./lmcache_3fs.json in the current directory).
      Values loaded from it act as defaults for the cluster params below;
      explicit keys in the --l2-adapter JSON override them.
    - cluster_id/chain_table_id/base_dir/force_use_tcp/dir_permission/
      file_permission: 3FS cluster identity & layout. See C++ defaults.
    - token: 3FS user token for server-side authentication (default '').
      Supports 'file://<path>' to load from a token file. Only required
      when the meta server enables `authenticate`; must match process uid/gid.
    """

    def __init__(
        self,
        num_workers: int = 8,
        chunk_size: int = 4 * 1024 * 1024,  # 4 MB; keys span ceil(len/cs) chunks
        meta_addr: str = "RDMA://192.168.1.100:8000",
        mgmtd_addr: str = "RDMA://192.168.1.100:8000",
        max_capacity_gb: float = 0,
        transport_pool_max_connections: int = 8,
        max_batch_bytes: int = 32 * 1024 * 1024,  # 32 MB; 4MB bottlenecked batch_io
        max_concurrent_requests: int = 256,
        max_concurrent_requests_per_server: int = 64,
        per_op_workers: Optional[dict] = None,
        config_file: Optional[str] = None,
        cluster_id: str = "stage",
        chain_table_id: int = 1,
        base_dir: str = "lmcache",
        force_use_tcp: bool = False,
        dir_permission: int = 0o755,
        file_permission: int = 0o644,
        token: str = "",
    ):
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.meta_addr = meta_addr
        self.mgmtd_addr = mgmtd_addr
        self.max_capacity_gb = max_capacity_gb
        self.transport_pool_max_connections = transport_pool_max_connections
        self.max_batch_bytes = max_batch_bytes
        self.max_concurrent_requests = max_concurrent_requests
        self.max_concurrent_requests_per_server = max_concurrent_requests_per_server
        self.per_op_workers = per_op_workers or {}
        self.config_file = config_file
        self.cluster_id = cluster_id
        self.chain_table_id = chain_table_id
        self.base_dir = base_dir
        self.force_use_tcp = force_use_tcp
        self.dir_permission = dir_permission
        self.file_permission = file_permission
        self.token = token

    @classmethod
    def from_dict(cls, d: dict) -> "RawHf3fsL2AdapterConfig":
        cp = _resolve_cluster_params(d)  # config file merged w/ explicit JSON

        num_workers = d.get("num_workers", 4)
        if not isinstance(num_workers, int) or num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        chunk_size = d.get("chunk_size", 4 * 1024 * 1024)
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        meta_addr = cp.get("meta_addr", "RDMA://192.168.1.100:8000")
        if not isinstance(meta_addr, str) or not meta_addr:
            raise ValueError("meta_addr must be a non-empty string")

        mgmtd_addr = cp.get("mgmtd_addr", "RDMA://192.168.1.100:8000")
        if not isinstance(mgmtd_addr, str) or not mgmtd_addr:
            raise ValueError("mgmtd_addr must be a non-empty string")

        cluster_id = cp.get("cluster_id", "stage")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError("cluster_id must be a non-empty string")

        chain_table_id = cp.get("chain_table_id", 1)
        if not isinstance(chain_table_id, int) or chain_table_id <= 0:
            raise ValueError("chain_table_id must be a positive integer")

        base_dir = cp.get("base_dir", "lmcache")
        if not isinstance(base_dir, str) or not base_dir:
            raise ValueError("base_dir must be a non-empty string")

        force_use_tcp = cp.get("force_use_tcp", False)
        if not isinstance(force_use_tcp, bool):
            raise ValueError("force_use_tcp must be a boolean")

        dir_permission = _parse_permission(cp.get("dir_permission", 0o755))
        if dir_permission is None:
            raise ValueError(
                "dir_permission must be an octal int or octal string (e.g. '0755')"
            )

        file_permission = _parse_permission(cp.get("file_permission", 0o644))
        if file_permission is None:
            raise ValueError(
                "file_permission must be an octal int or octal string (e.g. '0644')"
            )

        token = cp.get("token", "")
        if not isinstance(token, str):
            raise ValueError("token must be a string")

        max_capacity_gb = d.get("max_capacity_gb", 0)
        if not isinstance(max_capacity_gb, (int, float)):
            raise ValueError("max_capacity_gb must be a number")

        transport_pool_max_connections = d.get("transport_pool_max_connections", 8)
        if (
            not isinstance(transport_pool_max_connections, int)
            or transport_pool_max_connections <= 0
        ):
            raise ValueError(
                "transport_pool_max_connections must be a positive integer"
            )

        max_batch_bytes = d.get("max_batch_bytes", 32 * 1024 * 1024)  # 32 MB default
        if not isinstance(max_batch_bytes, int) or max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be a positive integer")

        max_concurrent_requests = d.get("max_concurrent_requests", 256)
        if not isinstance(max_concurrent_requests, int) or max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be a positive integer")

        per_server = d.get("max_concurrent_requests_per_server", 64)
        if not isinstance(per_server, int) or per_server < 0:
            raise ValueError(
                "max_concurrent_requests_per_server must be a non-negative integer"
            )

        per_op_workers = d.get("per_op_workers", {})
        if not isinstance(per_op_workers, dict):
            raise ValueError(
                "per_op_workers must be a dict mapping lane -> worker count"
            )
        for k, v in per_op_workers.items():
            if k not in ("lookup", "retrieve", "store", "delete"):
                raise ValueError(
                    f"per_op_workers key must be one of "
                    f"lookup/retrieve/store/delete, got {k!r}"
                )
            if not isinstance(v, int) or v <= 0:
                raise ValueError(f"per_op_workers[{k!r}] must be a positive integer")

        return cls(
            num_workers=num_workers,
            chunk_size=chunk_size,
            meta_addr=meta_addr,
            mgmtd_addr=mgmtd_addr,
            max_capacity_gb=max_capacity_gb,
            transport_pool_max_connections=transport_pool_max_connections,
            max_batch_bytes=max_batch_bytes,
            max_concurrent_requests=max_concurrent_requests,
            max_concurrent_requests_per_server=per_server,
            per_op_workers=per_op_workers,
            config_file=cp.get("config_file"),
            cluster_id=cluster_id,
            chain_table_id=chain_table_id,
            base_dir=base_dir,
            force_use_tcp=force_use_tcp,
            dir_permission=dir_permission,
            file_permission=file_permission,
            token=token,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "3FS native L2 adapter config fields:\n"
            "- num_workers (int): C++ worker threads for I/O (default 4, >0)\n"
            "- chunk_size (int): Chunk size in bytes (default 4MB; should "
            "be <= typical key size and divide it evenly)\n"
            "- meta_addr (str): Meta server address (default RDMA://192.168.1.100:8000)\n"
            "- mgmtd_addr (str): Mgmtd server address (default RDMA://192.168.1.100:8000)\n"
            "- max_capacity_gb (float): Max L2 capacity in GB for eviction "
            "(default 0 = disabled)\n"
            "- transport_pool_max_connections (int): Max connections per IO "
            "worker (default 1, increase for better concurrency)\n"
            "- max_batch_bytes (int): Max bytes per StorageClient "
            "batchRead/batchWrite RPC (default 32MB, increase to reduce "
            "batch overhead)\n"
            "- max_concurrent_requests (int): Max concurrent StorageClient "
            "read requests (default 32)"
            "\n- max_concurrent_requests_per_server (int): Max concurrent "
            "StorageClient requests per storage node (0 = 3FS default 8, "
            "raise to increase read concurrency)"
            "\n- per_op_workers (dict): Map from lane "
            "(lookup/retrieve/store/delete) to dedicated worker count, "
            "isolating read vs write worker threads (default {})."
            "\n- config_file (str): Path to the 3FS cluster config JSON "
            "(default $LMCACHE_3FS_CONFIG or ./lmcache_3fs.json in the "
            "current directory)."
            "\n- cluster_id (str): 3FS cluster identifier (default 'stage')."
            "\n- chain_table_id (int): ChainTableId for the storage layout "
            "(default 1)."
            "\n- base_dir (str): 3FS root-relative KV directory "
            "(default 'lmcache')."
            "\n- force_use_tcp (bool): Force TCP instead of RDMA "
            "(default false)."
            "\n- dir_permission / file_permission (octal int or '0xxx' "
            "string): created dir/file perms (default 0755/0644)."
            "\n- token (str): 3FS user token for server-side authentication "
            "(default ''). Also accepts a file path prefix 'file://<path>' "
            "to load the token from disk, e.g. 'file:///opt/3fs/etc/token.txt'."
        )


def _create_raw_hf3fs_l2_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create a NativeConnectorL2Adapter backed by the
    native C++ HF3fsConnector."""
    try:
        # Import HF3fsConnector from lmcache package
        # First Party
        from lmcache.lmcache_raw3fs import (  # type: ignore
            HF3fsConnector,
        )
    except ImportError as e:
        raise RuntimeError(
            "HF3fs L2 adapter requires the C++ HF3fs connector. "
            "Build with: BUILD_WITH_RAW_HF3FS=1 pip install -e ."
        ) from e

    # Lazy import to avoid circular dependency
    # First Party
    from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import (  # noqa: E501
        NativeConnectorL2Adapter,
    )

    assert isinstance(config, RawHf3fsL2AdapterConfig)
    token = config.token
    if token.startswith("file://"):
        # Standard
        from pathlib import Path

        token = Path(token[len("file://") :]).read_text(encoding="utf-8").strip()

    native_client = HF3fsConnector(
        num_workers=config.num_workers,
        meta_addr=config.meta_addr,
        mgmtd_addr=config.mgmtd_addr,
        chunk_size=config.chunk_size,
        transport_pool_max_connections=config.transport_pool_max_connections,
        max_batch_bytes=config.max_batch_bytes,
        max_concurrent_requests=config.max_concurrent_requests,
        max_concurrent_requests_per_server=config.max_concurrent_requests_per_server,
        per_op_workers=config.per_op_workers,
        cluster_id=config.cluster_id,
        chain_table_id=config.chain_table_id,
        base_dir=config.base_dir,
        force_use_tcp=config.force_use_tcp,
        dir_permission=config.dir_permission,
        file_permission=config.file_permission,
        token=token,
    )

    return NativeConnectorL2Adapter(
        native_client,
        MemoryTags(MemorySegment.STORAGE, MemoryResidency.REMOTE),
        type_name="RawHf3fsL2Adapter",
        extra_status={
            "meta_addr": config.meta_addr,
            "mgmtd_addr": config.mgmtd_addr,
            "num_workers": config.num_workers,
            "chunk_size": config.chunk_size,
            "base_dir": config.base_dir,
        },
    )


# Self-register config type and adapter factory
register_l2_adapter_type("raw_hf3fs", RawHf3fsL2AdapterConfig)
register_l2_adapter_factory("raw_hf3fs", _create_raw_hf3fs_l2_adapter)
