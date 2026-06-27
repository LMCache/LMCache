# SPDX-License-Identifier: Apache-2.0
"""
File-system based L2 adapter using aiofiles for async I/O.

Stores KV cache objects as raw tensor bytes on disk (no metadata
header). New ``.data`` filenames encode all ObjectKey fields so they
can be reversed on startup; legacy filenames without ``object_group_id``
are treated as ``object_group_id == 0`` for backward-compatible hits.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import asyncio
import os
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
        L2AdapterListener,
    )
    from lmcache.v1.memory_management import MemoryObj

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_type,
)
from lmcache.v1.distributed.l2_adapters.factory import (
    register_l2_adapter_factory,
)
from lmcache.v1.platform import create_event_notifier

logger = init_logger(__name__)

_KEY_SEP = "@"
# ``@`` in both ``model_name`` and ``cache_salt`` is rejected by
# ObjectKey.__post_init__, so splitting on ``@`` is unambiguous.
# Kept in sync with native_connector_l2_adapter.py and
# csrc/storage_backends/fs/connector.cpp.
_PATH_SLASH_REPLACEMENT = "-SEP-"
_FILE_EXT = ".data"
_DELETE_WAIT_TIMEOUT_SECONDS = 30.0


def _readinto_full(
    f,  # typing: IO[bytes]
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Loop readinto() until *buf* is full or EOF.

    A single ``readinto()`` may return fewer bytes than
    *len(buf)* even when more data is available.  This
    helper keeps reading until the buffer is completely
    filled or the file reaches EOF.

    Returns:
        Total number of bytes read.
    """
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


async def _async_readinto_full(
    f,  # aiofiles async file handle
    buf: Union[bytearray, memoryview, bytes],
) -> int:
    """Async version of :func:`_readinto_full`."""
    mv = memoryview(buf) if not isinstance(buf, memoryview) else buf
    total = 0
    while total < len(mv):
        n = await f.readinto(mv[total:])
        if n is None or n == 0:
            break
        total += n
    return total


def _object_key_to_filename(key: ObjectKey) -> str:
    """Build a reversible, filesystem-safe filename.

    Unsalted::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>.data

    Salted (trailing ``cache_salt``)::

        <safe_model>@0x<kv_rank_hex>@<object_group_id_hex>@<chunk_hash_hex>@<cache_salt>.data

    ``kv_rank`` is written in ``0x`` prefixed hex so each byte
    of the bitmap ``(ws<<24)|(rank<<16)|(local_ws<<8)|local``
    is directly readable. ``object_group_id`` is written in plain hex.
    """
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    base = (
        f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}"
        f"{_KEY_SEP}{key.object_group_id:x}{_KEY_SEP}{key.chunk_hash.hex()}"
    )
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}{_FILE_EXT}"
    return f"{base}{_FILE_EXT}"


def _legacy_object_key_to_filename(key: ObjectKey) -> Optional[str]:
    """Build the pre-``object_group_id`` filesystem filename.

    Older FS L2 files used:

        <safe_model>@0x<kv_rank_hex>@<chunk_hash_hex>.data

    and, after ``cache_salt`` was added, optionally appended the salt.
    Those names did not encode ``object_group_id``. They can only be
    mapped safely to ``object_group_id == 0``.
    """
    if key.object_group_id != 0:
        return None
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    base = f"{safe_model}{_KEY_SEP}{key.kv_rank:#010x}{_KEY_SEP}{key.chunk_hash.hex()}"
    if key.cache_salt:
        return f"{base}{_KEY_SEP}{key.cache_salt}{_FILE_EXT}"
    return f"{base}{_FILE_EXT}"


def _build_object_key(
    safe_model: str,
    kv_rank_str: str,
    object_group_str: str,
    chunk_hash_hex: str,
    cache_salt: str,
) -> Optional[ObjectKey]:
    """Build an ``ObjectKey`` from decoded filename fields."""
    model_name = safe_model.replace(_PATH_SLASH_REPLACEMENT, "/")
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
        kv_rank = int(kv_rank_str, 16)
        object_group_id = int(object_group_str, 16)
        # ObjectKey.__post_init__ raises ValueError when the decoded
        # model_name / cache_salt violate the forbidden-char or length
        # invariants (e.g. a stray file from another tool on disk).
        # The contract here is to return None for anything unparsable,
        # so keep the constructor inside the try block.
        return ObjectKey(
            chunk_hash=chunk_hash,
            model_name=model_name,
            kv_rank=kv_rank,
            object_group_id=object_group_id,
            cache_salt=cache_salt,
        )
    except ValueError:
        return None


def _filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Reverse ``_object_key_to_filename``.

    Accepts both the 4-field unsalted shape and the 5-field salted
    shape (trailing ``cache_salt``). It also accepts the legacy
    3-field unsalted shape and maps it to ``object_group_id == 0`` so
    existing ``.data`` directories remain readable after the filename
    format gained an explicit object group field. Returns ``None`` for
    anything else. Since ``model_name`` is guaranteed not to contain
    ``@``, plain ``split`` suffices — no marker, no rsplit.
    """
    if not filename.endswith(_FILE_EXT):
        return None
    stem = filename[: -len(_FILE_EXT)]
    parts = stem.split(_KEY_SEP)
    if len(parts) == 3:
        safe_model, kv_rank_str, chunk_hash_hex = parts
        return _build_object_key(
            safe_model=safe_model,
            kv_rank_str=kv_rank_str,
            object_group_str="0",
            chunk_hash_hex=chunk_hash_hex,
            cache_salt="",
        )
    if len(parts) == 4:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex = parts
        parsed = _build_object_key(
            safe_model=safe_model,
            kv_rank_str=kv_rank_str,
            object_group_str=object_group_str,
            chunk_hash_hex=chunk_hash_hex,
            cache_salt="",
        )
        if parsed is not None:
            return parsed

        # Best-effort compatibility for legacy salted files:
        # <model>@0x<rank>@<chunk_hash>@<cache_salt>.data.
        return _build_object_key(
            safe_model=safe_model,
            kv_rank_str=kv_rank_str,
            object_group_str="0",
            chunk_hash_hex=object_group_str,
            cache_salt=chunk_hash_hex,
        )
    elif len(parts) == 5:
        safe_model, kv_rank_str, object_group_str, chunk_hash_hex, cache_salt = parts
    else:
        return None

    return _build_object_key(
        safe_model=safe_model,
        kv_rank_str=kv_rank_str,
        object_group_str=object_group_str,
        chunk_hash_hex=chunk_hash_hex,
        cache_salt=cache_salt,
    )


class FSL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the filesystem-backed L2 adapter.

    Fields:
    - base_path: directory for storing KV cache files.
    - relative_tmp_dir: optional relative sub-dir for
      temp files (same as fs_connector_relative_tmp_dir).
    - max_capacity_gb: aggregate capacity used by get_usage();
      0 disables aggregate eviction.
    """

    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
        max_capacity_gb: float = 0.0,
    ) -> None:
        """Initialize FSL2AdapterConfig.

        Args:
            base_path: Directory for storing KV cache files.
            relative_tmp_dir: Relative sub-dir under
                base_path for temp files during writes.
            read_ahead_size: If set, trigger filesystem
                readahead by issuing a small initial read
                of this many bytes before reading the rest.
            use_odirect: If True, bypass the OS page cache
                using O_DIRECT for both reads and writes.
                Requires buffer sizes aligned to the
                filesystem block size.
            max_capacity_gb: Maximum aggregate L2 capacity in
                GiB for usage tracking and eviction. A value of
                0 keeps aggregate eviction disabled.
        """
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "FSL2AdapterConfig":
        """Build an FS L2 adapter config from a JSON object.

        Args:
            d: Adapter JSON dict. Must include ``base_path`` and may
                include ``relative_tmp_dir``, ``read_ahead_size``,
                ``use_odirect``, and ``max_capacity_gb``.

        Returns:
            Parsed ``FSL2AdapterConfig``.

        Raises:
            ValueError: If a field is missing or has an invalid type.
        """
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")
        relative_tmp_dir = d.get("relative_tmp_dir", None)
        if relative_tmp_dir is not None:
            if not isinstance(relative_tmp_dir, str):
                raise ValueError("relative_tmp_dir must be a string")
        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        max_capacity_gb = d.get("max_capacity_gb", 0.0)
        if (
            not isinstance(max_capacity_gb, (int, float))
            or isinstance(max_capacity_gb, bool)
            or max_capacity_gb < 0
        ):
            raise ValueError("max_capacity_gb must be a non-negative number")
        cfg = cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
            max_capacity_gb=float(max_capacity_gb),
        )
        cfg.eviction_config = cls._parse_eviction_config(d)
        return cfg

    @classmethod
    def help(cls) -> str:
        """Return CLI help text for FS L2 adapter JSON fields."""
        return (
            "FS L2 adapter config fields:\n"
            "- base_path (str): directory for KV cache "
            "files (required)\n"
            "- relative_tmp_dir (str): relative sub-dir "
            "for temp files (optional, same as "
            "fs_connector_relative_tmp_dir)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes first "
            "(optional)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)\n"
            "- max_capacity_gb (float): max L2 capacity "
            "in GB for usage tracking / eviction "
            "(default 0 = disabled)"
        )


class FSL2Adapter(L2AdapterInterface):
    """
    File-system backed L2 adapter with async I/O via *aiofiles*.

    Each file stores **only** the raw tensor bytes (no metadata
    header), which gives maximum I/O throughput. New filenames encode
    the full ``ObjectKey``; legacy filenames without ``object_group_id``
    are still recognized as ``object_group_id == 0``.

    Thread safety is ensured via a lock for shared bookkeeping
    and an asyncio event loop running on a dedicated daemon
    thread.
    """

    def __init__(self, config: FSL2AdapterConfig) -> None:
        super().__init__(max_capacity_bytes=int(config.max_capacity_gb * (1024**3)))
        self._config = config
        base = config.base_path
        self._base_path = Path(base)
        self._base_path.mkdir(parents=True, exist_ok=True)

        # Temp-file strategy aligned with FSConnector:
        # if relative_tmp_dir is set, write to a sub-dir;
        # otherwise fall back to a .tmp suffix.
        self._relative_tmp_dir: Optional[Path] = None
        if config.relative_tmp_dir is not None:
            self._relative_tmp_dir = Path(config.relative_tmp_dir)
            if (
                self._relative_tmp_dir.is_absolute()
                or ".." in self._relative_tmp_dir.parts
            ):
                raise ValueError("Invalid relative_tmp_dir: " + config.relative_tmp_dir)
            (self._base_path / self._relative_tmp_dir).mkdir(
                parents=False, exist_ok=True
            )

        # I/O tuning options aligned with FSConnector
        self._read_ahead_size = config.read_ahead_size
        self._use_odirect = config.use_odirect
        self._os_disk_bs = 0
        if self._use_odirect:
            stat = os.statvfs(self._base_path)
            self._os_disk_bs = stat.f_bsize

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task bookkeeping
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Per-key and per-file byte tracking. ``_key_sizes`` stores the
        # aggregate bytes accounted for each ObjectKey; ``_key_file_sizes``
        # preserves the concrete data files backing that key. The latter is
        # needed for legacy filenames and for migration windows where both
        # legacy and current filenames exist for object_group_id=0.
        self._key_sizes: dict[ObjectKey, int] = {}
        self._key_file_sizes: dict[ObjectKey, dict[Path, int]] = {}

        # Refcounted locks held between lookup hit and submit_unlock().
        # Lookup pre-locks requested keys to close the small race where
        # eviction deletes a file after lookup starts but before the hit is
        # recorded. Misses are released before the lookup task completes.
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        self._recover_existing_files()

        # Background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, "
            "read_ahead_size=%s, use_odirect=%s, max_capacity_gb=%.2f, "
            "recovered_objects=%d",
            self._base_path,
            self._relative_tmp_dir,
            self._read_ahead_size,
            self._use_odirect,
            config.max_capacity_gb,
            len(self._key_sizes),
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # ------------------------------------------------------------------
    # Listener Interface
    # ------------------------------------------------------------------

    def register_listener(self, listener: "L2AdapterListener") -> None:
        """Register a listener and replay currently tracked keys.

        ``FSL2Adapter`` restores existing ``.data`` files during
        construction, before ``StorageManager`` creates the L2 eviction
        state. Replaying the current key snapshot lets a newly registered
        eviction listener initialize its LRU state from recovered files
        without touching base-class byte accounting again.
        """
        super().register_listener(listener)
        with self._lock:
            keys = list(self._key_sizes)
            sizes = [self._key_sizes[key] for key in keys]
        if keys:
            listener.on_l2_keys_stored(keys, sizes)

    # ------------------------------------------------------------------
    # Store Interface
    # ------------------------------------------------------------------

    def submit_store_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_store(keys, objects, task_id),
            self._loop,
        )
        return task_id

    def pop_completed_store_tasks(
        self,
    ) -> dict[L2TaskId, L2StoreResult]:
        """Pop all completed store tasks.

        Returns:
            dict[L2TaskId, L2StoreResult]: a dictionary mapping the task
            id to an ``L2StoreResult`` that encodes both the success flag
            and the bytes actually transferred.
        """
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(
        self, keys: list[ObjectKey], layout_desc: MemoryLayoutDesc
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()
            for key in keys:
                self._locked_keys[key] += 1

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(keys, task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        with self._lock:
            for key in keys:
                self._unlock_key_locked(key)

    # ------------------------------------------------------------------
    # Load Interface
    # ------------------------------------------------------------------

    def submit_load_task(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_load(keys, objects, task_id),
            self._loop,
        )
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_load_tasks.pop(task_id, None)

    # ------------------------------------------------------------------
    # Status Interface
    # ------------------------------------------------------------------

    def report_status(self) -> dict:
        """Return a status dict for the FS L2 adapter."""
        usage = self.get_usage()
        with self._lock:
            stored_object_count = len(self._key_sizes)
            stored_file_count = sum(
                len(file_sizes) for file_sizes in self._key_file_sizes.values()
            )
            locked_object_count = len(self._locked_keys)
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "FSL2Adapter",
            "base_path": str(self._base_path),
            "use_odirect": self._use_odirect,
            "event_loop_alive": self._loop_thread.is_alive(),
            "max_capacity_bytes": usage.total_capacity_bytes,
            "total_bytes_used": usage.total_bytes_used,
            "usage_fraction": usage.usage_fraction,
            "stored_object_count": stored_object_count,
            "stored_file_count": stored_file_count,
            "locked_object_count": locked_object_count,
        }

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete unlocked keys from filesystem L2 storage.

        Args:
            keys: Object keys to delete.

        Note:
            Keys locked by lookup/load are skipped for this eviction
            cycle. The controller may retry them in a later cycle.
            Listener notification is emitted by the event-loop delete
            task after files are removed, so timeout in this synchronous
            wrapper does not drop the eventual deletion event.
        """
        if not keys:
            return

        with self._lock:
            deletable = [key for key in keys if self._locked_keys.get(key, 0) == 0]

        if not deletable:
            return

        fut = asyncio.run_coroutine_threadsafe(
            self._execute_delete(deletable),
            self._loop,
        )
        try:
            fut.result(timeout=_DELETE_WAIT_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            logger.warning(
                "FSL2Adapter delete did not complete within %.1fs for %d keys; "
                "the event-loop task is still running",
                _DELETE_WAIT_TIMEOUT_SECONDS,
                len(deletable),
            )
        except Exception as e:
            logger.warning("FSL2Adapter delete failed: %s", e)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class maintains aggregate and per-cache_salt byte totals from
    # ``_notify_keys_stored`` / ``_notify_keys_deleted``.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        async def _stop_tasks():
            tasks = [
                t
                for t in asyncio.all_tasks(self._loop)
                if t is not asyncio.current_task()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(_stop_tasks(), self._loop)
            try:
                fut.result(timeout=5)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)

        self._loop_thread.join()
        self._loop.close()

        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()
        logger.info("FSL2Adapter closed")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _get_next_task_id(self) -> L2TaskId:
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    def _unlock_key_locked(self, key: ObjectKey) -> None:
        """Decrease one lock refcount. Must be called under ``_lock``."""
        if key not in self._locked_keys:
            return
        if self._locked_keys[key] <= 1:
            del self._locked_keys[key]
        else:
            self._locked_keys[key] -= 1

    def _key_to_path(self, key: ObjectKey) -> Path:
        return self._base_path / _object_key_to_filename(key)

    def _key_to_legacy_path(self, key: ObjectKey) -> Optional[Path]:
        filename = _legacy_object_key_to_filename(key)
        if filename is None:
            return None
        return self._base_path / filename

    def _key_candidate_paths(self, key: ObjectKey) -> list[Path]:
        """Return known and compatible paths for ``key``.

        The current filename is always included. For ``object_group_id=0``,
        the legacy pre-object-group filename is included as a fallback so
        existing cache directories continue to hit.
        """
        candidates: list[Path] = []
        with self._lock:
            candidates.extend(self._key_file_sizes.get(key, {}).keys())

        candidates.append(self._key_to_path(key))
        legacy_path = self._key_to_legacy_path(key)
        if legacy_path is not None:
            candidates.append(legacy_path)

        deduped: list[Path] = []
        seen: set[Path] = set()
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    def _record_key_file_locked(
        self,
        key: ObjectKey,
        path: Path,
        size: int,
    ) -> bool:
        """Record a file for ``key``. Must be called under ``_lock``.

        Returns:
            ``True`` when this file was newly accounted, ``False`` if it
            was already tracked.
        """
        file_sizes = self._key_file_sizes.setdefault(key, {})
        if path in file_sizes:
            return False
        file_sizes[path] = size
        self._key_sizes[key] = self._key_sizes.get(key, 0) + size
        return True

    async def _track_existing_file(
        self,
        key: ObjectKey,
        path: Path,
    ) -> Optional[int]:
        """Account an existing on-disk file if this adapter has not seen it."""
        try:
            stat_result = await aiofiles.os.stat(path)
        except OSError:
            return None

        size = stat_result.st_size
        if size <= 0:
            return None

        with self._lock:
            if not self._record_key_file_locked(key, path, size):
                return None
        return size

    async def _find_existing_path(
        self,
        key: ObjectKey,
    ) -> Optional[Path]:
        """Return the first existing current or legacy file for ``key``."""
        for path in self._key_candidate_paths(key):
            try:
                if await aiofiles.os.path.exists(path):
                    return path
            except OSError:
                logger.debug("Failed to check path existence: %s", path)
        return None

    def _recover_existing_files(self) -> None:
        """Scan ``base_path`` and restore usage accounting from ``.data`` files."""
        recovered: list[tuple[float, ObjectKey, Path, int]] = []
        try:
            entries = list(self._base_path.iterdir())
        except OSError:
            logger.exception("Failed to scan FS L2 directory %s", self._base_path)
            return

        for entry in entries:
            try:
                if not entry.is_file() or not entry.name.endswith(_FILE_EXT):
                    continue
                key = _filename_to_object_key(entry.name)
                if key is None:
                    logger.warning(
                        "Skipping unparsable FS L2 data file during recovery: %s",
                        entry.name,
                    )
                    continue
                stat_result = entry.stat()
                size = stat_result.st_size
                if size <= 0:
                    logger.warning(
                        "Skipping empty FS L2 data file during recovery: %s",
                        entry.name,
                    )
                    continue
                recovered.append((stat_result.st_mtime, key, entry, size))
            except OSError:
                logger.warning(
                    "Skipping FS L2 data file that could not be stat'ed: %s",
                    entry,
                )

        recovered.sort(key=lambda item: item[0])
        recovered_keys: list[ObjectKey] = []
        recovered_sizes: list[int] = []
        with self._lock:
            for _, key, path, size in recovered:
                if self._record_key_file_locked(key, path, size):
                    recovered_keys.append(key)
                    recovered_sizes.append(size)

        if recovered_keys:
            self._notify_keys_stored(recovered_keys, recovered_sizes)
            logger.info(
                "Recovered %d FS L2 data files (%d bytes) from %s",
                len(recovered_keys),
                sum(recovered_sizes),
                self._base_path,
            )

    async def _key_exists_on_disk(
        self,
        key: ObjectKey,
    ) -> bool:
        """Check whether the file for *key* exists on disk.

        Uses ``aiofiles.os.path.exists`` so the check is
        non-blocking and always reflects the real FS state,
        which is critical for multi-node shared-FS setups.
        """
        return await self._find_existing_path(key) is not None

    def _key_to_file_and_tmp_path(self, key: ObjectKey) -> tuple[Path, Path]:
        """Return ``(final_path, tmp_path)``.

        When ``relative_tmp_dir`` is configured, the temp file
        is placed under that sub-directory (same behaviour as
        ``FSConnector._get_file_and_tmp_path``).  Otherwise a
        ``.tmp`` suffix is used.
        """
        fname = _object_key_to_filename(key)
        final = self._base_path / fname
        if self._relative_tmp_dir is not None:
            tmp = self._base_path / self._relative_tmp_dir / fname
        else:
            tmp = final.with_suffix(".tmp")
        return final, tmp

    # ---- O_DIRECT helpers -----------------------------------------------

    def _read_with_odirect(
        self,
        file_path: Path,
        dst_buf: Union[bytearray, memoryview, bytes],
    ) -> int:
        """Synchronous O_DIRECT read into *dst_buf*.

        Returns the number of bytes actually read.
        Runs in an executor (not on the event loop).
        """
        fd = -1
        size = len(dst_buf)
        try:
            aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
            if not aligned:
                logger.warning(
                    "Cannot use O_DIRECT for %s, size is not aligned.",
                    file_path,
                )
                with open(file_path, "rb") as f:
                    return _readinto_full(f, dst_buf)

            fd = os.open(
                str(file_path),
                os.O_RDONLY | getattr(os, "O_DIRECT", 0),
            )
            with os.fdopen(fd, "rb", buffering=0) as fdo:
                fd = -1  # now managed by fdopen
                return _readinto_full(fdo, dst_buf)
        except Exception:
            logger.exception("Failed to O_DIRECT read %s", file_path)
            return 0
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _write_with_odirect(self, file_path: Path, buf: bytes) -> None:
        """Synchronous O_DIRECT write of *buf*.

        Runs in an executor (not on the event loop).
        """
        fd = -1
        try:
            fd = os.open(
                str(file_path),
                os.O_CREAT | os.O_WRONLY | getattr(os, "O_DIRECT", 0),
                0o644,
            )
            os.write(fd, buf)
        except Exception:
            logger.exception("Failed to O_DIRECT write %s", file_path)
            raise
        finally:
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass

    # ---- store ----------------------------------------------------------

    async def _execute_store(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        success = True
        bytes_written = 0
        stored_keys: list[ObjectKey] = []
        stored_sizes: list[int] = []
        accessed_keys: list[ObjectKey] = []
        try:
            for key, obj in zip(keys, objects, strict=True):
                file_path, tmp_path = self._key_to_file_and_tmp_path(key)

                # Skip if already stored on disk, including legacy
                # pre-object_group_id filenames. Track the file if it
                # appeared after startup so usage accounting remains
                # conservative.
                existing_path = await self._find_existing_path(key)
                if existing_path is not None:
                    tracked_size = await self._track_existing_file(
                        key,
                        existing_path,
                    )
                    if tracked_size is not None:
                        stored_keys.append(key)
                        stored_sizes.append(tracked_size)
                    accessed_keys.append(key)
                    continue
                buf = obj.byte_array
                size = len(buf)

                try:
                    # Decide whether O_DIRECT is usable
                    do_odirect = self._use_odirect
                    if do_odirect:
                        aligned = self._os_disk_bs > 0 and size % self._os_disk_bs == 0
                        if not aligned:
                            logger.warning(
                                "Cannot use O_DIRECT for "
                                "writing size %d, not "
                                "aligned to block size "
                                "%d.",
                                size,
                                self._os_disk_bs,
                            )
                            do_odirect = False

                    if do_odirect:
                        await self._loop.run_in_executor(
                            None,
                            self._write_with_odirect,
                            tmp_path,
                            buf,
                        )
                    else:
                        async with aiofiles.open(tmp_path, "wb") as f:
                            await f.write(buf)

                    await aiofiles.os.replace(tmp_path, file_path)
                    bytes_written += size
                    with self._lock:
                        is_new_file = self._record_key_file_locked(
                            key,
                            file_path,
                            size,
                        )
                    if is_new_file:
                        stored_keys.append(key)
                        stored_sizes.append(size)
                    else:
                        accessed_keys.append(key)
                    logger.debug(
                        "FSL2Adapter stored key %s (%d bytes)",
                        file_path.name,
                        size,
                    )
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to store %s",
                        file_path,
                    )
                    if await aiofiles.os.path.exists(tmp_path):
                        await aiofiles.os.unlink(tmp_path)
                    success = False
        except Exception:
            logger.exception(
                "FSL2Adapter store task %s failed",
                task_id,
            )
            success = False

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)
        if accessed_keys:
            self._notify_keys_accessed(accessed_keys)
        self._store_efd.notify()

    # ---- lookup ---------------------------------------------------------

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        hit_keys: list[ObjectKey] = []
        missed_keys: list[ObjectKey] = []
        tracked_keys: list[ObjectKey] = []
        tracked_sizes: list[int] = []
        for i, key in enumerate(keys):
            path = await self._find_existing_path(key)
            if path is None:
                missed_keys.append(key)
                continue
            bitmap.set(i)
            hit_keys.append(key)
            tracked_size = await self._track_existing_file(key, path)
            if tracked_size is not None:
                tracked_keys.append(key)
                tracked_sizes.append(tracked_size)

        with self._lock:
            for key in missed_keys:
                self._unlock_key_locked(key)
            self._completed_lookup_tasks[task_id] = bitmap
        if tracked_keys:
            self._notify_keys_stored(tracked_keys, tracked_sizes)
        if hit_keys:
            self._notify_keys_accessed(hit_keys)
        self._lookup_efd.notify()

    # ---- load -----------------------------------------------------------

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        loaded_keys: list[ObjectKey] = []
        tracked_keys: list[ObjectKey] = []
        tracked_sizes: list[int] = []
        for i, key in enumerate(keys):
            file_path = await self._find_existing_path(key)
            if file_path is None:
                continue
            tracked_size = await self._track_existing_file(key, file_path)
            if tracked_size is not None:
                tracked_keys.append(key)
                tracked_sizes.append(tracked_size)
            try:
                dst_buf = objects[i].byte_array
                expected = len(dst_buf)
                num_read: Optional[int] = None

                # O_DIRECT path (sync, via executor)
                if self._use_odirect:
                    num_read = await self._loop.run_in_executor(
                        None,
                        self._read_with_odirect,
                        file_path,
                        dst_buf,
                    )
                    if num_read != expected:
                        logger.warning(
                            "Incomplete O_DIRECT read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read or 0,
                        )
                    else:
                        bitmap.set(i)
                        loaded_keys.append(key)
                        logger.debug(
                            "FSL2Adapter loaded key %s (%d bytes, O_DIRECT)",
                            file_path.name,
                            num_read,
                        )
                    continue

                # Standard async path with optional
                # read-ahead
                expected = len(dst_buf)
                async with aiofiles.open(file_path, "rb") as f:
                    if self._read_ahead_size is None:
                        num_read = await _async_readinto_full(f, dst_buf)
                    else:
                        if not isinstance(dst_buf, memoryview):
                            dst_buf = memoryview(dst_buf)
                        # Trigger readahead with a
                        # small initial read
                        ra = self._read_ahead_size
                        n_head = await _async_readinto_full(f, dst_buf[:ra])
                        if n_head == ra:
                            n_tail = await _async_readinto_full(f, dst_buf[ra:])
                            num_read = n_head + n_tail
                        else:
                            num_read = n_head

                    if num_read != expected:
                        logger.warning(
                            "Incomplete read for %s: expected %d, got %d",
                            file_path.name,
                            expected,
                            num_read,
                        )
                        continue

                    bitmap.set(i)
                    loaded_keys.append(key)
                    logger.debug(
                        "FSL2Adapter loaded key %s (%d bytes)",
                        file_path.name,
                        num_read,
                    )
            except FileNotFoundError:
                continue
            except Exception:
                logger.exception(
                    "FSL2Adapter failed to load %s",
                    file_path,
                )
                continue

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        if tracked_keys:
            self._notify_keys_stored(tracked_keys, tracked_sizes)
        if loaded_keys:
            self._notify_keys_accessed(loaded_keys)
        self._load_efd.notify()

    # ---- delete ---------------------------------------------------------

    async def _execute_delete(
        self,
        keys: list[ObjectKey],
    ) -> None:
        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []

        try:
            for key in keys:
                with self._lock:
                    if self._locked_keys.get(key, 0) > 0:
                        continue
                    tracked_files = dict(self._key_file_sizes.get(key, {}))

                candidate_paths = self._key_candidate_paths(key)
                accounted_deleted_size = 0
                removed_any_path = False
                paths_to_forget: list[Path] = []
                for path in candidate_paths:
                    tracked_size = tracked_files.get(path)
                    try:
                        await aiofiles.os.unlink(path)
                        removed_any_path = True
                        if tracked_size is not None:
                            accounted_deleted_size += tracked_size
                            paths_to_forget.append(path)
                        logger.debug("FSL2Adapter deleted key file %s", path.name)
                    except FileNotFoundError:
                        if tracked_size is not None:
                            accounted_deleted_size += tracked_size
                            paths_to_forget.append(path)
                        continue
                    except OSError:
                        logger.warning("FSL2Adapter failed to delete %s", path)
                        continue

                if not removed_any_path and accounted_deleted_size == 0:
                    continue

                with self._lock:
                    file_sizes = self._key_file_sizes.get(key)
                    if file_sizes is not None:
                        for path in paths_to_forget:
                            file_sizes.pop(path, None)
                        if not file_sizes:
                            self._key_file_sizes.pop(key, None)

                    if accounted_deleted_size > 0:
                        remaining = (
                            self._key_sizes.get(key, 0) - accounted_deleted_size
                        )
                        if remaining > 0:
                            self._key_sizes[key] = remaining
                        else:
                            self._key_sizes.pop(key, None)

                deleted_keys.append(key)
                deleted_sizes.append(accounted_deleted_size)
        except Exception:
            logger.exception("FSL2Adapter delete task failed")

        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)


# Self-register config type and adapter factory
register_l2_adapter_type("fs", FSL2AdapterConfig)


def _create_fs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create an FSL2Adapter from config."""
    return FSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("fs", _create_fs_adapter)
