# SPDX-License-Identifier: Apache-2.0
"""
File-system based L2 adapter using aiofiles for async I/O.

Stores KV cache objects as raw tensor bytes on disk (no metadata
header).  Each ObjectKey maps to a separate ``.data`` file whose
name encodes all key fields so it can be reversed on startup.
"""

# Standard
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union
import asyncio
import os
import threading

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.logging import init_logger
from lmcache.native_storage_ops import Bitmap
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters.base import (
    L2AdapterInterface,
    L2TaskId,
)
from lmcache.v1.distributed.l2_adapters.config import (
    L2AdapterConfigBase,
    register_l2_adapter_factory,
    register_l2_adapter_type,
)
from lmcache.v1.memory_management import MemoryObj

logger = init_logger(__name__)

_KEY_SEP = "@"
_PATH_SLASH_REPLACEMENT = "-SEP-"
_FILE_EXT = ".data"


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

    Format follows CacheEngineKey.to_string() convention::

        <model_name>@<kv_rank_hex>@<chunk_hash_hex>.data

    ``kv_rank`` is written in ``0x`` prefixed hex so each byte
    of the bitmap ``(ws<<24)|(rank<<16)|(local_ws<<8)|local``
    is directly readable.
    """
    safe_model = key.model_name.replace("/", _PATH_SLASH_REPLACEMENT)
    return (
        f"{safe_model}"
        f"{_KEY_SEP}{key.kv_rank:#010x}"
        f"{_KEY_SEP}{key.chunk_hash.hex()}"
        f"{_FILE_EXT}"
    )


def _filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Reverse ``_object_key_to_filename``.

    Returns ``None`` when the filename cannot be parsed.
    """
    stem = filename
    if stem.endswith(_FILE_EXT):
        stem = stem[: -len(_FILE_EXT)]
    else:
        return None

    # Split by ``@``.  Layout:
    #   <model_name> @ <kv_rank> @ <chunk_hash_hex>
    # model_name itself may contain ``@``, so we split
    # from the right to reliably isolate the last two
    # fields (kv_rank and chunk_hash).
    parts = stem.rsplit(_KEY_SEP, 2)
    if len(parts) != 3:
        return None

    safe_model, kv_rank_str, chunk_hash_hex = parts
    try:
        chunk_hash = bytes.fromhex(chunk_hash_hex)
        kv_rank = int(kv_rank_str, 16)
    except ValueError:
        return None
    model_name = safe_model.replace(_PATH_SLASH_REPLACEMENT, "/")
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name=model_name,
        kv_rank=kv_rank,
    )


class FSL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the filesystem-backed L2 adapter.

    Fields:
    - base_path: directory for storing KV cache files.
    - relative_tmp_dir: optional relative sub-dir for
      temp files (same as fs_connector_relative_tmp_dir).
    - scan_on_startup: scan existing files on startup
      to populate the in-memory index. Default False.
    """

    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        scan_on_startup: bool = False,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
    ):
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.scan_on_startup = scan_on_startup
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect

    @classmethod
    def from_dict(cls, d: dict) -> "FSL2AdapterConfig":
        base_path = d.get("base_path")
        if not isinstance(base_path, str) or not base_path:
            raise ValueError("base_path must be a non-empty string")
        relative_tmp_dir = d.get("relative_tmp_dir", None)
        if relative_tmp_dir is not None:
            if not isinstance(relative_tmp_dir, str):
                raise ValueError("relative_tmp_dir must be a string")
        scan_on_startup = d.get("scan_on_startup", False)
        if not isinstance(scan_on_startup, bool):
            raise ValueError("scan_on_startup must be a boolean")
        read_ahead_size = d.get("read_ahead_size", None)
        if read_ahead_size is not None:
            if not isinstance(read_ahead_size, int) or read_ahead_size <= 0:
                raise ValueError("read_ahead_size must be a positive integer")
        use_odirect = d.get("use_odirect", False)
        if not isinstance(use_odirect, bool):
            raise ValueError("use_odirect must be a boolean")
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            scan_on_startup=scan_on_startup,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
        )

    @classmethod
    def help(cls) -> str:
        return (
            "FS L2 adapter config fields:\n"
            "- base_path (str): directory for KV cache "
            "files (required)\n"
            "- relative_tmp_dir (str): relative sub-dir "
            "for temp files (optional, same as "
            "fs_connector_relative_tmp_dir)\n"
            "- scan_on_startup (bool): scan existing "
            "files on startup (optional, default false)\n"
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes first "
            "(optional)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)"
        )


class FSL2Adapter(L2AdapterInterface):
    """
    File-system backed L2 adapter with async I/O via *aiofiles*.

    Each file stores **only** the raw tensor bytes (no metadata
    header), which gives maximum I/O throughput.  The file name
    itself encodes the full ``ObjectKey`` so it is reversible.

    Thread safety is ensured via a lock for shared bookkeeping
    and an asyncio event loop running on a dedicated daemon
    thread.
    """

    def __init__(self, config: FSL2AdapterConfig):
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

        self._store_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._lookup_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)
        self._load_efd = os.eventfd(0, os.EFD_NONBLOCK | os.EFD_CLOEXEC)

        # In-memory index: which keys are on disk
        self._known_keys: set[ObjectKey] = set()
        self._locked_keys: dict[ObjectKey, int] = defaultdict(int)

        # Task bookkeeping
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, bool] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        # Scan existing files to populate _known_keys
        if config.scan_on_startup:
            self._scan_existing_files()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, scan_on_startup=%s, "
            "read_ahead_size=%s, use_odirect=%s, "
            "found %d existing files",
            self._base_path,
            self._relative_tmp_dir,
            config.scan_on_startup,
            self._read_ahead_size,
            self._use_odirect,
            len(self._known_keys),
        )

    # ------------------------------------------------------------------
    # Event Fd Interface
    # ------------------------------------------------------------------

    def get_store_event_fd(self) -> int:
        return self._store_efd

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd

    def get_load_event_fd(self) -> int:
        return self._load_efd

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
    ) -> dict[L2TaskId, bool]:
        with self._lock:
            completed = self._completed_store_tasks
            self._completed_store_tasks = {}
        return completed

    # ------------------------------------------------------------------
    # Lookup and Lock Interface
    # ------------------------------------------------------------------

    def submit_lookup_and_lock_task(self, keys: list[ObjectKey]) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        self._loop.call_soon_threadsafe(self._execute_lookup, keys, task_id)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        def _unlock(ks: list[ObjectKey]) -> None:
            for k in ks:
                if k not in self._locked_keys:
                    continue
                if self._locked_keys[k] <= 1:
                    del self._locked_keys[k]
                else:
                    self._locked_keys[k] -= 1

        self._loop.call_soon_threadsafe(_unlock, keys)

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

        os.close(self._store_efd)
        os.close(self._lookup_efd)
        os.close(self._load_efd)
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

    def _key_to_path(self, key: ObjectKey) -> Path:
        return self._base_path / _object_key_to_filename(key)

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

    def _scan_existing_files(self) -> None:
        """Populate _known_keys from ``.data`` files on disk.

        File names are reversible so we can reconstruct the
        original ``ObjectKey`` for each file.
        """
        for entry in self._base_path.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.endswith(_FILE_EXT):
                continue
            key = _filename_to_object_key(entry.name)
            if key is not None:
                self._known_keys.add(key)

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
        try:
            for key, obj in zip(keys, objects, strict=True):
                # Skip if already stored
                if key in self._known_keys:
                    continue

                file_path, tmp_path = self._key_to_file_and_tmp_path(key)
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
                    self._known_keys.add(key)
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
            self._completed_store_tasks[task_id] = success
        os.eventfd_write(self._store_efd, 1)

    # ---- lookup ---------------------------------------------------------

    def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            if key not in self._known_keys:
                continue
            bitmap.set(i)
            self._locked_keys[key] += 1

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        os.eventfd_write(self._lookup_efd, 1)

    # ---- load -----------------------------------------------------------

    async def _execute_load(
        self,
        keys: list[ObjectKey],
        objects: list[MemoryObj],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            file_path = self._key_to_path(key)
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
        os.eventfd_write(self._load_efd, 1)


# Self-register config type and adapter factory
register_l2_adapter_type("fs", FSL2AdapterConfig)
register_l2_adapter_factory(
    "fs",
    lambda config, l1_memory_desc=None: FSL2Adapter(config),
)
