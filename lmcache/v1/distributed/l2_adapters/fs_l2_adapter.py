# SPDX-License-Identifier: Apache-2.0
"""
File-system based L2 adapter using aiofiles for async I/O.

Stores KV cache objects as raw tensor bytes on disk (no metadata
header).  Each ObjectKey maps to a separate ``.data`` file whose
name encodes all key fields so it can be reversed on startup.
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union
import asyncio
import ctypes
import errno
import os
import threading

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.distributed.internal_api import (
        L1MemoryDesc,
    )
    from lmcache.v1.memory_management import MemoryObj

# Third Party
import aiofiles
import aiofiles.os

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    MemoryResidency,
    MemorySegment,
    MemoryTags,
    ObjectKey,
)
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.key_codec import decode_fs_filename, encode_fs_filename
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

# errnos with which a filesystem or device refuses O_DIRECT, at open() or
# at the transfer itself. Any of these means "use buffered I/O instead",
# not "this key is lost". (ENOTSUP equals EOPNOTSUPP on Linux.)
_DIRECT_IO_REFUSED_ERRNOS = frozenset({errno.EINVAL, errno.EOPNOTSUPP})


def _device_logical_block_size(path: Path) -> int:
    """Return the logical block size of the block device holding *path*.

    O_DIRECT alignment is set by the device's logical block size, not by
    ``statvfs().f_bsize``, which is the filesystem's preferred I/O size and
    is often larger (4096 on a disk whose logical block size is 512).

    Returns 0 when it cannot be determined, for example on a filesystem
    without a backing block device or on a platform without sysfs.
    """
    try:
        st = os.stat(path)
    except OSError:
        return 0
    dev = Path(f"/sys/dev/block/{os.major(st.st_dev)}:{os.minor(st.st_dev)}")
    # A partition has no queue/ of its own; its parent disk does.
    for queue in (dev / "queue", dev / ".." / "queue"):
        try:
            return int((queue / "logical_block_size").read_text())
        except (OSError, ValueError):
            continue
    return 0


def _buffer_is_aligned(
    buf: Union[bytearray, memoryview, bytes],
    align: int,
) -> bool:
    """Whether *buf*'s start address is a multiple of *align*.

    O_DIRECT needs the buffer address aligned as well as the offset and
    length. A read-only or empty buffer exposes no address to check, so it
    counts as unaligned and takes the buffered path.
    """
    if align <= 0:
        return False
    try:
        ptr = ctypes.addressof((ctypes.c_byte * 1).from_buffer(buf))
    except (TypeError, ValueError, BufferError):
        return False
    return ptr % align == 0


def _write_full(fd: int, buf: Union[bytearray, memoryview, bytes]) -> None:
    """Loop ``os.write`` until all of *buf* is written.

    A single ``os.write`` may write fewer bytes than asked for. Ignoring
    that would publish a truncated file under the key's final name.
    """
    mv = memoryview(buf).cast("B")
    total = 0
    while total < len(mv):
        n = os.write(fd, mv[total:])
        if n <= 0:
            raise OSError(errno.EIO, "write returned no progress")
        total += n


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
    """Return the filename shared by Python and native FS."""
    return encode_fs_filename(key)


def _filename_to_object_key(
    filename: str,
) -> Optional[ObjectKey]:
    """Recover an ObjectKey, or return None for a foreign/invalid filename."""
    try:
        return decode_fs_filename(filename)
    except ValueError:
        return None


class FSL2AdapterConfig(L2AdapterConfigBase):
    """
    Config for the filesystem-backed L2 adapter.

    Fields:
    - base_path: directory for storing KV cache files.
    - relative_tmp_dir: optional relative sub-dir for
      temp files (same as fs_connector_relative_tmp_dir).
    - max_capacity_gb: declared L2 capacity in GB (0 = unbounded), used for
      usage accounting; disk usage is bounded only when the adapter spec also
      carries an ``eviction`` block, as for fs_native.
    """

    def __init__(
        self,
        base_path: str,
        relative_tmp_dir: Optional[str] = None,
        read_ahead_size: Optional[int] = None,
        use_odirect: bool = False,
        max_capacity_gb: float = 0,
    ):
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
            max_capacity_gb: Capacity in GB for L2 usage accounting
                and capacity eviction. 0 disables both.
        """
        self.base_path = base_path
        self.relative_tmp_dir = relative_tmp_dir
        self.read_ahead_size = read_ahead_size
        self.use_odirect = use_odirect
        self.max_capacity_gb = max_capacity_gb

    @classmethod
    def from_dict(cls, d: dict) -> "FSL2AdapterConfig":
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
        max_capacity_gb = d.get("max_capacity_gb", 0)
        if (
            isinstance(max_capacity_gb, bool)
            or not isinstance(max_capacity_gb, (int, float))
            or max_capacity_gb < 0
        ):
            raise ValueError("max_capacity_gb must be a non-negative number")
        if max_capacity_gb > 0 and d.get("eviction") is None:
            logger.warning(
                "fs: max_capacity_gb=%s is declared for %s but no 'eviction' "
                "block is configured, so it is used for usage accounting only "
                "and will NOT bound disk usage.",
                max_capacity_gb,
                base_path,
            )
        return cls(
            base_path=base_path,
            relative_tmp_dir=relative_tmp_dir,
            read_ahead_size=read_ahead_size,
            use_odirect=use_odirect,
            max_capacity_gb=max_capacity_gb,
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
            "- read_ahead_size (int): trigger fs "
            "readahead by reading this many bytes first "
            "(optional)\n"
            "- use_odirect (bool): bypass page cache "
            "via O_DIRECT (optional, default false)\n"
            "- max_capacity_gb (float): capacity for usage "
            "accounting and capacity eviction (optional, "
            "default 0 = unbounded)"
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

    _DELETE_CONCURRENCY = 64

    def __init__(self, config: FSL2AdapterConfig):
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
        # O_DIRECT fallbacks repeat for every object of a misaligned or
        # refusing deployment, so each kind is logged once per adapter.
        self._warned_unaligned = False
        self._warned_refused = False
        if self._use_odirect:
            self._os_disk_bs = (
                _device_logical_block_size(self._base_path)
                or os.statvfs(self._base_path).f_bsize
            )

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        # Task bookkeeping
        self._next_task_id: L2TaskId = 0
        self._completed_store_tasks: dict[L2TaskId, L2StoreResult] = {}
        self._completed_lookup_tasks: dict[L2TaskId, Bitmap] = {}
        self._completed_load_tasks: dict[L2TaskId, Bitmap] = {}
        self._lock = threading.Lock()

        # Background asyncio event loop
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        logger.info(
            "Initialized FSL2Adapter with base_path=%s, "
            "relative_tmp_dir=%s, "
            "read_ahead_size=%s, use_odirect=%s",
            self._base_path,
            self._relative_tmp_dir,
            self._read_ahead_size,
            self._use_odirect,
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
        self, keys: list[ObjectKey], group_layout_descs: dict[int, MemoryLayoutDesc]
    ) -> L2TaskId:
        with self._lock:
            task_id = self._get_next_task_id()

        asyncio.run_coroutine_threadsafe(
            self._execute_lookup(keys, task_id),
            self._loop,
        )
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Bitmap | None:
        with self._lock:
            return self._completed_lookup_tasks.pop(task_id, None)

    def submit_unlock(self, keys: list[ObjectKey]) -> None:
        # No-op: the FS adapter tracks no per-key locks — a delete
        # racing the lookup→load window degrades the load to a miss.
        pass

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
        return {
            "is_healthy": self._loop_thread.is_alive(),
            "type": "FSL2Adapter",
            "base_path": str(self._base_path),
            "use_odirect": self._use_odirect,
            "event_loop_alive": self._loop_thread.is_alive(),
        }

    # ------------------------------------------------------------------
    # Eviction Interface
    # ------------------------------------------------------------------

    def delete(self, keys: list[ObjectKey]) -> None:
        """Delete each key's backing file and notify listeners.

        Args:
            keys: The object keys to delete.

        Note:
            No per-key locks: a delete racing a load turns that load
            into a miss; racing a store of the same key may leave the
            key re-stored.
        """
        if not keys:
            return
        try:
            fut = asyncio.run_coroutine_threadsafe(
                self._execute_delete(keys),
                self._loop,
            )
            deleted_keys, deleted_sizes = fut.result(timeout=30.0)
        except Exception as e:
            logger.warning("FSL2Adapter delete failed: %s", e)
            return
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    # ``get_usage()`` is inherited from ``L2AdapterInterface``. The base
    # class maintains byte totals via ``_notify_keys_stored`` /
    # ``_notify_keys_deleted``. Without ``max_capacity_gb`` (default 0) the
    # FS adapter declares no capacity, so ``supports_global_eviction`` returns
    # ``False`` and ``usage_fraction == -1.0`` — the eviction controller treats
    # this as "no eviction signal" and skips the adapter entirely.

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def get_memory_tags(self) -> MemoryTags:
        """Return local storage tags; remote when the adapter is shared."""
        residency = (
            MemoryResidency.REMOTE if self._config.shared else MemoryResidency.LOCAL
        )
        return MemoryTags(MemorySegment.STORAGE, residency)

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

    def _key_to_path(self, key: ObjectKey) -> Path:
        return self._base_path / _object_key_to_filename(key)

    async def _key_exists_on_disk(
        self,
        key: ObjectKey,
    ) -> bool:
        """Check whether the file for *key* exists on disk.

        Uses ``aiofiles.os.path.exists`` so the check is
        non-blocking and always reflects the real FS state,
        which is critical for multi-node shared-FS setups.
        """
        path = self._key_to_path(key)
        return await aiofiles.os.path.exists(path)

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

    def _odirect_usable(
        self, file_path: Path, buf: Union[bytearray, memoryview, bytes]
    ) -> bool:
        """Whether *buf* meets O_DIRECT's length and address alignment."""
        bs = self._os_disk_bs
        if bs > 0 and len(buf) % bs == 0 and _buffer_is_aligned(buf, bs):
            return True
        if not self._warned_unaligned:
            self._warned_unaligned = True
            logger.warning(
                "Cannot use O_DIRECT for %s: buffer of %d bytes is not aligned "
                "to block size %d, using buffered I/O. Further unaligned "
                "buffers will not be logged.",
                file_path.name,
                len(buf),
                bs,
            )
        return False

    def _warn_refused(self, op: str, file_path: Path, e: OSError) -> None:
        if self._warned_refused:
            return
        self._warned_refused = True
        logger.warning(
            "O_DIRECT %s of %s refused (%s), using buffered I/O. Further "
            "refusals will not be logged.",
            op,
            file_path.name,
            errno.errorcode.get(e.errno or 0, e.errno),
        )

    def _read_with_odirect(
        self,
        file_path: Path,
        dst_buf: Union[bytearray, memoryview, bytes],
    ) -> int:
        """Synchronous O_DIRECT read into *dst_buf*.

        Falls back to a buffered read when *dst_buf* is not aligned, or when
        the filesystem refuses O_DIRECT. Any other error, including a missing
        file, propagates to the caller, so a failed read is not mistaken for
        a short one.

        Returns the number of bytes actually read.
        Runs in an executor (not on the event loop).
        """
        if self._odirect_usable(file_path, dst_buf):
            try:
                fd = os.open(
                    str(file_path),
                    os.O_RDONLY | getattr(os, "O_DIRECT", 0),
                )
                try:
                    fdo = os.fdopen(fd, "rb", buffering=0)
                except BaseException:
                    os.close(fd)
                    raise
                with fdo:
                    return _readinto_full(fdo, dst_buf)
            except OSError as e:
                if e.errno not in _DIRECT_IO_REFUSED_ERRNOS:
                    raise
                self._warn_refused("read", file_path, e)
        with open(file_path, "rb") as f:
            return _readinto_full(f, dst_buf)

    def _write_with_odirect(
        self, file_path: Path, buf: Union[bytearray, memoryview, bytes]
    ) -> None:
        """Synchronous O_DIRECT write of *buf*.

        Falls back to a buffered write when *buf* is not aligned, or when
        the filesystem refuses O_DIRECT. The buffered write starts over, so a
        partial O_DIRECT write never survives. Any other error propagates.

        Runs in an executor (not on the event loop).
        """
        if self._odirect_usable(file_path, buf):
            try:
                fd = os.open(
                    str(file_path),
                    os.O_CREAT | os.O_WRONLY | os.O_TRUNC | getattr(os, "O_DIRECT", 0),
                    0o644,
                )
                try:
                    _write_full(fd, buf)
                finally:
                    os.close(fd)
                return
            except OSError as e:
                if e.errno not in _DIRECT_IO_REFUSED_ERRNOS:
                    raise
                self._warn_refused("write", file_path, e)
        with open(file_path, "wb") as f:
            f.write(buf)

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
        try:
            for key, obj in zip(keys, objects, strict=True):
                file_path, tmp_path = self._key_to_file_and_tmp_path(key)

                # Skip if already stored on disk
                if await aiofiles.os.path.exists(file_path):
                    continue
                buf = obj.byte_array
                size = len(buf)

                try:
                    if self._use_odirect:
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
                    stored_keys.append(key)
                    stored_sizes.append(size)
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

        if stored_keys:
            self._notify_keys_stored(stored_keys, stored_sizes)

        with self._lock:
            self._completed_store_tasks[task_id] = L2StoreResult(success, bytes_written)
        self._store_efd.notify()

    # ---- lookup ---------------------------------------------------------

    async def _execute_lookup(
        self,
        keys: list[ObjectKey],
        task_id: L2TaskId,
    ) -> None:
        bitmap = Bitmap(len(keys))
        for i, key in enumerate(keys):
            if not await self._key_exists_on_disk(key):
                continue
            bitmap.set(i)

        with self._lock:
            self._completed_lookup_tasks[task_id] = bitmap
        self._lookup_efd.notify()

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

        loaded_keys = [keys[i] for i in bitmap.get_indices_list()]
        if loaded_keys:
            self._notify_keys_accessed(loaded_keys)

        with self._lock:
            self._completed_load_tasks[task_id] = bitmap
        self._load_efd.notify()

    # ---- delete ---------------------------------------------------------

    async def _execute_delete(
        self, keys: list[ObjectKey]
    ) -> tuple[list[ObjectKey], list[int]]:
        """Unlink each key's file concurrently; return removed keys + sizes."""
        sem = asyncio.Semaphore(self._DELETE_CONCURRENCY)

        async def _delete_one(key: ObjectKey) -> tuple[ObjectKey, int] | None:
            file_path = self._key_to_path(key)
            async with sem:
                try:
                    size = (await aiofiles.os.stat(file_path)).st_size
                    await aiofiles.os.unlink(file_path)
                except FileNotFoundError:
                    return None
                except Exception:
                    logger.exception(
                        "FSL2Adapter failed to delete %s",
                        file_path,
                    )
                    return None
            return key, size

        results = await asyncio.gather(*(_delete_one(k) for k in keys))

        deleted_keys: list[ObjectKey] = []
        deleted_sizes: list[int] = []
        for res in results:
            if res is not None:
                key, size = res
                deleted_keys.append(key)
                deleted_sizes.append(size)
        return deleted_keys, deleted_sizes


# Self-register config type and adapter factory
register_l2_adapter_type("fs", FSL2AdapterConfig)


def _create_fs_adapter(
    config: L2AdapterConfigBase,
    l1_memory_desc: "Optional[L1MemoryDesc]" = None,
) -> L2AdapterInterface:
    """Create an FSL2Adapter from config."""
    return FSL2Adapter(config)  # type: ignore[arg-type]


register_l2_adapter_factory("fs", _create_fs_adapter)
