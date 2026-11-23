# SPDX-License-Identifier: Apache-2.0
"""Destructive, opt-in UECC test on a real NVMe namespace.

The test corrupts one payload LBA with the NVMe Write Uncorrectable
command and verifies that the raw-block core latches the device for the
life of the open handle and serves misses afterwards. The corrupted range
is backed up before the adapter is created and restored in a ``finally``
block. Expect the drive's error log, and possibly its SMART media-error
count, to increase because real reads are made to fail.

Set LMCACHE_TEST_RAW_BLOCK_DESTRUCTIVE=1 and LMCACHE_TEST_UECC_CHAR_DEVICE
to opt in. The lab's lmcache_reserved_nvme_wwid pytest fixture must acquire
exclusive use across all hosts and local consumers, yield the reserved
namespace's sysfs WWID, and hold the lease through fixture teardown. A local
block-device claim and fd scan alone cannot fence NVMe passthrough opens.

Set LMCACHE_TEST_UECC_BACKUP_DIR to an existing directory on a separate disk
(default: working directory). The recovery file starts with one JSON line,
followed by exactly the saved bytes. It is retained if verification fails.
"""

# Standard
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import ctypes
import fcntl
import hashlib
import json
import mmap
import os
import re
import stat
import struct

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.l2_adapters.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    encode_object_key,
)
from lmcache.v1.distributed.l2_adapters.raw_block.io_engine import IoUringBlockIO
from tests.v1.distributed.raw_block.test_utils import (
    make_empty_memory_obj,
    make_memory_obj,
    make_object_key,
    memory_obj_bytes,
)

DESTRUCTIVE_ENV = "LMCACHE_TEST_RAW_BLOCK_DESTRUCTIVE"
CHAR_DEVICE_ENV = "LMCACHE_TEST_UECC_CHAR_DEVICE"

BLOCK_ALIGN = 4096
CAPACITY_BYTES = 8 * 1024 * 1024
SLOT_BYTES = 256 * 1024
META_TOTAL_BYTES = 1 * 1024 * 1024
MAX_DATA_TRANSFER_SIZE = 64 * 1024
PAYLOAD_A_LEN = 150_000
PAYLOAD_B_LEN = 512

# NVMe status mask ignoring the CRD, More and DNR bits: SCT 2 / SC 0x81.
UECC_STATUS_MASK = 0x07FF
UECC_STATUS_VALUE = 0x0281

# Identify Controller offsets and opcodes (NVM Command Set).
_ONCS_OFFSET = 520
_IDENTIFY_OPCODE = 0x06
_IDENTIFY_CNS_CONTROLLER = 1
_WRITE_UNCORRECTABLE_OPCODE = 0x04
_NVME_IOCTL_MAGIC = 0x4E


class _NvmePassthruCmd(ctypes.Structure):
    """``struct nvme_passthru_cmd`` from the kernel NVMe uapi."""

    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("rsvd1", ctypes.c_uint16),
        ("nsid", ctypes.c_uint32),
        ("cdw2", ctypes.c_uint32),
        ("cdw3", ctypes.c_uint32),
        ("metadata", ctypes.c_uint64),
        ("addr", ctypes.c_uint64),
        ("metadata_len", ctypes.c_uint32),
        ("data_len", ctypes.c_uint32),
        ("cdw10", ctypes.c_uint32),
        ("cdw11", ctypes.c_uint32),
        ("cdw12", ctypes.c_uint32),
        ("cdw13", ctypes.c_uint32),
        ("cdw14", ctypes.c_uint32),
        ("cdw15", ctypes.c_uint32),
        ("timeout_ms", ctypes.c_uint32),
        ("result", ctypes.c_uint32),
    ]


def _iowr(nr: int, size: int) -> int:
    return (3 << 30) | (size << 16) | (_NVME_IOCTL_MAGIC << 8) | nr


_PASSTHRU_SIZE = ctypes.sizeof(_NvmePassthruCmd)
_NVME_IOCTL_ADMIN_CMD = _iowr(0x41, _PASSTHRU_SIZE)
_NVME_IOCTL_IO_CMD = _iowr(0x43, _PASSTHRU_SIZE)


def _identify(fd: int, nsid: int, cns: int) -> bytes:
    """Read Identify data through the same direct ioctl used for gating."""
    data = ctypes.create_string_buffer(4096)
    cmd = _NvmePassthruCmd(
        opcode=_IDENTIFY_OPCODE,
        nsid=nsid,
        cdw10=cns,
        addr=ctypes.addressof(data),
        data_len=len(data),
    )
    try:
        status = fcntl.ioctl(fd, _NVME_IOCTL_ADMIN_CMD, cmd)
    except OSError as exc:
        pytest.skip(f"Identify CNS={cns} NSID={nsid} ioctl failed: {exc}")
    if status:
        pytest.skip(f"Identify CNS={cns} NSID={nsid} returned NVMe status {status:#x}")
    return data.raw


def _namespace(fd: int) -> tuple[int, int, bytes, bytes]:
    try:
        nsid = fcntl.ioctl(fd, 0x4E40)  # NVME_IOCTL_ID
    except OSError as exc:
        pytest.skip(f"Get namespace ID ioctl failed: {exc}")
    if nsid <= 0:
        pytest.skip(f"Invalid namespace identifier {nsid}")
    data = _identify(fd, nsid, 0)
    fmt = (data[26] & 0x0F) | ((data[26] & 0x60) >> 1)
    if fmt > data[25]:
        pytest.skip(f"Invalid active NVMe LBA format {fmt}")
    metadata_size, shift = struct.unpack_from("<HB", data, 128 + 4 * fmt)
    lba_size = 1 << shift
    if metadata_size or not (512 <= lba_size <= BLOCK_ALIGN):
        pytest.skip(f"Unsupported format {lba_size=} {metadata_size=}")
    nguid, eui = data[104:120], data[120:128]
    if not any(nguid + eui):
        pytest.skip("Identify Namespace lacks a stable NGUID or EUI64")
    return nsid, lba_size, nguid, eui


def _check_unused(block: Path, forbidden: set[tuple[int, int]], own: set[int]) -> None:
    if list(block.glob("*/partition")) or any((block / "holders").iterdir()):
        pytest.skip(f"{block.name} has partitions or holders")
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        if line.split()[2] == (block / "dev").read_text().strip():
            pytest.skip(f"Mounted namespace: {line}")
    for line in Path("/proc/swaps").read_text().splitlines()[1:]:
        swap = os.stat(line.split()[0])
        if (stat.S_IFMT(swap.st_mode), swap.st_rdev) in forbidden:
            pytest.skip(f"Active swap: {line}")
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            for entry in (proc / "fd").iterdir():
                if int(proc.name) == os.getpid() and int(entry.name) in own:
                    continue
                try:
                    info = entry.stat()
                except FileNotFoundError:
                    continue
                if (stat.S_IFMT(info.st_mode), info.st_rdev) in forbidden:
                    pytest.skip(f"Namespace already open at {entry}")
        except (FileNotFoundError, ProcessLookupError):
            continue


def _sys_device(kind: str, device: int) -> Path:
    return Path(f"/sys/dev/{kind}/{os.major(device)}:{os.minor(device)}").resolve(
        strict=True
    )


def _check_backup_disk(directory: Path, block: Path) -> None:
    disk = _sys_device("block", directory.stat().st_dev)
    if (disk / "partition").exists():
        disk = disk.parent
    # Require a simple, independently identifiable local backup disk.
    if (
        disk == block
        or disk.parent == block.parent
        or "virtual" in disk.parts
        or any((disk / "slaves").iterdir())
    ):
        pytest.skip(f"Cannot establish independent backup storage on {disk}")


def _read_range(fd: int) -> bytes:
    with mmap.mmap(-1, CAPACITY_BYTES) as buf:
        if os.preadv(fd, [buf], 0) != CAPACITY_BYTES:
            raise OSError("Short direct read of the backup region")
        return buf[:]


def _restore_range(fd: int, start: int, data: bytes, align: int) -> None:
    if start < 0 or start % align or not data or len(data) % align:
        raise ValueError(
            "Restore requires an aligned offset and a whole number of LBAs"
        )
    with mmap.mmap(-1, len(data)) as buf:
        buf[:] = data
        pos = 0
        with memoryview(buf) as view:
            while pos < len(data):
                with view[pos:] as remaining:
                    written = os.pwritev(fd, [remaining], start + pos)
                if written <= 0 or written % align:
                    raise OSError(f"Incomplete direct restore at byte {start + pos}")
                pos += written
    os.fsync(fd)


def _write_backup(path: Path, metadata: dict[str, int | str], data: bytes) -> None:
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
    ) as f:
        f.write(json.dumps(metadata).encode() + b"\n")
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _exercise(core: RawBlockCore, char_fd: int, nsid: int, lba_size: int) -> None:
    key_a = encode_object_key(make_object_key(900))
    key_b = encode_object_key(make_object_key(901))
    payload_a = bytes(i & 0xFF for i in range(PAYLOAD_A_LEN))
    payload_b = b"b" * PAYLOAD_B_LEN
    assert core.put_many(
        [key_a, key_b],
        [make_memory_obj(payload_a), make_memory_obj(payload_b)],
    ).results == [True, True]

    # Clean read: both keys round-trip before the corruption.
    dst_a = make_empty_memory_obj(PAYLOAD_A_LEN)
    dst_b = make_empty_memory_obj(PAYLOAD_B_LEN)
    assert core.load_many_into([key_a.encoded, key_b.encoded], [dst_a, dst_b]) == [
        True,
        True,
    ]
    assert memory_obj_bytes(dst_a) == payload_a
    assert memory_obj_bytes(dst_b) == payload_b

    entry_offset = core.entry_offset(key_a.encoded)
    assert entry_offset is not None
    payload_offset = entry_offset + core.header_bytes
    inject_offset = payload_offset + MAX_DATA_TRANSFER_SIZE
    if (
        not core.data_base_offset() <= entry_offset < payload_offset <= inject_offset
        or inject_offset % lba_size
        or inject_offset + lba_size
        > min(CAPACITY_BYTES, payload_offset + len(payload_a))
    ):
        raise ValueError("Target LBA is outside the backed-up logical payload")
    inject_lba = inject_offset // lba_size
    cmd = _NvmePassthruCmd(
        opcode=_WRITE_UNCORRECTABLE_OPCODE,
        nsid=nsid,
        cdw10=inject_lba & 0xFFFFFFFF,
        cdw11=(inject_lba >> 32) & 0xFFFFFFFF,
        cdw12=0,
    )
    try:
        result = fcntl.ioctl(char_fd, _NVME_IOCTL_IO_CMD, cmd)
    except OSError as exc:
        raise RuntimeError(f"Write Uncorrectable ioctl failed: {exc}") from exc
    if result:
        raise RuntimeError(f"Write Uncorrectable returned NVMe status {result:#x}")

    # Load through LMCache: the first UECC latches the device and the
    # load misses, even though the sibling chunks read fine.
    reloaded_a = make_empty_memory_obj(PAYLOAD_A_LEN)
    assert core.load_many_into([key_a.encoded], [reloaded_a]) == [False]

    status = core.report_status()
    assert status["is_healthy"] is False
    worker_error = status["worker_error"]
    assert worker_error is not None
    status_match = re.search(r"status=0x([0-9a-fA-F]+)", worker_error)
    assert status_match is not None
    masked_status = int(status_match.group(1), 16) & UECC_STATUS_MASK
    assert masked_status == UECC_STATUS_VALUE, worker_error
    chunk_match = re.search(r"offset=(\d+) len=(\d+)", worker_error)
    assert chunk_match is not None
    chunk_offset = int(chunk_match.group(1))
    chunk_len = int(chunk_match.group(2))
    assert chunk_offset <= inject_offset
    assert inject_offset + lba_size <= chunk_offset + chunk_len

    # The healthy sibling key misses too: the whole device is blocked.
    reloaded_b = make_empty_memory_obj(PAYLOAD_B_LEN)
    assert core.load_many_into([key_b.encoded], [reloaded_b]) == [False]


def _open_core(char_path: Path) -> RawBlockCore:
    return RawBlockCore(
        RawBlockCoreConfig(
            device_path=str(char_path),
            capacity_bytes=CAPACITY_BYTES,
            block_align=BLOCK_ALIGN,
            header_bytes=BLOCK_ALIGN,
            slot_bytes=SLOT_BYTES,
            use_odirect=False,
            enable_zero_copy=False,
            meta_total_bytes=META_TOTAL_BYTES,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=3600,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            meta_verify_on_load=True,
            load_checkpoint_on_init=False,
            io_engine="io_uring",
            use_uring_cmd=True,
            iouring_queue_depth=64,
            max_data_transfer_size=MAX_DATA_TRANSFER_SIZE,
        )
    )


@pytest.mark.no_shared_allocator
def test_uecc_write_uncorrectable_latches_raw_block_core(
    request: pytest.FixtureRequest,
) -> None:
    """Trip the open core on one leased namespace and verify exact recovery."""
    if os.environ.get(DESTRUCTIVE_ENV) != "1":
        pytest.skip(f"Set {DESTRUCTIVE_ENV}=1 to run the destructive UECC test")
    target = os.environ.get(CHAR_DEVICE_ENV, "")
    if not target:
        pytest.skip(f"Set {CHAR_DEVICE_ENV} to the dedicated NVMe namespace")
    try:
        reserved = request.getfixturevalue("lmcache_reserved_nvme_wwid")
    except pytest.FixtureLookupError as exc:
        if exc.argname != "lmcache_reserved_nvme_wwid":
            raise
        pytest.skip("No lab reservation fixture for exclusive NVMe namespace use")
    pytest.importorskip("lmcache_rust_raw_block_io")
    directory = Path(os.environ.get("LMCACHE_TEST_UECC_BACKUP_DIR", os.getcwd()))
    with ExitStack() as stack:

        def open_fd(path: Path, flags: int) -> int:
            fd = os.open(path, flags)
            stack.callback(os.close, fd)
            return fd

        try:
            char_path = Path(target).resolve(strict=True)
            if not re.fullmatch(r"ng\d+n\d+", char_path.name):
                pytest.skip(f"Expected NVMe namespace character device: {char_path}")
            block_path = char_path.with_name("nvme" + char_path.name[2:])
            if not stat.S_ISCHR(char_path.stat().st_mode) or not stat.S_ISBLK(
                block_path.stat().st_mode
            ):
                pytest.skip(f"Invalid NVMe namespace pair: {char_path}, {block_path}")
            block_fd = open_fd(block_path, os.O_RDWR | os.O_EXCL | os.O_DIRECT)
            verify_fd = open_fd(block_path, os.O_RDONLY | os.O_DIRECT)
            char_fd = open_fd(char_path, os.O_RDWR)
            char_num = os.fstat(char_fd).st_rdev
            block_num = os.fstat(block_fd).st_rdev
            char = _sys_device("char", char_num)
            block = _sys_device("block", block_num)
            wwid = (block / "wwid").read_text().strip()
            if (
                not wwid
                or reserved != wwid
                or char.parent != block.parent
                or os.fstat(verify_fd).st_rdev != block_num
            ):
                pytest.skip("Namespace pair or exclusive lease does not match")
            _check_unused(
                block,
                {(stat.S_IFCHR, char_num), (stat.S_IFBLK, block_num)},
                {block_fd, verify_fd, char_fd},
            )
            _check_backup_disk(directory, block)
            identity = _namespace(char_fd)
            nsid, lba_size, nguid, eui = identity
            if (
                identity != _namespace(block_fd)
                or nsid != int((block / "nsid").read_text())
                or lba_size != int((block / "queue/logical_block_size").read_text())
            ):
                pytest.skip(
                    "Namespace identity, NSID or LBA size differs between paths"
                )
            oncs = struct.unpack_from(
                "<H", _identify(char_fd, 0, _IDENTIFY_CNS_CONTROLLER), _ONCS_OFFSET
            )[0]
            if not oncs & 2:
                pytest.skip(f"Write Uncorrectable unsupported: ONCS={oncs:#x}")
            if int((block / "size").read_text()) * 512 < CAPACITY_BYTES:
                pytest.skip("Namespace is smaller than the bounded backup region")
        except OSError as exc:
            pytest.skip(f"Cannot establish hardware-test prerequisites: {exc}")

        saved = _read_range(block_fd)
        digest = hashlib.sha256(saved).hexdigest()
        metadata: dict[str, int | str] = {
            "wwid": wwid,
            "nguid": nguid.hex(),
            "eui64": eui.hex(),
            "nsid": nsid,
            "lba_size": lba_size,
            "offset": 0,
            "length": CAPACITY_BYTES,
            "sha256": digest,
            "device": str(block_path),
        }
        backup = directory / f"lmcache-uecc-{nguid.hex()}-{eui.hex()}.backup"
        _write_backup(backup, metadata, saved)
        print(f"Synced recovery file {backup.resolve()}: {metadata}", flush=True)
        stopped = False
        try:
            core = _open_core(char_path)
            engine = core._io_engine
            if not isinstance(engine, IoUringBlockIO) or engine._raw is None:
                core.close()
                raise RuntimeError("Missing native io_uring handle")
            raw = engine._raw

            # The engine normally logs and suppresses native close errors.
            # Require the native drain/join to return before restoring bytes.
            def close_and_confirm_drain() -> None:
                nonlocal stopped
                raw.close()
                stopped = True

            with patch.object(engine, "close", close_and_confirm_drain):
                try:
                    if (raw.nvme_lba_size(), raw.nvme_nsid()) != (lba_size, nsid):
                        raise RuntimeError(
                            "Core opened a different namespace or LBA format"
                        )
                    _exercise(core, char_fd, nsid, lba_size)
                finally:
                    core.close()
        finally:
            if not stopped:
                raise RuntimeError(f"Adapter quiescence unconfirmed; retain {backup}")
            try:
                with backup.open("rb") as f:
                    if json.loads(f.readline()) != metadata:
                        raise ValueError("Recovery manifest changed")
                    restore = f.read()
                if (
                    len(restore) != CAPACITY_BYTES
                    or hashlib.sha256(restore).hexdigest() != digest
                ):
                    raise ValueError(
                        "Recovery data failed checksum or length validation"
                    )
                _restore_range(block_fd, 0, restore, lba_size)
                actual = _read_range(verify_fd)
                if actual != restore:
                    first = next(
                        i
                        for i, (a, b) in enumerate(zip(actual, restore, strict=True))
                        if a != b
                    )
                    raise ValueError(f"First differing absolute byte offset {first}")
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"Restore verification failed: {exc}. Backup: {backup}"
                ) from exc
        backup.unlink()
