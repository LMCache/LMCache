# SPDX-License-Identifier: Apache-2.0
# Standard
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
import asyncio
import fcntl


def lock_path_for_file(file_path: Path) -> Path:
    return file_path.with_name(file_path.name + ".lock")


def lock_path_for_chunk_hash(file_path: Path, chunk_hash: int) -> Path:
    return file_path.with_name(f"{chunk_hash}.lock")


@contextmanager
def exclusive_flock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_file
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def async_exclusive_flock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_path, "a+")
    try:
        await asyncio.to_thread(fcntl.flock, lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_file
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    finally:
        lock_file.close()
