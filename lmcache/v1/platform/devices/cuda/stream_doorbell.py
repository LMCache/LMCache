# SPDX-License-Identifier: Apache-2.0
"""Doorbell signals between CUDA streams: 32-bit words in host memory that
a stream can write (``cuStreamWriteValue32``) and wait on
(``cuStreamWaitValue32``).

Unlike a stream event, a doorbell word can also be written by the host, so
a load that is not stream-ordered (a thread pool reading files, a network
read) can release the stream that waits for it. The stream-ordered loads
in the tree gain nothing from it over events, so it is opt-in: set
``LMCACHE_STAGING_DOORBELL=1`` to use it for the staging banks.
"""

# Future
from __future__ import annotations

# Standard
import ctypes
import mmap
import os

# First Party
from lmcache.logging import init_logger
from lmcache.v1.platform import stream as platform_stream
from lmcache.v1.platform.base.staging_lane import BANKS_PER_LANE, BankSignals

logger = init_logger(__name__)

ENV_STAGING_DOORBELL = "LMCACHE_STAGING_DOORBELL"

_WORD_BYTES = 4
_WORD_MASK = 0xFFFFFFFF
# cuda.h: CU_MEMHOSTREGISTER_DEVICEMAP, CU_STREAM_WAIT_VALUE_EQ,
# CU_STREAM_WRITE_VALUE_DEFAULT.
_MEMHOSTREGISTER_DEVICEMAP = 0x02
_WAIT_VALUE_EQ = 0x1
_WRITE_VALUE_DEFAULT = 0x0
_WORDS_PER_LANE = 2 * BANKS_PER_LANE


def staging_doorbell_requested() -> bool:
    """Return whether the environment asks for doorbell bank signals."""
    return os.environ.get(ENV_STAGING_DOORBELL, "").lower() in ("1", "true", "yes")


class StreamDoorbell:
    """One page of doorbell words registered with the current CUDA context.

    Construct and close it with the owning device's context current. The
    words start at zero.

    Raises:
        RuntimeError: If the driver library is missing or refuses to map
            host memory for stream access.
    """

    def __init__(self) -> None:
        try:
            self._driver = ctypes.CDLL("libcuda.so.1")
        except OSError as error:
            raise RuntimeError("the CUDA driver library is not loadable") from error
        for function in (
            self._driver.cuStreamWaitValue32_v2,
            self._driver.cuStreamWriteValue32_v2,
        ):
            function.argtypes = [
                ctypes.c_void_p,
                ctypes.c_ulonglong,
                ctypes.c_uint32,
                ctypes.c_uint,
            ]
            function.restype = ctypes.c_int
        self._page = mmap.mmap(-1, mmap.PAGESIZE)
        self._host_address = ctypes.addressof(ctypes.c_char.from_buffer(self._page))
        self._check(
            self._driver.cuMemHostRegister_v2(
                ctypes.c_void_p(self._host_address),
                ctypes.c_size_t(mmap.PAGESIZE),
                ctypes.c_uint(_MEMHOSTREGISTER_DEVICEMAP),
            ),
            "cuMemHostRegister",
        )
        device_address = ctypes.c_ulonglong()
        self._check(
            self._driver.cuMemHostGetDevicePointer_v2(
                ctypes.byref(device_address),
                ctypes.c_void_p(self._host_address),
                ctypes.c_uint(0),
            ),
            "cuMemHostGetDevicePointer",
        )
        self._device_address = device_address.value
        self._words_taken = 0

    def take_words(self, count: int) -> int:
        """Reserve ``count`` consecutive words.

        Args:
            count: Number of words.

        Returns:
            The index of the first reserved word.

        Raises:
            RuntimeError: If the page has fewer than ``count`` words left.
        """
        first = self._words_taken
        if (first + count) * _WORD_BYTES > mmap.PAGESIZE:
            raise RuntimeError("the doorbell page has no free words")
        self._words_taken = first + count
        return first

    def write(self, stream_handle: int, word: int, value: int) -> None:
        """Queue a write of ``value`` to ``word`` on a stream.

        Args:
            stream_handle: Native handle of the stream.
            word: Word index.
            value: The 32-bit value to write.

        Raises:
            RuntimeError: If the driver rejects the operation.
        """
        self._check(
            self._driver.cuStreamWriteValue32_v2(
                ctypes.c_void_p(stream_handle),
                self._device_address + word * _WORD_BYTES,
                value,
                _WRITE_VALUE_DEFAULT,
            ),
            "cuStreamWriteValue32",
        )

    def wait_equal(self, stream_handle: int, word: int, value: int) -> None:
        """Queue a wait on a stream until ``word`` equals ``value``.

        Args:
            stream_handle: Native handle of the stream.
            word: Word index.
            value: The 32-bit value to wait for.

        Raises:
            RuntimeError: If the driver rejects the operation.
        """
        self._check(
            self._driver.cuStreamWaitValue32_v2(
                ctypes.c_void_p(stream_handle),
                self._device_address + word * _WORD_BYTES,
                value,
                _WAIT_VALUE_EQ,
            ),
            "cuStreamWaitValue32",
        )

    def read(self, word: int) -> int:
        """Return the current value of ``word``, read from the host."""
        return ctypes.c_uint32.from_address(
            self._host_address + word * _WORD_BYTES
        ).value

    def close(self) -> None:
        """Unregister the page. No stream may still wait on or write it."""
        result = self._driver.cuMemHostUnregister(ctypes.c_void_p(self._host_address))
        if result != 0:
            logger.warning("cuMemHostUnregister failed with CUDA error %d", result)

    @staticmethod
    def _check(result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed with CUDA error {result}")


def open_stream_doorbell(device: object, stream: object) -> StreamDoorbell:
    """Create a doorbell and check that a stream of ``device`` can write it.

    Blocks until ``stream`` has drained. Call with the device's context
    current.

    Args:
        device: The device whose streams will use the doorbell.
        stream: A stream of the device to run the check on.

    Returns:
        The doorbell.

    Raises:
        RuntimeError: If the driver cannot map the page or the device does
            not support stream writes to it.
    """
    doorbell = StreamDoorbell()
    try:
        word = doorbell.take_words(1)
        doorbell.write(platform_stream.stream_handle(device, stream), word, 1)
        platform_stream.synchronize_stream(device, stream)
        if doorbell.read(word) != 1:
            raise RuntimeError("a stream write did not reach the doorbell word")
    except RuntimeError:
        doorbell.close()
        raise
    return doorbell


class DoorbellBankSignals(BankSignals):
    """Bank signals as doorbell words: a "staged" and a "free" word per bank.

    Each word holds the bank's round number. A load waits until the "free"
    word equals the last round, then writes the next round to the "staged"
    word; the scatter waits for that value and writes it to the "free"
    word. Waiting for equality keeps the scheme correct when the 32-bit
    round number wraps.

    Args:
        doorbell: The doorbell page to take the words from.
        device: The device that owns the lane's streams.

    Raises:
        RuntimeError: If the page has no free words.
    """

    def __init__(self, doorbell: StreamDoorbell, device: object) -> None:
        self._doorbell = doorbell
        self._device = device
        self._first_word = doorbell.take_words(_WORDS_PER_LANE)
        self._rounds = [0] * BANKS_PER_LANE

    def wait_free(self, bank: int, stream: object) -> None:
        self._doorbell.wait_equal(
            self._handle(stream), self._free_word(bank), self._rounds[bank]
        )

    def signal_staged(self, bank: int, stream: object) -> None:
        next_round = (self._rounds[bank] + 1) & _WORD_MASK
        self._doorbell.write(self._handle(stream), self._staged_word(bank), next_round)
        # Advance only once the write is queued, so a failed call leaves the
        # bank's words and round number in step.
        self._rounds[bank] = next_round

    def wait_staged(self, bank: int, stream: object) -> None:
        self._doorbell.wait_equal(
            self._handle(stream), self._staged_word(bank), self._rounds[bank]
        )

    def signal_free(self, bank: int, stream: object) -> None:
        self._doorbell.write(
            self._handle(stream), self._free_word(bank), self._rounds[bank]
        )

    def _handle(self, stream: object) -> int:
        return platform_stream.stream_handle(self._device, stream)

    def _staged_word(self, bank: int) -> int:
        return self._first_word + 2 * bank

    def _free_word(self, bank: int) -> int:
        return self._first_word + 2 * bank + 1
