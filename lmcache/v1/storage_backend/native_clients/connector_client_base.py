# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
import asyncio
import concurrent.futures

NativeClientT = TypeVar("NativeClientT")

# Turns the per-key result mask of one completion into the value its future
# resolves to. Bound at submit time, so a completion can never be silently
# re-interpreted -- or discarded -- based on an op name at drain time.
CompletionDecoder = Callable[[Optional[Sequence[int]]], Optional[list[bool]]]


def _discard_mask(mask: Optional[Sequence[int]]) -> None:
    """Decode a completion that carries no per-key results (SET).

    Args:
        mask: Ignored; the native layer reports no per-key results for SET.

    Returns:
        None.
    """
    return None


def _decode_mask(num_keys: int) -> CompletionDecoder:
    """Build a decoder returning one bool per submitted key.

    The native layer reports ``1`` when a key hit (EXISTS found it, GET filled
    its buffer) and ``0`` when it missed. A batch completes with ``ok=True``
    even when individual keys miss, so this mask is the only signal that
    separates hits from misses: partial hits are the normal case for a cache,
    and reading a missed key's buffer would yield uninitialised memory dressed
    up as data. A missing or mis-sized mask is therefore an error, never an
    implicit "all hit".

    Args:
        num_keys: Number of keys submitted in the batch.

    Returns:
        A decoder producing a ``list[bool]`` of length ``num_keys``.
    """

    def decode(mask: Optional[Sequence[int]]) -> list[bool]:
        if mask is None or len(mask) != num_keys:
            raise RuntimeError(
                f"native connector returned {0 if mask is None else len(mask)} "
                f"per-key results for a {num_keys}-key batch; cannot tell hits "
                f"from misses"
            )
        return [bool(bit) for bit in mask]

    return decode


class ConnectorClientBase(Generic[NativeClientT]):
    """Asyncio bridge to a pybind-wrapped native connector.

    Submissions return a native future id; completions arrive as a batch on the
    connector's eventfd and are routed back to the waiting future. Two contracts
    matter for subclasses and callers:

    - **Per-key results.** A batch completes successfully even when individual
      keys miss, so GET and EXISTS resolve to one bool per key rather than to a
      single batch-level verdict. Buffers of missed keys are never written and
      must not be read.
    - **Buffer lifetime.** Native workers hold raw pointers into the caller's
      buffers until the corresponding completion is drained, so this class
      keeps a reference to every submitted buffer until its future resolves,
      and stops the native workers before failing futures in bulk.
    """

    def __init__(
        self,
        native_client: NativeClientT,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.loop = loop or asyncio.get_running_loop()
        self._client: NativeClientT = native_client
        self._fd = int(self._client.event_fd())  # type: ignore[attr-defined]
        self._closed = False
        # Keepalive refs prevent buffers passed to native code from being
        # garbage-collected while C++ worker threads still hold raw pointers.
        self._pending: Dict[
            int,
            Tuple[
                Union[asyncio.Future, concurrent.futures.Future],
                CompletionDecoder,
                Tuple[Any, ...],
            ],
        ] = {}
        self.loop.add_reader(self._fd, self._on_ready)

    def _on_ready(self) -> None:
        if self._closed:
            return

        try:
            while True:
                items = self._client.drain_completions()  # type: ignore[attr-defined]
                if not items:
                    return

                for future_id, ok, error, result_bools in items:
                    fid = int(future_id)
                    entry = self._pending.pop(fid, None)
                    if entry is None:
                        continue

                    fut, decode, _keepalive = entry
                    if fut.done():
                        continue

                    if not ok:
                        fut.set_exception(RuntimeError(str(error)))
                        continue

                    try:
                        fut.set_result(decode(result_bools))
                    except Exception as e:
                        # A malformed completion poisons only its own future.
                        fut.set_exception(e)
        except Exception as e:
            self._shutdown_native(best_effort=True)
            try:
                self._client.close()  # type: ignore[attr-defined]
            finally:
                self._fail_all(RuntimeError(f"native drain_completions failed: {e}"))

    def _fail_all(self, exc: Exception) -> None:
        for fid, (fut, _, _keepalive) in list(self._pending.items()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    def _shutdown_native(self, best_effort: bool = False) -> None:
        try:
            self._closed = True
            self.loop.remove_reader(self._fd)
        except Exception:
            if not best_effort:
                raise

    def _register_future_async(
        self,
        decode: CompletionDecoder,
        future_id: int,
        keepalive: Tuple[Any, ...] = (),
    ) -> asyncio.Future:
        fut = self.loop.create_future()
        self._pending[int(future_id)] = (fut, decode, keepalive)
        return fut

    def _register_future_sync(
        self,
        decode: CompletionDecoder,
        future_id: int,
        keepalive: Tuple[Any, ...] = (),
    ) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._pending[int(future_id)] = (fut, decode, keepalive)
        return fut

    async def get(self, key: str, buf: memoryview) -> bool:
        """Read one key into ``buf``.

        Args:
            key: Key to read.
            buf: Destination buffer, written only on a hit.

        Returns:
            True if the key was found and ``buf`` filled; False on a miss, in
            which case ``buf`` still holds its previous content.
        """
        return (await self.batch_get([key], [buf]))[0]

    async def set(self, key: str, buf: memoryview) -> None:
        return await self.batch_set([key], [buf])

    async def exists(self, key: str) -> bool:
        results = await self.batch_exists([key])
        return results[0]

    async def batch_get(self, keys: list[str], bufs: list[memoryview]) -> list[bool]:
        """Read ``keys`` into ``bufs``, tolerating per-key misses.

        Args:
            keys: Keys to read.
            bufs: Destination buffers, one per key; ``bufs[i]`` is written only
                if ``keys[i]`` hit.

        Returns:
            One bool per key: True if the key hit and its buffer was filled,
            False on a miss. Buffers of missed keys are untouched and must not
            be read.

        Raises:
            ValueError: If ``keys`` and ``bufs`` have different lengths.
            RuntimeError: If the batch failed, or if the native layer did not
                report exactly one result per key.
        """
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_async(
            _decode_mask(len(keys)), future_id, (keys, tuple(bufs))
        )
        return await fut

    async def batch_set(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_async(_discard_mask, future_id, (keys, tuple(bufs)))
        return await fut

    async def batch_exists(self, keys: list[str]) -> list[bool]:
        future_id = int(self._client.submit_batch_exists(keys))  # type: ignore[attr-defined]
        fut = self._register_future_async(_decode_mask(len(keys)), future_id)
        return await fut

    async def batched_exists(self, keys: list[str]) -> list[bool]:
        return await self.batch_exists(keys)

    def get_sync(self, key: str, buf: memoryview) -> bool:
        """Blocking variant of :meth:`get`.

        Args:
            key: Key to read.
            buf: Destination buffer, written only on a hit.

        Returns:
            True if the key was found and ``buf`` filled; False on a miss.
        """
        return self.batch_get_sync([key], [buf])[0]

    def set_sync(self, key: str, buf: memoryview) -> None:
        return self.batch_set_sync([key], [buf])

    def exists_sync(self, key: str) -> bool:
        results = self.batch_exists_sync([key])
        return results[0]

    def batch_get_sync(self, keys: list[str], bufs: list[memoryview]) -> list[bool]:
        """Blocking variant of :meth:`batch_get`.

        Args:
            keys: Keys to read.
            bufs: Destination buffers, one per key.

        Returns:
            One bool per key: True if the key hit and its buffer was filled,
            False on a miss.

        Raises:
            ValueError: If ``keys`` and ``bufs`` have different lengths.
            RuntimeError: If the batch failed, or if the native layer did not
                report exactly one result per key.
        """
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_get(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_sync(
            _decode_mask(len(keys)), future_id, (keys, tuple(bufs))
        )
        return fut.result()

    def batch_set_sync(self, keys: list[str], bufs: list[memoryview]) -> None:
        if len(keys) != len(bufs):
            raise ValueError("keys and bufs length mismatch")
        future_id = int(self._client.submit_batch_set(keys, bufs))  # type: ignore[attr-defined]
        fut = self._register_future_sync(_discard_mask, future_id, (keys, tuple(bufs)))
        return fut.result()

    def batch_exists_sync(self, keys: list[str]) -> list[bool]:
        future_id = int(self._client.submit_batch_exists(keys))  # type: ignore[attr-defined]
        fut = self._register_future_sync(_decode_mask(len(keys)), future_id)
        return fut.result()

    def batched_exists_sync(self, keys: list[str]) -> list[bool]:
        return self.batch_exists_sync(keys)

    def close(self) -> None:
        if not self._closed:
            self._shutdown_native(best_effort=True)
            # Join native workers first: a failed future lets its caller free
            # the buffers C++ may still be writing into.
            self._client.close()  # type: ignore[attr-defined]
            self._fail_all(RuntimeError("Client closed"))
