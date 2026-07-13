# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Any, Generic, Optional, TypeVar
import threading

# First Party
from lmcache import torch_dev, torch_device_type
from lmcache.logging import init_logger

logger = init_logger(__name__)

T = TypeVar("T")


class MessagingFuture(Generic[T]):
    def __init__(self):
        self.is_done_ = threading.Event()
        self.result_ = None
        # Callbacks fired on result-set.  Used to chain work without
        # blocking the MQ polling thread or worker thread on
        # ``future.result(timeout=...)``.  See ``add_done_callback``.
        self._callbacks: list = []
        # Guards the is_done_/callback-list transition so a callback
        # registered concurrently with set_result() is either appended
        # before the drain or run synchronously -- never lost.
        self._callback_lock = threading.Lock()

    def add_done_callback(self, fn) -> None:
        """Register a callback to run once ``set_result`` is called.

        If the future is already done, the callback runs synchronously
        (in the caller's thread).  Otherwise, it runs in the thread
        that calls ``set_result`` (the shared polling loop).  Keeps the
        callback chain short to avoid stalling the loop.
        """
        with self._callback_lock:
            if not self.is_done_.is_set():
                self._callbacks.append(fn)
                return
        # Already done: run synchronously, outside the lock.
        try:
            fn(self.result_)
        except Exception:
            # Do not propagate from callback -- one bad callback
            # would otherwise kill the polling loop.
            logger.exception("MessagingFuture done-callback raised")

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        return self.is_done_.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            bool: True if the future is done, False if the timeout was reached.
        """
        return self.is_done_.wait(timeout)

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds.
                If None, wait indefinitely.

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise TimeoutError("Future result not available within timeout")
        return self.result_

    def set_result(self, result: T) -> None:
        """Mark the future done and fire any registered callbacks.

        NOT to be called by users directly -- only the messaging
        system should set results.
        """
        with self._callback_lock:
            self.result_ = result
            self.is_done_.set()
            # Snapshot-and-swap under the lock: callbacks registered
            # after this point see is_done_ set and run synchronously
            # in their own thread; callbacks in the snapshot run below,
            # outside the lock (a callback may itself call
            # add_done_callback without deadlocking).
            callbacks, self._callbacks = self._callbacks, []
        for fn in callbacks:
            try:
                fn(result)
            except Exception:
                # Do not propagate from callback -- one bad callback
                # would otherwise kill the polling loop.
                logger.exception("MessagingFuture done-callback raised")

    def to_cuda_future(
        self,
        device: Any | None = None,
    ) -> "CUDAMessagingFuture":
        # TODO: need extra type checking for the future type
        return CUDAMessagingFuture.FromMessagingFuture(self, device)  # type: ignore


class CUDAMessagingFuture(MessagingFuture[T]):
    """
    The future class that wraps both result and a CUDA IPC event.
    The `query`, `wait`, and `result` methods will pend on both the
    original future and the CUDA event.
    The original future should return tuple[bytes, T], where the first
    element is the serialized CUDA event.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
    ) -> None:
        super().__init__()
        self.raw_future_ = raw_future
        self.event_: Any | None = None
        self.result_: T | None = None
        self.device_ = device if device is not None else torch_dev.current_device()

    def _on_raw_future_complete(self):
        """
        Update the CUDA event and result when the raw future is complete.
        """
        event_bytes, result = self.raw_future_.result()
        self.result_ = result

        # Not all backends support interprocess Events (CUDA IPC specific)
        if not hasattr(torch_dev, "Event") or not hasattr(
            torch_dev.Event, "from_ipc_handle"
        ):
            raise RuntimeError(
                f"Backend '{torch_device_type}' does not support interprocess "
                "Events (Event.from_ipc_handle not available). "
                "Multiprocess IPC requires CUDA."
            )
        self.event_ = torch_dev.Event.from_ipc_handle(self.device_, event_bytes)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done, with the CUDA stream.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the CUDA event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            bool: True if the future is done, False if the timeout was reached.

        Raises:
            ValueError: if the timeout is not None.

        Notes:
            This function does not support waiting for a specific time.
        """
        if self.event_:
            self.event_.synchronize()
            return True

        flag = self.raw_future_.wait(timeout)
        if not flag:
            return False

        self._on_raw_future_complete()

        assert self.event_ is not None
        self.event_.synchronize()

        return True

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the CUDA event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise TimeoutError(
                "CUDAMessagingFuture result not available within timeout"
            )

        assert self.result_ is not None
        return self.result_

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        if self.event_:
            return self.event_.query()

        if self.raw_future_.query():
            self._on_raw_future_complete()
            assert self.event_ is not None
            return self.event_.query()

        return False

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "CUDAMessagingFuture does not support set_result directly"
        )

    @staticmethod
    def FromMessagingFuture(
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
    ) -> "CUDAMessagingFuture[T]":
        return CUDAMessagingFuture(raw_future, device)
