# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Callable
from concurrent.futures import CancelledError
from typing import Any, Generic, Optional, TypeVar, cast
import threading

# First Party
from lmcache import torch_dev
from lmcache.utils import init_logger, lmcache_deprecate
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend

T = TypeVar("T")
logger = init_logger(__name__)


class MessagingFuture(Generic[T]):
    def __init__(self) -> None:
        self.is_done_ = threading.Event()
        self.result_: T | None = None
        self._exception: BaseException | None = None
        self._completion_lock = threading.RLock()
        self._done_callbacks: list[Callable[["MessagingFuture[T]"], None]] = []
        self._retained_references: list[object] = []

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
            raise LMCacheTimeoutError("Future result not available within timeout")
        with self._completion_lock:
            if self._exception is not None:
                raise self._exception
            return cast(T, self.result_)

    def set_result(self, result: T) -> None:
        """
        Set the result of the future and mark it as done. This function is NOT
        SUPPOSED TO BE CALLED by users directly. It should be only called by
        the messaging system when the result is available.

        Args:
            result (T): The result to set.
        """
        callbacks: list[Callable[[MessagingFuture[T]], None]]
        with self._completion_lock:
            if self.is_done_.is_set():
                return
            self.result_ = result
            self.is_done_.set()
            callbacks = self._done_callbacks
            self._done_callbacks = []
        self._run_done_callbacks(callbacks)

    def set_exception(self, exception: BaseException) -> None:
        """Complete the future with ``exception``.

        Args:
            exception: Error raised by subsequent calls to :meth:`result`.
        """
        callbacks: list[Callable[[MessagingFuture[T]], None]]
        with self._completion_lock:
            if self.is_done_.is_set():
                return
            self._exception = exception
            self.is_done_.set()
            callbacks = self._done_callbacks
            self._done_callbacks = []
        self._run_done_callbacks(callbacks)

    def cancel(self) -> bool:
        """Cancel an unfinished future.

        Returns:
            True if this call cancelled the future, or False if it had already
            reached a terminal state.
        """
        callbacks: list[Callable[[MessagingFuture[T]], None]]
        with self._completion_lock:
            if self.is_done_.is_set():
                return False
            self._exception = CancelledError("Message queue client closed")
            self.is_done_.set()
            callbacks = self._done_callbacks
            self._done_callbacks = []
        self._run_done_callbacks(callbacks)
        return True

    def cancelled(self) -> bool:
        """Return whether this future completed through cancellation."""
        with self._completion_lock:
            return self.is_done_.is_set() and isinstance(
                self._exception, CancelledError
            )

    def exception(self, timeout: Optional[float] = None) -> BaseException | None:
        """Return the terminal exception, waiting up to ``timeout`` seconds.

        Args:
            timeout: Maximum time to wait, or None to wait indefinitely.

        Returns:
            The terminal exception, or None after successful completion.

        Raises:
            TimeoutError: If the future is not done within ``timeout``.
        """
        if not self.wait(timeout):
            raise LMCacheTimeoutError("Future result not available within timeout")
        with self._completion_lock:
            return self._exception

    def add_done_callback(
        self,
        callback: Callable[["MessagingFuture[T]"], None],
    ) -> None:
        """Invoke ``callback`` once this future reaches a terminal state.

        The callback runs synchronously in the thread that completes the
        future. If the future is already done, it runs before this method
        returns.

        Args:
            callback: Callable receiving this future.
        """
        with self._completion_lock:
            if not self.is_done_.is_set():
                self._done_callbacks.append(callback)
                return
        self._run_done_callbacks([callback])

    def retain_reference(self, value: object) -> None:
        """Keep a resource alive for at least the lifetime of this future.

        Async callers can use this for resources, such as exported IPC events,
        whose validity must extend until a remote operation completes.

        Args:
            value: Resource whose lifetime must be tied to this future.
        """
        self._retained_references.append(value)

    def _run_done_callbacks(
        self,
        callbacks: list[Callable[["MessagingFuture[T]"], None]],
    ) -> None:
        """Run terminal callbacks without holding the completion lock."""
        for callback in callbacks:
            try:
                callback(self)
            except Exception:
                logger.exception("MessagingFuture done callback failed")

    def to_device_future(
        self,
        device: Any | None = None,
    ) -> "DeviceMessagingFuture":
        """Wrap this future in a device-aware future.

        Args:
            device: The device whose event backend orders completion. Defaults
                to the active device.

        Returns:
            A DeviceMessagingFuture pending on both this future and the event.
        """
        # TODO: need extra type checking for the future type
        return DeviceMessagingFuture.FromMessagingFuture(self, device)  # type: ignore

    @lmcache_deprecate("Use to_device_future() instead")
    def to_cuda_future(
        self,
        device: Any | None = None,
    ) -> "DeviceMessagingFuture[T]":
        """Return a device-aware future using the deprecated CUDA name.

        Args:
            device: Device on which the completion event will be imported.

        Returns:
            A device-aware future wrapping this messaging future.
        """
        return self.to_device_future(device)


class DeviceMessagingFuture(MessagingFuture[T]):
    """
    The future class that wraps both a result and a device IPC event.
    The `query`, `wait`, and `result` methods pend on both the original
    future and the device event, ordered through the platform event backend.
    The original future should return tuple[bytes, T], where the first
    element is the serialized device event handle. An empty handle means the
    remote side submitted no device work, so completion of the original future
    is also terminal completion of this future.
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
        self._raw_response_processed = False
        self.device_ = device if device is not None else torch_dev.current_device()
        self._event_backend = get_event_ipc_backend(self.device_)
        self._event_backend.check_event_support(self.device_)

    def _on_raw_future_complete(self) -> None:
        """
        Update the device event and result when the raw future is complete.
        """
        with self._completion_lock:
            if self._raw_response_processed or self.is_done_.is_set():
                return
            event_bytes, result = self.raw_future_.result()
            self.result_ = result
            self.event_ = (
                self._event_backend.import_event(event_bytes, self.device_)
                if event_bytes
                else None
            )
            self._raw_response_processed = True

        if self.event_ is None:
            MessagingFuture.set_result(self, result)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the future to be done, ordered through the device event.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the device event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            bool: True if the future is done, False if the timeout was reached.

        Notes:
            This function does not support waiting for a specific time.
        """
        if self.is_done_.is_set():
            return True

        if not self._raw_response_processed:
            flag = self.raw_future_.wait(timeout)
            if not flag:
                return False

            try:
                self._on_raw_future_complete()
            except BaseException as error:
                MessagingFuture.set_exception(self, error)
                return True

        if self.is_done_.is_set():
            return True

        if self.event_ is None:
            MessagingFuture.set_result(self, cast(T, self.result_))
            return True

        try:
            self._event_backend.synchronize_event(self.event_, self.device_)
        except BaseException as error:
            MessagingFuture.set_exception(self, error)
            return True
        MessagingFuture.set_result(self, cast(T, self.result_))
        return True

    def result(self, timeout: Optional[float] = None) -> T:
        """
        Get the result of the future.

        Args:
            timeout (Optional[float]): Maximum time to wait for the UNDERLYING
                RAW FUTURE in seconds. The exact timeout is not guaranteed
                when waiting on the device event. (NOTE: this could be improved
                with careful threading management)

        Returns:
            T: The result of the future.

        Raises:
            TimeoutError: If the future is not done within the timeout.
        """
        flag = self.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError(
                "DeviceMessagingFuture result not available within timeout"
            )

        with self._completion_lock:
            if self._exception is not None:
                raise self._exception
            return cast(T, self.result_)

    def query(self) -> bool:
        """
        Check if the future is done.

        Returns:
            bool: True if the future is done, False otherwise.
        """
        if self.is_done_.is_set():
            return True

        if not self._raw_response_processed and self.raw_future_.query():
            try:
                self._on_raw_future_complete()
            except BaseException as error:
                MessagingFuture.set_exception(self, error)
                return True

        if self.is_done_.is_set():
            return True

        if self._raw_response_processed and self.event_ is None:
            MessagingFuture.set_result(self, cast(T, self.result_))
            return True

        if self.event_ is not None and self._event_backend.query_event(self.event_):
            MessagingFuture.set_result(self, cast(T, self.result_))
            return True

        return False

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "DeviceMessagingFuture does not support set_result directly"
        )

    def cancel(self) -> bool:
        """Cancel both the raw MQ request and this device-aware future."""
        self.raw_future_.cancel()
        return MessagingFuture.cancel(self)

    @staticmethod
    def FromMessagingFuture(
        raw_future: MessagingFuture[tuple[bytes, T]],
        device: Any | None = None,
    ) -> "DeviceMessagingFuture[T]":
        return DeviceMessagingFuture(raw_future, device)


# Backward-compatible alias for existing imports.
CUDAMessagingFuture = DeviceMessagingFuture
