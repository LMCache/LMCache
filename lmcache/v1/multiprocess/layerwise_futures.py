# SPDX-License-Identifier: Apache-2.0
"""Future type for the layer-wise KV retrieve path.

Kept out of :mod:`lmcache.v1.multiprocess.futures` so the default transfer
path never imports layer-wise machinery. All partial-result handling lives
here, inside the future itself, rather than in the message queue.
"""

# Standard
from typing import Any, Optional, TypeVar
import queue
import struct

# First Party
from lmcache import torch_dev
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.platform.base.event_ipc import get_event_ipc_backend
from lmcache.v1.platform.base.event_pool import EventPool

T = TypeVar("T")


class LayerwiseDeviceMessagingFuture(MessagingFuture[T]):
    """Future that carries per-layer IPC events for layerwise KV loading.

    The server answers a layer-wise retrieve with one message per layer
    batch.  This future installs itself as the raw future's delivery sink,
    so buffering the intermediate messages and deciding when the request is
    complete happens entirely here -- the message queue only sees "the
    result says it is not done yet".  :meth:`wait_for_layer` imports and
    waits on each handle as soon as it arrives, overlapping H2D transfer of
    later batches with GPU attention of earlier layers.
    """

    def __init__(
        self,
        raw_future: MessagingFuture[tuple[bytes, bool, T]],
        device: Any | None = None,
        event_pool: EventPool | None = None,
    ) -> None:
        super().__init__()
        self.raw_future_ = raw_future
        self.result_: T | None = None
        self.device_ = device if device is not None else torch_dev.current_device()
        self._event_backend = get_event_ipc_backend(self.device_)
        self._event_backend.check_event_support(self.device_)
        self._last_waited_event: object | None = None
        self._event_pool = event_pool
        self._layer_event_map: dict[int, Any] = {}
        self._partial_queue: "queue.Queue[bytes | None]" = queue.Queue()
        # Must happen before the request is submitted; see
        # LMCacheLayerwiseTransferContext.submit_retrieve.
        raw_future.set_delivery_sink(self._deliver_frame)

    # ------------------------------------------------------------------
    # Incremental delivery
    # ------------------------------------------------------------------

    def _deliver_frame(self, response: "tuple[bytes, bool, T]") -> bool:
        """Consume one response frame; return True on the closing frame.

        Installed on the raw future via
        :meth:`MessagingFuture.set_delivery_sink`, so the notion of an
        incomplete answer is confined to this class.
        """
        payload, is_final, _ = response
        if not is_final:
            self._partial_queue.put(payload)
            return False
        # Wake any thread blocked in _drain_until_layer() so it observes
        # the outcome instead of waiting out the timeout.
        self._partial_queue.put(None)
        self.raw_future_.set_result(response)
        return True

    def _import_partial(self, b_data: bytes) -> None:
        """Import the event described by one intermediate frame."""
        assert self._event_pool is not None
        first_layer, count, pool_idx = struct.unpack("<3i", b_data)
        evt = self._event_pool.event_at(pool_idx)
        for i in range(first_layer, first_layer + count):
            self._layer_event_map[i] = evt

    def _drain_until_layer(self, target_layer_idx: int) -> None:
        """Block-drain the partial queue until *target_layer_idx* is available."""
        while target_layer_idx not in self._layer_event_map:
            try:
                b_data = self._partial_queue.get(timeout=60)
            except queue.Empty:
                raise LMCacheTimeoutError(
                    f"Timed out waiting for the event of layer {target_layer_idx}"
                ) from None
            if b_data is None:
                # Sentinel from _deliver_frame(): the closing frame arrived
                # and no further frames are coming.  Break out so the caller
                # can discover the outcome via the raw future.
                break
            self._import_partial(b_data)

    def _drain_remaining(self) -> None:
        """Non-blocking drain of any queued intermediate frames."""
        while True:
            try:
                b_data = self._partial_queue.get_nowait()
            except queue.Empty:
                break
            if b_data is None:
                break
            self._import_partial(b_data)

    def _resolve_final(self) -> None:
        """Extract the success flag from the final ZMQ response."""
        if self.result_ is not None:
            return
        _, _, result = self.raw_future_.result()
        self.result_ = result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait_for_layer(self, layer_idx: int) -> None:
        """Make the current stream wait for a specific layer's transfer."""
        evt = self._layer_event_map.get(layer_idx)
        if evt is None:
            self._drain_until_layer(layer_idx)
            evt = self._layer_event_map.get(layer_idx)
        if evt is not None and evt is not self._last_waited_event:
            current_stream = torch_dev.current_stream(self.device_)
            self._event_backend.wait_event(evt, current_stream)
            self._last_waited_event = evt

    def wait(self, timeout: Optional[float] = None) -> bool:
        flag = self.raw_future_.wait(timeout)
        if not flag:
            return False
        self._resolve_final()
        self._drain_remaining()
        if self._layer_event_map:
            last_layer = max(self._layer_event_map.keys())
            self._event_backend.synchronize_event(
                self._layer_event_map[last_layer], self.device_
            )
        return True

    def result(self, timeout: Optional[float] = None) -> T:
        flag = self.wait(timeout)
        if not flag:
            raise LMCacheTimeoutError(
                "LayerwiseDeviceMessagingFuture result not available within timeout"
            )
        assert self.result_ is not None
        return self.result_

    def query(self) -> bool:
        if not self.raw_future_.query():
            return False
        self._resolve_final()
        self._drain_remaining()
        if self._layer_event_map:
            last_layer = max(self._layer_event_map.keys())
            return self._event_backend.query_event(self._layer_event_map[last_layer])
        return True

    def set_result(self, result: T) -> None:
        raise NotImplementedError(
            "LayerwiseDeviceMessagingFuture does not support set_result"
        )

    @property
    def num_layers(self) -> int:
        return len(self._layer_event_map)
