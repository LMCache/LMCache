# SPDX-License-Identifier: Apache-2.0
"""Tests for the lmcache.v1.platform cross-platform abstraction layer."""

# Standard
import select


class TestCapabilities:
    """Test platform capability detection flags."""

    def test_has_cuda_is_bool(self):
        # First Party
        from lmcache.v1.platform import HAS_CUDA

        assert isinstance(HAS_CUDA, bool)

    def test_has_eventfd_is_bool(self):
        # First Party
        from lmcache.v1.platform import HAS_EVENTFD

        assert isinstance(HAS_EVENTFD, bool)


class TestOps:
    """Test centralized lmc_ops import."""

    def test_lmc_ops_has_page_buffer_shape_desc(self):
        # First Party
        from lmcache.v1.platform import lmc_ops

        assert hasattr(lmc_ops, "PageBufferShapeDesc")

    def test_lmc_ops_has_alloc_pinned_ptr(self):
        # First Party
        from lmcache.v1.platform import lmc_ops

        assert hasattr(lmc_ops, "alloc_pinned_ptr")

    def test_lmc_ops_has_free_pinned_ptr(self):
        # First Party
        from lmcache.v1.platform import lmc_ops

        assert hasattr(lmc_ops, "free_pinned_ptr")

    def test_lmc_ops_has_transfer_direction(self):
        # First Party
        from lmcache.v1.platform import lmc_ops

        assert hasattr(lmc_ops, "TransferDirection")


class TestMemoryPinner:
    """Test the MemoryPinner abstraction."""

    def test_create_returns_memory_pinner(self):
        # First Party
        from lmcache.v1.platform import MemoryPinner
        from lmcache.v1.platform.memory_pinner import (
            create_memory_pinner,
        )

        pinner = create_memory_pinner()
        assert isinstance(pinner, MemoryPinner)
        pinner.close()

    def test_noop_pinner_is_safe(self):
        """NoOpMemoryPinner operations do not raise."""
        # First Party
        from lmcache.v1.platform.memory_pinner import (
            NoOpMemoryPinner,
        )

        pinner = NoOpMemoryPinner()
        pinner.pin(0, 1024)
        pinner.unpin(0)
        pinner.close()
        # Idempotent close
        pinner.close()

    def test_create_pinner_matches_platform(self):
        """Factory returns correct type based on CUDA availability."""
        # First Party
        from lmcache.v1.platform import HAS_CUDA
        from lmcache.v1.platform.memory_pinner import (
            CudaMemoryPinner,
            NoOpMemoryPinner,
            create_memory_pinner,
        )

        pinner = create_memory_pinner()
        if HAS_CUDA:
            assert isinstance(pinner, CudaMemoryPinner)
        else:
            assert isinstance(pinner, NoOpMemoryPinner)
        pinner.close()


class TestEventNotifierFromPlatform:
    """Test EventNotifier imported via the platform package."""

    def test_create_from_platform(self):
        # First Party
        from lmcache.v1.platform import (
            EventNotifier,
            create_event_notifier,
        )

        n = create_event_notifier()
        try:
            assert isinstance(n, EventNotifier)
            assert n.fileno() >= 0
        finally:
            n.close()

    def test_notify_consume_from_platform(self):
        # First Party
        from lmcache.v1.platform import create_event_notifier

        with create_event_notifier() as n:
            n.notify()
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(100)
            assert len(events) > 0
            n.consume()

    def test_consume_fd_from_platform(self):
        # First Party
        from lmcache.v1.platform import (
            consume_fd,
            create_event_notifier,
        )

        with create_event_notifier() as n:
            n.notify()
            consume_fd(n.fileno())
            poller = select.poll()
            poller.register(n.fileno(), select.POLLIN)
            events = poller.poll(50)
            assert len(events) == 0


class TestBackwardCompatImport:
    """Ensure old import paths still work."""

    def test_old_event_notifier_import(self):
        """lmcache.v1.event_notifier still re-exports."""
        # First Party
        from lmcache.v1.event_notifier import (
            EventNotifier,
            create_event_notifier,
        )

        n = create_event_notifier()
        assert isinstance(n, EventNotifier)
        n.close()
