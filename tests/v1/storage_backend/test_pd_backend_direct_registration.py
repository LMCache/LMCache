# SPDX-License-Identifier: Apache-2.0
"""
Tests for PDBackend.register_external_kv_caches direct KV registration.

Verifies that:
1. The method is a no-op when pd_direct_registration is False
2. Memory regions are correctly built from KV cache tensors
3. NixlAgentWrapper.from_memory_regions creates correct descriptors
"""

# Standard
from unittest.mock import MagicMock, patch
import threading

# Third Party
import torch

# First Party
from lmcache.v1.storage_backend.pd_backend import PDBackend


class TestRegisterExternalKVCaches:
    """Tests for the register_external_kv_caches method."""

    def _make_backend_stub(self, direct_registration: bool = True):
        """Create a minimal PDBackend-like object for testing."""
        backend = MagicMock(spec=PDBackend)
        backend.use_direct_registration = direct_registration
        backend.external_kv_caches = None
        backend.tp_rank = 0
        backend._nixl_backends = ["UCX"]
        backend.data = {}
        backend.data_lock = threading.Lock()

        # Mock memory allocator
        backend.memory_allocator = MagicMock()
        backend.memory_allocator.gpu_allocator = MagicMock()
        backend.memory_allocator.gpu_allocator.align_bytes = 4096

        # Mock transfer channel
        backend.transfer_channel = MagicMock()
        backend.transfer_channel.nixl_wrapper = MagicMock()
        backend.transfer_channel.nixl_wrapper.agent = MagicMock()
        backend.transfer_channel.nixl_wrapper.reg_descs = MagicMock()
        backend.transfer_channel.nixl_wrapper.xfer_handler = MagicMock()

        return backend

    def test_noop_when_disabled(self):
        """register_external_kv_caches is a no-op when direct_registration=False."""
        backend = self._make_backend_stub(direct_registration=False)

        kv_caches = {
            "layer_0": torch.zeros(100, dtype=torch.bfloat16),
        }

        # Call the real method on the mock
        PDBackend.register_external_kv_caches(backend, kv_caches)

        # Should not set external_kv_caches
        assert backend.external_kv_caches is None

    def test_stores_kv_caches(self):
        """register_external_kv_caches stores the KV cache dict."""
        backend = self._make_backend_stub(direct_registration=True)

        kv_caches = {
            "layer_0": torch.zeros(100, dtype=torch.bfloat16),
            "layer_1": torch.zeros(100, dtype=torch.bfloat16),
        }

        with patch(
            "lmcache.v1.transfer_channel.nixl_channel.NixlAgentWrapper"
        ) as MockWrapper:
            mock_instance = MagicMock()
            mock_instance.agent = MagicMock()
            MockWrapper.from_memory_regions.return_value = mock_instance

            PDBackend.register_external_kv_caches(backend, kv_caches)

        assert backend.external_kv_caches is kv_caches

    def test_builds_memory_regions(self):
        """Memory regions are correctly built from KV cache tensors."""
        backend = self._make_backend_stub(direct_registration=True)

        t1 = torch.zeros(100, dtype=torch.bfloat16)
        t2 = torch.zeros(200, dtype=torch.bfloat16)
        kv_caches = {"layer_0": t1, "layer_1": t2}

        with patch(
            "lmcache.v1.transfer_channel.nixl_channel.NixlAgentWrapper"
        ) as MockWrapper:
            mock_instance = MagicMock()
            mock_instance.agent = MagicMock()
            MockWrapper.from_memory_regions.return_value = mock_instance

            PDBackend.register_external_kv_caches(backend, kv_caches)

            # Verify from_memory_regions was called
            MockWrapper.from_memory_regions.assert_called_once()
            call_kwargs = MockWrapper.from_memory_regions.call_args

            regions = call_kwargs.kwargs.get(
                "memory_regions", call_kwargs[1].get("memory_regions")
            )
            if regions is None:
                regions = call_kwargs[0][0]

            assert len(regions) == 2
            # Each region should be (base_addr, size, dev_id, meta)
            for region in regions:
                assert len(region) == 4
                base_addr, size, dev_id, meta = region
                assert isinstance(base_addr, int)
                assert size > 0
                assert dev_id == 0  # tp_rank
                assert meta in ("layer_0", "layer_1")

    def test_custom_page_size(self):
        """Custom page_size is passed to from_memory_regions."""
        backend = self._make_backend_stub(direct_registration=True)

        kv_caches = {"layer_0": torch.zeros(100, dtype=torch.bfloat16)}

        with patch(
            "lmcache.v1.transfer_channel.nixl_channel.NixlAgentWrapper"
        ) as MockWrapper:
            mock_instance = MagicMock()
            mock_instance.agent = MagicMock()
            MockWrapper.from_memory_regions.return_value = mock_instance

            PDBackend.register_external_kv_caches(backend, kv_caches, page_size=8192)

            call_kwargs = MockWrapper.from_memory_regions.call_args
            # Check page_size was passed
            if call_kwargs.kwargs:
                assert call_kwargs.kwargs.get("page_size") == 8192
            else:
                assert call_kwargs[1].get("page_size") == 8192


class TestNixlAgentWrapperFromMemoryRegions:
    """Tests for NixlAgentWrapper.from_memory_regions classmethod."""

    def test_creates_wrapper_with_multiple_regions(self):
        """from_memory_regions creates a wrapper with multiple memory regions."""
        # First Party
        from lmcache.v1.transfer_channel.nixl_channel import NixlAgentWrapper

        # We can't test with real NIXL (needs GPU), but we can verify
        # the method signature and logic by mocking nixl
        with patch("lmcache.v1.transfer_channel.nixl_channel.uuid") as mock_uuid:
            mock_uuid.uuid4.return_value = "test-agent-id"

            with patch.dict(
                "sys.modules",
                {
                    "nixl": MagicMock(),
                    "nixl._api": MagicMock(),
                },
            ):
                # Mock the nixl imports
                mock_nixl_api = MagicMock()
                mock_agent = MagicMock()
                mock_nixl_api.nixl_agent.return_value = mock_agent
                mock_agent.get_reg_descs.return_value = "reg_descs"
                mock_agent.get_xfer_descs.return_value = "xfer_descs"
                mock_agent.prep_xfer_dlist.return_value = "xfer_handler"

                with patch.dict("sys.modules", {"nixl._api": mock_nixl_api}):
                    regions = [
                        (0x1000, 4096, 0, "layer_0"),
                        (0x2000, 4096, 0, "layer_1"),
                    ]

                    wrapper = NixlAgentWrapper.from_memory_regions(
                        memory_regions=regions,
                        page_size=1024,
                        tp_rank=0,
                        backends=["UCX"],
                    )

                    assert wrapper.agent is mock_agent
                    assert wrapper.reg_descs == "reg_descs"
                    assert wrapper.xfer_descs == "xfer_descs"
                    assert wrapper.xfer_handler == "xfer_handler"

                    # Verify register_memory was called
                    mock_agent.register_memory.assert_called_once()

                    # Verify xfer descriptors were created at page_size granularity
                    # 2 regions * 4096/1024 = 8 descriptors
                    xfer_call = mock_agent.get_xfer_descs.call_args
                    xfer_desc_list = xfer_call[0][0]
                    assert len(xfer_desc_list) == 8


class TestConfigField:
    """Test that pd_direct_registration config field exists."""

    def test_config_field_default(self):
        """pd_direct_registration defaults to False."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults()
        assert config.pd_direct_registration is False

    def test_config_field_set(self):
        """pd_direct_registration can be set to True."""
        # First Party
        from lmcache.v1.config import LMCacheEngineConfig

        config = LMCacheEngineConfig.from_defaults(pd_direct_registration=True)
        assert config.pd_direct_registration is True
