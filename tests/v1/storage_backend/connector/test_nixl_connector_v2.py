# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
from unittest.mock import MagicMock, patch
import types

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.memory_management import MemoryFormat, MemoryObj


class MockNixlAgent:
    """Mock NIXL agent for testing."""

    def __init__(self):
        self.registered_memory = []
        self.remote_agents = {}
        self.dlist_handles = {}
        self.xfer_handles = {}
        self.notifications = []

    def register_memory(
        self,
        reg_list,
        mem_type: Optional[str] = None,
        is_sorted: bool = False,
        backends: Optional[list[str]] = None,
    ):
        """Register memory with the NIXL agent."""
        self.registered_memory.append(reg_list)
        mock_desc = MagicMock()
        mock_desc.trim.return_value = MagicMock()
        return mock_desc

    def deregister_memory(self, dereg_list, backends: Optional[list[str]] = None):
        """Deregister memory from the NIXL agent."""
        if dereg_list in self.registered_memory:
            self.registered_memory.remove(dereg_list)

    def get_serialized_descs(self, descs) -> bytes:
        """Get serialized descriptors."""
        return b"mock_serialized_descs"

    def get_agent_metadata(self):
        """Get agent metadata."""
        return b"mock_agent_metadata"

    def add_remote_agent(self, metadata: bytes) -> bytes:
        """Add a remote agent."""
        peer_name = "mock_peer_name"
        self.remote_agents[peer_name] = metadata
        return peer_name.encode("utf-8")

    def remove_remote_agent(self, peer_name):
        """Remove a remote agent."""
        if peer_name in self.remote_agents:
            del self.remote_agents[peer_name]


class MockSideChannel:
    """Mock side channel for testing."""

    def __init__(self):
        self.closed = False

    def close(self):
        """Close the side channel."""
        self.closed = True

    def send(self, data: bytes):
        """Send data to the specific sender."""
        pass


@pytest.fixture
def mock_nixl_config():
    """Create a mock NixlConfig for testing."""
    nixl_pkg = types.ModuleType("nixl")
    nixl_pkg.__path__ = []
    nixl_api_mod = types.ModuleType("nixl._api")
    nixl_api_mod.nixl_agent = MagicMock(return_value=MockNixlAgent())

    with patch.dict(
        "sys.modules",
        {
            "nixl": nixl_pkg,
            "nixl._api": nixl_api_mod,
        },
    ):
        # First Party
        from lmcache.v1.storage_backend.connector.nixl_connector_v2 import NixlConfig

        config = NixlConfig(
            role="sender",
            receiver_host="127.0.0.1",
            receiver_port=12346,
            buffer_size=1024 * 1024,
            buffer_device="cuda:0",
            enable_gc=True,
        )
        return config


@pytest.fixture
def mock_nixl_pipe(mock_nixl_config):
    """Create a mock NixlPipe for testing."""
    nixl_pkg = types.ModuleType("nixl")
    nixl_pkg.__path__ = []
    nixl_api_mod = types.ModuleType("nixl._api")
    nixl_api_mod.nixl_agent = MagicMock(return_value=MockNixlAgent())

    with patch.dict(
        "sys.modules",
        {
            "nixl": nixl_pkg,
            "nixl._api": nixl_api_mod,
        },
    ):
        # First Party
        from lmcache.v1.storage_backend.connector.nixl_connector_v2 import NixlPipe

        pipe = NixlPipe(
            nixl_config=mock_nixl_config,
            side_channel=MockSideChannel(),
            sender_meta=b"mock_sender_meta",
        )
        return pipe


class TestNixlPipeBatchedAllocateForWrite:
    """Test cases for NixlPipe.batched_allocate_for_write method."""

    def test_batched_allocate_for_write_success(self, mock_nixl_pipe):
        """Test successful batch allocation for write."""
        # Arrange
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        batch_size = 4
        fmt = MemoryFormat.KV_2LTD

        result = mock_nixl_pipe.batched_allocate_for_write(
            shape=shape, dtype=dtype, batch_size=batch_size, fmt=fmt
        )

        assert result is not None
        assert len(result) == batch_size
        for obj in result:
            assert isinstance(obj, MemoryObj)
            assert obj.metadata.shape == shape
            assert obj.metadata.dtype == dtype
            assert obj.metadata.fmt == fmt

    def test_batched_allocate_for_write_allocator_arise_assertion_error(
        self, mock_nixl_pipe
    ):
        """Test when allocator raises assertion error (buffer full scenario)."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        batch_size = 1000
        fmt = MemoryFormat.KV_2LTD

        with pytest.raises(AssertionError):
            mock_nixl_pipe.batched_allocate_for_write(
                shape=shape, dtype=dtype, batch_size=batch_size, fmt=fmt
            )

    def test_batched_allocate_for_write_different_formats(self, mock_nixl_pipe):
        """Test batch allocation with different memory formats."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        batch_size = 2

        formats = [
            MemoryFormat.KV_2LTD,
            MemoryFormat.KV_2TD,
            MemoryFormat.KV_T2D,
            MemoryFormat.KV_MLA_FMT,
        ]

        for fmt in formats:
            result = mock_nixl_pipe.batched_allocate_for_write(
                shape=shape, dtype=dtype, batch_size=batch_size, fmt=fmt
            )

            assert result is not None
            assert len(result) == batch_size
            for obj in result:
                assert isinstance(obj, MemoryObj)
                assert obj.metadata.shape == shape
                assert obj.metadata.dtype == dtype
                assert obj.metadata.fmt == fmt

    def test_batched_allocate_for_write_different_batch_sizes(self, mock_nixl_pipe):
        """Test batch allocation with different batch sizes."""
        shape = torch.Size([2, 16, 8, 128])
        dtype = torch.bfloat16
        fmt = MemoryFormat.KV_2LTD

        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            result = mock_nixl_pipe.batched_allocate_for_write(
                shape=shape, dtype=dtype, batch_size=batch_size, fmt=fmt
            )

            assert result is not None
            assert len(result) == batch_size
            for obj in result:
                assert isinstance(obj, MemoryObj)


if __name__ == "__main__":
    pytest.main([__file__])
