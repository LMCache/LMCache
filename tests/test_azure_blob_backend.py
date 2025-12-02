# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Azure Blob Storage backend."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
import numpy as np

from lmcache.utils import CacheEngineKey


class TestAzureBlobBackend:
    """Test suite for AzureBlobBackend."""
    
    @pytest.fixture
    def mock_blob_service(self):
        """Fixture for mocking BlobServiceClient."""
        with patch(
            'lmcache.storage_backend.azure_blob_backend.BlobServiceClient'
        ) as mock:
            yield mock
    
    @pytest.fixture
    def backend_config(self):
        """Fixture for backend configuration."""
        return {
            "account_url": "https://test.blob.core.windows.net",
            "container_name": "test-container",
            "credential_mode": "account_key",
            "account_key": "test_key",
        }
    
    @pytest.fixture
    def sample_key(self):
        """Fixture for sample CacheEngineKey."""
        return CacheEngineKey(
            model_name="test-model",
            layer_id=0,
            seq_hash="test_hash_123"
        )
    
    @pytest.fixture
    def sample_kv_chunk(self):
        """Fixture for sample KV cache tensor."""
        return torch.randn(128, 64, dtype=torch.float16)
    
    def test_backend_initialization(self, mock_blob_service, backend_config):
        """Test that backend initializes correctly."""
        from lmcache.storage_backend.azure_blob_backend import AzureBlobBackend
        
        backend = AzureBlobBackend(**backend_config)
        assert backend.account_url == "https://test.blob.core.windows.net"
        assert backend.container_name == "test-container"
        assert backend.credential_mode == "account_key"
    
    def test_blob_name_generation(self, mock_blob_service, backend_config, sample_key):
        """Test blob name generation from CacheEngineKey."""
        from lmcache.storage_backend.azure_blob_backend import AzureBlobBackend
        
        backend = AzureBlobBackend(**backend_config)
        blob_name = backend._get_blob_name(sample_key)
        
        assert "lmcache-kv/" in blob_name
        assert "test-model" in blob_name
        assert "layer_0" in blob_name
        assert "test_hash_123" in blob_name
        assert blob_name.endswith(".bin")
    
    def test_put_chunk(self, mock_blob_service, backend_config, sample_key, sample_kv_chunk):
        """Test putting a KV cache chunk."""
        from lmcache.storage_backend.azure_blob_backend import AzureBlobBackend
        
        backend = AzureBlobBackend(**backend_config)
        backend.container_client = MagicMock()
        
        # Mock blob client
        mock_blob_client = MagicMock()
        backend.container_client.get_blob_client.return_value = mock_blob_client
        
        # Call put
        backend.put(sample_key, sample_kv_chunk)
        
        # Verify upload was called
        backend.container_client.get_blob_client.assert_called()
        mock_blob_client.upload_blob.assert_called_once()
        mock_blob_client.set_blob_tags.assert_called_once()
    
    def test_get_chunk(self, mock_blob_service, backend_config, sample_key):
        """Test getting a KV cache chunk."""
        from lmcache.storage_backend.azure_blob_backend import AzureBlobBackend
        
        backend = AzureBlobBackend(**backend_config)
        backend.container_client = MagicMock()
        
        # Create mock data
        test_data = torch.randn(128, 64, dtype=torch.float16).cpu().numpy().tobytes()
        
        # Mock blob client and stream
        mock_blob_client = MagicMock()
        mock_stream = MagicMock()
        mock_stream.readall.return_value = test_data
        mock_blob_client.download_blob.return_value = mock_stream
        backend.container_client.get_blob_client.return_value = mock_blob_client
        
        # Call get
        result = backend.get(sample_key)
        
        # Verify download was called
        mock_blob_client.download_blob.assert_called_once()
        assert result is not None
    
    def test_contains_chunk(self, mock_blob_service, backend_config, sample_key):
        """Test checking if chunk exists."""
        from lmcache.storage_backend.azure_blob_backend import AzureBlobBackend
        
        backend = AzureBlobBackend(**backend_config)
        backend.container_client = MagicMock()
        
        # Mock blob client
        mock_blob_client = MagicMock()
        backend.container_client.get_blob_client.return_value = mock_blob_client
        mock_blob_client.get_blob_properties.return_value = {}
        
        # Call contains
        result = backend.contains(sample_key)
        
        assert result is True
        mock_blob_client.get_blob_properties.assert_called_once()
    
    def test_close_backend(self, mock_blob_service, backend_config):
        """Test closing backend."""
        from lmcache.storage_backend.azure_blob_backend import AzureBlobBackend
        
        backend = AzureBlobBackend(**backend_config)
        backend.blob_service_client = MagicMock()
        
        # Call close
        backend.close()
        
        # Verify close was called
        backend.blob_service_client.close.assert_called_once()
