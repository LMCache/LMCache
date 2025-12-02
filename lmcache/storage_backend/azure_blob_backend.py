# SPDX-License-Identifier: Apache-2.0
"""Azure Blob Storage backend for LMCache KV cache storage.

This module provides an implementation of the LMCBackendInterface
for storing and retrieving KV cache chunks using Azure Blob Storage.

Supported credential modes:
- account_key: Storage account key authentication
- connection_string: Full connection string
- managed_identity: Azure Managed Identity (default for AKS)
- sas_token: Shared Access Signature token
"""

# Standard
from typing import Optional
import logging
import time

# Third Party
import torch
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError, AzureError

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from .abstract_backend import LMCBackendInterface

logger = init_logger(__name__)


class AzureBlobBackend(LMCBackendInterface):
    """Azure Blob Storage backend for LMCache KV cache storage.
    
    This backend stores KV cache chunks in Azure Blob Storage, enabling
    cost-effective, scalable storage for long-context LLM serving.
    
    Configuration parameters:
        account_url (str): Azure Storage account URL
            e.g., https://<account>.blob.core.windows.net
        container_name (str): Container name for KV cache storage
            default: "lmcache-kv-cache"
        credential_mode (str): Authentication mode
            options: "account_key", "connection_string", "managed_identity", "sas_token"
            default: "managed_identity"
        account_key (str, optional): Storage account key for account_key mode
        connection_string (str, optional): Full connection string for connection_string mode
        sas_token (str, optional): SAS token for sas_token mode
        max_concurrency (int): Max concurrent uploads/downloads
            default: 4
        chunk_upload_size (int): Upload chunk size in bytes
            default: 4194304 (4MB)
        blob_prefix (str): Prefix for blob names
            default: "lmcache-kv/"
        ttl_hours (int): Time-to-live for cached blobs in hours
            default: 24
        enable_compression (bool): Enable gzip compression
            default: False
    """
    
    def __init__(
        self,
        account_url: str,
        container_name: str = "lmcache-kv-cache",
        credential_mode: str = "managed_identity",
        account_key: Optional[str] = None,
        connection_string: Optional[str] = None,
        sas_token: Optional[str] = None,
        max_concurrency: int = 4,
        chunk_upload_size: int = 4194304,
        blob_prefix: str = "lmcache-kv/",
        ttl_hours: int = 24,
        enable_compression: bool = False,
        dst_device: str = "cuda",
    ):
        """Initialize Azure Blob Storage backend."""
        super().__init__(dst_device=dst_device)
        
        self.account_url = account_url
        self.container_name = container_name
        self.credential_mode = credential_mode
        self.max_concurrency = max_concurrency
        self.chunk_upload_size = chunk_upload_size
        self.blob_prefix = blob_prefix
        self.ttl_hours = ttl_hours
        self.enable_compression = enable_compression
        
        # Initialize Blob Service Client
        try:
            self.blob_service_client = self._create_blob_service_client(
                account_url=account_url,
                credential_mode=credential_mode,
                account_key=account_key,
                connection_string=connection_string,
                sas_token=sas_token,
            )
            self.container_client = self.blob_service_client.get_container_client(
                container_name
            )
            logger.info(
                f"AzureBlobBackend initialized: {account_url}/{container_name}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure Blob Backend: {e}")
            raise
    
    def _create_blob_service_client(self, **kwargs):
        """Create Azure Blob Service client based on credential mode."""
        account_url = kwargs.get("account_url")
        credential_mode = kwargs.get("credential_mode")
        
        if credential_mode == "connection_string":
            return BlobServiceClient.from_connection_string(
                kwargs.get("connection_string")
            )
        elif credential_mode == "account_key":
            return BlobServiceClient(
                account_url=account_url,
                credential=kwargs.get("account_key"),
            )
        elif credential_mode == "sas_token":
            return BlobServiceClient(
                account_url=account_url,
                credential=kwargs.get("sas_token"),
            )
        elif credential_mode == "managed_identity":
            try:
                from azure.identity import DefaultAzureCredential
                credential = DefaultAzureCredential()
                return BlobServiceClient(
                    account_url=account_url,
                    credential=credential,
                )
            except ImportError:
                logger.error(
                    "azure-identity package required for managed_identity mode. "
                    "Install it with: pip install azure-identity"
                )
                raise
        else:
            raise ValueError(f"Unknown credential_mode: {credential_mode}")
    
    def _get_blob_name(self, key: CacheEngineKey) -> str:
        """Generate blob name from CacheEngineKey."""
        # Format: blob_prefix/model_id/layer_id/seq_hash
        return (
            f"{self.blob_prefix}"
            f"{key.model_name}/"
            f"layer_{key.layer_id}/"
            f"{key.seq_hash}.bin"
        )
    
    def put(
        self,
        key: CacheEngineKey,
        kv_chunk: torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """Store KV cache chunk in Azure Blob Storage.
        
        Args:
            key: Cache engine key identifying the chunk
            kv_chunk: KV cache tensor to store
            blocking: Whether to block until upload completes
        """
        try:
            # Convert tensor to bytes
            data = kv_chunk.cpu().detach().numpy().tobytes()
            
            # Apply compression if enabled
            if self.enable_compression:
                import gzip
                data = gzip.compress(data)
            
            blob_name = self._get_blob_name(key)
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Upload blob
            blob_client.upload_blob(
                data,
                overwrite=True,
            )
            
            # Set metadata tags for lifecycle management
            tags = {
                "model": key.model_name,
                "layer": str(key.layer_id),
                "seq_hash": str(key.seq_hash),
                "created_time": str(int(time.time())),
                "compressed": str(self.enable_compression),
            }
            blob_client.set_blob_tags(tags)
            
            logger.debug(f"Stored chunk {blob_name} in Azure Blob Storage")
        except Exception as e:
            logger.error(f"Failed to put chunk {key}: {e}")
            raise
    
    def get(
        self,
        key: CacheEngineKey,
    ) -> Optional[torch.Tensor]:
        """Retrieve KV cache chunk from Azure Blob Storage.
        
        Args:
            key: Cache engine key identifying the chunk
            
        Returns:
            KV cache tensor or None if not found
        """
        try:
            blob_name = self._get_blob_name(key)
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Download blob
            stream = blob_client.download_blob()
            data = stream.readall()
            
            # Decompress if needed
            if self.enable_compression:
                import gzip
                data = gzip.decompress(data)
            
            # Convert bytes back to tensor
            kv_tensor = torch.from_numpy(
                __import__('numpy').frombuffer(data, dtype='float16')
            ).to(self.dst_device)
            
            logger.debug(f"Retrieved chunk {blob_name} from Azure Blob Storage")
            return kv_tensor
        except ResourceNotFoundError:
            logger.debug(f"Chunk {key} not found in Azure Blob Storage")
            return None
        except Exception as e:
            logger.error(f"Failed to get chunk {key}: {e}")
            return None
    
    def contains(self, key: CacheEngineKey) -> bool:
        """Check if KV cache chunk exists in storage.
        
        Args:
            key: Cache engine key identifying the chunk
            
        Returns:
            True if chunk exists, False otherwise
        """
        try:
            blob_name = self._get_blob_name(key)
            blob_client = self.container_client.get_blob_client(blob_name)
            blob_client.get_blob_properties()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking existence of chunk {key}: {e}")
            return False
    
    def evict_expired(self, ttl_seconds: Optional[int] = None) -> int:
        """Evict blobs older than TTL.
        
        Args:
            ttl_seconds: Time-to-live in seconds (uses ttl_hours if not specified)
            
        Returns:
            Number of evicted blobs
        """
        if ttl_seconds is None:
            ttl_seconds = self.ttl_hours * 3600
        
        current_time = time.time()
        evicted_count = 0
        
        try:
            for blob in self.container_client.list_blobs():
                if not blob.name.startswith(self.blob_prefix):
                    continue
                
                if blob.creation_time:
                    age_seconds = current_time - blob.creation_time.timestamp()
                    if age_seconds > ttl_seconds:
                        blob_client = self.container_client.get_blob_client(
                            blob.name
                        )
                        blob_client.delete_blob()
                        evicted_count += 1
                        logger.debug(f"Evicted expired blob {blob.name}")
        except Exception as e:
            logger.error(f"Error during eviction: {e}")
        
        return evicted_count
    
    def close(self) -> None:
        """Close connections and cleanup resources."""
        try:
            if self.blob_service_client:
                self.blob_service_client.close()
                logger.info("AzureBlobBackend closed")
        except Exception as e:
            logger.error(f"Error closing Azure Blob Backend: {e}")
