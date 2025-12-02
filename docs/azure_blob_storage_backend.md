# Azure Blob Storage Backend for LMCache

## Overview

This document provides comprehensive guidance on using the Azure Blob Storage backend with LMCache for efficient KV cache storage in long-context LLM serving scenarios on Azure infrastructure.

The Azure Blob Storage backend enables:
- **Cost-effective storage** for KV cache with multiple storage tier options
- **Seamless Azure integration** with support for Managed Identity in AKS
- **Enterprise-grade reliability** with built-in replication and durability
- **Scalable capacity** without GPU memory constraints
- **Multi-authentication support** for various deployment scenarios

## Prerequisites

### Python Dependencies
```bash
pip install lmcache azure-storage-blob azure-identity
```

### Azure Resources
1. **Azure Storage Account** - Create via Azure CLI or Portal
2. **Storage Container** - For KV cache blobs (default: `lmcache-kv-cache`)
3. **Authentication credentials** based on your chosen method

### Azure CLI Commands
```bash
# Create a storage account (if not already exists)
az storage account create \
  --name <storage_account_name> \
  --resource-group <resource_group> \
  --location <region>

# Create a container
az storage container create \
  --account-name <storage_account_name> \
  --name lmcache-kv-cache
```

## Authentication Methods

### 1. Managed Identity (Recommended for AKS)

Best practice for Kubernetes deployments. No credentials required.

**Configuration:**
```yaml
backend_config:
  azure_blob:
    account_url: "https://<storage_account>.blob.core.windows.net"
    container_name: "lmcache-kv-cache"
    credential_mode: "managed_identity"
```

**AKS Setup:**
```bash
# Enable managed identity on AKS cluster
az aks update \
  --resource-group <resource_group> \
  --name <cluster_name> \
  --enable-managed-identity

# Create pod identity binding
az identity create \
  --resource-group <resource_group> \
  --name lmcache-identity

# Grant storage permissions
az role assignment create \
  --assignee <identity_principal_id> \
  --role "Storage Blob Data Contributor" \
  --scope /subscriptions/<subscription>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<account>
```

### 2. Account Key

Simple authentication using storage account key.

**Configuration:**
```yaml
backend_config:
  azure_blob:
    account_url: "https://<storage_account>.blob.core.windows.net"
    container_name: "lmcache-kv-cache"
    credential_mode: "account_key"
    account_key: "${AZURE_STORAGE_ACCOUNT_KEY}"
```

**Get Account Key:**
```bash
az storage account keys list \
  --account-name <storage_account_name> \
  --query "[0].value" --output tsv
```

### 3. Connection String

Full connection string authentication.

**Configuration:**
```yaml
backend_config:
  azure_blob:
    credential_mode: "connection_string"
    connection_string: "${AZURE_STORAGE_CONNECTION_STRING}"
```

### 4. SAS Token

Time-limited token-based authentication.

**Configuration:**
```yaml
backend_config:
  azure_blob:
    account_url: "https://<storage_account>.blob.core.windows.net"
    credential_mode: "sas_token"
    sas_token: "${AZURE_STORAGE_SAS_TOKEN}"
```

## Configuration

### Complete Configuration Example

```yaml
cache_config:
  # KV chunk size (128KB)
  chunk_size: 131072
  
  # Local GPU cache (4GB) - hot data
  local_gpu_size: 4294967296
  
  # Local CPU cache (8GB) - warm data
  local_cpu_size: 8589934592
  
  # Remote storage backends
  remote_backends:
    - name: "azure_blob"
      priority: 100
      enabled: true

backend_config:
  azure_blob:
    # Azure Storage account URL
    account_url: "https://<storage_account>.blob.core.windows.net"
    
    # Container for KV cache storage
    container_name: "lmcache-kv-cache"
    
    # Authentication mode
    credential_mode: "managed_identity"  # or account_key, connection_string, sas_token
    
    # Performance tuning
    max_concurrency: 8              # Concurrent upload/download operations
    chunk_upload_size: 4194304      # 4MB per chunk
    
    # Blob naming and organization
    blob_prefix: "lmcache-kv/"      # Prefix for all blob names
    
    # TTL and lifecycle
    ttl_hours: 24                   # Keep blobs for 24 hours
    enable_compression: true        # Enable gzip compression
```

### Performance Tuning

```yaml
# For high-throughput scenarios
backend_config:
  azure_blob:
    max_concurrency: 16             # Increase concurrent operations
    chunk_upload_size: 8388608      # 8MB chunks
    enable_compression: false       # Skip compression for speed

# For cost-optimized scenarios
backend_config:
  azure_blob:
    max_concurrency: 4              # Conservative concurrency
    chunk_upload_size: 2097152      # 2MB chunks
    enable_compression: true        # Compress for storage savings
    ttl_hours: 72                   # Longer retention
```

## Usage Examples

### With vLLM

```python
from vllm import LLM, SamplingParams
from lmcache.lmcache import LMCache

# Initialize LMCache with Azure Blob backend
lmcache = LMCache.load_from_config(
    config_file="lmcache_config_azure.yaml"
)

# Create vLLM engine with LMCache
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    lmcache=lmcache,
)

# Make requests - KV cache will be stored in Azure Blob
outputs = llm.generate(
    prompts=["Explain machine learning"],
    sampling_params=SamplingParams(temperature=0.7, max_tokens=512),
)

for output in outputs:
    print(output.outputs[0].text)
```

### With SGLang

```python
import sglang as sgl
from lmcache.lmcache import LMCache

# Initialize backend
lmcache = LMCache.load_from_config(
    config_file="lmcache_config_azure.yaml"
)

# Create SGLang runtime
runtime = sgl.Runtime(
    model="meta-llama/Llama-2-70b-hf",
    lmcache=lmcache,
)

# Use SGLang for inference
response = runtime.generate(
    "Generate a poem about Azure clouds",
    max_new_tokens=512,
)

print(response)
```

## Monitoring and Debugging

### Enable Logging

```python
import logging

# Enable debug logging for Azure backend
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('lmcache.storage_backend.azure_blob_backend')
logger.setLevel(logging.DEBUG)
```

### Monitor Azure Storage

```bash
# View blob statistics
az storage blob list \
  --account-name <storage_account> \
  --container-name lmcache-kv-cache \
  --output table

# Monitor storage metrics
az monitor metrics list-definitions \
  --resource /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.Storage/storageAccounts/<account>
```

## Performance Considerations

### Latency
- **Network latency**: Typically 10-50ms for Azure Blob operations
- **Serialization**: Tensor to bytes conversion adds 5-10ms per chunk
- **Compression**: Optional compression adds 5-20ms overhead

### Throughput
- **Sequential uploads**: ~100-200 MB/s per connection
- **Concurrent uploads**: Up to 1-2 GB/s with 8+ concurrent connections
- **Storage tier impact**: Hot tier offers best performance, Cold tier has higher latency

### Cost Optimization
- Use **Cool/Cold storage tiers** for infrequently accessed cache
- Enable **compression** to reduce data transfer costs
- Set appropriate **TTL** to minimize storage costs
- Use **lifecycle policies** for automatic tiering

## Cost Estimation

### Example Calculation (for 70B model, 4k context)
```
# Per-request cost estimation:
- KV cache size: ~500MB per request
- Write cost: $0.01/GB = $0.005 per write
- Read cost: $0.001/GB = $0.0005 per read
- Storage (24hr TTL): ~$0.0001 per request

# Total per 1000 requests: ~$6.00
```

## Troubleshooting

### Authentication Failures

**Issue**: "Authentication failed"

**Solutions**:
- Verify credentials in configuration
- Check service principal permissions: `Storage Blob Data Contributor`
- For Managed Identity, verify pod identity binding
- Check Azure CLI: `az account show`

### Blob Not Found

**Issue**: "ResourceNotFoundError: The specified blob does not exist"

**Solutions**:
- Verify container name matches configuration
- Check blob naming prefix consistency
- Ensure proper permissions for blob operations

### Timeout Issues

**Issue**: "Operation timed out"

**Solutions**:
- Increase `max_concurrency` in config
- Check network connectivity to Azure
- Monitor Azure Storage metrics for throttling
- Consider using premium storage for higher throughput

### High Latency

**Issue**: Slow KV cache retrieval

**Solutions**:
- Use Hot storage tier instead of Cool/Cold
- Increase `max_concurrency` for parallel operations
- Enable compression to reduce data transfer
- Deploy LMCache closer to Azure region

## Best Practices

1. **Use Managed Identity** on AKS for secure, credential-less authentication
2. **Enable compression** to reduce storage costs and transfer time
3. **Set appropriate TTL** based on workload patterns
4. **Monitor storage metrics** for performance bottlenecks
5. **Use lifecycle policies** for automatic storage tier transition
6. **Configure alerts** for storage account quota and performance
7. **Test with your workload** before production deployment
8. **Use regional endpoints** to minimize latency

## Kubernetes Deployment Example

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lmcache-azure-config
data:
  lmcache_config.yaml: |
    cache_config:
      chunk_size: 131072
      local_gpu_size: 4294967296
      local_cpu_size: 8589934592
      remote_backends:
        - name: "azure_blob"
          priority: 100
          enabled: true
    backend_config:
      azure_blob:
        account_url: "https://<storage_account>.blob.core.windows.net"
        container_name: "lmcache-kv-cache"
        credential_mode: "managed_identity"
        max_concurrency: 8
        enable_compression: true

---
apiVersion: v1
kind: Pod
metadata:
  name: llm-inference
spec:
  serviceAccountName: lmcache-sa
  containers:
  - name: inference
    image: vllm:latest
    env:
    - name: LMCACHE_CONFIG_FILE
      value: /etc/lmcache/config.yaml
    volumeMounts:
    - name: lmcache-config
      mountPath: /etc/lmcache
  volumes:
  - name: lmcache-config
    configMap:
      name: lmcache-azure-config
```

## References

- [Azure Blob Storage Documentation](https://learn.microsoft.com/en-us/azure/storage/blobs/)
- [Azure Identity Authentication](https://learn.microsoft.com/en-us/python/api/azure-identity/)
- [LMCache Documentation](https://docs.lmcache.ai/)
- [vLLM Integration Guide](https://docs.vllm.ai/)

## Support and Contributions

For issues, feature requests, or contributions related to the Azure Blob Storage backend:
- Open an issue on the [LMCache GitHub](https://github.com/LMCache/LMCache/issues)
- Check existing issues for known problems
- Provide detailed configuration and error messages for troubleshooting
