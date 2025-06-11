# External Connector for LMCache

LMCache supports custom external storage backends via Python modules. This connector type allows integrating any key-value store with LMCache.

## Requirements

1. Implement a connector class inheriting from `BaseConnector` (see `base_connector.py`)
2. Place your module in Python import path

## Configuration

Specify module and class name by `remote_url` in `backend_type.yaml`, and the remote_url should contain
- **Module Path**: Specify the Python module path (e.g., `adt`)
- **Connector Name**: Provide the class name of the connector (e.g., `adt_kv_connector`)

## Example YAML Configuration

This example use adt-kv as an example which is an internal lmcache remote connector.

```yaml
remote_url: "external://host:0/adt.adt_kv_connector/?connector_name=AdtKVConnector"
extra_config:
  adt_kv_secret: "---"
  adt_kv_connections: 4
  adt_kv_workers: 32
```

## Start vLLM with the external adt-kv connector

```shell
VLLM_USE_V1=0 \
LMCACHE_USE_EXPERIMENTAL=True LMCACHE_TRACK_USAGE=false \
LMCACHE_CONFIG_FILE=backend_type.yaml \
vllm serve /disc/f/models/opt-125m/ \
           --served-model-name "facebook/opt-125m" \
           --enforce-eager  \
           --port 8000 \
           --gpu-memory-utilization 0.8 \
           --kv-transfer-config '{"kv_connector":"LMCacheConnector","kv_role":"kv_both"}' \
           --trust-remote-code
```
