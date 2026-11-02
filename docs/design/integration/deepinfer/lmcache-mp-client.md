# DeepInfer LMCache MP client

DeepInfer integrates through `lmcache.integration.deepinfer`. Its scheduler and
worker roles share one `LMCacheMPConnector`, which selects the matching adapter
from its `role`. Lifecycle handling, cache mapping, and transfer
specifications are all maintained in that directory. The integration depends
only on LMCache's engine-neutral multiprocess core; it does not import another
serving-engine integration. The existing multiprocess server, hashing, lookup,
transfer, and sparse recurrent-state behavior remain shared, so no
DeepInfer-specific server module or object type is needed.

The adapter discovers cache geometry from the live DeepInfer cache objects:

- locally owned `ica_cache` and `ca_cache` tensors are registered as one full
  attention engine group and share DeepInfer block IDs;
- shared layer aliases are deduplicated by tensor identity;
- the source-token coverage of each page comes from DeepInfer's configured
  block size, while physical compressed slots come from each tensor shape;
- a canonical full runtime-state staging tensor is registered as a recurrent
  group. Historical chunks use null block ID 0 and only the final chunk points
  at the request's state row.

The last rule makes a prefix recoverable only when full CA/ICA data and an
endpoint state snapshot exist on every rank. Start the MP server with
`--separate-object-groups`; this lets the recurrent group remain sparse without
weakening the full-attention prefix requirement.

DeepInfer packs and restores state through its public
`select_prefix_state_to_store(...)` and
`load_full_state_to_cache(...)` helpers. The full transfer format includes
stored SWA, YoCo consumer SWA, and canonical boundary compression state rather
than the smaller persistent-disk state format. This avoids hard-coding layer
counts or assuming that raw ring-buffer rows are portable snapshots.

Example DeepInfer configuration:

```python
kv_transfer_config = {
    "connector": "LMCacheMPConnector",
    "connector_module_path": (
        "lmcache.integration.deepinfer.lmcache_mp_connector"
    ),
    "server_url": "tcp://127.0.0.1:5555",
    "transfer_mode": "lmcache_driven",
    "mq_timeout": 300.0,
}
```

`connector_module_path` selects this out-of-tree implementation instead of a
connector with the same name in DeepInfer's built-in registry.

The server chunk size must be a positive multiple of the DeepInfer block size.
The current DeepInfer model uses 128-token blocks, so a 128-token MP chunk is
the simplest configuration.
