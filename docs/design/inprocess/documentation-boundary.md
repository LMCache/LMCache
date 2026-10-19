# Documentation boundaries between deployment modes

Classify a page by the engine, configuration, and interface used by its examples.
The presence of separate worker processes, a standalone starter, a shared storage
product, or the word “legacy” in an IPC description does not identify the
LMCache deployment mode. Read configuration blocks and follow the implementation
before moving a page. Split mixed pages at the interface boundary.

| Surface | Implementation ownership | Documentation home |
| --- | --- | --- |
| MP Python storage plugins | `lmcache/v1/distributed/l2_adapters/plugin_l2_adapter.py`, `L2AdapterInterface`, `--l2-adapter` | `docs/source/developer_guide/extending_lmcache/storage_plugins.rst` (development), `docs/source/mp/l2_storage/plugin.rst` (configuration) |
| MP native plugins | `native_plugin_l2_adapter.py` and `NativeConnectorL2Adapter` | Main developer native connector guide and MP plugin reference |
| Old storage plugins | `lmcache/v1/storage_backend/abstract_backend.py`, `StoragePluginInterface`, engine YAML | `docs/source/legacy/developer_guide/extending_lmcache/storage_plugins.rst` |
| Shared native connector framework | `csrc/storage_backends/`, `ConnectorClientBase` and `NativeConnectorL2Adapter` | Main native connector guide, with separate mode-specific steps |
| Common HTTP diagnostics | MP `http_apis/common_api.py` imports the embedded server's common routers | `docs/source/internal_api_server/common_apis.rst` |
| Version routes | MP `http_apis/info_api.py` re-exports `internal_api_server/vllm/version_api.py` | Shared common API reference; adapter-specific inference endpoints remain legacy |
| Periodic thread monitoring | Shared `PeriodicThreadRegistry`, used by MP sessions and modules as well as the embedded engine | `docs/source/production/observability/periodic_thread_api.rst` |
| Basic Check CLI | `basic_check.py` dispatches both `test_l2_adapter` and old storage checks | Main developer usage entrypoint; old backend modes and YAML remain legacy |
| Runtime plugins | `MPRuntimePluginLauncher` wraps shared `RuntimePluginLauncher` | Main developer guide, separate CLI/aggregate JSON and engine YAML/role filtering |
| Usage telemetry | Shared `lmcache/usage_telemetry/` with distinct deployment-mode messages | Main developer guide |
| MUSA transfer support | Shared device implementation; MP IPC mode differs from embedded connector setup | Shared install/MP launch guide, separate legacy YAML launch |
| Hidden-state store | Constructed by `lmcache/v1/cache_engine.py`, not the MP server | Legacy implementation guide; main non-KV page describes availability |
| Controller management | Old `cache_controller` and `api_server` versus MP coordinator/HTTP APIs | Old event/worker schemas remain legacy; current APIs under MP |
| Model recipes | Standalone server and `LMCacheMPConnector` | Main recipes; compression links point to MP availability, never silently substitute old CacheGen YAML |
| Docker and Kubernetes | Image/cache permissions are shared; `LMCacheConnectorV1` and Production Stack CPU-buffer examples are embedded recipes | Common deployment prerequisites remain active; old launch examples remain legacy |

A diagnostic route can be shared while its state or metrics are mode-specific.
For example, MP scripts use `app.state.engine`, embedded scripts use
`app.state.lmcache_adapter.lmcache_engine`, and the shared `/metrics/reset` route
resets legacy Prometheus instruments rather than MP OpenTelemetry counters.

Keep current documentation discoverable through the active toctrees and
`docs/source/llms.txt`. Do not label similarly named MP features as exact
replacements for deprecated controller operations. Keep old command schemas and
configuration examples explicitly scoped to the old implementation.

For a migration, audit every English RST page, the main and legacy toctrees,
installation/quickstart routing, recipe feature links, literal includes, and
`llms.txt`. Do not edit generated `docs/source/locale/zh_CN/` translations.
Validate using the documentation dependencies from `requirements/docs.txt`, a
clean Sphinx HTML build with `SPHINXOPTS='-W --keep-going'`, and the repository's
pre-commit checks.
