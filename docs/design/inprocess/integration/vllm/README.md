# vLLM in-process integration

The modules under `lmcache/inprocess/integration/vllm/` implement LMCache's
deprecated in-process integration with vLLM. This package boundary keeps the
in-process entry points separate from the multiprocess integration under
`lmcache/integration/vllm/`.

The in-process modules may depend on shared LMCache components, including the
cache engine, memory management, GPU connectors, and utilities. Shared modules
must not depend on `lmcache.inprocess`; this one-way dependency keeps the
in-process implementation removable without pulling multiprocess code with it.

Compatibility modules remain at the previous `lmcache.integration.vllm`
locations because upstream vLLM and existing `kv_connector_module_path`
configurations import those paths. New direct imports and dynamic connector
configurations should use the `lmcache.inprocess.integration.vllm` paths.
