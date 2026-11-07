# MP-mode sleep and KV-cache release

vLLM can give its KV cache memory back to the device and take it again
later without restarting: `/sleep`, `/release_kv_cache_memory`, and the
extensible KV cache's release during a weight update. Before releasing
the memory vLLM calls the connector's `unregister_kv_caches()`, and after
mapping memory again it calls `restore_kv_caches()`. This document covers
what `LMCacheMPConnector` does in those two hooks. The connector only
delegates. The logic lives in `LMCacheMPWorkerAdapter`
(`lmcache/integration/vllm/vllm_multi_process_adapter.py`).

## Why the hooks are needed

In LMCache-driven transfer the server imports the worker's KV cache
(CUDA IPC or VMM handles) and copies blocks on its own streams. With
CUDA VMM memory (`lmcache.mp.use_vmm_api`, vLLM's
`--enable-extensible-kv-cache`) an import keeps the physical memory
alive, so a server that still holds it makes vLLM's release free
nothing. After wake-up the KV cache sits at the same virtual addresses
but on new physical chunks, so the old import would also point at the
wrong memory. See
[vmm_cuda_ipc.md](../../v1/platform/devices/cuda/vmm_cuda_ipc.md#engine-remaps-sleep-mode)
for the driver side.

## Flow

```
vLLM worker                          adapter                               server
-----------                          -------                               ------
sleep / discard / weight update
  unregister_kv_caches() ----------> [registration lock]
                                     wait for in-flight stores/retrieves
                                       (each future's server-side device
                                        work done, mq_timeout in total)
                                     detach transfer_ctx, mark released
                                     transfer_ctx.unregister() ---------> drop context, close imports
                                     transfer_ctx.close()
  release_physical()                 (memory freed: nothing imports it)
...
wake_up / weight update done
  recommit()                         (new physical chunks)
  restore_kv_caches() -------------> [registration lock]
                                     server healthy?
                                       yes: new transfer context, wrap
                                            the same tensors again -----> import the new memory
                                       no:  clear released and leave it
                                            to heartbeat recovery
```

Both hooks are idempotent: `unregister_kv_caches()` returns at once when
already unregistered or before `register_kv_caches()`, and
`restore_kv_caches()` returns at once unless unregistered. vLLM relies on
this, for example `wake_up(tags=["weights"])` followed by
`wake_up(tags=["kv_cache"])` calls restore only once, and
`release_kv_cache_memory` followed by `/wake_up` pairs one unregister with
one restore.

## Design points

- **Wait before unregistering.** The server handles a store or retrieve
  on a worker thread, while it handles `UNREGISTER_KV_CACHE` on its main
  loop. Unregistering while a transfer is in flight could unmap memory
  under a running copy. The adapter therefore waits for every tracked
  store and retrieve future first, including the server's completion
  event. The futures stay tracked, so `get_finished()` still reports
  them after wake-up. A failed transfer counts as done. A transfer still
  running after `mq_timeout` is logged and the unregistration proceeds.
- **Transfers while unregistered fail loudly.** The adapter drops its
  transfer context, so `submit_store_request()`,
  `submit_retrieve_request()` and `create_recorded_event()` raise
  `RuntimeError` naming `restore_kv_caches()` instead of reaching
  released memory. vLLM does not schedule while asleep, so this only
  catches misuse.
- **Heartbeat recovery stays out of the way.** A server restart during
  sleep takes the heartbeat's unhealthy-to-healthy edge and runs the
  recover callback. While the caches are released the callback skips
  their re-registration and still reports healthy. It shares a lock
  with both hooks, so a recovery cycle cannot register memory that the
  engine is releasing. After restore, recovery works as before.
- **Waking while the server is down degrades instead of failing.** If
  the heartbeat already marked the server unhealthy, restore clears the
  released state without contacting the server. Stores are dropped and
  retrieves fail, as in any outage, until the next successful ping's
  recover callback registers the caches. If the server looked healthy
  but does not answer the registration within `mq_timeout`, restore
  raises `ConnectionError` and stays unregistered, so calling it again
  retries.
- **Both transfer modes.** The hooks only use `TransferContext.unregister()`,
  `close()` and a fresh registration, so they behave the same for
  LMCache-driven and engine-driven transfer. In engine-driven mode the
  server never imports GPU memory, and the round trip releases and
  re-creates the shared-memory staging context.

## Out of scope

- The experimental Q ring (`transfer_intermediate_tensors`) is memory
  LMCache allocates itself. Sleep does not release it, so the hooks leave
  its registration alone.

## Testing

- Unit tests: `tests/v1/test_vllm_mp_adapter.py` (the
  `unregister`/`restore` group, both transfer modes with real transfer
  contexts and a stubbed request client) and
  `tests/v1/test_mp_connector_kv_roles.py::test_worker_sleep_hooks_drop_and_redo_registration`.
- End to end: vLLM with `--enable-extensible-kv-cache --enable-sleep-mode`
  and `lmcache.mp.use_vmm_api: true`. Send a prompt, `/sleep?level=1`,
  `/wake_up`, then send the prompt twice more. vLLM should report
  `Sleep mode freed` close to the KV cache size, and the server should log
  `Unregistered KV cache` followed by `Registered KV cache`. vLLM's sleep
  pauses with `clear_cache`, which also clears the LMCache server, so the
  first prompt after wake-up stores through the new registration and the
  second one hits LMCache (`vllm:external_prefix_cache_hits_total`) with
  identical output.
