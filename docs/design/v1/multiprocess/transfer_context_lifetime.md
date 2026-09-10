# MP transfer context lifetime

This shared Device-DAX prerequisite protects registered GPU contexts in the MP
`LMCacheDrivenTransferModule` during standard STORE/RETRIEVE handlers.

## Role and authority

| Field | Contract |
| --- | --- |
| Capability and domain | MP transfer executor; owns local context admission and teardown |
| Deployment and policy | Existing MP server module; explicit unregister, liveness reaping or shutdown |
| Resource and scope | One registered GPU context per worker instance in one MP server |
| Request path | Standard STORE/RETRIEVE handlers and asynchronous lifecycle operations |
| Inputs/freshness | Current context registry, handler counts and existing heartbeat/grace thresholds |
| Decisions and state | Module decides whether to admit a handler or begin draining; state is local and ephemeral |
| Derived state | Context/status snapshots; snapshots do not reserve lifetime |
| Executor/effects | Module calls context close, layout unregister and device memory collection |
| Identity/locality | Existing worker instance ID; context entry identifies the registration being used |
| Consistency | One condition serializes admission; a separate lock serializes registration and cleanup |
| Durability/recovery | No persistent state; a failed close retains the draining entry for retry |
| Fencing | Draining entries and a closing module reject standard STORE/RETRIEVE admission and new registrations |
| Public contracts | Existing register/unregister, STORE/RETRIEVE, reap and close APIs |
| Failure/readiness | Missing/draining transfers return the existing failure result; cleanup errors propagate |
| Non-responsibilities | Shared-region ownership, allocation policy, payload visibility, coordinator deployment and fleet placement |

Storage operations and future shared-object authority remain separate from local
context lifetime. MP messages, normal heartbeat thresholds and storage
configuration are unchanged. Draining cleanup is retried despite a fresh PING;
there is no stored state to migrate.

## Ordering

```text
register -> active -> draining -> closed -> removed
              |
       admit STORE/RETRIEVE -> increment active handlers
              |
       return or raise -> decrement active handlers -> notify cleanup
```

Lookup and admission share one condition. Teardown stops admission, then waits
for handlers with the condition released. Device cleanup runs outside it.
The cleanup lock serializes registration and teardown. Module close rejects new
registrations and includes a registration already under construction.

Existing platform context close synchronizes the transfer stream before releasing
GPU IPC mappings. Completion dispatch stays alive until all contexts close.
Cleanup retains unfinished entries for retry without repeating a successful close.
The module drops its handler and registry references before memory collection;
external snapshots or exception tracebacks may still retain references.
Checksum context snapshots exclude draining entries; diagnostic entry snapshots
retain them for retry inspection. Neither snapshot reserves lifetime, and this
PR does not lease nonstandard Blend or diagnostic operations.

A later MP shutdown change must drain request-queue and lookup/event work before
closing shared storage. DAX mappings must stay registered through transfer
completion and visibility barriers; shared-mode activation depends on these steps.

## Verification

Tests race public MP handlers against unregister, reaping and close; check
admission, reference release, retry after PING and concurrent registration; exclude
draining checksum contexts; and retain existing transfer and liveness behavior.
They establish MP lifecycle contracts, not physical DAX sharing or performance.
