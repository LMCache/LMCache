Native CPU KV Events for vLLM
=============================

MP publishes CPU placement through vLLM's existing KV-event publisher.
vLLM attaches its native block identities to STORE metadata;
MP emits a Stored transition only when every required KV rank and object group
has a committed, readable L1 copy. Eviction of a required object emits Removed
with the same native identities. Removing one of two L1 copies keeps the
placement available.

Enable native KV events in vLLM and keep MP observability and a positive
``--kv-event-log-size`` enabled. Workers require the
``kv_event_native_bindings_v1`` capability. They validate one immutable identity
profile on every rank's cache registration, including native block size, hash
algorithm, seed semantics, and representation. Old chunk-hash consumers cannot
use this stream.
An incompatible profile or missing capability fails explicitly. The elected
subscriber also checks the profile when opening or restoring its stream.

Supported deployment
--------------------

Use vLLM with dense text-only full attention, LMCache-driven device transfer,
private CPU/L1 storage, and DP=PP=DCP=1. TP size follows the serving configuration;
native events impose no additional TP-size cap. One private MP must own the
serving replica's complete KV-rank set. LMCache chunk size C must be an integer multiple
of the engine's native event block size B. Complete chunks advertise all C/B
native blocks and their original parent chain; incomplete terminal chunks do
not advertise CPU placement.

Event-enabled hybrid/SWA/Mamba, MLA, DCP, multimodal, LoRA, salted, speculative,
and shared/L2 configurations are outside this profile. Unsupported model and
topology configurations fail at startup. Explicitly disabling engine KV events
retains the ordinary inference/cache path.

vLLM obtains identities from scheduler ``Request.block_hashes`` and carries
them to workers in connector metadata. Its direct event hook and MultiConnector
worker-metadata bridge carry each transition through one delivery path.
vLLM uses its existing connector APIs and requires no companion engine patch.
CPU stores declare group zero and the ``full_attention`` cache kind, allowing
routers to index startup replay before any GPU store event arrives.
Requests with salted, LoRA, multimodal, embedding or session namespaces retain
ordinary caching without native CPU-event bindings. They are not rejected by
an LMCache admission hook.

Delivery and recovery
---------------------

vLLM uses ``MPKVEventSubscriber`` over ZMQ or gRPC. One elected worker
subscribes per engine/MP attachment. Other TP ranks remain in server-event mode.
Both synchronous and asynchronous gRPC servers are supported.
The subscriber owns its stream and receiver thread; the engine owns the client,
registration, heartbeat, native publisher, and advertised CPU identity ledger.
The vLLM buffered-event gauge counts pending stream records and invalidations
even when vLLM stops draining. Generated/drained counters track native publication.

Reconnect invalidates prior CPU advertisements and replays the current retained
log from its beginning. Log truncation, local queue overflow, and a new MP
incarnation also invalidate CPU advertisements. Server EventBus loss clears the
server's completeness projection before new residency can be advertised.
These removals are CPU-scoped and leave native GPU placements intact.

There is no snapshot service: a cold subscriber can miss warm entries whose
records have left the retained log. Increasing ``--kv-event-log-size`` extends
replay coverage but does not guarantee recovery of every warm entry. Protocol
failures remain visible; the connector never switches to local STORE
announcements, lookup polling, or guessed native hashes.

vLLM drains on engine steps, so an idle worker may receive an MP eviction before
publishing its removal. Active-step convergence and request-free idle convergence
are separate measurements. Pause behavior must also be measured separately.

An MP process restart also loses its device-transfer registrations.
Event-stream reconnection alone does not restore the transfer path.
vLLM's existing transfer recovery runs after its heartbeat observes an
unhealthy-to-healthy transition. A short restart between successful pings can
miss that transition. Keep MP down until its attached clients report degraded
mode, then start MP and verify all GPU registrations and the subscription before
resuming inference; otherwise restart the engine to recreate its registrations.

Verification
------------

A passing routed cache-load result must
correlate router selection, the selected engine's GPU match, MP L1 lookup,
completed CPU-to-GPU transfer, and the resulting generation. Verify GPU-only,
CPU-only, mixed-prefix, eviction, reconnect, and restart cases separately;
idle publication and retained replay have the limitations described above.
