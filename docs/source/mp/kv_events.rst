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
private CPU/L1 storage, and DP=PP=DCP=PCP=1. Validation targets TP1 and TP2.
Readiness uses the registered KV-rank set. One private MP must own the
serving replica's complete KV-rank set. LMCache chunk size C must be an integer multiple
of the engine's native event block size B. Complete chunks advertise all C/B
native blocks and their original parent chain; incomplete terminal chunks do
not advertise CPU placement.

Native block visibility is a routing hint. Lookup still requires matching
complete LMCache chunks: a shorter prefix or new branch may overlap native
blocks without providing an MP cache hit. Mixed GPU/CPU retrieval preserves
the already-resident GPU range using the existing partial-chunk load path;
this feature adds no independent subchunk lookup.

Event-enabled hybrid/SWA/Mamba, MLA, DCP, multimodal, LoRA, salted, speculative,
and shared/L2 configurations are outside this profile. Unsupported model and
topology configurations fail at startup. Explicitly disabling engine KV events
retains the ordinary inference/cache path.

vLLM obtains identities from scheduler ``Request.block_hashes`` and carries
them to workers in connector metadata. Its direct event hook and MultiConnector
worker-metadata bridge carry each transition through one delivery path.
Event-enabled operation requires vLLM's connector ``validate_request`` hook,
called during input preprocessing before scheduler admission. MultiConnector
must forward it to its children. Builds without this API fail event-enabled
startup; events-disabled inference retains its existing behavior.
CPU stores declare group zero and the ``full_attention`` cache kind, allowing
routers to index startup replay before any GPU store event arrives.
Requests with salted, LoRA, multimodal, embedding, session, or resumable inputs
receive a request-validation error. This rejection must not stop the engine.
Missing native hashes for an otherwise supported STORE are errors too.
The old ``lmcache.mp.hash_algorithm`` worker option no longer controls events;
the server's storage hash remains independent of the engine's native identity.

Delivery and recovery
---------------------

vLLM uses ``MPKVEventSubscriber`` over ZMQ or gRPC. One elected worker
subscribes per engine/MP attachment. Other TP ranks remain in server-event mode.
Both synchronous and asynchronous gRPC servers are supported.
The subscriber owns its stream and receiver thread; the engine owns the client,
registration, heartbeat, native publisher, and advertised CPU identity ledger.
Every native-event rank starts heartbeat after registration, including before
the first inference request, so idle transfer registrations remain live.
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
