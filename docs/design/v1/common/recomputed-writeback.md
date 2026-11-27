# Recomputed KV uses ordinary STORE

A successful lookup sets the engine adapter's stored-prefix cursor. A later
partial delivery can contradict it for individual chunks. The adapters now keep
those chunks separately; they do not rewind the cursor or discard the healthy
suffix. A committed normal forward makes those exact chunks eligible for the
same asynchronous STORE used for any newly computed KV.

There is no special server request, external-acknowledgment contract, forced
overwrite or retry policy. L1 admission, deduplication, background L2 offload,
failures and retries follow the existing LMCache configuration and ordinary
STORE behavior. The adapters retain the GPU source using their existing pinning
mechanism until ordinary transfer completion. Normal STORE acknowledgment means
L1 admission, not CMX persistence. External completion is checked by traces, not
used to change STORE semantics.

Revisioned range tickets prevent an older STORE completion from clearing newer
failure evidence. Failed submissions retain the range for later normal store
opportunities without retaining the GPU allocation after transfer completion.
The retry backlog contains only content identity and range metadata. No new
retry loop, timeout, or persistence policy is introduced.

A delivery failure is not proof of physical object loss. Such failures invalidate
the adapter's already-stored assumption conservatively. Lower-layer read misses
retain their normal miss/eviction behavior. A healthy L1 copy may still deduplicate
an ordinary STORE; the CMX qualification uses the existing skip_l1 read/write
policies, so completed reads release their L1 staging copies and regenerated
chunks follow the usual CMX PUT path. Healthy chunks are not explicitly stored.

Scope: single-GPU full attention. vLLM subtracts in-flight tokens from its
computed position before making a missing range eligible. GPU qualification
uses synchronous scheduling, as required by the existing selective repair path.
Hybrid state and multiple KV groups are outside this implementation.
No change to inference scheduling policy or attention kernels is required.
