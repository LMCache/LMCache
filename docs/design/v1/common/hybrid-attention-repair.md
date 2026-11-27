# Hybrid attention delivery coverage

The existing `repair` engine policy also needs to distinguish attention KV
delivery failures from healthy recurrent checkpoints in a hybrid model.
Recovery of missing recurrent state is out of scope. There is no new CLI.

`RetrieveCoverage` retains its aggregate loaded/failed chunk partition and
adds optional `groups: list[RetrieveGroupCoverage]`. Each group names its
object-group ID, engine-group IDs, registered kind (`attention` or `recurrent`),
first required chunk, and loaded/failed chunks. Its partition covers only
`[first_chunk, total_chunks)`, because recurrent state may need only the last
checkpoint. Aggregate failures must equal the union of group failures.
Legacy replies omit groups and retain their existing meaning for one group.

The server derives identities and kinds from registration. Partial coverage
requires separated object groups for hybrid models, full chunks and a terminal
GPU completion event. A client must validate every expected engine group and
drain that event before treating any recurrent checkpoint as available.
An attention-only repair is allowed only when all required recurrent delivery
succeeded. Unknown or recurrent failures keep ordinary recompute behavior.

Engines own replay and scratch state, checkpoint lifetimes and commit. LMCache
tracks failed attention ranges per engine group, separately from stored-prefix
cursors. After engine commit it submits the usual STORE with null block IDs
for non-target groups, using existing all-null-group suppression. No special
overwrite, retry, persistence acknowledgment or backend policy is introduced.

Hybrid selective range STORE requires separated attention/recurrent object
groups. Ordinary STORE rejects partially null components of a single atomic
external object, so an unsupported co-packed layout cannot replace healthy
state with null pages. The dirty range remains eligible through the existing
normal STORE outcome bookkeeping. Unknown delivery errors conservatively
mark attention ranges dirty; delivery loss alone does not prove CMX object
loss. Normal layer lookup/dedup/offload rules decide whether STORE reaches L2.

vLLM's subpaged attention and opaque state page views must retain the resolved
NHD/HND axis order during format detection. Re-viewing bytes as synthetic NHD
while declaring HND can turn a 576-token block into 576 heads. The adapter
normalizes only addressing metadata, preserving the allocation and byte order.
