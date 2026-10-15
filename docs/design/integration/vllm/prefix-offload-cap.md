# Qualifying The Existing Request Offload Cap

Refs [#5783](https://github.com/LMCache/LMCache/issues/5783). No new policy API
or scheduling mechanism is introduced.

## Existing Contract

`LMCacheMPRequestTracker` reads `lmcache.max_offload_tokens` from
`SamplingParams.extra_args["kv_transfer_params"]`. The caller supplies a
nonnegative integer. `GetStoreMetadata` limits the computed, allocated,
available prompt range by this cap before emitting complete LMCache chunks.
For a cold request, cap zero produces no STORE; a cap not divisible by chunk
size admits only its complete leading chunks. Later chunked-prefill steps
cannot extend the stored range beyond that cap. This changes neither cache
lookup nor computation of the request's remaining tokens.

Eager and lazy modes receive these same bounded candidates. Lazy mode may
defer, coalesce or discard them, but cannot widen them. GPU block protection
belongs to the existing LazyOffloadManager; the cap creates no new owner.
Reset/preemption drops buffered candidates before they can be submitted.

## Acceptance Tests

`tests/v1/test_mp_offload_cap.py` constructs actual vLLM Request, BlockPool and
hashed cache blocks, and initializes the real connector with a mocked adapter
transport. Public lookup/allocation/build_connector_meta/request-finished and
completion entry points drive actual tracker/metadata and lazy-manager logic.
Both new-request and cached-request eager submissions are checked. It simulates
chunked-prefill advances starting below a chunk and at a storable boundary,
checking the computed-token bound immediately after every step. Modes are eager, FIFO
and EVICTION_AWARE; caps are zero, smaller than one chunk, one complete chunk,
a non-aligned multi-chunk prefix, and uncapped. A zero-token scheduler step
must not emit buffered stores. In-flight source blocks must have exact positive
reference counts and cannot be handed to a competing allocation. Completion
must leave every original block at reference count zero, not just a matching
free-queue length. Reset cases must emit nothing.

CUDA cases initialize real paged GPU tensors, copy the selected metadata
ranges into a real StorageManager's L1 reservations. Write keys are derived
from emitted token identity using the real TokenHasher; lookup keys come from
an independent expected token sequence. Tests finish writes, query a
prefix lookup and copy the retained payload back to CUDA. Independently
constructed expected values verify block mapping and token ranges; exact
byte counts verify that excluded tokens never cross this test's D2H path.
A guard-filled GPU suffix must remain unchanged. cap=0 also exercises the
zero-hit completion path rather than treating a skipped copy as test success.

The configuration and scheduler output envelopes and adapter/receipt transport
are test doubles; constructor state and dispatch are not bypassed. Torch copies
stand in for the MP worker/IPC executor. No model,
full connector server, multi-worker receipt protocol, cancellation of an
in-flight device operation, compiled CUDA transfer kernel or serving metric
is qualified by these tests. Reset tests cover buffered preemption, not a
transport abort. GPU tests require a real CUDA runtime and are marked cuda;
absence of vLLM skips the module explicitly, not a passing qualification.

## Follow-On Experiment

Use the existing cap as the input for a separately measured prefix selector.
Any online score must use prior observations, account for prefix dependency,
and distinguish unknown recompute/transfer cost from zero cost. The cap is
an admission limit, not a reuse predictor and not the lazy drain policy.
Full serving experiments must separately report eager/lazy behavior, GPU APC,
TTFT/ITL, goodput, errors, actual bytes and fixed model/layout/revision.
No performance benefit or pooled safety follows from this functional suite.
