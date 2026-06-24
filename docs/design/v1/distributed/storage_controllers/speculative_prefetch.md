# Speculative Prefetch

Predictive ("speculative") prefetching: guess which cache keys a future request
will need and warm them into L1 from a slower L2 tier *before* the request
arrives. This is the software-portable idea from **CXL-SpecKV: A Disaggregated
FPGA Speculative KV-Cache for Datacenter LLM Serving**
([arXiv:2512.11920](https://arxiv.org/abs/2512.11920)), adapted to LMCache's
tiered storage. The paper's other two techniques map onto existing LMCache
features and are out of scope here: CXL memory disaggregation is already
available through the Maru backend (`storage_backend/maru_backend.py`), and
hardware KV compression has a software analog in the existing SERDE path.

## Motivation

Today's `PrefetchController` is **reactive**: a caller hands it an explicit list
of keys via `submit_prefetch_request(...)`, and it looks them up across L2
adapters, plans loads, reserves L1, and loads. Nothing *predicts* which keys to
prefetch ahead of an actual request — so the first request for a not-yet-cached
prefix always pays the full L2 load latency. Predictive prefetch hides that
latency for access patterns that are partially predictable: multi-turn chat
continuations, RAG documents that co-occur, and the ordered chunks of one
prompt.

## Components

### `SpeculativePrefetcher` (`speculative_prefetcher.py`)

The predictor — the "brain". It is intentionally **generic over a hashable key
type and imports nothing from the rest of LMCache**, so it is unit-testable in
isolation (no torch / native extensions) and reusable for any key
representation.

It learns two cheap signals online from the observed access stream, the
software stand-in for CXL-SpecKV's LSTM sequence predictor:

1. **First-order Markov successor model** — recency-decayed transition weights
   `P(next = B | last = A)`. Captures "a request for chunk `A` is usually
   followed by chunk `B`".
2. **Popularity prior** — a small-weight global access-frequency term that
   ranks broadly-hot keys when the Markov model has no evidence for the current
   context.

API: `observe(key)` / `observe_sequence(keys)` to update the model;
`predict(recent, k)` / `predict_keys(...)` to query ranked next-key guesses with
confidence scores in `[0, 1]`. Updates and queries are amortized O(1) /
O(out-degree); memory is bounded by `max_sources` (least-informative source
evicted) and per-source pruning of decayed weights.

Tuning knobs: `max_predictions`, `min_confidence` (precision/recall trade-off),
`decay` (adaptation speed), `popularity_weight`, `max_sources`.

### Integration into `PrefetchController`

The controller gains an optional `speculator` parameter (default `None` → no
behavior change, no overhead):

- `submit_prefetch_request(...)` additionally feeds the request's keys to the
  predictor via `observe_sequence` (under a dedicated lock, since submission is
  called from arbitrary external threads). This **only updates the model**; it
  does not itself issue any load.
- `predict_prefetch_keys(recent, max_keys)` returns the predictor's ranked
  guesses.

### Configuration

`StorageManagerConfig` (`distributed/config.py`) exposes
`enable_speculative_prefetch` (default `False`) plus
`speculative_prefetch_max_keys`, `speculative_prefetch_min_confidence`, and
`speculative_prefetch_decay`, with matching `--enable-speculative-prefetch` /
`--speculative-prefetch-*` CLI flags. `StorageManager` constructs a
`SpeculativePrefetcher` from these and passes it to the `PrefetchController`
only when enabled.

## Design boundary: prediction vs. issuing loads

This change implements and wires the **prediction** half end-to-end (learn the
access stream; expose ranked predictions). It deliberately does **not** have the
controller auto-issue speculative loads internally, for a correctness reason:

`PrefetchController` read-locks every loaded key in L1 and expects the submitter
to release it via `query_prefetch_result(...)`. A load with no consumer would
leak L1 read-locks and pin memory. Therefore predictions are surfaced through
`predict_prefetch_keys(...)` so a caller can issue them through the **normal,
lock-owned** `submit_prefetch_request(...)` path, where the existing
retained-bitmap lifecycle releases them correctly.

The natural follow-up is a thin driver (e.g. in the MP connector or storage
manager) that, on a real request, calls `predict_prefetch_keys()` and submits a
capped speculative prefetch — loaded as **temporary** (non-retained) so eviction
can reclaim it — turning the prediction into an actual latency win. That driver,
and an end-to-end benchmark of hit-rate vs. wasted bandwidth, are intentionally
left to a separate PR so they can be validated against a running engine.

## Prediction signal options (future work)

The current predictor keys off raw key-access order. Stronger signals to layer
in later, mirroring the systems surveyed alongside CXL-SpecKV (PCR, KVFlow,
PRESERVE): explicit multi-turn session/`cache_salt` continuation, RAG document
co-occurrence graphs, and prefix-tree popularity.

## Testing

`tests/v1/distributed/test_speculative_prefetcher.py` covers the predictor's
contract directly (Markov learning, context exclusion, no self-transitions,
popularity fallback, recency decay, confidence filtering, bounded memory,
deterministic ordering, and arbitrary hashable keys). Because the predictor is
dependency-free, these run without torch.
