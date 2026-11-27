# Experimental MP Holemiss delivery evidence

The benchmark uses the real MP server, CUDA IPC event ordering and native
object-group copies. RETRIEVE may return terminal chunk coverage in addition
to its legacy boolean status. Indices refer to the original chunk range;
healthy chunks after a missing chunk retain their original GPU destinations.
The vLLM integration captures the scheduler transfer generation in broadcast
metadata and emits evidence only after the returned GPU completion event.
Unknown failures retain the conservative whole-transfer failure behavior.

The request-scoped fault is injected after L2 prefetch at the GPU-delivery
boundary. L1 is cleared between waves to require real CMX reads. This tests
loss between successful lookup/prefetch and GPU delivery; it is not a failed
NVMe GET or physical deletion. Exactly one chunk is withheld, with all other
chunks delivered normally. The injection is an external benchmark harness.

The measured native backend is the retained 128-bit-key/reactor extension
(SHA256 a635432b6a43e545743d5d770049f2c77c9b224d67ada1756055fce59e307a21).
The factory's reactor/B8/depth128 arguments require that extension; they are
not an ABI supported by an unpatched upstream NIXL client build. The MP
GPU/native transfer extensions were rebuilt against PyTorch 2.14/cu132.

Benchmark receipts and fault controls live in /dev/shm during measured work.
Metrics, event snapshots and output artifacts are persisted after timing.
Every fill wave drains L2 writes; an entire recovery pass precedes the timed
cohorts. Each timed wave requires one failed GPU delivery and 62 successful
chunk deliveries per request, and all repair transactions must commit.
