# CUDA Engine-Driven Store Transfer Benchmark

## Workload

- Hardware: 4x NVIDIA L20
- Model: Qwen3-0.6B
- Requests: 200
- Input length: 10,000 tokens per request
- Output length: 1 token per request
- L1 cache: 80 GiB shared memory
- Store path: engine-driven, two async store workers
- vLLM chunked prefill: disabled
- Seed: `20260902`

## Results

All timing values are seconds per store. `P50` and `P95` are percentiles over
the recorded store operations.

| Path | Queue wait mean / P50 / P95 | Worker pre-copy mean / P50 / P95 | D2H enqueue mean / P50 / P95 | D2H wait mean / P50 / P95 | D2H total mean / P50 / P95 | Staging-to-SHM mean / P50 / P95 | Throughput / mean TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| C++ ops -> pinned SHM | 0.624 / 0.216 / 1.152s | 0.013 / 0.010 / 0.020s | 0.440 / 0.124 / 0.924s | 0.075 / 0.074 / 0.090s | **0.514 / 0.180 / 1.006s** | - | 35,731 tok/s / 28.94s |
| Torch fallback -> pinned SHM | 0.916 / 0.961 / 1.296s | 0.010 / 0.007 / 0.021s | 0.326 / 0.257 / 0.582s | 0.133 / 0.075 / 0.433s | **0.459 / 0.372 / 0.694s** | - | 39,572 tok/s / 26.75s |
| Torch fallback -> direct unpinned SHM | 36.390 / 19.303 / 80.153s | 0.024 / 0.020 / 0.042s | 2.293 / 1.085 / 4.840s | 0.003 / 0.000 / 0.007s | **2.296 / 1.085 / 4.840s** | - | 9,855 tok/s / 114.56s |
| Torch fallback -> pinned staging -> unpinned SHM | 1.416 / 1.263 / 2.699s | 0.018 / 0.009 / 0.017s | 0.221 / 0.198 / 0.504s | 0.114 / 0.090 / 0.388s | **0.334 / 0.280 / 0.636s** | **0.134 / 0.043 / 0.399s** | 39,304 tok/s / 26.77s |

The staging-to-SHM measurement contains 7,802 individual CPU copies:

| CPU copy metric | Value |
|---|---:|
| Mean | 3.463 ms |
| P50 | 1.097 ms |
| P95 | 13.454 ms |
| Maximum | 70.160 ms |

## Transport Verification

| Path | Block-transfer mode | SHM pinned |
|---|---|---|
| C++ ops -> pinned SHM | `ptr` | `True` |
| Torch fallback -> pinned SHM | `tensor` | `True` |
| Torch fallback -> direct unpinned SHM | `tensor` | `False` |
| Torch fallback -> pinned staging -> unpinned SHM | `tensor` | `False` |

## Timing Definitions

- **Queue wait**: Time from submitting a store task until an async store worker
  begins processing it.
- **Worker pre-copy**: Worker time before calling the paged-KV gather/copy
  operation, including `prepare_store`, output-buffer selection, and staging
  view allocation.
- **D2H enqueue**: Time spent submitting paged-KV gather and device-to-host
  work to the copy stream.
- **D2H wait**: Time in `gather_done.synchronize()` after the event recorded
  on the copy stream.
- **D2H total**: `D2H enqueue + D2H wait`; it measures completed GPU-to-host
  paged-KV transfer work, but includes paged gather and layout packing and is
  not a pure PCIe DMA-bandwidth measurement.
- **Staging-to-SHM**: CPU copies from pinned staging buffers to unpinned shared
  memory after D2H completion.

