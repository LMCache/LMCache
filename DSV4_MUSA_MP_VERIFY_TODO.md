# DeepSeek V4 MUSA MP Verification TODO

This checklist is for a real MUSA host. The current CPU tests validate the
layout algorithm, but they do not validate TorchMUSA pointer reconstruction,
device copies, or the full multiprocess serving path.

## 1. Host And Build

- [ ] Record the host, driver, MUSA runtime, Python, PyTorch, TorchMUSA,
  vLLM, and LMCache revisions.
- [ ] Confirm that the host has a supported MUSA device and that all worker
  processes can see it.

```bash
python - <<'PY'
import torch

assert hasattr(torch, "musa"), "TorchMUSA is not installed"
assert torch.musa.is_available(), "MUSA is not available"
print("torch:", torch.__version__)
print("musa device count:", torch.musa.device_count())
print("musa device 0:", torch.musa.get_device_name(0))
PY
```

- [ ] Install this branch in the same environment used by vLLM. If the
  package build would select CUDA extensions on the MUSA host, use the
  MUSA-supported source-only/native-extension configuration documented by
  that host image.
- [ ] Confirm that `lmcache` reports `musa` as its device and that importing
  `lmcache.v1.platform.devices.musa.device_ops` succeeds.

## 2. Focused MUSA Tests

- [ ] Run the MUSA adapter tests on the real host:

```bash
python -m pytest -q \
  tests/v1/platform/devices/musa/test_musa_mp_block_transfer.py \
  tests/v1/platform/devices/musa/test_pointer_transfer.py \
  tests/v1/platform/devices/musa/test_format_capabilities.py
```

- [ ] Confirm that the real-device tests are not all skipped. The CPU-only
  DSV4 fallback test is useful for layout regression, but it is not evidence
  of a MUSA transfer.
- [ ] Confirm that the MUSA pointer test reconstructs each
  `NL_X_NB_BSV_BSS` page as `[NB, block_stride_elems]` with stride
  `(block_stride_elems, 1)`, and each staging object as
  `[NL, tokens, 132]`.
- [ ] Run an actual device round trip with at least:
  - `BS=64`, `row_width=132`;
  - a tight block stride (`8448` bytes);
  - a padded block stride greater than `8448`;
  - source block and destination block with different in-block offsets;
  - more than one layer and more than one LMCache object.
- [ ] Verify byte-exact equality for both the 128-byte value region and the
  4-byte scale region. Do not only compare the logical `[NB, BS, 132]` view,
  because it hides the blocked physical ordering.

## 3. MP Pointer Path

- [ ] Start the LMCache server in MP mode with
  `LMCACHE_MP_TRANSFER_MODE=lmcache_driven` and the MUSA device affinity
  variable required by the host image.
- [ ] Start the MUSA vLLM worker with a DSV4 model and enable the LMCache
  v1 multiprocess connector. Use a small reproducible prompt first.
- [ ] Capture the LMCache and worker logs and confirm:
  - the KV format is `NL_X_NB_BSV_BSS`;
  - the group shape contains `BS=64`, `HS=132`, `kv_size=1`, and the actual
    `block_stride_elems`;
  - the MUSA MP capability predicate accepts the format;
  - the native MUSA ABI is not selected for this format;
  - the torch fallback handles the D2H store and H2D retrieve.
- [ ] Send the same deterministic request twice, resetting the engine-local
  prefix cache between requests while keeping LMCache data. Compare the
  generated output and verify that the second request has a non-zero
  LMCache retrieve.
- [ ] Repeat with a prompt that crosses a block boundary and with a retrieve
  whose first requested block has a non-zero prefix skip.
- [ ] Repeat with at least two tensor-parallel workers if the MUSA vLLM
  integration supports DSV4 TP on the target host.

The existing
`.buildkite/k3_tests/multiprocess/scripts/workloads/vllm/dsv4-flash-tp.sh`
is an NVIDIA/CUDA DeepSeek-V4-Flash workload. Do not treat a skipped or
CUDA-only run of that script as MUSA evidence. Reuse its two-request
correctness structure only after the vLLM MUSA launcher and device affinity
are confirmed.

## 4. Failure Collection

- [ ] Save the exact command lines, environment variables, revisions, and
  model configuration.
- [ ] Save LMCache server logs, all worker logs, and the pytest output.
- [ ] If the result fails, record whether it failed during pointer
  reconstruction, capability selection, staging allocation, D2H store, H2D
  retrieve, or model output comparison.
- [ ] Record whether native transfer was disabled as expected. Do not enable
  an unimplemented native blocked-scale ABI to make the test pass.

## Acceptance

The MUSA MP verification is complete only when the focused tests execute on a
real MUSA device, the pointer-path round trip is byte-exact for values and
scales, and the two-request DSV4 serving test proves a non-vacuous LMCache
retrieve with identical deterministic output.
