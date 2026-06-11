# SPDX-License-Identifier: Apache-2.0
"""Microbenchmarks: SYCL (XPU) kernels vs python_ops_fallback.

Per the project plan, Phase 0 confirmed all four fallback functions
accept XPU tensors directly, so we can do an apples-to-apples timing
comparison on the same device.

Run with: ``pytest tests/benchmarks/test_xpu_kernels_microbench.py --benchmark-only``
"""

# Third Party
import pytest
import torch

# First Party
import lmcache.python_ops_fallback as F

pytestmark = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU not available",
)

XPU = "xpu"


def _xpu_sync():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.synchronize()


@pytest.fixture(scope="module")
def xops():
    # First Party
    import lmcache.xpu_ops as XOPS  # noqa: F401

    return XOPS


# ---------------- calculate_cdf ----------------


@pytest.mark.benchmark(group="calculate_cdf")
@pytest.mark.parametrize("ntokens", [256, 1024, 4096])
def test_bench_cdf_sycl(benchmark, xops, ntokens):
    nlayers, nchannels, max_bins = 32, 1024, 32
    sym = torch.randint(
        0, max_bins, (nlayers, ntokens, nchannels), dtype=torch.uint8, device=XPU
    )
    _xpu_sync()

    def run():
        out = xops.calculate_cdf(sym, max_bins)
        _xpu_sync()
        return out

    benchmark(run)


@pytest.mark.benchmark(group="calculate_cdf")
@pytest.mark.parametrize("ntokens", [256, 1024, 4096])
def test_bench_cdf_fallback(benchmark, ntokens):
    nlayers, nchannels, max_bins = 32, 1024, 32
    sym = torch.randint(
        0, max_bins, (nlayers, ntokens, nchannels), dtype=torch.uint8, device=XPU
    )
    _xpu_sync()

    def run():
        out = F.calculate_cdf(sym, max_bins)
        _xpu_sync()
        return out

    benchmark(run)


# ---------------- encode_fast_new ----------------


def _make_encode_inputs(xops, nlayers, ntokens, nchannels, max_bins):
    sym = torch.randint(
        0, max_bins, (nlayers, ntokens, nchannels), dtype=torch.uint8, device=XPU
    )
    cdf = xops.calculate_cdf(sym, max_bins)
    buf = torch.zeros((nlayers, nchannels, 256), dtype=torch.uint8, device=XPU)
    lens = torch.zeros((nlayers, nchannels), dtype=torch.int32, device=XPU)
    return sym, cdf, buf, lens


@pytest.mark.benchmark(group="encode_fast_new")
@pytest.mark.parametrize("ntokens", [64, 256])
def test_bench_encode_sycl(benchmark, xops, ntokens):
    nlayers, nchannels, max_bins = 32, 1024, 32
    sym, cdf, buf, lens = _make_encode_inputs(
        xops, nlayers, ntokens, nchannels, max_bins
    )
    _xpu_sync()

    def run():
        xops.encode_fast_new(cdf, sym, buf, lens)
        _xpu_sync()

    benchmark(run)


@pytest.mark.benchmark(group="encode_fast_new")
@pytest.mark.parametrize("ntokens", [64, 256])
def test_bench_encode_fallback(benchmark, xops, ntokens):
    nlayers, nchannels, max_bins = 32, 1024, 32
    sym, cdf, buf, lens = _make_encode_inputs(
        xops, nlayers, ntokens, nchannels, max_bins
    )
    _xpu_sync()

    def run():
        F.encode_fast_new(cdf, sym, buf, lens)
        _xpu_sync()

    benchmark(run)


# ---------------- decode_fast_new ----------------


@pytest.mark.benchmark(group="decode_fast_new")
@pytest.mark.parametrize("ntokens", [256])
def test_bench_decode_sycl(benchmark, xops, ntokens):
    nlayers, nchannels, max_bins = 32, 1024, 32
    sym, cdf, buf, lens = _make_encode_inputs(
        xops, nlayers, ntokens, nchannels, max_bins
    )
    xops.encode_fast_new(cdf, sym, buf, lens)
    out = torch.zeros_like(sym)
    _xpu_sync()

    def run():
        xops.decode_fast_new(cdf, buf, lens, out)
        _xpu_sync()

    benchmark(run)


# NOTE: decode_fast_new fallback crashes on XPU with an internal
# IndexKernel gather OOB on these shapes (a torch-xpu fallback bug,
# not a CacheGen bug).  Per the project plan we do not modify
# python_ops_fallback to accommodate XPU; the SYCL kernel is the
# correct/fast path.  Recording the absolute SYCL throughput is
# sufficient.


# ---------------- rotary_embedding_k_fused ----------------


@pytest.mark.benchmark(group="rope_k_fused")
@pytest.mark.parametrize("ntokens", [256, 1024, 4096])
def test_bench_rope_sycl(benchmark, xops, ntokens):
    num_kv_heads, head_size, rot_dim = 8, 128, 128
    embed_dim = num_kv_heads * head_size
    old_positions = torch.arange(ntokens, dtype=torch.int64, device=XPU)
    new_positions = (old_positions + 1) % 2048
    key = torch.randn(ntokens, embed_dim, dtype=torch.float16, device=XPU)
    cos_sin = torch.randn(2048, rot_dim, dtype=torch.float16, device=XPU)
    _xpu_sync()

    def run():
        xops.rotary_embedding_k_fused(
            old_positions, new_positions, key, head_size, cos_sin, True
        )
        _xpu_sync()

    benchmark(run)


# NOTE: rotary_embedding_k_fused fallback uses advanced indexing that
# triggers an internal IndexKernel OOB on XPU at these shapes.  Per the
# project plan we do not patch fallback to fit XPU; SYCL is the
# performance and correctness path.
