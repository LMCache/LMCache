# SPDX-License-Identifier: Apache-2.0
"""Probe the exact FlashAttention 2 kernel that rejected cu132 PTX in CI."""

# Standard
import argparse
import ctypes

# Third Party
from vllm.vllm_flash_attn import flash_attn_varlen_func
import torch


def main() -> None:
    """Report the CUDA driver API and verify a numerical FA2 kernel result.

    Raises:
        RuntimeError: If the driver query or CUDA execution fails. The optional
            diagnostic flag permits only the known unsupported PTX error.
        AssertionError: If FlashAttention returns an incorrect result.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-unsupported-ptx",
        action="store_true",
        help="Allow the known PTX error when diagnosing the unconfigured driver.",
    )
    args = parser.parse_args()
    driver_version = ctypes.c_int()
    driver = ctypes.CDLL("libcuda.so.1")
    status = driver.cuDriverGetVersion(ctypes.byref(driver_version))
    if status != 0:
        raise RuntimeError(f"cuDriverGetVersion failed with CUDA error {status}")
    print("CUDA driver API:", driver_version.value, flush=True)
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    qkv = torch.ones((16, 4, 64), dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([0, 16], dtype=torch.int32, device="cuda")
    try:
        result = flash_attn_varlen_func(
            qkv,
            qkv,
            qkv,
            max_seqlen_q=16,
            cu_seqlens_q=lengths,
            max_seqlen_k=16,
            cu_seqlens_k=lengths,
            causal=True,
            fa_version=2,
        )
        torch.cuda.synchronize()
    except RuntimeError as exc:
        if not args.allow_unsupported_ptx or "unsupported toolchain" not in str(exc):
            raise
        print("Reproduced the original unsupported PTX toolchain error", flush=True)
        return
    torch.testing.assert_close(result, qkv, rtol=0.02, atol=0.02)
    print("FlashAttention 2 CUDA kernel and numerical check passed", flush=True)


if __name__ == "__main__":
    main()
