# SPDX-License-Identifier: Apache-2.0
# SPDX-License-Identifier: Apache-2.0
"""Probe the exact FlashAttention 2 kernel that rejected cu132 PTX in CI."""

# Standard
import argparse
import ctypes

# Third Party
from vllm.vllm_flash_attn import flash_attn_varlen_func
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-unsupported-ptx", action="store_true")
    args = parser.parse_args()
    driver_version = ctypes.c_int()
    driver = ctypes.CDLL("libcuda.so.1")
    assert driver.cuDriverGetVersion(ctypes.byref(driver_version)) == 0
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
