# SPDX-License-Identifier: Apache-2.0
"""Validate an installed LMCache wheel against the ROCm torch 2.10 ABI."""

# Standard
import importlib.util
import os

# Third Party
import torch

# First Party
import lmcache.cuda_ops


def _validate_atom_integration() -> bool:
    """Validate ATOM adapters when the installed LMCache version provides them."""
    if importlib.util.find_spec("lmcache.integration.atom") is None:
        return False

    # First Party
    from lmcache.integration.atom import (  # noqa: PLC0415
        AtomMPSchedulerAdapter,
        AtomMPWorkerAdapter,
    )

    assert AtomMPSchedulerAdapter.__module__.startswith("lmcache.integration.atom")
    assert AtomMPWorkerAdapter.__module__.startswith("lmcache.integration.atom")
    return True


def _validate_gpu_kernels() -> str:
    """Run small LMCache D2H and H2D HIP kernel transfers."""
    blocks, block_size, heads, head_size = 2, 16, 2, 8
    elements = blocks * block_size * heads * head_size
    key = torch.arange(elements, dtype=torch.float32, device="cuda").reshape(
        blocks, block_size, heads, head_size
    )
    value = key + 10_000
    slots = torch.tensor([0, 3, 17, 31], dtype=torch.int64, device="cuda")
    host = torch.empty(
        (2, 1, len(slots), heads * head_size),
        dtype=torch.float32,
        pin_memory=True,
    )

    lmcache.cuda_ops.load_and_reshape_flash(host, key, value, slots, 0)
    torch.cuda.synchronize()
    expected_key = (
        key.reshape(-1, heads, head_size)[slots].cpu().reshape(len(slots), -1)
    )
    expected_value = (
        value.reshape(-1, heads, head_size)[slots].cpu().reshape(len(slots), -1)
    )
    assert torch.equal(host[0, 0], expected_key)
    assert torch.equal(host[1, 0], expected_value)

    key_back = torch.zeros_like(key)
    value_back = torch.zeros_like(value)
    lmcache.cuda_ops.reshape_and_cache_back_flash(host, key_back, value_back, slots, 0)
    torch.cuda.synchronize()
    assert torch.equal(
        key_back.reshape(-1, heads, head_size)[slots],
        key.reshape(-1, heads, head_size)[slots],
    )
    assert torch.equal(
        value_back.reshape(-1, heads, head_size)[slots],
        value.reshape(-1, heads, head_size)[slots],
    )

    properties = torch.cuda.get_device_properties(0)
    arch = getattr(properties, "gcnArchName", "unknown").split(":", 1)[0]
    assert arch in {"gfx942", "gfx950"}, arch
    return f"{properties.name} {arch}"


def main() -> None:
    """Validate native extension loading, adapters, and optional GPU access."""
    torch_prefix = os.environ.get("TORCH_VERSION_PREFIX", "2.10.")
    rocm_prefix = os.environ.get("ROCM_VERSION_PREFIX", "7.2")

    assert torch.__version__.startswith(torch_prefix), torch.__version__
    assert torch.version.hip is not None, "installed torch is not a ROCm build"
    assert torch.version.hip.startswith(rocm_prefix), torch.version.hip
    atom_available = _validate_atom_integration()

    if os.environ.get("LMCACHE_ROCM_REQUIRE_GPU", "0") == "1":
        assert torch.cuda.is_available(), "ROCm GPU is not available"
        print("ROCm GPU and LMCache kernels:", _validate_gpu_kernels())

    print(
        "OK: ROCm torch 2.10 wheel ABI;",
        "ATOM adapters",
        "available" if atom_available else "not in this release;",
        "torch",
        torch.__version__,
        "hip",
        torch.version.hip,
        "cuda_ops",
        lmcache.cuda_ops.__file__,
    )


if __name__ == "__main__":
    main()
