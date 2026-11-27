# SPDX-License-Identifier: Apache-2.0
"""Validate the exact AMD ROCm 7.2.4 / torch 2.10 build ABI."""

# Standard
from importlib.metadata import distribution
import argparse
import importlib.util
import os
import sys

# Third Party
import torch

DEFAULT_TORCH_VERSION = "2.10.0+rocm7.2.4.git3d3aa833"
DEFAULT_TORCH_GIT_VERSION = "3d3aa833db84eed6b7f5595cb5f162c2f78300a4"
DEFAULT_HIP_VERSION = "7.2.53211"
DEFAULT_ROCM_VERSION = "7.2.4"
DEFAULT_PYTHON_ABI = "cp312"
DEFAULT_CXX11_ABI = "1"
DEFAULT_WHEEL_TAG = "cp312-cp312-manylinux_2_39_x86_64"


def _required_value(name: str, default: str) -> str:
    """Return an expected ABI value, allowing the workflow to override it."""
    return os.environ.get(name, default)


def _validate_runtime_abi() -> None:
    """Require the exact torch, ROCm, Python, and libstdc++ ABI tuple."""
    expected_torch = _required_value("EXPECTED_TORCH_VERSION", DEFAULT_TORCH_VERSION)
    expected_git = _required_value(
        "EXPECTED_TORCH_GIT_VERSION", DEFAULT_TORCH_GIT_VERSION
    )
    expected_hip = _required_value("EXPECTED_HIP_VERSION", DEFAULT_HIP_VERSION)
    expected_rocm = _required_value("EXPECTED_ROCM_VERSION", DEFAULT_ROCM_VERSION)
    expected_python_abi = _required_value("EXPECTED_PYTHON_ABI", DEFAULT_PYTHON_ABI)
    expected_cxx11_abi = _required_value("EXPECTED_CXX11_ABI", DEFAULT_CXX11_ABI)

    actual_python_abi = f"cp{sys.version_info.major}{sys.version_info.minor}"
    actual_cxx11_abi = str(int(torch._C._GLIBCXX_USE_CXX11_ABI))
    assert torch.__version__ == expected_torch, (torch.__version__, expected_torch)
    assert torch.version.git_version == expected_git, (
        torch.version.git_version,
        expected_git,
    )
    assert torch.version.hip == expected_hip, (torch.version.hip, expected_hip)
    assert torch.version.rocm == expected_rocm, (torch.version.rocm, expected_rocm)
    assert actual_python_abi == expected_python_abi, (
        actual_python_abi,
        expected_python_abi,
    )
    assert actual_cxx11_abi == expected_cxx11_abi, (
        actual_cxx11_abi,
        expected_cxx11_abi,
    )

    print(
        "ROCm torch 2.10 ABI:",
        "torch",
        torch.__version__,
        "torch_git",
        torch.version.git_version,
        "hip",
        torch.version.hip,
        "rocm",
        torch.version.rocm,
        "python_abi",
        actual_python_abi,
        "cxx11abi",
        actual_cxx11_abi,
    )


def _validate_installed_wheel_tag() -> None:
    """Require the installed wheel to advertise the supported Python/platform ABI."""
    expected_tag = _required_value("EXPECTED_WHEEL_TAG", DEFAULT_WHEEL_TAG)
    wheel_metadata = distribution("lmcache").read_text("WHEEL")
    assert wheel_metadata is not None, "installed lmcache has no WHEEL metadata"
    assert f"Tag: {expected_tag}" in wheel_metadata, wheel_metadata


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


def _validate_gpu_kernels() -> None:
    """Run small LMCache D2H and H2D HIP kernel transfers."""
    # First Party
    from lmcache.v1.common.utils import make_page_buffer_shape_desc
    from lmcache.v1.platform.native_block_transfer import transfer_with_address_kernel
    import lmcache.cuda_ops  # noqa: PLC0415
    import lmcache.lmcache_native as native

    blocks, block_size, heads, head_size = 2, 16, 2, 8
    elements = 2 * blocks * block_size * heads * head_size
    pages = torch.arange(elements, dtype=torch.float32, device="cuda").reshape(
        2, blocks, block_size, heads, head_size
    )
    fmt = native.EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    desc = make_page_buffer_shape_desc([pages], fmt, 0, 1, blocks, block_size)
    host = torch.empty(
        (2, 1, block_size, heads, head_size), pin_memory=True, dtype=pages.dtype
    )

    def transfer(
        pool: torch.Tensor, block: int, direction: native.TransferDirection
    ) -> None:
        transfer_with_address_kernel(
            lmcache.cuda_ops.multi_layer_block_kv_transfer,
            [pool],
            [host],
            [block],
            pool.device,
            direction,
            desc,
            block_size,
            fmt,
            0,
        )
        torch.cuda.synchronize()

    transfer(pages, 1, native.TransferDirection.D2H)
    assert torch.equal(host[:, 0], pages[:, 1].cpu())
    restored = torch.zeros_like(pages)
    transfer(restored, 0, native.TransferDirection.H2D)
    assert torch.equal(restored[:, 0], pages[:, 1])
    assert torch.count_nonzero(restored[:, 1]).item() == 0


def main() -> None:
    """Validate native extension loading, adapters, and optional GPU access."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runtime-only",
        action="store_true",
        help="validate the base environment before LMCache is installed",
    )
    args = parser.parse_args()

    _validate_runtime_abi()
    if args.runtime_only:
        return

    # First Party
    import lmcache.cuda_ops  # noqa: PLC0415

    _validate_installed_wheel_tag()
    atom_available = _validate_atom_integration()

    if os.environ.get("LMCACHE_ROCM_REQUIRE_GPU", "0") == "1":
        assert torch.cuda.is_available(), "ROCm GPU is not available"
        properties = torch.cuda.get_device_properties(0)
        arch = getattr(properties, "gcnArchName", "unknown").split(":", 1)[0]
        assert arch in {"gfx942", "gfx950"}, arch
        _validate_gpu_kernels()
        print("ROCm GPU and LMCache kernels:", properties.name, arch)

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
