# SPDX-License-Identifier: Apache-2.0
"""Compatibility coverage for the pre-devices CUDA import path."""


def test_legacy_cuda_ipc_wrapper_import_reexports_current_classes() -> None:
    """Legacy CUDA imports should resolve to the current device namespace."""
    # First Party
    from lmcache.v1.platform.cuda.ipc_wrapper import (
        CudaIPCWrapper as LegacyCudaIPCWrapper,
    )
    from lmcache.v1.platform.cuda.ipc_wrapper import (
        RawCudaIPCWrapper as LegacyRawCudaIPCWrapper,
    )
    from lmcache.v1.platform.cuda.ipc_wrapper import (
        VmmCudaIPCWrapper as LegacyVmmCudaIPCWrapper,
    )
    from lmcache.v1.platform.devices.cuda.ipc_wrapper import (
        CudaIPCWrapper,
        RawCudaIPCWrapper,
        VmmCudaIPCWrapper,
    )

    assert LegacyCudaIPCWrapper is CudaIPCWrapper
    assert LegacyRawCudaIPCWrapper is RawCudaIPCWrapper
    assert LegacyVmmCudaIPCWrapper is VmmCudaIPCWrapper


def test_legacy_cuda_package_import_reexports_current_spec() -> None:
    """The legacy CUDA package should still expose CudaDeviceSpec."""
    # First Party
    from lmcache.v1.platform.cuda import CudaDeviceSpec as LegacyCudaDeviceSpec
    from lmcache.v1.platform.devices.cuda import CudaDeviceSpec

    assert LegacyCudaDeviceSpec is CudaDeviceSpec
