# SPDX-License-Identifier: Apache-2.0
"""Optional direct-libibverbs extension for the LMCache L1 transfer channel."""

# Standard
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile


class RdmaL1Profile(StorageBackendProfile):
    """Build the LMCache-owned RC RDMA transport without NIXL or Mooncake."""

    name = "rdma_l1"
    env_var = "BUILD_WITH_RDMA_L1"

    def detect(self) -> bool:
        """Keep the native transport opt-in even when rdma-core is installed."""
        return False

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        # Third Party
        from setuptools import Extension
        from torch.utils import cpp_extension

        return [
            Extension(
                "lmcache.rdma_l1_ops",
                sources=[
                    "csrc/rdma/pybind.cpp",
                    "csrc/rdma/rdma_transport.cpp",
                ],
                include_dirs=["csrc/rdma", *cpp_extension.include_paths()],
                libraries=["ibverbs"],
                extra_compile_args=extra_cxx_flags + ["-O3", "-std=c++17", "-pthread"],
                extra_link_args=["-pthread"],
                language="c++",
            )
        ]
