# SPDX-License-Identifier: Apache-2.0
"""Build-only bridge for the opt-in DAX-Coordinated L1 extension.

PR1 temporarily subclasses the L2-oriented ``StorageBackendProfile`` only to
reuse its additive native-extension discovery and ``BUILD_WITH_*`` handling.
This does not classify DAX-Coordinated L1 as an L2 runtime storage backend.
It is never auto-detected: operators must explicitly select it with
``BUILD_WITH_DAX_COORDINATED_L1=1`` on Linux x86-64.
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING
import platform

if TYPE_CHECKING:
    # Third Party
    from setuptools.extension import Extension

# First Party
from setup_extensions.storage_backend_profiles import StorageBackendProfile

ROOT_DIR = Path(__file__).resolve().parents[2]
DAX_COORDINATED_L1_SOURCES = (
    "csrc/dax_coordinated_l1/index_core.cpp",
    "csrc/dax_coordinated_l1/peterson.cpp",
    "csrc/dax_coordinated_l1/pybind.cpp",
    "csrc/dax_coordinated_l1/visibility_x86.cpp",
)
_SUPPORTED_SYSTEMS = frozenset({"linux"})
_SUPPORTED_MACHINES = frozenset({"x86_64", "amd64"})


def is_dax_coordinated_l1_build_host_supported(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> bool:
    """Return whether the build host supports the x86 visibility code."""
    current_system = (system or platform.system()).lower()
    current_machine = (machine or platform.machine()).lower()
    return (
        current_system in _SUPPORTED_SYSTEMS and current_machine in _SUPPORTED_MACHINES
    )


class DaxCoordinatedL1BuildProfile(StorageBackendProfile):
    """Temporary build-discovery adapter for the Device-DAX L1 extension."""

    name = "dax_coordinated_l1"
    env_var = "BUILD_WITH_DAX_COORDINATED_L1"

    def detect(self) -> bool:
        """Do not infer a hardware-qualified runtime setup at build time."""
        return False

    def build(self, extra_cxx_flags: list[str]) -> list["Extension"]:
        """Build the explicitly selected DAX-Coordinated L1 extension."""
        if not is_dax_coordinated_l1_build_host_supported():
            raise RuntimeError(
                f"{self.env_var}=1 requires a Linux x86-64 build host; "
                f"got {platform.system()} {platform.machine()}"
            )

        # Third Party
        from torch.utils import cpp_extension

        return [
            cpp_extension.CppExtension(
                "lmcache.lmcache_dax_coordinated_l1",
                sources=list(DAX_COORDINATED_L1_SOURCES),
                include_dirs=[str(ROOT_DIR / "csrc")],
                extra_compile_args={
                    "cxx": extra_cxx_flags + ["-O3", "-std=c++17"],
                },
            ),
        ]
