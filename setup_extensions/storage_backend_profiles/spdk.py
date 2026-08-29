# SPDX-License-Identifier: Apache-2.0
"""SPDK storage backend build profile.

Builds the ``liblmcache_spdk.so`` shared library containing the SPDK C++
implementation for NVMe-over-Fabrics I/O with hugepage memory support.

The library is built via CMake and installed alongside the Python package.

Environment variables:
    SPDK_ROOT          - SPDK installation directory (default: /opt/spdk)
    DPDK_ROOT          - DPDK build directory (default: $SPDK_ROOT/dpdk)
    LMCACHE_SPDK_DIR   - Pre-built SPDK library directory (for skip-build)
    SKIP_SPDK_BUILD    - Set to "1" to skip SPDK library build entirely
"""

# Standard
from pathlib import Path
from typing import TYPE_CHECKING, Optional
import os
import subprocess
import sys

if TYPE_CHECKING:
    # Third Party
    pass


# First Party
from setup_extensions.build_profiles import BuildProfile


class SPDKProfile(BuildProfile):
    """SPDK storage backend build profile."""

    name = "spdk"
    env_var = "BUILD_WITH_SPDK"

    def __init__(self) -> None:
        """Initialize SPDK profile with path configuration."""
        self._spdk_root: Optional[str] = None
        self._dpdk_root: Optional[str] = None
        self._library_path: Optional[str] = None

    def detect(self) -> bool:
        """Detect SPDK by locating SPDK and DPDK directories.

        Checks for SPDK build/lib and DPDK build/lib directories.

        Returns:
            True if SPDK and DPDK are found, False otherwise.
        """
        spdk_root = os.environ.get("SPDK_ROOT", "/opt/spdk")
        dpdk_root = os.environ.get("DPDK_ROOT", os.path.join(spdk_root, "dpdk"))

        # Check for SPDK build artifacts
        spdk_lib_dir = os.path.join(spdk_root, "build", "lib")
        dpdk_lib_dir = os.path.join(dpdk_root, "build", "lib")
        spdk_include_dir = os.path.join(spdk_root, "build", "include")
        dpdk_include_dir = os.path.join(dpdk_root, "build", "include")

        if not os.path.isdir(spdk_lib_dir):
            return False
        if not os.path.isdir(dpdk_lib_dir):
            return False
        if not os.path.isdir(spdk_include_dir):
            return False
        if not os.path.isdir(dpdk_include_dir):
            return False

        self._spdk_root = spdk_root
        self._dpdk_root = dpdk_root
        return True

    def is_explicitly_requested(self) -> bool:
        """Check if SPDK build was explicitly requested."""
        if os.environ.get("SKIP_SPDK_BUILD") == "1":
            return False
        return super().is_explicitly_requested()

    def build(
        self,
        extra_cxx_flags: Optional[list[str]] = None,
    ) -> tuple[list, dict]:
        """Build the SPDK shared library via CMake.

        Args:
            extra_cxx_flags: Additional C++ flags (ignored for SPDK).

        Returns:
            Empty lists (SPDK is built as a standalone shared library,
            not a Python extension).
        """
        if os.environ.get("SKIP_SPDK_BUILD") == "1":
            print("Skipping SPDK library build (SKIP_SPDK_BUILD=1)")
            return [], {}

        # Check for pre-built library
        prebuilt_dir = os.environ.get("LMCACHE_SPDK_DIR")
        if prebuilt_dir and os.path.isfile(
            os.path.join(prebuilt_dir, "liblmcache_spdk.so")
        ):
            print(f"Using pre-built SPDK library from {prebuilt_dir}")
            self._library_path = os.path.join(prebuilt_dir, "liblmcache_spdk.so")
            return [], {}

        spdk_root = self._spdk_root or os.environ.get("SPDK_ROOT", "") or ""
        dpdk_root = self._dpdk_root or os.environ.get("DPDK_ROOT", "") or ""

        # Use defaults if empty strings
        if not spdk_root:
            spdk_root = "/opt/spdk"
        if not dpdk_root:
            dpdk_root = os.path.join(spdk_root, "dpdk")

        # Verify paths
        spdk_lib_dir = os.path.join(spdk_root, "build", "lib")
        dpdk_lib_dir = os.path.join(dpdk_root, "build", "lib")

        if not os.path.isdir(spdk_lib_dir):
            print(
                f"warning: SPDK library directory not found: {spdk_lib_dir}",
                file=sys.stderr,
            )
            print("Set SPDK_ROOT or skip with SKIP_SPDK_BUILD=1")
            return [], {}

        if not os.path.isdir(dpdk_lib_dir):
            print(
                f"warning: DPDK library directory not found: {dpdk_lib_dir}",
                file=sys.stderr,
            )
            print("Set DPDK_ROOT or skip with SKIP_SPDK_BUILD=1")
            return [], {}

        print(f"Building SPDK library (SPDK_ROOT={spdk_root}, DPDK_ROOT={dpdk_root})")

        # Build using CMake
        cmake_lists = (
            Path(__file__).parent.parent.parent
            / "csrc"
            / "storage_backends"
            / "raw_block"
            / "CMakeLists.txt"
        )
        if not cmake_lists.exists():
            print(
                f"warning: CMakeLists.txt not found at {cmake_lists}",
                file=sys.stderr,
            )
            return [], {}

        try:
            build_dir = cmake_lists.parent / "build_spdk"
            build_dir.mkdir(exist_ok=True)

            # Configure CMake
            cmake_cmd = [
                "cmake",
                str(cmake_lists.parent),
                "-DCMAKE_BUILD_TYPE=Release",
                f"-DSPDK_ROOT={spdk_root}",
                f"-DDPDK_ROOT={dpdk_root}",
            ]

            print(f"Running: {' '.join(cmake_cmd)}")
            subprocess.run(
                cmake_cmd,
                cwd=str(build_dir),
                check=True,
                capture_output=True,
                text=True,
            )

            # Build
            make_cmd = ["make", "-j"]
            print(f"Running: {' '.join(make_cmd)}")
            subprocess.run(
                make_cmd,
                cwd=str(build_dir),
                check=True,
                capture_output=True,
                text=True,
            )

            # Check for built library
            lib_path = build_dir / "liblmcache_spdk.so"
            if lib_path.exists():
                self._library_path = str(lib_path)
                print(f"SPDK library built successfully: {lib_path}")
            else:
                print("warning: SPDK library not found after build", file=sys.stderr)

        except subprocess.CalledProcessError as e:
            print(
                f"error: SPDK build failed: {e}",
                file=sys.stderr,
            )
            print(f"stdout: {e.stdout}" if e.stdout else "")
            print(f"stderr: {e.stderr}" if e.stderr else "")
            return [], {}
        except FileNotFoundError:
            print(
                "warning: cmake not found in PATH, skipping SPDK library build",
                file=sys.stderr,
            )
            return [], {}

        return [], {}

    def library_path(self) -> Optional[str]:
        """Return the path to the built SPDK library.

        Returns:
            Path to liblmcache_spdk.so, or None if not built.
        """
        if self._library_path:
            return self._library_path

        # Check common locations
        script_dir = Path(__file__).parent.parent.parent
        build_dir = (
            script_dir / "csrc" / "storage_backends" / "raw_block" / "build_spdk"
        )
        if (build_dir / "liblmcache_spdk.so").exists():
            return str(build_dir / "liblmcache_spdk.so")

        return None

    def default_cxx_flags(self) -> list[str]:
        """Return default C++ flags for SPDK builds."""
        flags = []
        if self._spdk_root:
            flags.append(f"-I{self._spdk_root}/build/include")
        if self._dpdk_root:
            flags.append(f"-I{self._dpdk_root}/build/include")
        return flags
