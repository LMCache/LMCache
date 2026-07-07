# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import subprocess
import sys
import textwrap

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_import_script(script: str) -> None:
    """Run an import check in a fresh Python subprocess.

    Args:
        script: Python source code to execute with ``python -c``.

    Raises:
        AssertionError: If the subprocess exits with a non-zero status.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_memory_allocators_package_import_is_lazy() -> None:
    """Importing the allocator package does not eagerly import submodules."""
    _run_import_script(
        """
        import sys

        import lmcache.v1.memory_allocators as allocators

        allocator_modules = {
            "ad_hoc_memory_allocator",
            "buffer_allocator",
            "cu_file_memory_allocator",
            "devdax_memory_allocator",
            "gpu_memory_allocator",
            "hip_file_memory_allocator",
            "host_memory_allocator",
            "lazy_memory_allocator",
            "mixed_memory_allocator",
            "paged_cpu_gpu_memory_allocator",
            "paged_tensor_memory_allocator",
            "pin_memory_allocator",
            "tensor_memory_allocator",
        }
        loaded = sorted(
            name
            for name in sys.modules
            if name.startswith("lmcache.v1.memory_allocators.")
            and name.rsplit(".", 1)[-1] in allocator_modules
        )
        assert loaded == [], loaded
        assert "LazyMemoryAllocator" in allocators.__all__
        assert "TensorMemoryAllocator" in allocators.__all__
        """
    )


def test_allocator_submodule_first_import_preserves_old_surface() -> None:
    """Importing concrete allocator modules first does not create a cycle."""
    _run_import_script(
        """
        from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
            PagedAddressManager,
            PagedTensorMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.tensor_memory_allocator import (
            TensorMemoryAllocator,
        )
        from lmcache.v1.memory_management import (
            PagedAddressManager as OldPagedAddressManager,
            PagedTensorMemoryAllocator as OldPagedTensorMemoryAllocator,
            TensorMemoryAllocator as OldTensorMemoryAllocator,
        )

        assert PagedAddressManager is OldPagedAddressManager
        assert PagedTensorMemoryAllocator is OldPagedTensorMemoryAllocator
        assert TensorMemoryAllocator is OldTensorMemoryAllocator
        """
    )


def test_allocator_package_first_import_preserves_old_surface() -> None:
    """Package-level lazy exports match the old memory_management exports."""
    _run_import_script(
        """
        from lmcache.v1.memory_allocators import (
            DevDaxMemoryAllocator,
            TensorMemoryAllocator,
        )
        from lmcache.v1.memory_management import (
            DevDaxMemoryAllocator as OldDevDaxMemoryAllocator,
            TensorMemoryAllocator as OldTensorMemoryAllocator,
        )

        assert DevDaxMemoryAllocator is OldDevDaxMemoryAllocator
        assert TensorMemoryAllocator is OldTensorMemoryAllocator
        """
    )


def test_lazy_allocator_new_and_old_import_paths_match() -> None:
    """The moved lazy allocator remains available from both import paths."""
    _run_import_script(
        """
        from lmcache.v1.lazy_memory_allocator import (
            LazyMemoryAllocator as OldLazyMemoryAllocator,
        )
        from lmcache.v1.memory_allocators import (
            LazyMemoryAllocator as PackageLazyMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.lazy_memory_allocator import (
            LazyMemoryAllocator,
        )

        assert LazyMemoryAllocator is OldLazyMemoryAllocator
        assert LazyMemoryAllocator is PackageLazyMemoryAllocator
        """
    )


def test_memory_management_all_keeps_allocator_compatibility_names() -> None:
    """The old memory_management import surface remains listed in __all__."""
    _run_import_script(
        """
        import lmcache.v1.memory_management as memory_management

        expected = {
            "AdHocMemoryAllocator",
            "AddressManager",
            "BufferAllocator",
            "CuFileMemoryAllocator",
            "DevDaxMemoryAllocator",
            "GPUMemoryAllocator",
            "HipFileMemoryAllocator",
            "HostMemoryAllocator",
            "MemoryAllocatorInterface",
            "MemoryFormat",
            "MemoryObj",
            "MemoryObjMetadata",
            "MixedMemoryAllocator",
            "PagedAddressManager",
            "PagedCpuGpuMemoryAllocator",
            "PagedTensorMemoryAllocator",
            "PinMemoryAllocator",
            "TensorMemoryAllocator",
            "TensorMemoryObj",
        }
        missing = expected - set(memory_management.__all__)
        assert missing == set(), missing
        assert memory_management.PagedAddressManager.__name__ == "PagedAddressManager"
        """
    )
