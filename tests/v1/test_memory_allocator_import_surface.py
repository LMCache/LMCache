# SPDX-License-Identifier: Apache-2.0

# Standard
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import textwrap


def _run_import_script(script: str) -> None:
    """Run an import check in a fresh Python subprocess.

    Args:
        script: Python source code to execute with ``python -c``.

    Raises:
        AssertionError: If the subprocess exits with a non-zero status.
    """
    spec = importlib.util.find_spec("lmcache")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot find lmcache for the subprocess")
    env = os.environ.copy()
    import_root = str(Path(spec.origin).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (import_root, env.get("PYTHONPATH")) if part
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"subprocess failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_memory_allocators_package_has_no_allocator_exports() -> None:
    """Allocator classes are imported from their concrete submodules."""
    _run_import_script(
        """
        import sys

        import lmcache.v1.memory_allocators as allocators

        loaded = sorted(
            name
            for name in sys.modules
            if name.startswith("lmcache.v1.memory_allocators.")
        )
        assert loaded == [], loaded
        assert not hasattr(allocators, "__all__")
        assert not hasattr(allocators, "__getattr__")
        assert not hasattr(allocators, "TensorMemoryAllocator")
        """
    )


def test_allocator_submodule_first_import_keeps_core_surface() -> None:
    """Importing concrete allocator modules first does not create a cycle."""
    _run_import_script(
        """
        import sys

        from lmcache.v1.memory_allocators.paged_tensor_memory_allocator import (
            PagedAddressManager,
            PagedTensorMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.tensor_memory_allocator import (
            TensorMemoryAllocator,
        )
        from lmcache.v1.memory_management import (
            MemoryAllocatorInterface,
            MemoryFormat,
        )

        assert PagedAddressManager.__name__ == "PagedAddressManager"
        assert PagedTensorMemoryAllocator.__name__ == "PagedTensorMemoryAllocator"
        assert TensorMemoryAllocator.__name__ == "TensorMemoryAllocator"
        assert MemoryAllocatorInterface.__name__ == "MemoryAllocatorInterface"
        assert MemoryFormat.KV_2LTD.name == "KV_2LTD"
        assert not any(name.startswith("lmcache.inprocess") for name in sys.modules)
        """
    )


def test_multiple_allocator_submodule_imports_keep_core_surface() -> None:
    """Concrete allocator imports do not disturb core memory_management imports."""
    _run_import_script(
        """
        from lmcache.v1.memory_allocators.devdax_memory_allocator import (
            DevDaxMemoryAllocator,
        )
        from lmcache.v1.memory_allocators.tensor_memory_allocator import (
            TensorMemoryAllocator,
        )
        from lmcache.v1.memory_management import (
            MemoryObj,
            TensorMemoryObj,
        )

        assert DevDaxMemoryAllocator.__name__ == "DevDaxMemoryAllocator"
        assert TensorMemoryAllocator.__name__ == "TensorMemoryAllocator"
        assert MemoryObj.__name__ == "MemoryObj"
        assert TensorMemoryObj.__name__ == "TensorMemoryObj"
        """
    )


def test_memory_management_all_keeps_only_core_names() -> None:
    """The memory_management import surface lists only core memory types."""
    _run_import_script(
        """
        import lmcache.v1.memory_management as memory_management

        allocator_names = {
            "AdHocMemoryAllocator",
            "BufferAllocator",
            "CuFileMemoryAllocator",
            "DevDaxMemoryAllocator",
            "GPUMemoryAllocator",
            "HipFileMemoryAllocator",
            "HostMemoryAllocator",
            "MixedMemoryAllocator",
            "PagedAddressManager",
            "PagedCpuGpuMemoryAllocator",
            "PagedTensorMemoryAllocator",
            "PinMemoryAllocator",
            "TensorMemoryAllocator",
        }
        expected = {
            "AddressManager",
            "BytesBufferMemoryObj",
            "FreeBlock",
            "GDSMemoryObject",
            "MemoryAllocatorInterface",
            "MemoryFormat",
            "MemoryObj",
            "MemoryObjMetadata",
            "TensorMemoryObj",
            "torch_device_type",
        }
        missing = expected - set(memory_management.__all__)
        assert missing == set(), missing
        assert allocator_names.isdisjoint(memory_management.__all__)

        assert not hasattr(memory_management, "TensorMemoryAllocator")
        """
    )
