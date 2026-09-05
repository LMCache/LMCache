#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fail if LMCache's compiled extension is missing or cannot be loaded.

``CudaDeviceOps.ensure_native()`` swallows an ``ImportError`` from
``lmcache.cuda_ops`` and falls back to the torch baseline. Loading the file
directly is what actually runs ``dlopen``, so a torch ABI mismatch surfaces as
the ``undefined symbol`` error it really is.

The probed name must keep ``cuda_ops`` as its last component -- CPython derives the
``PyInit_`` symbol from it.  No GPU needed: symbols resolve when the extension
loads, not when a kernel launches.

Usage: ``assert_native_ops.py [module]``  (default ``lmcache.cuda_ops``)
"""

# Future
from __future__ import annotations

# Standard
from pathlib import Path
import importlib.util
import sys

# Third Party
# Must precede the load below: importing torch is what puts libc10/libtorch on
# the process's library search path.
import torch


def main() -> int:
    """Load the extension and report whether it resolved. Returns 0 on success."""
    module = sys.argv[1] if len(sys.argv) > 1 else "lmcache.cuda_ops"
    package, _, stem = module.rpartition(".")

    spec = importlib.util.find_spec(package)
    if spec is None or not spec.origin:
        print(f"::error::{package} is not installed")
        return 1

    matches = sorted(Path(spec.origin).parent.glob(f"{stem}.*.so"))
    if not matches:
        print(f"::error::no compiled {module} is installed")
        return 1

    ext_spec = importlib.util.spec_from_file_location(module, matches[0])
    if ext_spec is None:
        print(f"::error::could not build a module spec for {matches[0]}")
        return 1

    try:
        importlib.util.module_from_spec(ext_spec)  # dlopen + symbol resolution
    except ImportError as exc:
        print(
            f"::error::{module} is installed but failed to load against torch "
            f"{torch.__version__}; LMCache would silently fall back to the torch "
            f"baseline for all ops. {exc}"
        )
        return 1

    print(f"OK: {module} loaded against torch {torch.__version__} ({matches[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
