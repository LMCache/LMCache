# SPDX-License-Identifier: Apache-2.0
"""The transfer-context package keeps its documented re-export surface.

The worker-side contexts moved to ``lmcache.v1.client``, but
``lmcache.v1.multiprocess.transfer_context`` promises that imports through the
package path keep working without naming the sub-module, and code outside this
repository imports them from there.
"""

# Standard
import subprocess
import sys

# Third Party
import pytest

# First Party
import lmcache.v1.client.async_engine_driven as async_engine_driven
import lmcache.v1.client.worker_transfer as worker_transfer
import lmcache.v1.multiprocess.transfer_context as transfer_context

RE_EXPORTED = [
    "AsyncEngineDrivenTransferContext",
    "EngineDrivenTransferContext",
    "LMCacheDrivenTransferContext",
    "TransferContext",
    "create_transfer_context",
]
"""Every name the package re-exported before the worker-side move."""

IMPORT_ORDERS = [
    "from lmcache.v1.multiprocess.transfer_context import TransferContext",
    "import lmcache.v1.client.worker_transfer\n"
    "from lmcache.v1.multiprocess.transfer_context import TransferContext",
]
"""Which module each case imports first; both must resolve."""


@pytest.mark.parametrize("name", RE_EXPORTED)
def test_the_package_re_exports_the_worker_side_names(name: str) -> None:
    """MP-1: each name is the object the module that now defines it has."""
    expected = getattr(async_engine_driven, name, None) or getattr(
        worker_transfer, name
    )
    assert getattr(transfer_context, name) is expected


@pytest.mark.parametrize("imports", IMPORT_ORDERS)
def test_either_import_order_resolves(imports: str) -> None:
    """MP-2: whichever module the import machinery reaches first, the package
    still answers.

    A fresh interpreter per case: this one has both modules imported already,
    so it cannot show which order broke.
    """
    completed = subprocess.run(
        [sys.executable, "-c", f"{imports}\nprint(TransferContext.__name__)"],
        capture_output=True,
        timeout=60,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "TransferContext" in completed.stdout
