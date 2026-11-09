# SPDX-License-Identifier: Apache-2.0


def test_coordinator_legacy_transfer_context_import():
    from lmcache.v1.client.worker_transfer import (
        TransferContext as CurrentTransferContext,
    )
    from lmcache.v1.multiprocess.transfer_context import TransferContext

    assert TransferContext is CurrentTransferContext
