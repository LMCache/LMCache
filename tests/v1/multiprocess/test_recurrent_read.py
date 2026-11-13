# SPDX-License-Identifier: Apache-2.0

# Standard
from unittest.mock import Mock

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.multiprocess.recurrent_read import reserve_complete_recurrent_reads


def test_reserves_only_complete_checkpoints() -> None:
    manager = Mock(spec=L1ManagerInterface)
    complete = tuple(
        ObjectKey(b"complete", "model", rank, group)
        for rank in range(2)
        for group in range(2)
    )
    partial = tuple(
        ObjectKey(b"partial", "model", rank, group)
        for rank in range(2)
        for group in range(2)
    )
    resident = {key: Mock(spec=MemoryObj) for key in (*complete, *partial[:-1])}
    manager.reserve_read.return_value = {
        **{key: (L1Error.SUCCESS, obj) for key, obj in resident.items()},
        partial[-1]: (L1Error.KEY_NOT_EXIST, None),
    }

    selected, objects = reserve_complete_recurrent_reads(
        manager, {16: complete, 32: (*partial, complete[0]), 48: ()}
    )

    assert selected == {16: complete}
    assert objects == {key: resident[key] for key in complete}
    manager.reserve_read.assert_called_once_with([*complete, *partial])
    manager.finish_read.assert_called_once_with(list(partial[:-1]))
