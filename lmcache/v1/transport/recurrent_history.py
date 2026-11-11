# SPDX-License-Identifier: Apache-2.0
"""Messages for the opt-in recurrent history restore extension."""

# Future
from __future__ import annotations

# Third Party
import msgspec

# First Party
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

HISTORY_CONFIG = "lmcache.mp.recurrent_history"
HISTORY_METHOD = "lmcache.recurrent_history"


class HistoryRequest(msgspec.Struct, kw_only=True):
    """Prepare, query, cancel, or acknowledge a checkpoint reservation.

    Args:
        action: prepare, status, cancel, or acknowledge.
        key: Full scheduler prefix key, required for prepare.
        already_local: Checkpoint token positions already cached on device.
        lease: Server reservation token, required for other actions.
    """

    action: str
    key: IPCCacheServerKey | None = None
    already_local: list[int] = []
    lease: str = ""


class HistoryResponse(msgspec.Struct, kw_only=True):
    """A history plan or completion result.

    Args:
        lease: Server reservation token.
        positions: Complete earlier checkpoints in ascending token order.
        groups: Engine group IDs that need private device destinations.
        settled: All started copies have reached a device fence.
        successful: All required copies succeeded and were not canceled.
    """

    lease: str = ""
    positions: list[int] = []
    groups: list[int] = []
    settled: bool = False
    successful: bool = False


class HistoryCopy(msgspec.Struct, kw_only=True):
    """Private destination blocks supplied with a worker's primary retrieve.

    Args:
        lease: Token of the reserved complete CPU checkpoints.
        destinations: Token positions mapped to engine groups and block IDs.
    """

    lease: str
    destinations: dict[int, dict[int, int]]
