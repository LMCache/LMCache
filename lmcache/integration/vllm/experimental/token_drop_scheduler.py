# SPDX-License-Identifier: Apache-2.0
"""Track each request's resident KV length separately from its logical length.

The allocator updates it after successful allocation.
Scheduler-side updates after compaction and cleanup are handled in follow-up PRs.
"""

_resident_kv_tokens_by_request_id: dict[str, int] = {}


def get_resident_kv_tokens(request_id: str) -> int | None:
    return _resident_kv_tokens_by_request_id.get(request_id)


def set_resident_kv_tokens(request_id: str, num_tokens: int) -> None:
    _resident_kv_tokens_by_request_id[request_id] = num_tokens


def clear_resident_kv_tokens(request_id: str) -> None:
    _resident_kv_tokens_by_request_id.pop(request_id, None)
