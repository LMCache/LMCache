# SPDX-License-Identifier: Apache-2.0
"""Store each request's physically resident KV length independently of logical progress.

The adaptor reads it for allocation; reclaim and preemption update or clear it.
"""

_resident_kv_tokens_by_request_id: dict[str, int] = {}


def get_resident_kv_tokens(request_id: str) -> int | None:
    return _resident_kv_tokens_by_request_id.get(request_id)


def set_resident_kv_tokens(request_id: str, num_tokens: int) -> None:
    _resident_kv_tokens_by_request_id[request_id] = num_tokens


def clear_resident_kv_tokens(request_id: str) -> None:
    _resident_kv_tokens_by_request_id.pop(request_id, None)
