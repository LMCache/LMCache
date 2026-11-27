# SPDX-License-Identifier: Apache-2.0
"""Handlers registered by the legacy ZMQ server tests."""

# First Party
from lmcache.v1.common.custom_types import IPCCacheServerKey

KeyType = IPCCacheServerKey


def noop_handler() -> str:
    """
    Dummy handler for NOOP requests.
    Takes no arguments and returns a simple string response.
    """
    return "NOOP_OK"


def store_handler(
    key: KeyType, gpu_id: int, gpu_block_ids: list[list[int]], ipc_handle: bytes
) -> tuple[bytes, bool]:
    """
    Dummy handler for STORE requests.

    Args:
        key: Cache key to store
        gpu_id: GPU device ID
        gpu_block_ids: GPU block IDs per KV cache group
        ipc_handle: CUDA event IPC handle

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(ipc_handle, bytes), (
        f"Expected ipc_handle to be bytes, got {type(ipc_handle)}"
    )
    return b"\x01" * 64, True


def retrieve_handler(
    key: KeyType,
    gpu_id: int,
    gpu_block_ids: list[list[int]],
    event_handler: bytes,
    skip_first_n_tokens: int = 0,
) -> tuple[bytes, bool]:
    """
    Dummy handler for RETRIEVE requests.

    Args:
        key: Cache key to retrieve
        gpu_id: GPU device ID
        gpu_block_ids: GPU block IDs per KV cache group
        event_handler: CUDA event IPC handle
        skip_first_n_tokens: Number of tokens to skip at retrieve start

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(event_handler, bytes), (
        f"Expected event_handler to be bytes, got {type(event_handler)}"
    )
    assert isinstance(skip_first_n_tokens, int), (
        f"Expected skip_first_n_tokens to be int, got {type(skip_first_n_tokens)}"
    )
    return b"\x01" * 64, True


def lookup_handler(key: KeyType, tp_size: int) -> None:
    """
    Dummy handler for LOOKUP requests.

    Args:
        key: Cache key to look up (request_id embedded in the key)
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None: LOOKUP registers the job server-side; poll via QUERY_PREFETCH_STATUS.
    """
    # In a real implementation, this would look up the key in the cache
    # For testing, we just validate the input
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"
