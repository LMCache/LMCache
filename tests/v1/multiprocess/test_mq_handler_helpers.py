# SPDX-License-Identifier: Apache-2.0
"""
Helper handler functions for MessageQueue tests.

These handlers are defined at module level to allow them to be pickled
and passed between processes during multiprocessing tests.
"""

# Standard
from typing import Optional

# First Party
from lmcache.v1.multiprocess.custom_types import KVCache
from lmcache.v1.multiprocess.protocol import KeyType

# ==============================================================================
# NOOP Request Handlers
# ==============================================================================


def noop_handler() -> str:
    """
    Dummy handler for NOOP requests.
    Takes no arguments and returns a simple string response.
    """
    return "NOOP_OK"


# ==============================================================================
# REGISTER_KV_CACHE Request Handlers
# ==============================================================================


def register_kv_cache_handler(gpu_id: int, kv_cache: KVCache) -> None:
    """
    Dummy handler for REGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID
        kv_cache: List of CudaIPCWrapper objects representing KV cache

    Returns:
        None
    """
    # In a real implementation, this would register the KV cache
    # For testing, we just validate the inputs are received correctly
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(kv_cache, list), (
        f"Expected kv_cache to be list, got {type(kv_cache)}"
    )
    # No return value (returns None implicitly)


# ==============================================================================
# UNREGISTER_KV_CACHE Request Handlers
# ==============================================================================


def unregister_kv_cache_handler(gpu_id: int) -> None:
    """
    Dummy handler for UNREGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID

    Returns:
        None
    """
    # In a real implementation, this would unregister the KV cache for the given GPU
    # For testing, we just validate the input is received correctly
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    # No return value (returns None implicitly)


# ==============================================================================
# STORE Request Handlers
# ==============================================================================


def store_handler(
    keys: list[KeyType], gpu_id: int, gpu_block_ids: list[int], ipc_handle: bytes
) -> tuple[bytes, bool]:
    """
    Dummy handler for STORE requests.

    Args:
        keys: List of cache keys to store
        gpu_id: GPU device ID
        gpu_block_ids: List of GPU block IDs

    Returns:
        bool: True if store succeeded
    """
    # In a real implementation, this would store KV cache data
    # For testing, we just validate the inputs are received correctly
    assert isinstance(keys, list), f"Expected keys to be list, got {type(keys)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert isinstance(ipc_handle, bytes), (
        f"Expected ipc_handle to be bytes, got {type(ipc_handle)}"
    )
    # Return success
    return b"\x01" * 64, True


# ==============================================================================
# RETRIEVE Request Handlers
# ==============================================================================


def retrieve_handler(
    keys: list[KeyType], gpu_id: int, gpu_block_ids: list[int], event_handler: bytes
) -> tuple[bytes, list[bool]]:
    """
    Dummy handler for RETRIEVE requests.

    Args:
        keys: List of cache keys to retrieve
        gpu_id: GPU device ID
        gpu_block_ids: List of GPU block IDs

    Returns:
        bool: True if retrieve succeeded
    """
    # In a real implementation, this would retrieve KV cache data
    # For testing, we just validate the inputs are received correctly
    assert isinstance(keys, list), f"Expected keys to be list, got {type(keys)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert isinstance(event_handler, bytes), (
        f"Expected event_handler to be bytes, got {type(event_handler)}"
    )
    # Return success
    return b"\x01" * 64, [True for _ in keys]


# ==============================================================================
# LOOKUP Request Handlers
# ==============================================================================


def lookup_handler(keys: list[KeyType], lock: Optional[bool]) -> list[bool]:
    """
    Dummy handler for LOOKUP requests.

    Args:
        keys: List of cache keys to look up
        lock: Optional flag to lock the keys

    Returns:
        list[bool]: List indicating whether each key was found
    """
    # In a real implementation, this would look up keys in the cache
    # For testing, we just validate the inputs and return dummy results
    assert isinstance(keys, list), f"Expected keys to be list, got {type(keys)}"
    assert lock is None or isinstance(lock, bool), (
        f"Expected lock to be None or bool, got {type(lock)}"
    )
    # Return a result for each key (alternating True/False for testing)
    return [i % 2 == 0 for i in range(len(keys))]
