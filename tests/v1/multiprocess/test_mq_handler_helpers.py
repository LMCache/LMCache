# SPDX-License-Identifier: Apache-2.0
"""
Helper handler functions for MessageQueue tests.

These handlers are defined at module level to allow them to be pickled
and passed between processes during multiprocessing tests.
"""

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


def register_kv_cache_handler(
    gpu_id: int, kv_cache: KVCache, model_name: str, world_size: int
) -> None:
    """
    Dummy handler for REGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID
        kv_cache: List of CudaIPCWrapper objects representing KV cache
        model_name: Name of the model associated with this KV cache
        world_size: World size associated with this KV cache

    Returns:
        None
    """
    # In a real implementation, this would register the KV cache
    # For testing, we just validate the inputs are received correctly
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(kv_cache, list), (
        f"Expected kv_cache to be list, got {type(kv_cache)}"
    )
    assert isinstance(model_name, str), (
        f"Expected model_name to be str, got {type(model_name)}"
    )
    assert isinstance(world_size, int), (
        f"Expected world_size to be int, got {type(world_size)}"
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
    key: KeyType, gpu_id: int, gpu_block_ids: list[int], ipc_handle: bytes
) -> tuple[bytes, bool]:
    """
    Dummy handler for STORE requests.

    Args:
        key: Cache key to store
        gpu_id: GPU device ID
        gpu_block_ids: List of GPU block IDs
        ipc_handle: CUDA event IPC handle

    Returns:
        tuple[bytes, bool]: (event handle, success flag)
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(gpu_id, int), f"Expected gpu_id to be int, got {type(gpu_id)}"
    assert isinstance(gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(gpu_block_ids)}"
    )
    assert isinstance(ipc_handle, bytes), (
        f"Expected ipc_handle to be bytes, got {type(ipc_handle)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# RETRIEVE Request Handlers
# ==============================================================================


def retrieve_handler(
    key: KeyType,
    gpu_id: int,
    gpu_block_ids: list[int],
    event_handler: bytes,
    skip_first_n_tokens: int = 0,
) -> tuple[bytes, bool]:
    """
    Dummy handler for RETRIEVE requests.

    Args:
        key: Cache key to retrieve
        gpu_id: GPU device ID
        gpu_block_ids: List of GPU block IDs
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
    assert isinstance(event_handler, bytes), (
        f"Expected event_handler to be bytes, got {type(event_handler)}"
    )
    assert isinstance(skip_first_n_tokens, int), (
        f"Expected skip_first_n_tokens to be int, got {type(skip_first_n_tokens)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# LOOKUP Request Handlers
# ==============================================================================


def lookup_handler(key: KeyType) -> int:
    """
    Dummy handler for LOOKUP requests.

    Args:
        key: Cache key to look up (request_id embedded in the key)

    Returns:
        int: Number of matched chunks (always returns 1 for testing)
    """
    # In a real implementation, this would look up the key in the cache
    # For testing, we just validate the input and return a dummy result
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    return 1


# ==============================================================================
# FREE_LOOKUP_LOCKS Request Handlers
# ==============================================================================


def free_locks_handler(key: KeyType) -> None:
    """
    Dummy handler for FREE_LOOKUP_LOCKS requests.

    Args:
        key: Cache key whose read locks should be released

    Returns:
        None
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
