# SPDX-License-Identifier: Apache-2.0
"""
Helper handler functions for MessageQueue tests.

These handlers are defined at module level to allow them to be pickled
and passed between processes during multiprocessing tests.
"""

# First Party
from lmcache.v1.multiprocess.custom_types import KVCacheRegistration
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


def register_kv_cache_handler(registration: KVCacheRegistration) -> None:
    """
    Dummy handler for REGISTER_KV_CACHE requests.

    Args:
        registration: Structured registration payload

    Returns:
        None
    """
    # In a real implementation, this would register the KV cache
    # For testing, we just validate the inputs are received correctly
    assert isinstance(registration, KVCacheRegistration), (
        f"Expected KVCacheRegistration, got {type(registration)}"
    )
    assert isinstance(registration.instance_id, int), (
        f"Expected instance_id to be int, got {type(registration.instance_id)}"
    )
    assert isinstance(registration.model_name, str), (
        f"Expected model_name to be str, got {type(registration.model_name)}"
    )
    assert isinstance(registration.world_size, int), (
        f"Expected world_size to be int, got {type(registration.world_size)}"
    )
    assert isinstance(registration.engine_type, str), (
        f"Expected engine_type to be str, got {type(registration.engine_type)}"
    )
    assert isinstance(registration.block_size, int), (
        f"Expected block_size to be int, got {type(registration.block_size)}"
    )


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
    layer_begin: int = -1,
    layer_end: int = -1,
) -> tuple[bytes, bool]:
    """
    Dummy handler for RETRIEVE requests.

    Args:
        key: Cache key to retrieve
        gpu_id: GPU device ID
        gpu_block_ids: List of GPU block IDs
        event_handler: CUDA event IPC handle
        skip_first_n_tokens: Number of tokens to skip at retrieve start
        layer_begin: Inclusive layer index, or -1 for all layers
        layer_end: Exclusive layer index, or -1 for all layers

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
    assert isinstance(layer_begin, int), (
        f"Expected layer_begin to be int, got {type(layer_begin)}"
    )
    assert isinstance(layer_end, int), (
        f"Expected layer_end to be int, got {type(layer_end)}"
    )
    return b"\x01" * 64, True


# ==============================================================================
# LOOKUP Request Handlers
# ==============================================================================


def lookup_handler(key: KeyType, tp_size: int) -> int:
    """
    Dummy handler for LOOKUP requests.

    Args:
        key: Cache key to look up (request_id embedded in the key)
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        int: Number of matched chunks (always returns 1 for testing)
    """
    # In a real implementation, this would look up the key in the cache
    # For testing, we just validate the input and return a dummy result
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"
    return 1


# ==============================================================================
# FREE_LOOKUP_LOCKS Request Handlers
# ==============================================================================


def free_locks_handler(key: KeyType, tp_size: int) -> None:
    """
    Dummy handler for FREE_LOOKUP_LOCKS requests.

    Args:
        key: Cache key whose read locks should be released
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None
    """
    assert isinstance(key, KeyType), f"Expected key to be KeyType, got {type(key)}"
    assert isinstance(tp_size, int), f"Expected tp_size to be int, got {type(tp_size)}"
