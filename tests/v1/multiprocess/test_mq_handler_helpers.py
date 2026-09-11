# SPDX-License-Identifier: Apache-2.0
"""
Helper handler functions for MessageQueue tests.

These handlers are defined at module level to allow them to be pickled
and passed between processes during multiprocessing tests.
"""

# First Party
from lmcache.utils import EngineType
from lmcache.v1.multiprocess.custom_types import (
    BlockAllocationRecord,
)
from lmcache.v1.multiprocess.protocol import KeyType
from lmcache.v1.multiprocess.rpc_messages import (
    EventIpcHandleResult,
    FreeLookupLocksRequest,
    FreeLookupLocksResponse,
    LookupRequest,
    LookupResponse,
    NoopRequest,
    NoopResponse,
    RegisterKvCacheRequest,
    RegisterKvCacheResponse,
    ReportBlockAllocationRequest,
    ReportBlockAllocationResponse,
    RetrieveRequest,
    RetrieveResponse,
    StoreRequest,
    StoreResponse,
    UnregisterKvCacheRequest,
    UnregisterKvCacheResponse,
)

# ==============================================================================
# NOOP Request Handlers
# ==============================================================================


def noop_handler(request: NoopRequest) -> NoopResponse:
    """
    Dummy handler for NOOP requests.
    Takes no arguments and returns a simple string response.
    """
    return NoopResponse("NOOP_OK")


# ==============================================================================
# REGISTER_KV_CACHE Request Handlers
# ==============================================================================


def register_kv_cache_handler(
    request: RegisterKvCacheRequest,
) -> RegisterKvCacheResponse:
    """
    Dummy handler for REGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID
        kv_cache: List of CudaIPCWrapper objects representing KV cache
        model_name: Name of the model associated with this KV cache
        world_size: World size associated with this KV cache
        engine_type: Which serving engine produced the caches
        layout_hints: Engine-provided hints dict.
        engine_group_infos: Engine-neutral KV cache group metadata,
            msgspec-decoded from the request payload.

    Returns:
        None
    """
    # In a real implementation, this would register the KV cache
    # For testing, we just validate the inputs are received correctly
    assert isinstance(request.instance_id, int), (
        f"Expected gpu_id to be int, got {type(request.instance_id)}"
    )
    assert isinstance(request.kv_cache, list), (
        f"Expected kv_cache to be list, got {type(request.kv_cache)}"
    )
    assert isinstance(request.model_name, str), (
        f"Expected model_name to be str, got {type(request.model_name)}"
    )
    assert isinstance(request.world_size, int), (
        f"Expected world_size to be int, got {type(request.world_size)}"
    )
    assert isinstance(request.engine_type, EngineType), (
        f"Expected engine_type to be EngineType, got {type(request.engine_type)}"
    )
    assert isinstance(request.layout_hints, dict), (
        f"Expected layout_hints to be dict, got {type(request.layout_hints)}"
    )
    assert isinstance(request.engine_group_infos, list), (
        "Expected engine_group_infos to be a list, got "
        f"{type(request.engine_group_infos)}"
    )
    return RegisterKvCacheResponse()


# ==============================================================================
# UNREGISTER_KV_CACHE Request Handlers
# ==============================================================================


def unregister_kv_cache_handler(
    request: UnregisterKvCacheRequest,
) -> UnregisterKvCacheResponse:
    """
    Dummy handler for UNREGISTER_KV_CACHE requests.

    Args:
        gpu_id: GPU device ID

    Returns:
        None
    """
    # In a real implementation, this would unregister the KV cache for the given GPU
    # For testing, we just validate the input is received correctly
    assert isinstance(request.instance_id, int), (
        f"Expected gpu_id to be int, got {type(request.instance_id)}"
    )
    return UnregisterKvCacheResponse()


# ==============================================================================
# STORE Request Handlers
# ==============================================================================


def store_handler(request: StoreRequest) -> StoreResponse:
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
    assert isinstance(request.key, KeyType), (
        f"Expected key to be KeyType, got {type(request.key)}"
    )
    assert isinstance(request.instance_id, int), (
        f"Expected gpu_id to be int, got {type(request.instance_id)}"
    )
    assert isinstance(request.gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(request.gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in request.gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(request.event_ipc_handle, bytes), (
        f"Expected ipc_handle to be bytes, got {type(request.event_ipc_handle)}"
    )
    return StoreResponse(EventIpcHandleResult(b"\x01" * 64, True))


# ==============================================================================
# RETRIEVE Request Handlers
# ==============================================================================


def retrieve_handler(request: RetrieveRequest) -> RetrieveResponse:
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
    assert isinstance(request.key, KeyType), (
        f"Expected key to be KeyType, got {type(request.key)}"
    )
    assert isinstance(request.instance_id, int), (
        f"Expected gpu_id to be int, got {type(request.instance_id)}"
    )
    assert isinstance(request.gpu_block_ids, list), (
        f"Expected gpu_block_ids to be list, got {type(request.gpu_block_ids)}"
    )
    assert all(isinstance(block_ids, list) for block_ids in request.gpu_block_ids), (
        "Expected gpu_block_ids to be list[list[int]]"
    )
    assert isinstance(request.event_ipc_handle, bytes), (
        f"Expected event_handler to be bytes, got {type(request.event_ipc_handle)}"
    )
    assert isinstance(request.skip_first_n_tokens, int), (
        "Expected skip_first_n_tokens to be int, got "
        f"{type(request.skip_first_n_tokens)}"
    )
    return RetrieveResponse(EventIpcHandleResult(b"\x01" * 64, True))


# ==============================================================================
# LOOKUP Request Handlers
# ==============================================================================


def lookup_handler(request: LookupRequest) -> LookupResponse:
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
    assert isinstance(request.key, KeyType), (
        f"Expected key to be KeyType, got {type(request.key)}"
    )
    assert isinstance(request.tp_size, int), (
        f"Expected tp_size to be int, got {type(request.tp_size)}"
    )
    return LookupResponse()


# ==============================================================================
# FREE_LOOKUP_LOCKS Request Handlers
# ==============================================================================


def free_locks_handler(
    request: FreeLookupLocksRequest,
) -> FreeLookupLocksResponse:
    """
    Dummy handler for FREE_LOOKUP_LOCKS requests.

    Args:
        key: Cache key whose read locks should be released
        tp_size: Tensor-parallel size for MLA
            multi-reader locking

    Returns:
        None
    """
    assert isinstance(request.key, KeyType), (
        f"Expected key to be KeyType, got {type(request.key)}"
    )
    assert isinstance(request.tp_size, int), (
        f"Expected tp_size to be int, got {type(request.tp_size)}"
    )
    return FreeLookupLocksResponse()


# ==============================================================================
# REPORT_BLOCK_ALLOCATION Request Handlers
# ==============================================================================


def report_block_allocations_handler(
    request: ReportBlockAllocationRequest,
) -> ReportBlockAllocationResponse:
    """
    Dummy handler for REPORT_BLOCK_ALLOCATION requests.

    Args:
        instance_id: The scheduler instance ID.
        model_name: The model name from the adapter.
        records: List of BlockAllocationRecord with per-request
            block and token allocation deltas.

    Returns:
        None
    """
    assert isinstance(request.records, list), (
        f"Expected records to be list, got {type(request.records)}"
    )
    for rec in request.records:
        assert isinstance(rec, BlockAllocationRecord), (
            f"Expected BlockAllocationRecord, got {type(rec)}"
        )
        assert isinstance(rec.req_id, str)
        assert isinstance(rec.new_block_ids, list)
        assert isinstance(rec.new_token_ids, list)
    return ReportBlockAllocationResponse()
