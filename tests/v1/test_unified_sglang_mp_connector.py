# SPDX-License-Identifier: Apache-2.0
"""Small, server-free tests for the unified SGLang MP connector contracts."""

# Standard
from types import ModuleType
from unittest.mock import Mock, patch
import unittest

# Third Party
import torch

# First Party
from lmcache.integration.sglang.unified_lmcache_mp_connector import (
    LMCacheKVGroup,
    LMCacheLoadOperation,
    LMCacheLookupOperation,
    UnifiedLMCacheMPConnector,
)


class _Future:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self):
        return self.ready

    def retain_reference(self, value):
        self.value = value


class _ResultFuture(_Future):
    def __init__(self, ready: bool, value):
        super().__init__(ready)
        self.value = value

    def result(self, timeout=None):
        del timeout
        if not self.ready:
            raise TimeoutError("future is not ready")
        return self.value

    def wait_on_stream(self, stream, timeout=None):
        del timeout
        self.waited_stream = stream
        return self.result()

    def prepare(self, timeout=None):
        return self.result(timeout)


class _TransferContext:
    def __init__(self):
        self.store_args = None
        self.retrieve_args = None

    def submit_store(self, *args):
        self.store_args = args
        return _Future(True)

    def submit_retrieve(self, *args, **kwargs):
        self.retrieve_args = (args, kwargs)
        return _Future(True)


class _IPCCacheServerKey:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _RequestType:
    LOOKUP = object()
    QUERY_PREFETCH_STATUS = object()
    WAIT_PREFETCH_STATUS = object()
    FREE_LOOKUP_LOCKS = object()
    END_SESSION = object()


class TestUnifiedLMCacheMPConnector(unittest.TestCase):
    def setUp(self):
        self.connector = object.__new__(UnifiedLMCacheMPConnector)
        self.connector.page_size = 4

    def test_slots_to_blocks_accepts_noncontiguous_pages(self):
        slots = torch.tensor([4, 5, 6, 7, 12, 13, 14, 15])
        self.assertEqual(self.connector._slots_to_blocks(slots), [1, 3])

    def test_pp_parallel_geometry_uses_global_worker_ids(self):
        actual = []
        for pp_rank in range(2):
            for tp_rank in range(2):
                geometry = self.connector._resolve_parallel_geometry(
                    2, tp_rank, 2, pp_rank
                )
                actual.append((geometry[4], geometry[5]))

        self.assertEqual(actual, [(4, 0), (4, 1), (4, 2), (4, 3)])

    def test_mla_kv_geometry_collapses_only_tp_object_identity(self):
        actual = []
        for pp_rank in range(2):
            for tp_rank in range(2):
                geometry = self.connector._resolve_kv_geometry(
                    2, tp_rank, 2, pp_rank, mla_only=True
                )
                actual.append(geometry)

        self.assertEqual(actual, [(2, 0), (2, 0), (2, 1), (2, 1)])

    def test_parallel_geometry_rejects_invalid_pp_rank(self):
        with self.assertRaisesRegex(ValueError, "Invalid LMCache PP topology"):
            self.connector._resolve_parallel_geometry(2, 0, 2, 2)

    def test_parallel_all_reduce_covers_tp_then_pp(self):
        self.connector.tp_size = 2
        self.connector.pp_size = 2
        self.connector.tp_group = object()
        self.connector.pp_group = object()
        value = torch.tensor([1], dtype=torch.int32)

        with patch.object(torch.distributed, "all_reduce") as all_reduce:
            self.connector._parallel_all_reduce(value, torch.distributed.ReduceOp.MIN)

        self.assertEqual(all_reduce.call_count, 2)
        self.assertIs(
            all_reduce.call_args_list[0].kwargs["group"], self.connector.tp_group
        )
        self.assertIs(
            all_reduce.call_args_list[1].kwargs["group"], self.connector.pp_group
        )

    def test_aligned_empty_lookup_does_not_send_end_session(self):
        self.connector.chunk_size = 8
        self.connector._lookups = {}
        self.connector._active_sessions = set()
        self.connector._lookup_leader = True
        self.connector._mq_client = object()
        self.connector._send_request = Mock()
        self.connector._track_control_future = Mock()

        protocol = ModuleType("lmcache.v1.multiprocess.protocol")
        protocol.RequestType = _RequestType
        with patch.dict("sys.modules", {"lmcache.v1.multiprocess.protocol": protocol}):
            operation = self.connector.submit_lookup(
                "short-request",
                [1, 2, 3],
                local_hit_tokens=0,
                cache_salt="",
            )
            self.connector.end_session("short-request")

        self.assertEqual(operation.total_hit_tokens, 0)
        self.connector._send_request.assert_not_called()
        self.connector._track_control_future.assert_not_called()

    def test_submitted_lookup_tracks_server_session(self):
        self.connector.chunk_size = 4
        self.connector.kv_world_size = 1
        self.connector._lookups = {}
        self.connector._active_sessions = set()
        self.connector._lookup_leader = True
        self.connector._mq_client = object()
        self.connector._create_key = Mock(return_value=object())
        self.connector._send_request = Mock(return_value=_Future(False))
        self.connector._sync_success = lambda success: success

        protocol = ModuleType("lmcache.v1.multiprocess.protocol")
        protocol.RequestType = _RequestType
        with patch.dict("sys.modules", {"lmcache.v1.multiprocess.protocol": protocol}):
            self.connector.submit_lookup(
                "lookup-request",
                [1, 2, 3, 4],
                local_hit_tokens=0,
                cache_salt="",
            )

        self.assertIn("lookup-request", self.connector._active_sessions)

    def test_poll_lookup_uses_short_lived_status_queries(self):
        self.connector.chunk_size = 4
        self.connector._lookup_leader = True
        self.connector._mq_client = object()
        self.connector._sync_leader_int = lambda value: value
        pending_query = _ResultFuture(False, None)
        completed_query = _ResultFuture(True, 3)
        self.connector._send_request = Mock(
            side_effect=[pending_query, completed_query]
        )
        operation = LMCacheLookupOperation(
            request_id="lookup-request",
            token_ids=list(range(16)),
            local_hit_tokens=0,
            cache_salt="",
            submission_future=_ResultFuture(True, None),
        )

        protocol = ModuleType("lmcache.v1.multiprocess.protocol")
        protocol.RequestType = _RequestType
        with patch.dict("sys.modules", {"lmcache.v1.multiprocess.protocol": protocol}):
            # Submit one non-blocking status query. An outstanding query must
            # not be duplicated by subsequent scheduler polls.
            self.assertIsNone(self.connector.poll_lookup(operation))
            self.assertIsNone(self.connector.poll_lookup(operation))
            self.assertEqual(self.connector._send_request.call_count, 1)

            # A None response means prefetch is still running. The following
            # scheduler pass issues a fresh query, which returns the final hit.
            pending_query.ready = True
            self.assertIsNone(self.connector.poll_lookup(operation))
            self.assertIsNone(self.connector.poll_lookup(operation))
            self.assertEqual(self.connector.poll_lookup(operation), 12)

        self.assertEqual(self.connector._send_request.call_count, 2)
        for call in self.connector._send_request.call_args_list:
            self.assertIs(call.args[1], _RequestType.QUERY_PREFETCH_STATUS)
            self.assertEqual(call.args[2], ["lookup-request"])
        self.assertTrue(operation.locks_held)

    def test_slots_to_blocks_rejects_partial_page(self):
        with self.assertRaisesRegex(ValueError, "complete SGLang pages"):
            self.connector._slots_to_blocks(torch.tensor([4, 5, 6]))

    def test_slots_to_blocks_rejects_unaligned_page(self):
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            self.connector._slots_to_blocks(torch.tensor([5, 6, 7, 8]))

    def test_slots_to_blocks_accepts_explicit_dummy_page_for_load(self):
        slots = torch.tensor([0, 0, 0, 0, 8, 9, 10, 11])
        self.assertEqual(
            self.connector._slots_to_blocks(slots, allow_dummy_page=True), [0, 2]
        )

    def test_component_block_ids_expand_to_kernel_groups(self):
        self.connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
            LMCacheKVGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        self.connector._kernel_group_to_engine_group = (0, 1, 1)
        block_ids = self.connector._block_ids_for_transfer(
            [
                torch.tensor([4, 5, 6, 7, 12, 13, 14, 15]),
                torch.tensor([8, 9, 10, 11, 16, 17, 18, 19]),
            ],
            allow_dummy_page=False,
        )
        self.assertEqual(block_ids, [[1, 3], [2, 4], [2, 4]])

    def test_wire_view_folds_attention_page_into_one_opaque_row(self):
        tensor = torch.arange(4 * 2 * 3).reshape(4, 2, 3)

        wire = self.connector._to_wire_block_tensor(tensor, slots_per_block=4)

        self.assertEqual(tuple(wire.shape), (1, 1, 24))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())
        self.assertTrue(torch.equal(wire.reshape(-1), tensor.reshape(-1)))

    def test_wire_view_keeps_one_mamba_state_slot_per_block(self):
        tensor = torch.arange(5 * 2 * 3).reshape(5, 2, 3)

        wire = self.connector._to_wire_block_tensor(tensor, slots_per_block=1)

        self.assertEqual(tuple(wire.shape), (5, 1, 6))
        self.assertEqual(wire.data_ptr(), tensor.data_ptr())

    def test_group_info_specs_preserve_component_address_spaces(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector._kv_groups = (
            LMCacheKVGroup(
                "full",
                (
                    torch.empty(20, 1, 8),
                    torch.empty(20, 1, 8),
                ),
                tokens_per_block=4,
                slots_per_block=4,
            ),
            LMCacheKVGroup(
                "swa",
                (
                    torch.empty(12, 1, 8),
                    torch.empty(12, 1, 16),
                ),
                sliding_window_size=8,
                tokens_per_block=4,
                slots_per_block=4,
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0, 1, 1))
        self.assertEqual(
            [spec["layer_indices"] for spec in specs], [(0, 1), (2,), (3,)]
        )
        self.assertEqual([spec["sw_size_tokens"] for spec in specs], [-1, 8, 8])

    def test_group_info_specs_mark_mamba_as_recurrent_one_block_window(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._kv_groups = (
            LMCacheKVGroup(
                "mamba",
                (torch.empty(12, 1, 32),),
                sliding_window_size=256,
                tokens_per_block=256,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0,))
        self.assertEqual(specs[0]["tokens_per_block"], 256)
        self.assertEqual(specs[0]["sw_size_tokens"], 256)
        self.assertTrue(specs[0]["recurrent_state"])

    def test_group_info_specs_keep_dsa_sidecar_in_full_address_space(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._kv_groups = (
            LMCacheKVGroup(
                "full",
                (
                    torch.empty(3, 1, 64, dtype=torch.bfloat16),
                    torch.empty(3, 1, 528, dtype=torch.uint8),
                ),
                tokens_per_block=4,
                slots_per_block=4,
                tensor_rows_per_block=(1, 1),
            ),
        )

        specs, kernel_to_engine = connector._build_engine_group_info_specs()

        self.assertEqual(kernel_to_engine, (0, 0))
        self.assertEqual([spec["layer_indices"] for spec in specs], [(0,), (1,)])
        self.assertEqual([spec["engine_group_id"] for spec in specs], [0, 0])

    def test_submit_store_passes_list_of_block_ids_per_group(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector.chunk_size = 8
        connector.blocks_in_chunk = 2
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
            LMCacheKVGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = True
        connector._new_event = lambda: object()
        connector._create_key = Mock(return_value=object())
        connector._sync_success = lambda success: success
        connector._sync_leader_int = lambda value: value

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [
                torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
                torch.tensor([12, 13, 14, 15, 20, 21, 22, 23]),
            ],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertIn("request", connector._active_sessions)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [[1, 2], [3, 5]],
        )
        self.assertEqual(connector._create_key.call_args.kwargs["worker_id"], 0)

    def test_mla_non_writer_uses_collective_placeholder(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector.chunk_size = 8
        connector.blocks_in_chunk = 2
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0,)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = False
        connector._sync_success = lambda success: success
        # A peer TP0 submitted the shared MLA object.
        connector._sync_leader_int = lambda value: 1

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertIsNone(connector._transfer_ctx.store_args)
        self.assertTrue(operation.query())
        self.assertTrue(connector.complete_store(operation))
        self.assertIn("request", connector._active_sessions)

    def test_submit_store_defers_unmapped_swa_page(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 4
        connector.chunk_size = 8
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=4, slots_per_block=4),
            LMCacheKVGroup("swa", (), tokens_per_block=4, slots_per_block=4),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._is_kv_writer = True
        connector._sync_success = lambda success: success

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [
                torch.tensor([4, 5, 6, 7, 8, 9, 10, 11]),
                torch.tensor([0, 0, 0, 0, 12, 13, 14, 15]),
            ],
            cache_salt="",
        )

        self.assertIsNone(operation)
        self.assertNotIn("request", connector._active_sessions)
        self.assertNotIn("request", connector._store_submitted_tokens)

    def test_submit_store_accepts_dummy_mamba_blocks(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 1
        connector.chunk_size = 8
        connector.blocks_in_chunk = 8
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=1, slots_per_block=1),
            LMCacheKVGroup(
                "mamba",
                (),
                sliding_window_size=4,
                tokens_per_block=4,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._store_submitted_tokens = {}
        connector._active_sessions = set()
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._is_kv_writer = True
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()
        connector._sync_success = lambda success: success
        connector._sync_leader_int = lambda value: value

        operation = connector.submit_store(
            "request",
            list(range(8)),
            [torch.arange(1, 9), torch.tensor([0, 7])],
            cache_salt="",
        )

        self.assertIsNotNone(operation)
        self.assertEqual(
            connector._transfer_ctx.store_args[4],
            [list(range(1, 9)), [0, 7]],
        )

    def test_submit_load_uses_compressed_mamba_block_ids(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector.page_size = 1
        connector.chunk_size = 8
        connector.blocks_in_chunk = 8
        connector.sglang_worker_id = 0
        connector.kv_worker_id = 0
        connector.instance_id = 1
        connector._kv_groups = (
            LMCacheKVGroup("full", (), tokens_per_block=1, slots_per_block=1),
            LMCacheKVGroup(
                "mamba",
                (),
                sliding_window_size=4,
                tokens_per_block=4,
                slots_per_block=1,
                recurrent_state=True,
            ),
        )
        connector._kernel_group_to_engine_group = (0, 1)
        connector._kv_caches = {}
        connector._transfer_ctx = _TransferContext()
        connector._new_event = lambda: object()
        connector._create_key = lambda *args, **kwargs: object()
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=3,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )

        operation = connector.submit_load(
            lookup,
            [torch.arange(4, 9), torch.tensor([0, 7])],
            local_hit_tokens=3,
        )

        args, kwargs = connector._transfer_ctx.retrieve_args
        self.assertEqual(args[4], [[0, 0, 0, 4, 5, 6, 7, 8], [0, 7]])
        self.assertEqual(kwargs["skip_first_n_tokens"], 3)
        self.assertEqual(operation.start, 0)
        self.assertEqual(operation.end, 8)

    def test_prepare_load_orders_forward_stream_without_completing_future(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._mq_timeout = 10
        connector._sync_success = lambda success: success
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=0,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )
        future = _ResultFuture(True, True)
        operation = LMCacheLoadOperation(
            request_id="request",
            token_ids=list(range(8)),
            start=0,
            end=8,
            local_hit_tokens=0,
            device_indices=torch.arange(8),
            future=future,
            lookup=lookup,
        )

        self.assertTrue(connector.prepare_load_on_stream(operation, "forward"))
        self.assertEqual(future.waited_stream, "forward")
        self.assertIsNone(operation.result)
        self.assertTrue(lookup.locks_held)

    def test_free_lookup_locks_sends_one_leader_prefix_range(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._lookup_leader = True
        connector.kv_world_size = 2
        connector._mq_client = object()
        connector._track_control_future = Mock()
        connector._send_request = Mock(return_value=_Future(True))
        connector._create_key = Mock(return_value="prefix-key")
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=4,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )
        connector._lookups = {lookup.request_id: lookup}
        protocol = ModuleType("lmcache.v1.multiprocess.protocol")
        protocol.RequestType = _RequestType

        with patch.dict("sys.modules", {"lmcache.v1.multiprocess.protocol": protocol}):
            connector.free_lookup_locks("request", start=0, end=4)

        connector._create_key.assert_called_once_with(
            lookup, start=0, end=4, worker_id=None
        )
        connector._send_request.assert_called_once_with(
            connector._mq_client,
            _RequestType.FREE_LOOKUP_LOCKS,
            ["prefix-key", 2],
        )
        self.assertEqual(lookup.lock_start, 4)
        self.assertTrue(lookup.locks_held)
        self.assertIs(connector._lookups["request"], lookup)

    def test_complete_load_relies_on_retrieve_to_release_read_locks(self):
        connector = object.__new__(UnifiedLMCacheMPConnector)
        connector._sync_success = lambda success: success
        connector._free_lookup_locks = Mock()
        lookup = LMCacheLookupOperation(
            request_id="request",
            token_ids=list(range(8)),
            local_hit_tokens=0,
            cache_salt="",
            total_hit_tokens=8,
            locks_held=True,
        )
        connector._lookups = {lookup.request_id: lookup}
        operation = LMCacheLoadOperation(
            request_id="request",
            token_ids=list(range(8)),
            start=0,
            end=8,
            local_hit_tokens=0,
            device_indices=torch.arange(8),
            future=_ResultFuture(True, True),
            lookup=lookup,
        )

        self.assertTrue(connector.complete_load(operation))

        connector._free_lookup_locks.assert_not_called()
        self.assertFalse(lookup.locks_held)
        self.assertNotIn("request", connector._lookups)

    def test_completed_operation_does_not_requery_future(self):
        operation = object.__new__(LMCacheLoadOperation)
        operation.result = False
        operation.future = _Future(ready=True)
        self.assertTrue(operation.query())

    def test_create_key_declares_single_kv_reader(self):
        self.connector.model_name = "model"
        self.connector.kv_world_size = 2
        self.connector.num_kv_readers = 1
        operation = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="salt",
        )
        custom_types = ModuleType("lmcache.v1.multiprocess.custom_types")
        custom_types.IPCCacheServerKey = _IPCCacheServerKey

        with patch.dict(
            "sys.modules",
            {"lmcache.v1.multiprocess.custom_types": custom_types},
        ):
            key = self.connector._create_key(operation, start=0, end=4, worker_id=None)

        self.assertEqual(key.num_kv_readers, 1)
        self.assertEqual(key.world_size, 2)

    def test_create_key_reserves_one_read_lock_per_tp_replica(self):
        self.connector.model_name = "model"
        self.connector.kv_world_size = 1
        self.connector.num_kv_readers = 4
        operation = LMCacheLookupOperation(
            request_id="request",
            token_ids=[1, 2, 3, 4],
            local_hit_tokens=0,
            cache_salt="salt",
        )
        custom_types = ModuleType("lmcache.v1.multiprocess.custom_types")
        custom_types.IPCCacheServerKey = _IPCCacheServerKey

        with patch.dict(
            "sys.modules",
            {"lmcache.v1.multiprocess.custom_types": custom_types},
        ):
            key = self.connector._create_key(operation, start=0, end=4, worker_id=None)

        self.assertEqual(key.num_kv_readers, 4)
        self.assertEqual(key.world_size, 1)
