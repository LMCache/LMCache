# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the fused raw-block multiprocess retrieve path."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, call, patch
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.engine_module import ThreadPoolType
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    ContextEntry,
    LMCacheDrivenTransferModule,
    _fused_event_session_resource_key,
    _FusedDrainEvent,
    _FusedDrainState,
    _launch_staged_h2d_batch,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.protocol import (
    get_handler_type,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.base import HandlerType, RequestType
from lmcache.v1.multiprocess.protocols.engine import (
    FUSED_RAW_BLOCK_RETRIEVE_CAPABILITY,
)
import lmcache.c_ops as lmc_ops


def _make_key() -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="test-model",
        world_size=1,
        worker_id=0,
        token_ids=list(range(12)),
        start=4,
        end=12,
        request_id="request-1",
    )


def _make_object_key(chunk_id: int) -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_id),
        model_name="test-model",
        kv_rank=0,
    )


def _make_module() -> tuple[
    LMCacheDrivenTransferModule,
    MagicMock,
    MagicMock,
    MagicMock,
    MagicMock,
]:
    ctx = MagicMock()
    ctx.chunk_size = 4
    ctx.event_bus = MagicMock()
    object_keys = [_make_object_key(0), _make_object_key(1)]
    ctx.resolve_obj_keys.return_value = [object_keys]
    session = MagicMock()
    ctx.session_manager.get_or_create.return_value = session

    cache_context = MagicMock()
    cache_context.device = torch.device("cpu")
    cache_context.stream = MagicMock()
    cache_context.cupy_stream = object()
    cache_context.max_batch_size = 1
    cache_context.calculate_num_blocks.return_value = 2
    groups = cache_context.kv_layer_groups_manager
    groups.num_object_groups = 1
    groups.num_kernel_groups = 1

    event_backend = MagicMock()
    final_event = object()
    event_backend.create_event.return_value = final_event
    event_backend.export_event.side_effect = lambda event, _device: {
        final_event: b"final-event",
    }.get(event, event)

    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)
    module._ctx = ctx
    module._lock = threading.Lock()
    module._cache_contexts = {
        7: ContextEntry(
            cache_context=cache_context,
            model_name="test-model",
            world_size=1,
            event_backend=event_backend,
        )
    }
    module._fused_counter_lock = threading.Lock()
    module._fused_success_count = 0
    module._fused_pipelined_count = 0
    module._fused_staged_fallback_count = 0
    module._fused_failure_count = 0
    return module, ctx, cache_context, event_backend, session


def _invoke(
    module: LMCacheDrivenTransferModule,
    loaded: tuple[list[ObjectKey], list[MagicMock]] | None,
    *,
    callback_side_effect: Exception | None = None,
) -> tuple[bytes, tuple[int, bool]]:
    load_raw_block_prefix = cast(
        MagicMock,
        module.context.storage_manager.load_raw_block_prefix,
    )
    load_raw_block_prefix.return_value = loaded
    with (
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.device",
            return_value=nullcontext(),
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.stream",
            return_value=nullcontext(),
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.get_layout_desc",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "downsample_and_stage_block_ids",
            return_value=[MagicMock()],
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "transfer_kv_per_object_group"
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "submit_callback_to_stream",
            side_effect=callback_side_effect,
        ),
    ):
        return module.fused_raw_block_retrieve(
            _make_key(),
            7,
            [list(range(4))],
            b"producer-event",
            0,
        )


def test_fused_raw_block_protocol_and_handler_routing():
    assert (
        RequestType.FUSED_RAW_BLOCK_RETRIEVE.value
        == RequestType.GET_EXPERIMENTAL.value + 1
    )
    assert (
        RequestType.FUSED_RAW_BLOCK_DRAIN.value
        == RequestType.FUSED_RAW_BLOCK_RETRIEVE.value + 1
    )
    assert get_payload_classes(RequestType.FUSED_RAW_BLOCK_RETRIEVE) == [
        IPCCacheServerKey,
        int,
        list[list[int]],
        bytes,
        int,
    ]
    assert (
        get_response_class(RequestType.FUSED_RAW_BLOCK_RETRIEVE)
        == tuple[bytes, tuple[int, bool]]
    )
    assert (
        get_handler_type(RequestType.FUSED_RAW_BLOCK_RETRIEVE) is HandlerType.BLOCKING
    )
    assert get_payload_classes(RequestType.FUSED_RAW_BLOCK_DRAIN) == [str, int]
    assert get_response_class(RequestType.FUSED_RAW_BLOCK_DRAIN) is bool
    assert get_handler_type(RequestType.FUSED_RAW_BLOCK_DRAIN) is HandlerType.BLOCKING

    module, _, _, _, _ = _make_module()
    pools = {spec.request_type: spec.pool for spec in module.get_handlers()}
    assert pools[RequestType.FUSED_RAW_BLOCK_RETRIEVE] is ThreadPoolType.AFFINITY
    assert pools[RequestType.FUSED_RAW_BLOCK_DRAIN] is ThreadPoolType.AFFINITY


def test_capability_requires_exact_supported_raw_block_topology():
    ctx = MagicMock()
    ctx.storage_manager.supports_fused_raw_block_retrieve.return_value = True
    module = ManagementModule(ctx, experimental_transfer=["other-feature"])

    assert module.get_experimental() == [
        "other-feature",
        FUSED_RAW_BLOCK_RETRIEVE_CAPABILITY,
    ]

    ctx.storage_manager.supports_fused_raw_block_retrieve.return_value = False
    assert module.get_experimental() == ["other-feature"]


def test_status_reports_undersubscribed_affinity_pool():
    module, _, _, _, _ = _make_module()
    module._affinity_worker_count = 1
    module._cache_contexts[7].world_size = 8

    assert module.report_status()["affinity_pool"] == {
        "worker_count": 1,
        "max_registered_world_size": 8,
        "undersubscribed": True,
    }


def test_fused_retrieve_returns_final_event_and_retains_exporter():
    module, ctx, _, event_backend, session = _make_module()
    object_keys = ctx.resolve_obj_keys.return_value[0]
    memory_objs = [MagicMock(), MagicMock()]
    for memory_obj in memory_objs:
        memory_obj.get_size.return_value = 64

    response = _invoke(module, (object_keys, memory_objs))

    assert response[1] == (8, True)
    assert response[0] == b"final-event"
    event_backend.wait_event.assert_called_once_with(
        event_backend.import_event.return_value,
        module._cache_contexts[7].cache_context.stream,
    )
    event_backend.record_event.assert_called_once()
    session.retain_resources.assert_called_once()
    resource_key, resources = session.retain_resources.call_args.args
    assert resource_key == _fused_event_session_resource_key(0)
    assert len(resources) == 1
    drain_event, terminal_safe = resources[0].snapshot()
    assert drain_event == _FusedDrainEvent(
        backend=event_backend,
        event=event_backend.create_event.return_value,
        device=module._cache_contexts[7].cache_context.device,
    )
    assert terminal_safe is False
    assert session.retain_resources.call_args.kwargs == {"owner_id": 0}
    assert module.report_status()["fused_raw_block_retrieve"] == {
        "success_count": 1,
        "pipelined_count": 0,
        "staged_fallback_count": 1,
        "failure_count": 0,
    }


def test_fused_retrieve_automatically_pipelines_loaded_objects():
    module, ctx, cache_context, _, _ = _make_module()
    object_keys = ctx.resolve_obj_keys.return_value[0]
    memory_objs = [MagicMock(), MagicMock()]
    for memory_obj in memory_objs:
        memory_obj.get_size.return_value = 64
    cache_context.max_batch_size = 4

    def load_prefix(_keys, _layout, **kwargs):
        assert kwargs["completion_batch_size"] == 1
        callback = kwargs["on_batch_loaded"]
        callback(0, 1, object_keys[:1], memory_objs[:1])
        callback(1, 2, object_keys[1:], memory_objs[1:])
        return object_keys, memory_objs

    ctx.storage_manager.load_raw_block_prefix.side_effect = load_prefix
    staged_block_ids = torch.arange(4)
    with (
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.device",
            return_value=nullcontext(),
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.stream",
            return_value=nullcontext(),
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.get_layout_desc",
            return_value=MagicMock(),
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "downsample_and_stage_block_ids",
            return_value=[staged_block_ids],
        ),
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "lmcache_memcpy_async_h2d"
        ) as stage_h2d,
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "_launch_staged_h2d_batch"
        ) as launch,
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "transfer_kv_per_object_group"
        ) as fallback,
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
            "submit_callback_to_stream"
        ),
    ):
        response = module.fused_raw_block_retrieve(
            _make_key(),
            7,
            [list(range(4))],
            b"producer-event",
            0,
        )

    assert response[1] == (8, True)
    assert response[0] == b"final-event"
    assert stage_h2d.call_args_list == [
        call(
            memory_objs[0],
            cache_context.get_temp_object_group_buffer.return_value,
        ),
        call(
            memory_objs[1],
            cache_context.get_temp_object_group_buffer.return_value,
        ),
    ]
    assert cache_context.get_temp_object_group_buffer.call_args_list == [
        call(0, 0),
        call(1, 0),
    ]
    launch.assert_called_once()
    launch_args = launch.call_args
    assert torch.equal(launch_args.args[1][0], staged_block_ids)
    assert launch_args.kwargs == {
        "object_group_id": 0,
        "batch_len": 2,
        "skip_first_n_tokens": 0,
    }
    fallback.assert_not_called()
    assert module.report_status()["fused_raw_block_retrieve"] == {
        "success_count": 1,
        "pipelined_count": 1,
        "staged_fallback_count": 0,
        "failure_count": 0,
    }


def test_pipelined_stage_failure_records_final_event_and_cleans_handed_prefix():
    module, ctx, _, event_backend, session = _make_module()
    object_keys = ctx.resolve_obj_keys.return_value[0]
    memory_objs = [MagicMock(), MagicMock()]

    def load_prefix(_keys, _layout, **kwargs):
        kwargs["on_batch_loaded"](0, 1, object_keys[:1], memory_objs[:1])
        raise AssertionError("staging failure must escape the callback")

    ctx.storage_manager.load_raw_block_prefix.side_effect = load_prefix

    with patch(
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
        "submit_callback_to_stream"
    ) as submit_callback:
        with (
            patch(
                "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
                "torch_dev.device",
                return_value=nullcontext(),
            ),
            patch(
                "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
                "torch_dev.stream",
                return_value=nullcontext(),
            ),
            patch(
                "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
                "get_layout_desc",
                return_value=MagicMock(),
            ),
            patch(
                "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
                "downsample_and_stage_block_ids",
                return_value=[torch.arange(4)],
            ),
            patch(
                "lmcache.v1.multiprocess.modules.lmcache_driven_transfer."
                "lmcache_memcpy_async_h2d",
                side_effect=RuntimeError("staging failed"),
            ),
        ):
            response = module.fused_raw_block_retrieve(
                _make_key(),
                7,
                [list(range(4))],
                b"producer-event",
                0,
            )

    assert response[1] == (0, False)
    assert response[0] == b"final-event"
    event_backend.record_event.assert_called_once()
    submit_callback.assert_called_once_with(
        module._cache_contexts[7].cache_context.cupy_stream,
        "finish_raw_block_restore",
        object_keys[:1],
    )
    session.retain_resources.assert_called_once()
    resource_key, resources = session.retain_resources.call_args.args
    assert resource_key == _fused_event_session_resource_key(0)
    assert len(resources) == 1
    drain_event, terminal_safe = resources[0].snapshot()
    assert drain_event == _FusedDrainEvent(
        backend=event_backend,
        event=event_backend.create_event.return_value,
        device=module._cache_contexts[7].cache_context.device,
    )
    assert terminal_safe is False
    assert session.retain_resources.call_args.kwargs == {"owner_id": 0}
    assert module.report_status()["fused_raw_block_retrieve"]["failure_count"] == 1


def test_fused_raw_block_clean_miss_is_success_without_transfer():
    module, _, _, _, _ = _make_module()

    response = _invoke(module, ([], []))

    assert response[1] == (0, True)
    assert response[0] == b"final-event"


def test_cached_capability_after_adapter_deletion_becomes_clean_miss():
    module, _, _, _, _ = _make_module()

    response = _invoke(module, None)

    assert response[1] == (0, True)
    assert response[0] == b"final-event"
    assert module.report_status()["fused_raw_block_retrieve"] == {
        "success_count": 1,
        "pipelined_count": 0,
        "staged_fallback_count": 0,
        "failure_count": 0,
    }


def test_response_construction_failure_drains_final_event_server_side():
    module, _, _, event_backend, _ = _make_module()
    event_backend.export_event.side_effect = RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        _invoke(module, ([], []))

    event_backend.synchronize_event.assert_called_once_with(
        event_backend.create_event.return_value,
        module._cache_contexts[7].cache_context.device,
    )
    assert module.report_status()["fused_raw_block_retrieve"]["failure_count"] == 1


def test_rank_specific_drain_synchronizes_retained_final_events():
    module, ctx, cache_context, event_backend, session = _make_module()
    final_events = [object(), object()]
    drain_states = [_FusedDrainState(), _FusedDrainState()]
    for state, final_event in zip(drain_states, final_events, strict=True):
        state.publish_final(
            _FusedDrainEvent(event_backend, final_event, cache_context.device),
        )
    session.get_retained_resources.return_value = [
        drain_states[0],
        object(),
        drain_states[1],
    ]
    ctx.session_manager.get.return_value = session

    with patch(
        "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.device",
        return_value=nullcontext(),
    ):
        assert module.fused_raw_block_drain("request-1", 3) is True

    session.get_retained_resources.assert_called_once_with(
        _fused_event_session_resource_key(3)
    )
    assert event_backend.synchronize_event.call_args_list == [
        call(final_events[0], cache_context.device),
        call(final_events[1], cache_context.device),
    ]


def test_rank_specific_drain_reports_missing_session():
    module, ctx, _, _, _ = _make_module()
    ctx.session_manager.get.return_value = None

    assert module.fused_raw_block_drain("missing", 0) is False


def test_record_event_failure_fences_stream_and_cleans_temporary_l1():
    module, ctx, cache_context, event_backend, _ = _make_module()
    object_keys = ctx.resolve_obj_keys.return_value[0]
    memory_objs = [MagicMock(), MagicMock()]
    event_backend.record_event.side_effect = RuntimeError("record failed")

    with pytest.raises(RuntimeError, match="record failed"):
        _invoke(module, (object_keys, memory_objs))

    cache_context.stream.synchronize.assert_called_once_with()
    ctx.storage_manager.finish_raw_block_restore.assert_called_once_with(object_keys)


def test_end_event_publication_failure_fences_final_and_cleans_temporary_l1():
    module, ctx, cache_context, event_backend, _ = _make_module()
    object_keys = ctx.resolve_obj_keys.return_value[0]
    memory_objs = [MagicMock(), MagicMock()]
    ctx.event_bus.publish_on_stream.side_effect = [
        None,
        RuntimeError("end event failed"),
    ]

    with pytest.raises(RuntimeError, match="end event failed"):
        _invoke(module, (object_keys, memory_objs))

    event_backend.synchronize_event.assert_called_once_with(
        event_backend.create_event.return_value,
        cache_context.device,
    )
    ctx.storage_manager.finish_raw_block_restore.assert_called_once_with(object_keys)


def test_cleanup_callback_failure_fences_final_and_cleans_temporary_l1():
    module, ctx, cache_context, event_backend, _ = _make_module()
    object_keys = ctx.resolve_obj_keys.return_value[0]
    memory_objs = [MagicMock(), MagicMock()]

    with pytest.raises(RuntimeError, match="callback failed"):
        _invoke(
            module,
            (object_keys, memory_objs),
            callback_side_effect=RuntimeError("callback failed"),
        )

    event_backend.synchronize_event.assert_called_once_with(
        event_backend.create_event.return_value,
        cache_context.device,
    )
    ctx.storage_manager.finish_raw_block_restore.assert_called_once_with(object_keys)


def test_drain_state_retention_failure_occurs_before_any_stream_work():
    module, ctx, _, event_backend, session = _make_module()
    session.retain_resources.side_effect = RuntimeError("retain failed")

    with pytest.raises(RuntimeError, match="retain failed"):
        _invoke(module, ([], []))

    ctx.storage_manager.load_raw_block_prefix.assert_not_called()
    event_backend.create_event.assert_not_called()


@pytest.mark.parametrize(
    ("failure", "error_type", "error_match"),
    [
        ("missing_context", ValueError, "No GPU context"),
        ("missing_event_backend", RuntimeError, "no event backend"),
    ],
)
def test_preflight_failure_publishes_terminal_safe_drain_state(
    failure: str,
    error_type: type[Exception],
    error_match: str,
):
    module, ctx, _, event_backend, session = _make_module()
    if failure == "missing_context":
        module._cache_contexts.clear()
    else:
        module._cache_contexts[7].event_backend = None

    with pytest.raises(error_type, match=error_match):
        _invoke(module, ([], []))

    session.retain_resources.assert_called_once()
    resource_key, resources = session.retain_resources.call_args.args
    assert resource_key == _fused_event_session_resource_key(0)
    assert len(resources) == 1
    drain_state = resources[0]
    assert drain_state.snapshot() == (None, True)
    assert session.retain_resources.call_args.kwargs == {"owner_id": 0}
    event_backend.create_event.assert_not_called()
    ctx.storage_manager.load_raw_block_prefix.assert_not_called()

    ctx.session_manager.get.return_value = session
    session.get_retained_resources.return_value = [drain_state]
    assert module.fused_raw_block_drain("request-1", 0) is True
    event_backend.synchronize_event.assert_not_called()


def test_close_fences_retained_events_and_stream_before_dispatcher_teardown():
    module, ctx, cache_context, event_backend, session = _make_module()
    final_event = object()
    drain_state = _FusedDrainState()
    drain_state.publish_final(
        _FusedDrainEvent(event_backend, final_event, cache_context.device)
    )
    ctx.session_manager.sessions_snapshot.return_value = [session]
    session.get_retained_resources_by_prefix.return_value = [drain_state]
    module._device_host_func_dispatcher = MagicMock()
    call_order: list[str] = []
    event_backend.synchronize_event.side_effect = lambda *_args: call_order.append(
        "event"
    )
    cache_context.stream.synchronize.side_effect = lambda: call_order.append("stream")
    module._device_host_func_dispatcher.stop.side_effect = lambda: call_order.append(
        "dispatcher"
    )

    with (
        patch(
            "lmcache.v1.multiprocess.modules.lmcache_driven_transfer.torch_dev.device",
            return_value=nullcontext(),
        ),
        patch.object(
            module,
            "_release_entries",
            side_effect=lambda _entries: call_order.append("release"),
        ),
    ):
        module.close()

    assert call_order == ["event", "stream", "dispatcher", "release"]
    session.get_retained_resources_by_prefix.assert_called_once_with(
        "fused_raw_block_export_events."
    )
    event_backend.synchronize_event.assert_called_once_with(
        final_event,
        cache_context.device,
    )
    assert module.context_entries_snapshot() == {}


def _make_staged_context() -> MagicMock:
    cache_context = MagicMock()
    cache_context.device = torch.device("cpu")
    cache_context.lmcache_tokens_per_chunk = 4
    groups = cache_context.kv_layer_groups_manager
    groups.object_groups = [SimpleNamespace(kernel_group_indices=[0])]
    groups.get_subchunk_sw_size_tokens.return_value = 4
    cache_context.calculate_num_blocks.side_effect = (
        lambda token_count, _kernel_group_id: token_count // 2
    )
    cache_context.get_kernel_group_kv_pointers.return_value = object()
    cache_context.get_temp_kernel_group_buffer.side_effect = (
        lambda slot, _kernel_group_id: SimpleNamespace(
            data_ptr=lambda: 1000 + slot,
        )
    )
    shape_desc = object()
    cache_context.get_shape_desc.return_value = shape_desc
    cache_context.get_slots_per_chunk_in_sw.return_value = 4
    cache_context.get_engine_kv_format.return_value = object()
    return cache_context


def test_launch_staged_h2d_batch_uses_distinct_slots_and_prefix_skip():
    cache_context = _make_staged_context()
    block_ids = torch.arange(4)

    with patch.object(lmc_ops, "multi_layer_block_kv_transfer") as transfer:
        _launch_staged_h2d_batch(
            cache_context,
            [block_ids],
            object_group_id=0,
            batch_len=2,
            skip_first_n_tokens=4,
        )

    transfer.assert_called_once()
    args = transfer.call_args.args
    assert args[0] is cache_context.get_kernel_group_kv_pointers.return_value
    assert args[1] == [1000, 1001]
    assert torch.equal(args[2], block_ids)
    assert args[3] == cache_context.device
    assert args[4] is lmc_ops.TransferDirection.H2D
    assert args[5] is cache_context.get_shape_desc.return_value
    assert args[6] == 4
    assert args[7] is cache_context.get_engine_kv_format.return_value
    assert args[8] == 2
    assert cache_context.get_temp_kernel_group_buffer.call_args_list == [
        call(0, 0),
        call(1, 0),
    ]


def test_launch_staged_h2d_batch_skips_fully_covered_window():
    cache_context = _make_staged_context()
    with patch.object(lmc_ops, "multi_layer_block_kv_transfer") as transfer:
        _launch_staged_h2d_batch(
            cache_context,
            [torch.arange(4)],
            object_group_id=0,
            batch_len=2,
            skip_first_n_tokens=8,
        )

    transfer.assert_not_called()
    cache_context.get_temp_kernel_group_buffer.assert_not_called()
