# SPDX-License-Identifier: Apache-2.0

"""Tests for the two store-path trigger points of the Dynamo store hook.

Verifies that ``EngineDrivenTransferModule.commit_store`` and
``LMCacheDrivenTransferModule.store`` invoke ``publish_store_from_key`` with the
committed ``(key, object_keys)`` on their success branch when a publisher is
installed, and skip it when none is.

The heavy storage/GPU dependencies are mocked so the success branch is reached
without real hardware; the modules themselves are imported normally.
"""

# Standard
from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

# Third Party
import pytest  # noqa: F401  (kept for parity / future fixtures)

# First Party
from lmcache.v1.multiprocess.modules.engine_driven_transfer import (
    EngineDrivenTransferModule,
)
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)

# Hashable stand-in for ObjectKey (used as a dict key in the store path).
_ObjKey = namedtuple("_ObjKey", ["chunk_hash"])

_ED_MOD = "lmcache.v1.multiprocess.modules.engine_driven_transfer"
_LD_MOD = "lmcache.v1.multiprocess.modules.lmcache_driven_transfer"


def _make_key() -> SimpleNamespace:
    """A synthetic IPCCacheServerKey (only the fields the hook reads)."""
    return SimpleNamespace(request_id="req-1", token_ids=(0, 1, 2, 3), start=0, end=4)


# --------------------------------------------------------------------------- #
# Engine-driven path: commit_store
# --------------------------------------------------------------------------- #


def _make_engine_module(publisher: object) -> EngineDrivenTransferModule:
    """An EngineDrivenTransferModule wired to reach the success branch."""
    module = EngineDrivenTransferModule.__new__(EngineDrivenTransferModule)
    session = SimpleNamespace(extras={})
    module._ctx = SimpleNamespace(  # type: ignore[assignment]
        session_manager=SimpleNamespace(get_or_create=lambda _rid: session),
        chunk_size=16,
        dynamo_kv_publisher=publisher,
    )
    strategy = Mock()
    strategy.commit_store.return_value = True
    module._resolve_for_transfer = Mock(  # type: ignore[method-assign]
        return_value=(Mock(), strategy)
    )
    return module


def test_engine_driven_commit_store_triggers_hook() -> None:
    publisher = Mock()
    module = _make_engine_module(publisher)
    key = _make_key()
    object_keys = [_ObjKey(b"c0"), _ObjKey(b"c1")]
    module._resolve_single_group_obj_keys = Mock(  # type: ignore[method-assign]
        return_value=object_keys
    )

    with patch(f"{_ED_MOD}.publish_store_from_key") as hook:
        result = module.commit_store(key, instance_id=1, cpu_data=b"")

    assert result is True
    hook.assert_called_once_with(publisher, key, object_keys)


def test_engine_driven_commit_store_no_publisher_skips_hook() -> None:
    module = _make_engine_module(publisher=None)
    module._resolve_single_group_obj_keys = Mock(  # type: ignore[method-assign]
        return_value=[_ObjKey(b"c0")]
    )

    with patch(f"{_ED_MOD}.publish_store_from_key") as hook:
        result = module.commit_store(_make_key(), instance_id=1, cpu_data=b"")

    assert result is True
    hook.assert_not_called()


# --------------------------------------------------------------------------- #
# LMCache-driven path: store()
# --------------------------------------------------------------------------- #


def _make_lmcache_module(
    publisher: object,
) -> tuple[LMCacheDrivenTransferModule, _ObjKey]:
    """An LMCacheDrivenTransferModule wired to reach the store success branch."""
    module = LMCacheDrivenTransferModule.__new__(LMCacheDrivenTransferModule)

    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager.num_object_groups = 1
    cache_context.kv_layer_groups_manager.num_kernel_groups = 1
    cache_context.calculate_num_blocks.return_value = 1

    entry = SimpleNamespace(cache_context=cache_context, model_name="m")
    module.get_and_touch_context_entry = Mock(  # type: ignore[method-assign]
        return_value=entry
    )

    obj_key = _ObjKey(b"c0")
    storage_manager = MagicMock()
    storage_manager.reserve_write.return_value = {
        obj_key: MagicMock(get_size=lambda: 64)
    }

    module._ctx = SimpleNamespace(  # type: ignore[assignment]
        resolve_obj_keys=Mock(return_value=[[obj_key]]),
        chunk_size=16,
        storage_manager=storage_manager,
        event_bus=MagicMock(),
        dynamo_kv_publisher=publisher,
    )
    return module, obj_key


def _run_lmcache_store(
    module: LMCacheDrivenTransferModule,
) -> tuple[object, tuple[bytes, bool], MagicMock]:
    """Call store() with all heavy collaborators patched out."""
    key = _make_key()
    with (
        patch(f"{_LD_MOD}.torch_dev"),
        patch(f"{_LD_MOD}.check_interprocess_event_support"),
        patch(f"{_LD_MOD}.downsample_and_stage_block_ids", return_value=MagicMock()),
        patch(f"{_LD_MOD}.get_layout_desc", return_value=MagicMock()),
        patch(f"{_LD_MOD}.transfer_kv_per_object_group"),
        patch(f"{_LD_MOD}.submit_callback_to_stream"),
        patch(f"{_LD_MOD}.publish_store_from_key") as hook,
    ):
        result = module.store(
            key,
            instance_id=1,
            gpu_block_ids=[[0]],
            event_ipc_handle=b"",
        )
    return key, result, hook


def test_lmcache_driven_store_triggers_hook() -> None:
    module, obj_key = _make_lmcache_module(publisher=Mock())
    key, result, hook = _run_lmcache_store(module)

    assert result[1] is True
    hook.assert_called_once_with(module._ctx.dynamo_kv_publisher, key, [obj_key])


def test_lmcache_driven_store_no_publisher_skips_hook() -> None:
    module, _obj_key = _make_lmcache_module(publisher=None)
    _key, result, hook = _run_lmcache_store(module)

    assert result[1] is True
    hook.assert_not_called()
