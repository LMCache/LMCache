# SPDX-License-Identifier: Apache-2.0
"""通过公开请求结束接口验证混合对象组在LRU压力下保留完整前缀。"""

# Standard
from unittest.mock import MagicMock
import hashlib

# Third Party
import pytest

# First Party
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.api import ObjectKey, ipc_key_to_object_keys
from lmcache.v1.distributed.bitmap_ops.fold import fold_unfold_grouped
from lmcache.v1.distributed.eviction_policy.lru import LRUEvictionPolicy
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lookup import LookupModule


def make_hashes(label: str, count: int) -> list[bytes]:
    """生成确定性的合成分块摘要；不读取模型权重或真实请求。

    Args:
        label: 区分不同合成请求的固定标签。
        count: 生成的分块数量。

    Returns:
        按分块位置排列的SHA256摘要。
    """
    return [hashlib.sha256(f"{label}:{i}".encode()).digest() for i in range(count)]


def make_key(world_size: int, worker_id: int | None = None) -> IPCCacheServerKey:
    """构造仅用于元数据展开的合成IPC键，保留缓存隔离域。

    Args:
        world_size: 请求所需的KV分片数量。
        worker_id: 指定单个分片；None表示全部分片。

    Returns:
        不包含真实输入数据的IPC键。
    """
    return IPCCacheServerKey(
        model_name="synthetic-model",
        world_size=world_size,
        worker_id=worker_id,
        token_bytes=b"",
        start=0,
        end=0,
        request_id="synthetic-request",
        cache_salt="synthetic-tenant",
    )


def make_module(
    key: IPCCacheServerKey, hashes: list[bytes], groups: int
) -> tuple[LookupModule, MagicMock]:
    """模拟外部存储和会话边界，正常构造LookupModule并调用公开接口。

    Args:
        key: 待结束会话的IPC键。
        hashes: 会话按前缀顺序记录的分块摘要。
        groups: 布局注册表声明的对象组数。

    Returns:
        真实查询模块和可观测其存储、会话、事件交互的模拟上下文。
    """
    session = MagicMock()
    session.lookup_ipc_key = key
    session.get_hashes.return_value = hashes
    ctx = MagicMock()
    ctx.layout_desc_registry.find_attn_desc.return_value.num_object_groups = groups
    ctx.session_manager.remove.return_value = session
    return LookupModule(ctx), ctx


def matched_chunks(
    present: set[ObjectKey], per_group: list[list[ObjectKey]], attention_group: int
) -> int:
    """使用真实CPU位图内核合并全部分片；完整注意力必须保留连续前缀。

    Args:
        present: 当前仍可用的缓存对象集合。
        per_group: 各对象组按分块及KV分片排列的完整请求键。
        attention_group: 需要连续前缀的完整注意力组编号；另一组窗口为1。

    Returns:
        全部对象组与分片能够共同服务的前缀分块数。
    """
    rows: list[Bitmap] = []
    windows: list[int] = []
    for group_id, objects in enumerate(per_group):
        for rank in sorted({obj.kv_rank for obj in objects}):
            rank_keys = [obj for obj in objects if obj.kv_rank == rank]
            bitmap = Bitmap(len(rank_keys))
            for index, obj in enumerate(rank_keys):
                if obj in present:
                    bitmap.set(index)
            rows.append(bitmap)
            windows.append(-1 if group_id == attention_group else 1)
    return fold_unfold_grouped(rows, windows)[0]


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("count", [0, 1, 17])
def test_end_session_touches_complete_chunks(
    world_size: int, groups: int, count: int
) -> None:
    """请求结束须保持全部键的身份和前缀顺序，覆盖空输入及单组模型。"""
    key, hashes = make_key(world_size), make_hashes("matrix", count)
    per_group = ipc_key_to_object_keys(key, hashes, list(range(groups)))
    module, ctx = make_module(key, hashes, groups)
    module.end_session(key.request_id)
    touched = ctx.storage_manager.touch_cached_keys.call_args.args[0]
    expected = [obj for row in zip(*per_group, strict=True) for obj in row]
    assert touched == expected
    assert len(touched) == count * world_size * groups
    assert set(touched) == {obj for row in per_group for obj in row}
    ctx.session_manager.remove.assert_called_once_with(key.request_id)
    assert ctx.event_bus.publish.call_count == 2


def test_worker_specific_session_only_touches_its_rank() -> None:
    """工作进程键只展开自身分片，结束会话不得扩大到其他工作进程。"""
    key, hashes = make_key(4, worker_id=2), make_hashes("worker", 7)
    module, ctx = make_module(key, hashes, 2)
    module.end_session(key.request_id)
    touched = ctx.storage_manager.touch_cached_keys.call_args.args[0]
    assert len(touched) == 14
    assert len({obj.kv_rank for obj in touched}) == 1


@pytest.mark.parametrize("attention_group", [0, 1])
def test_repeated_eviction_retains_usable_hybrid_prefix(attention_group: int) -> None:
    """同等淘汰及外来写入压力下，两组均应保留前半段完整前缀。"""
    key, hashes = make_key(4), make_hashes("history", 100)
    per_group = ipc_key_to_object_keys(key, hashes, [0, 1])
    policy = LRUEvictionPolicy()
    present: set[ObjectKey] = set()
    for row in zip(*per_group, strict=True):
        policy.on_keys_created(list(row))
        present.update(row)
    module, ctx = make_module(key, hashes, 2)
    ctx.storage_manager.touch_cached_keys.side_effect = policy.on_keys_touched
    module.end_session(key.request_id)

    for step in range(1, 6):
        actions = policy.get_eviction_actions(0.1, lambda _: True)
        removed = [obj for action in actions for obj in action.keys]
        assert len(removed) == 80
        present.difference_update(removed)
        policy.on_keys_removed(removed)
        other = ipc_key_to_object_keys(key, make_hashes(f"other-{step}", 10), [0, 1])
        for row in zip(*other, strict=True):
            policy.on_keys_created(list(row))
            present.update(row)
        # 位图合并验证可服务前缀，避免只统计尚未被淘汰的零散对象。
        assert matched_chunks(present, per_group, attention_group) == 100 - step * 10

    assert [sum(obj in present for obj in row) for row in per_group] == [200, 200]


def test_end_session_keeps_locked_objects_ineligible_for_eviction() -> None:
    """触碰顺序不能改变原有锁过滤规则，锁定对象不得进入淘汰候选。"""
    key, hashes = make_key(4), make_hashes("locked", 5)
    per_group = ipc_key_to_object_keys(key, hashes, [0, 1])
    keys = [obj for row in per_group for obj in row]
    locked = set(per_group[0][:4] + per_group[1][:4])
    policy = LRUEvictionPolicy()
    policy.on_keys_created(keys)
    module, ctx = make_module(key, hashes, 2)
    ctx.storage_manager.touch_cached_keys.side_effect = policy.on_keys_touched
    module.end_session(key.request_id)
    actions = policy.get_eviction_actions(1.0, lambda obj: obj not in locked)
    removed = {obj for action in actions for obj in action.keys}
    assert removed == set(keys) - locked


@pytest.mark.parametrize("missing", [True, False])
def test_end_session_without_lookup_does_not_touch_storage(missing: bool) -> None:
    """不存在或尚未绑定查询键的会话只执行既有清理，不更新缓存顺序。"""
    key = make_key(1)
    module, ctx = make_module(key, [], 1)
    ctx.session_manager.remove.return_value = (
        None if missing else MagicMock(lookup_ipc_key=None)
    )
    module.end_session(key.request_id)
    ctx.storage_manager.touch_cached_keys.assert_not_called()
    ctx.session_manager.remove.assert_called_once_with(key.request_id)
