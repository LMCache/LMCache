# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for distributed tests."""

# Standard
from collections.abc import Callable
from typing import Any, ClassVar

# First Party
from lmcache.v1.distributed.api import (
    GroupedObjectKeys,
    MemoryLayoutDesc,
    ObjectKey,
    PrefetchTaskSpec,
)
from lmcache.v1.distributed.internal_api import L1ManagerInterface
from lmcache.v1.distributed.storage_controllers.write_policy import (
    WritePolicy,
    register_write_policy,
)
from lmcache.v1.platform import current_device_spec

ROUND_ROBIN_WRITE_POLICY = "test-round-robin"


SCRIPTED_WRITE_POLICY = "test-scripted"

WriteAnswer = Callable[
    [list[ObjectKey], list[L1ManagerInterface]], dict[int, list[ObjectKey]]
]


class RoundRobinWritePolicy(WritePolicy):
    """Spread keys over the L1 managers in turn, continuing across calls.

    The first key ever seen goes to the first manager, the next to the second,
    and so on, so single-key calls alternate managers too.
    """

    def __init__(self) -> None:
        self._next = 0

    def select_write_targets(
        self, keys: list[ObjectKey], l1_managers: list[L1ManagerInterface]
    ) -> dict[int, list[ObjectKey]]:
        targets: dict[int, list[ObjectKey]] = {}
        for key in keys:
            owner = l1_managers[self._next % len(l1_managers)].l1_manager_id
            self._next += 1
            targets.setdefault(owner, []).append(key)
        return targets


class ScriptedWritePolicy(WritePolicy):
    """Answer with whatever the test installed in ``ScriptedWritePolicy.answer``."""

    answer: ClassVar[WriteAnswer | None] = None

    def select_write_targets(
        self, keys: list[ObjectKey], l1_managers: list[L1ManagerInterface]
    ) -> dict[int, list[ObjectKey]]:
        assert ScriptedWritePolicy.answer is not None, "test did not script an answer"
        return ScriptedWritePolicy.answer(keys, l1_managers)


register_write_policy(ROUND_ROBIN_WRITE_POLICY, RoundRobinWritePolicy)
register_write_policy(SCRIPTED_WRITE_POLICY, ScriptedWritePolicy)


def should_use_lazy_alloc() -> bool:
    """Return whether the current platform supports lazy L1 allocation."""
    return current_device_spec.is_pin_supported


def single_row_spec(
    keys: list[ObjectKey], layout_desc: MemoryLayoutDesc, **spec_kwargs: Any
) -> PrefetchTaskSpec:
    """A prefetch spec with one key row: object group 0, a single kv rank.

    ``spec_kwargs`` are forwarded to :class:`PrefetchTaskSpec` (e.g.
    ``num_kv_readers``, ``fetching_policy``, ``lock_mode``).
    """
    return PrefetchTaskSpec(
        key_groups=[
            GroupedObjectKeys(keys=keys, object_group_id=0, layout_desc=layout_desc)
        ],
        **spec_kwargs,
    )


def ranked_spec(
    keys_by_rank: list[list[ObjectKey]],
    layout_desc: MemoryLayoutDesc,
    **spec_kwargs: Any,
) -> PrefetchTaskSpec:
    """A single-object-group prefetch spec with one key row per kv rank.

    ``keys_by_rank[r]`` is the chunk-ordered key list of kv rank ``r``.
    ``spec_kwargs`` are forwarded to :class:`PrefetchTaskSpec`.
    """
    return PrefetchTaskSpec(
        key_groups=[
            GroupedObjectKeys(keys=keys, object_group_id=0, layout_desc=layout_desc)
            for keys in keys_by_rank
        ],
        **spec_kwargs,
    )
