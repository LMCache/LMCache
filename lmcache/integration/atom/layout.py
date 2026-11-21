# SPDX-License-Identifier: Apache-2.0
"""Convert ATOM's store layout into LMCache engine groups.

ATOM describes its KV store as named tensors plus the kernel groups that read
them; LMCache registers the same information as :class:`EngineGroupInfo`
values. The descriptor shapes here mirror ``atom/kv_transfer/offload/mp/api.py``
in the external-connector design (v2), so ATOM can hand its own objects over
unchanged and LMCache owns the conversion to its registration payload.

The input is duck-typed: any object carrying ``tensor_names``,
``engine_group_id``, ``tokens_per_block`` and the optional fields works, which
keeps LMCache testable without importing ATOM.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# First Party
from lmcache.v1.common.group_view import EngineGroupInfo


@dataclass(frozen=True)
class AtomKernelGroup:
    """One ATOM kernel group, as the external-connector API defines it.

    Attributes:
        tensor_names: Names of the tensors this group reads.
        engine_group_id: Engine group these tensors live in; dense from 0.
        tokens_per_block: Tokens one engine block id covers in this group.
        sw_size_tokens: Sliding-window size in tokens, ``-1`` when unused.
        recurrent_state: Whether the pages hold recurrent state (Mamba/GDN).
        extra_object_group_tag: Connector-private extra-group tag, ``0`` for a
            regular group.
        tp_replicas: Consecutive KV ranks holding identical copies of this
            group; ATOM reports its collapse factor here.
        dcp_shards: Token shards those ranks are split into under decode
            context parallelism; ``1`` without DCP.
    """

    tensor_names: tuple[str, ...]
    engine_group_id: int
    tokens_per_block: int
    sw_size_tokens: int = -1
    recurrent_state: bool = False
    extra_object_group_tag: int = 0
    tp_replicas: int = 1
    dcp_shards: int = 1


@dataclass(frozen=True)
class AtomStoreLayout:
    """ATOM's store layout: the tensors, the groups that read them, state spec.

    Attributes:
        tensors: Named tensors ATOM registered, in registration order.
        groups: Kernel groups reading those tensors.
        native_state: ATOM's native-state convention object, passed through
            unread; the LMCache backend forwards it to the state path.
    """

    tensors: Mapping[str, Any]
    groups: tuple[AtomKernelGroup, ...] = field(default_factory=tuple)
    native_state: Any | None = None


def _field(group: Any, name: str) -> Any:
    """Read one descriptor field, failing loudly when it is absent."""
    try:
        return getattr(group, name)
    except AttributeError as exc:
        raise ValueError(
            f"ATOM kernel group {group!r} has no {name!r}; expected a descriptor "
            "with tensor_names, engine_group_id and tokens_per_block"
        ) from exc


def _optional_field(group: Any, name: str, default: Any) -> Any:
    """Read one descriptor field, falling back to the API's default."""
    return getattr(group, name, default)


def engine_group_infos_from_atom(layout: AtomStoreLayout) -> list[EngineGroupInfo]:
    """Build LMCache engine groups from an ATOM store layout.

    Tensor names are resolved to their registration index here, because that is
    what LMCache's registration payload carries; ATOM names its tensors and
    never has to know the ordering LMCache uses.

    Args:
        layout: ATOM's store layout: named tensors, kernel groups, optional
            native-state spec.

    Returns:
        One :class:`EngineGroupInfo` per kernel group, in input order.

    Raises:
        ValueError: If there are no groups or tensors, a group names an unknown
            or already claimed tensor, engine group ids are not dense from 0, a
            token count is not positive, or a group's replication factors are
            invalid.
    """
    descriptors = list(layout.groups)
    if not descriptors:
        raise ValueError("ATOM registered no kernel groups")
    tensor_names = list(layout.tensors)
    if not tensor_names:
        raise ValueError("ATOM registered no tensors")
    tensor_index = {name: index for index, name in enumerate(tensor_names)}
    if len(tensor_index) != len(tensor_names):
        raise ValueError("ATOM tensor names must be unique")

    infos: list[EngineGroupInfo] = []
    claimed: dict[str, int] = {}
    group_ids: set[int] = set()

    for index, descriptor in enumerate(descriptors):
        engine_group_id = int(_field(descriptor, "engine_group_id"))
        tokens_per_block = int(_field(descriptor, "tokens_per_block"))
        names = tuple(str(name) for name in _field(descriptor, "tensor_names"))

        if tokens_per_block < 1:
            raise ValueError(
                f"ATOM kernel group {index} covers {tokens_per_block} tokens per "
                "block; a group must cover at least one"
            )
        if not names:
            raise ValueError(f"ATOM kernel group {index} names no tensors")
        layer_indices: list[int] = []
        for name in names:
            if name not in tensor_index:
                raise ValueError(
                    f"ATOM kernel group {index} names tensor {name!r}, which is "
                    f"not registered (known: {sorted(tensor_index)[:4]}...)"
                )
            if name in claimed:
                raise ValueError(
                    f"tensor {name!r} appears in kernel groups "
                    f"{claimed[name]} and {index}"
                )
            claimed[name] = index
            layer_indices.append(tensor_index[name])
        group_ids.add(engine_group_id)

        tp_replicas = int(_optional_field(descriptor, "tp_replicas", 1))
        dcp_shards = int(_optional_field(descriptor, "dcp_shards", 1))
        if tp_replicas < 1:
            raise ValueError(
                f"engine group {engine_group_id}: tp_replicas must be >= 1, "
                f"got {tp_replicas}"
            )
        if dcp_shards < 1:
            raise ValueError(
                f"engine group {engine_group_id}: dcp_shards must be >= 1, "
                f"got {dcp_shards}"
            )
        if dcp_shards > tp_replicas:
            raise ValueError(
                f"engine group {engine_group_id}: dcp_shards ({dcp_shards}) cannot "
                f"exceed tp_replicas ({tp_replicas})"
            )

        infos.append(
            EngineGroupInfo(
                engine_group_id=engine_group_id,
                layer_indices=tuple(layer_indices),
                tokens_per_block=tokens_per_block,
                sw_size_tokens=int(_optional_field(descriptor, "sw_size_tokens", -1)),
                recurrent_state=bool(
                    _optional_field(descriptor, "recurrent_state", False)
                ),
                extra_object_group_tag=int(
                    _optional_field(descriptor, "extra_object_group_tag", 0)
                ),
                tp_replicas=tp_replicas,
                dcp_shards=dcp_shards,
            )
        )

    if group_ids != set(range(len(group_ids))):
        raise ValueError(
            f"ATOM engine group ids must be dense from 0; got {sorted(group_ids)}"
        )
    return infos


def page_kernel_groups(
    tensor_names: Sequence[Sequence[str]],
    *,
    tokens_per_block: int,
) -> tuple[AtomKernelGroup, ...]:
    """Build PAGE kernel groups from ATOM's per-group tensor names.

    PAGE registers every group as engine group ``0`` and lets LMCache repeat the
    group id across physically distinct tensors; this keeps that shape in one
    place instead of an inline comprehension.

    Args:
        tensor_names: Tensor names per registered group, in order.
        tokens_per_block: ATOM's block size in tokens.

    Returns:
        One :class:`AtomKernelGroup` per tensor group.

    Raises:
        ValueError: If there are no groups or a group is empty.
    """
    if not tensor_names:
        raise ValueError("ATOM published no PAGE tensor groups")
    groups: list[AtomKernelGroup] = []
    for index, names in enumerate(tensor_names):
        resolved = tuple(str(name) for name in names)
        if not resolved:
            raise ValueError(f"PAGE tensor group {index} is empty")
        groups.append(
            AtomKernelGroup(
                tensor_names=resolved,
                engine_group_id=0,
                tokens_per_block=tokens_per_block,
            )
        )
    return tuple(groups)


__all__ = [
    "AtomKernelGroup",
    "AtomStoreLayout",
    "engine_group_infos_from_atom",
    "page_kernel_groups",
]
