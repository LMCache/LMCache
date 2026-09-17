# SPDX-License-Identifier: Apache-2.0
"""Synthetic cache layouts: layer shapes, physical descriptions, then allocation."""

# Standard
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated, Literal
import hashlib
import math

# Third Party
import msgspec
import torch
import yaml

# First Party
from lmcache.utils import EngineType
from lmcache.v1.gpu_connector.utils import normalize_and_discover_per_layer_formats
from lmcache.v1.kv_layer_groups import KVLayerGroupsManager, group_layers_by_identity
from lmcache.v1.multiprocess.group_view import (
    EngineGroupInfo,
    expand_engine_block_ids,
    slice_block_ids_per_group,
)

PositiveInt = Annotated[int, msgspec.Meta(gt=0)]


def _round_up(n: int, alignment: int) -> int:
    return -(-n // alignment) * alignment


def _resolve_layer_for_tp(
    layer: "KDA | MHA | MLA | DSV4", tp_size: int
) -> "RankLocalLayer":
    """Resolve one global layer definition into rank-local cache geometry."""
    if isinstance(layer, KDA):
        if layer.num_heads % tp_size:
            raise ValueError("KDA num_heads must be divisible by TP")
        return RankLocalLayer(
            msgspec.structs.replace(layer, num_heads=layer.num_heads // tp_size),
            sharded=True,
        )
    if isinstance(layer, MHA):
        if layer.num_attention_heads % layer.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if layer.num_attention_heads % tp_size:
            raise ValueError("MHA num_attention_heads must be divisible by TP")
        local_attention_heads = layer.num_attention_heads // tp_size
        if layer.num_key_value_heads == 1:
            local_kv_heads, sharded = 1, False
        elif layer.num_key_value_heads < tp_size:
            raise ValueError(
                "replicated GQA with fewer KV heads than TP is unsupported"
            )
        elif layer.num_key_value_heads % tp_size:
            raise ValueError("MHA num_key_value_heads must be divisible by TP")
        else:
            local_kv_heads, sharded = (
                layer.num_key_value_heads // tp_size,
                tp_size > 1,
            )
        return RankLocalLayer(
            msgspec.structs.replace(
                layer,
                num_attention_heads=local_attention_heads,
                num_key_value_heads=local_kv_heads,
            ),
            sharded=sharded,
        )
    if isinstance(layer, (MLA, DSV4)):
        return RankLocalLayer(layer, sharded=False)
    raise TypeError(f"unsupported layer type: {type(layer).__name__}")


def _components(
    layer: "RankLocalLayer",
    tokens_per_block: int,
    window_tokens_per_block: int | None = None,
) -> list["ComponentSpec"]:
    """Resolve one rank-local layer into per-block component storage layouts."""
    definition = layer.definition
    if isinstance(definition, KDA):
        conv = (
            definition.num_heads
            * 3
            * definition.head_size
            * (definition.short_conv_kernel_size - 1)
        )
        state = definition.num_heads * definition.head_size**2
        return [
            ComponentSpec.recurrent(
                suffix="conv",
                dtype=torch.bfloat16,
                state_elements=conv,
                tokens_per_block=tokens_per_block,
                sharded=layer.sharded,
            ),
            ComponentSpec.recurrent(
                suffix="state",
                dtype=torch.float32,
                state_elements=state,
                tokens_per_block=tokens_per_block,
                sharded=layer.sharded,
            ),
        ]
    elif isinstance(definition, MHA):
        component_tokens_per_block = (
            math.gcd(tokens_per_block, definition.sliding_window)
            if definition.sliding_window is not None
            else tokens_per_block
        )
        return [
            ComponentSpec(
                suffix=suffix,
                dtype=torch.bfloat16,
                elements_per_slot=definition.num_key_value_heads * definition.head_dim,
                tokens_per_block=component_tokens_per_block,
                slots_per_block=component_tokens_per_block,
                sw_size_tokens=definition.sliding_window or -1,
                sharded=layer.sharded,
            )
            for suffix in ("key", "value")
        ]
    elif isinstance(definition, MLA):
        width = definition.kv_lora_rank + definition.qk_rope_head_dim
        return [
            ComponentSpec(
                suffix="mla",
                dtype=torch.bfloat16,
                elements_per_slot=width,
                tokens_per_block=tokens_per_block,
                slots_per_block=tokens_per_block,
                sharded=layer.sharded,
            )
        ]
    elif isinstance(definition, DSV4):
        compress_ratio = definition.compress_ratio
        # Supported encoded rows: fp8_ds_mla=584 bytes, DSV4 indexer=132 bytes.
        # State caches expose their actual 4/8-token cache block sizes below;
        # those values are page geometry, not a separate update-span concept.
        if tokens_per_block % compress_ratio:
            raise ValueError(
                "tokens_per_block must be divisible by compression ratio "
                f"{compress_ratio}"
            )
        result: list[ComponentSpec] = []
        if compress_ratio == 4:
            result += [
                ComponentSpec(
                    suffix="indexer_state",
                    dtype=torch.float32,
                    elements_per_slot=512,
                    tokens_per_block=4,
                    slots_per_block=4,
                    sw_size_tokens=8,
                    sharded=layer.sharded,
                ),
                ComponentSpec(
                    suffix="indexer",
                    dtype=torch.uint8,
                    elements_per_slot=132,
                    tokens_per_block=tokens_per_block,
                    slots_per_block=tokens_per_block // compress_ratio,
                    sharded=layer.sharded,
                ),
            ]
        swa_tokens_per_block = window_tokens_per_block or math.gcd(
            tokens_per_block, 128
        )
        result.append(
            ComponentSpec(
                suffix="swa",
                dtype=torch.uint8,
                elements_per_slot=584,
                tokens_per_block=swa_tokens_per_block,
                slots_per_block=swa_tokens_per_block,
                sw_size_tokens=128,
                sharded=layer.sharded,
            )
        )
        if compress_ratio > 1:
            state_tokens_per_block, state_elements_per_slot, state_sw_size_tokens = (
                (4, 2048, 8) if compress_ratio == 4 else (8, 1024, 128)
            )
            result += [
                ComponentSpec(
                    suffix="compressed",
                    dtype=torch.uint8,
                    elements_per_slot=584,
                    tokens_per_block=tokens_per_block,
                    slots_per_block=tokens_per_block // compress_ratio,
                    sharded=layer.sharded,
                ),
                ComponentSpec(
                    suffix="compressor_state",
                    dtype=torch.float32,
                    elements_per_slot=state_elements_per_slot,
                    tokens_per_block=state_tokens_per_block,
                    slots_per_block=state_tokens_per_block,
                    sw_size_tokens=state_sw_size_tokens,
                    sharded=layer.sharded,
                ),
            ]
        return result
    raise TypeError(f"unsupported layer type: {type(definition).__name__}")


def _make_unplaced_tensor_spec(
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    tokens_per_block: int,
    sw_size_tokens: int = -1,
    recurrent_state: bool = False,
    sharded: bool = False,
) -> "TensorSpec":
    """Build a tensor spec before group IDs and storage placement are assigned."""
    group_template = EngineGroupInfo(
        # Replaced with the outer group index by
        # _assign_tensor_group_placement().
        engine_group_id=0,
        tokens_per_block=tokens_per_block,
        sw_size_tokens=sw_size_tokens,
        recurrent_state=recurrent_state,
    )
    return TensorSpec(
        name=name,
        dtype=dtype,
        shape=shape,
        group=group_template,
        sharded=sharded,
        stride=math.prod(shape) * dtype.itemsize,
    )


def _build_component_tensor_groups(
    layers: list["RankLocalLayer"],
    tokens_per_block: int,
    window_tokens_per_block: int | None = None,
) -> list[list["TensorSpec"]]:
    """Resolve component tensors and group matching cache semantics."""
    groups: dict[tuple[int, int, bool], list[TensorSpec]] = defaultdict(list)
    for index, layer in enumerate(layers):
        for component in _components(layer, tokens_per_block, window_tokens_per_block):
            name = f"model.layers.{index}.{component.suffix}"
            shape = (component.slots_per_block, component.elements_per_slot)
            key = (
                component.tokens_per_block,
                component.sw_size_tokens,
                component.recurrent_state,
            )
            groups[key].append(
                _make_unplaced_tensor_spec(
                    name=name,
                    shape=shape,
                    dtype=component.dtype,
                    tokens_per_block=component.tokens_per_block,
                    sw_size_tokens=component.sw_size_tokens,
                    recurrent_state=component.recurrent_state,
                    sharded=component.sharded,
                )
            )
    return list(groups.values())


def _basic_layout(
    layers: list["RankLocalLayer"], layout: "Layout"
) -> list["TensorSpec"]:
    """Build the engine-independent layout with one owner per tensor."""
    if layout.kv_cache_dtype is not None:
        raise ValueError("basic uses layer dtypes; kv_cache_dtype requires vllm")
    return _assign_tensor_group_placement(
        _build_component_tensor_groups(layers, layout.block_size),
        "per_tensor",
    )


def _assign_tensor_group_placement(
    groups: list[list["TensorSpec"]],
    placement: Literal["per_tensor", "shared_by_position", "packed_by_group"],
    alignment: int = 1,
) -> list["TensorSpec"]:
    """Assign storage owners, offsets, strides, ID spaces, and group IDs.

    ``per_tensor`` gives every tensor a dedicated owner. ``shared_by_position``
    shares an owner between tensors at the same position in different engine
    groups. ``packed_by_group`` packs each group's tensors at distinct offsets
    in one shared owner.
    """
    shares_owners = placement != "per_tensor"
    sizes = [[_round_up(t.stride, alignment) for t in group] for group in groups]
    stride = (
        max(map(sum, sizes))
        if placement == "packed_by_group"
        else max(max(s) for s in sizes)
    )
    result: list[TensorSpec] = []
    for gid, group in enumerate(groups):
        offset = 0
        for slot, tensor in enumerate(group):
            pool = (
                (0 if placement == "packed_by_group" else slot)
                if shares_owners
                else len(result)
            )
            result.append(
                replace(
                    tensor,
                    pool=pool,
                    offset=offset,
                    stride=stride if shares_owners else tensor.stride,
                    domain=0 if shares_owners else gid,
                    group=msgspec.structs.replace(tensor.group, engine_group_id=gid),
                )
            )
            if placement == "packed_by_group":
                offset += sizes[gid][slot]
    return result


def _vllm_layout(
    layers: list["RankLocalLayer"], layout: "Layout"
) -> list["TensorSpec"]:
    kinds = {type(layer.definition) for layer in layers}
    if kinds == {DSV4}:
        if layout.kv_cache_dtype != "fp8_ds_mla" or layout.block_size != 256:
            raise ValueError("vllm DSV4 requires fp8_ds_mla and block_size=256")
        groups = _build_component_tensor_groups(layers, 256, 64)
        full = [t for g in groups for t in g if t.group.sw_size_tokens < 0]
        windows = [g for g in groups if g[0].group.sw_size_tokens > 0]
        if not full:
            raise ValueError("vllm DSV4 requires compressed attention")
        sizes = [
            max(Counter(_round_up(t.stride, 576) for t in g).values())
            for g in [full, *windows]
        ]
        tuples = min(
            range(sizes[0], max(sizes) + 1),
            key=lambda n: (sum((-s) % n for s in sizes), -n),
        )
        groups = [full]
        for pages in windows:
            by_size: dict[int, list[TensorSpec]] = defaultdict(list)
            for t in pages:
                by_size[_round_up(t.stride, 576)].append(t)
            rows = list(zip(*by_size.values(), strict=True))
            count = math.ceil(len(rows) / tuples)
            groups.extend(
                [t for row in rows[j::count] for t in row] for j in range(count)
            )
        return _assign_tensor_group_placement(groups, "packed_by_group", 576)
    if kinds != {KDA, MLA} or layout.kv_cache_dtype != "bfloat16":
        raise ValueError("vllm supports DSV4 or KDA+MLA bfloat16, not this mix")
    if layout.block_size != 256:
        raise ValueError("vllm KDA+MLA derives block size; leave block_size at default")
    if (
        len({x.definition.num_heads for x in layers if isinstance(x.definition, KDA)})
        != 1
    ):
        raise ValueError("vllm requires uniform KDA head counts")
    kda = next(x for x in layers if isinstance(x.definition, KDA))
    state_bytes = sum(
        component.elements_per_slot
        * component.slots_per_block
        * component.dtype.itemsize
        for component in _components(kda, 1)
    )
    mla = next(_components(x, 1)[0] for x in layers if isinstance(x.definition, MLA))
    row_bytes = mla.elements_per_slot * mla.dtype.itemsize
    tokens_per_block = _round_up(math.ceil(state_bytes / row_bytes), 128)
    buckets: dict[bool, list[TensorSpec]] = defaultdict(list)
    for index, layer in enumerate(layers):
        state = isinstance(layer.definition, KDA)
        name = f"model.layers.{index}.{'state' if state else 'mla'}"
        shape = (
            (tokens_per_block, 1, row_bytes)
            if state
            else (tokens_per_block, mla.elements_per_slot)
        )
        dtype = torch.int8 if state else mla.dtype
        buckets[state].append(
            _make_unplaced_tensor_spec(
                name=name,
                shape=shape,
                dtype=dtype,
                tokens_per_block=tokens_per_block,
                sw_size_tokens=tokens_per_block if state else -1,
                recurrent_state=state,
                sharded=layer.sharded,
            )
        )
    sizes = [len(g) for g in buckets.values()]
    group_size = max(sizes) if max(sizes) < min(sizes) * 1.5 else min(sizes)
    groups = []
    for pages in buckets.values():
        count = math.ceil(len(pages) / group_size)
        groups.extend(pages[j::count] for j in range(count))
    return _assign_tensor_group_placement(groups, "shared_by_position")


def load_model_layout(path: str | Path) -> "ModelLayout":
    """Parse a YAML path; raise ValueError for invalid fields, layers or geometry."""
    try:
        with Path(path).open() as stream:
            data = yaml.load(stream, Loader=_UniqueLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    model = data.get("model") if isinstance(data, dict) else None
    if isinstance(model, dict) and "preset" in model:
        if len(model) != 1:
            raise ValueError("model.preset cannot accompany layer definitions")
        if model["preset"] == "kimi_k3":
            model = Model(
                {"linear": KDA(96), "latent": MLA()},
                ["linear", "linear", "linear", "latent"] * 23 + ["latent"],
            )
        elif model["preset"] == "deepseek_v4_flash":
            model = Model(
                {"swa": DSV4(1), "c4": DSV4(4), "c128": DSV4(128)},
                ["swa", "swa"] + ["c4", "c128"] * 20 + ["c4"],
            )
        else:
            raise ValueError("unknown model.preset")
        data["model"] = msgspec.to_builtins(model)
    spec = msgspec.convert(data, ModelLayout)
    order = spec.model.layer_types
    if not order or any(n not in spec.model.layer_definitions for n in order):
        raise ValueError("model.layer_types must be nonempty and reference definitions")
    resolve_rank_local_tensor_specs(spec)
    return spec


def resolve_rank_local_tensor_specs(
    spec: "ModelLayout",
) -> tuple[list["TensorSpec"], EngineType]:
    """Resolve a model layout into placed tensor specs for one TP rank.

    Args:
        spec: Global model, parallelism, and cache-layout configuration.

    Returns:
        The rank-local tensor specs with storage placement assigned, together
        with the engine type that consumes those specs.

    Raises:
        ValueError: If TP cannot partition a sharded layer or if the requested
            cache layout is incompatible with the model.
    """
    layers = [
        _resolve_layer_for_tp(spec.model.layer_definitions[name], spec.parallel.tp_size)
        for name in spec.model.layer_types
    ]
    if spec.layout.mode == "basic":
        return _basic_layout(layers, spec.layout), EngineType.MOCK
    if spec.layout.mode == "vllm":
        return _vllm_layout(layers, spec.layout), EngineType.VLLM
    raise ValueError(f"unsupported layout mode: {spec.layout.mode}")


def allocate_layout(
    specs: list["TensorSpec"], nb: int, chunk: int, device: str, engine: EngineType
) -> "ModelCache":
    """Allocate described pools/views and production groups; reject invalid bounds.

    specs comes from resolve_rank_local_tensor_specs; nb includes null block 0,
    chunk is the actual server token chunk, device is a Torch device string.
    Returns owners, registered caches and a group manager; raises ValueError for
    invalid geometry.
    """
    strides: dict[int, int] = {}
    for t in specs:
        size = math.prod(t.shape) * t.dtype.itemsize
        if (
            nb < 1
            or chunk < 1
            or chunk % t.group.tokens_per_block
            or min(t.shape) < 1
            or t.offset < 0
            or t.offset + size > t.stride
            or t.offset % t.dtype.itemsize
            or t.stride % t.dtype.itemsize
        ):
            raise ValueError("invalid capacity, chunk_size, view bounds or alignment")
        if strides.setdefault(t.pool, t.stride) != t.stride:
            raise ValueError("views in one pool must share a block stride")
    owners = {
        pool: torch.zeros(nb * stride, dtype=torch.uint8, device=device)
        for pool, stride in strides.items()
    }
    caches = {
        t.name: owners[t.pool]
        .view(t.dtype)
        .as_strided(
            (nb, *t.shape),
            (
                t.stride // t.dtype.itemsize,
                *(math.prod(t.shape[i + 1 :]) for i in range(len(t.shape))),
            ),
            t.offset // t.dtype.itemsize,
        )
        for t in specs
    }
    layers: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(specs):
        layers[t.group.engine_group_id].append(i)
    tensors, formats = normalize_and_discover_per_layer_formats(
        list(caches.values()), list(layers.values()), engine, {"kv_layout": "NHD"}
    )
    identities = group_layers_by_identity(
        tensors, formats, [t.group.engine_group_id for t in specs]
    )
    infos = [
        msgspec.structs.replace(specs[indices[0]].group, layer_indices=tuple(indices))
        for _, indices in identities
    ]
    manager = KVLayerGroupsManager(
        tensors, formats, infos, chunk, separate_object_groups=True
    )
    return ModelCache(specs, owners, caches, infos, manager)


class _UniqueLoader(yaml.SafeLoader):
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Reject duplicate YAML keys instead of silently overwriting values."""
        mapping = super().construct_mapping(node, deep=deep)
        if len(mapping) != len(node.value):
            raise ValueError(f"duplicate YAML key at {node.start_mark}")
        return mapping


class KDA(msgspec.Struct, tag="kda", tag_field="kind", forbid_unknown_fields=True):
    num_heads: PositiveInt
    head_size: Literal[128] = 128
    short_conv_kernel_size: Literal[4] = 4


class MHA(msgspec.Struct, tag="mha", tag_field="kind", forbid_unknown_fields=True):
    """Standard explicit K/V attention, including MHA, GQA and MQA."""

    num_attention_heads: PositiveInt
    num_key_value_heads: PositiveInt
    head_dim: PositiveInt
    sliding_window: PositiveInt | None = None


class MLA(msgspec.Struct, tag="mla", tag_field="kind", forbid_unknown_fields=True):
    kv_lora_rank: Literal[512] = 512
    qk_rope_head_dim: Literal[64] = 64


class DSV4(
    msgspec.Struct, tag="dsv4_attention", tag_field="kind", forbid_unknown_fields=True
):
    compress_ratio: Literal[1, 4, 128]


@dataclass(frozen=True)
class RankLocalLayer:
    """One layer after TP geometry and cache ownership have been resolved."""

    definition: KDA | MHA | MLA | DSV4
    sharded: bool


class Model(msgspec.Struct, forbid_unknown_fields=True):
    layer_definitions: dict[str, KDA | MHA | MLA | DSV4]
    layer_types: list[str]


class Parallel(msgspec.Struct, forbid_unknown_fields=True):
    tp_size: PositiveInt = 1


class Allocation(msgspec.Struct, forbid_unknown_fields=True):
    num_blocks: PositiveInt


class Layout(msgspec.Struct, forbid_unknown_fields=True):
    mode: Literal["basic", "vllm"] = "basic"
    block_size: PositiveInt = 256
    kv_cache_dtype: Literal["bfloat16", "fp8_ds_mla"] | None = None


class ModelLayout(msgspec.Struct, forbid_unknown_fields=True):
    model: Model
    allocation: Allocation
    parallel: Parallel = msgspec.field(default_factory=Parallel)
    layout: Layout = msgspec.field(default_factory=Layout)

    @property
    def cache_name(self) -> str:
        """Hash effective layer/layout/TP semantics, excluding names and capacity."""
        fields = (
            self.parallel,
            self.layout,
            [self.model.layer_definitions[n] for n in self.model.layer_types],
        )
        encoded = msgspec.json.encode(fields)
        return "server-bench-layout-v3-" + hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True, kw_only=True)
class ComponentSpec:
    """Resolved storage layout and cache semantics for one layer component.

    ``tokens_per_block`` is the logical token coverage of one engine block ID.
    ``slots_per_block`` is the number of physical rows stored in that block,
    and ``elements_per_slot`` is the row width measured in ``dtype`` elements.
    ``sw_size_tokens`` follows :class:`EngineGroupInfo`; ``-1`` means that the
    component has no retained sliding window. ``sharded`` records whether TP
    ranks own distinct payloads for this component.
    """

    suffix: str
    dtype: torch.dtype
    elements_per_slot: int
    tokens_per_block: int
    slots_per_block: int
    sw_size_tokens: int = -1
    recurrent_state: bool = False
    sharded: bool = False

    @classmethod
    def recurrent(
        cls,
        *,
        suffix: str,
        dtype: torch.dtype,
        state_elements: int,
        tokens_per_block: int,
        sharded: bool,
    ) -> "ComponentSpec":
        """Lay out one fixed recurrent snapshot across an engine block.

        Args:
            suffix: Component suffix appended to the layer name.
            dtype: Storage dtype of the recurrent state.
            state_elements: Total elements in one unpadded state snapshot.
            tokens_per_block: Logical tokens covered by one engine block ID.
            sharded: Whether TP ranks own distinct state payloads.

        Returns:
            A resolved recurrent component with 16-byte-aligned physical rows.
        """
        elements_per_slot = _round_up(
            math.ceil(state_elements / tokens_per_block), 16 // dtype.itemsize
        )
        return cls(
            suffix=suffix,
            dtype=dtype,
            elements_per_slot=elements_per_slot,
            tokens_per_block=tokens_per_block,
            slots_per_block=tokens_per_block,
            sw_size_tokens=tokens_per_block,
            recurrent_state=True,
            sharded=sharded,
        )


@dataclass(frozen=True)
class TensorSpec:
    """Final block-first view; offset/stride are bytes, shape excludes NB."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    group: EngineGroupInfo
    sharded: bool = False
    pool: int = 0
    offset: int = 0
    stride: int = 0
    domain: int = 0


@dataclass
class ModelCache:
    """Own backing allocations until unregister; reuse production window metadata."""

    specs: list[TensorSpec]
    owners: dict[int, torch.Tensor]
    caches: dict[str, torch.Tensor]
    groups: list[EngineGroupInfo]
    manager: KVLayerGroupsManager

    def block_ids(
        self, total: int, sequence: int, start: int, end: int
    ) -> list[list[int]]:
        """Map a logical token interval to group IDs; reject misalignment/capacity."""
        spans = {t.group.engine_group_id: t.group.tokens_per_block for t in self.specs}
        domains = {t.group.engine_group_id: t.domain for t in self.specs}
        if not 0 <= start <= end <= total or any(
            v % b for b in spans.values() for v in (total, start, end)
        ):
            raise ValueError("invalid or unaligned token range")
        sizes: dict[int, int] = defaultdict(int)
        for gid, span in spans.items():
            sizes[domains[gid]] += total // span
        needed = max(sizes.values())
        capacity = next(iter(self.caches.values())).shape[0]
        if needed >= capacity:
            raise ValueError(
                f"request needs {needed} blocks plus null block; capacity={capacity}"
            )
        cursors = dict.fromkeys(sizes, 1 + sequence * needed % (capacity - needed))
        ids = {}
        for gid, span in spans.items():
            offset, count = cursors[domains[gid]], total // span
            ids[gid] = list(range(offset, offset + count))
            cursors[domains[gid]] += count
        return expand_engine_block_ids(
            self.groups,
            slice_block_ids_per_group(ids, list(spans.values()), start, end),
        )

    def fill(self, total: int, sequence: int, rank: int) -> None:
        """Fill request blocks by component/block/shard identity; retain sentinels."""
        for owner in self.owners.values():
            owner.fill_(165)
        tensors = list(self.caches.values())
        for group, ids in zip(
            self.groups, self.block_ids(total, sequence, 0, total), strict=True
        ):
            for index in group.layer_indices:
                t, tensor = self.specs[index], tensors[index]
                values = torch.arange(math.prod(t.shape), device=tensor.device)
                for logical, block in enumerate(ids):
                    identity = (
                        f"{sequence}:{t.name}:{rank if t.sharded else 0}:{logical}"
                    )
                    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16)
                    tensor[block].copy_(((values ^ seed) % 251 + 1).reshape(t.shape))

    def restore_views(
        self, total: int, sequence: int, start: int, end: int, chunk: int
    ) -> Iterator[torch.Tensor]:
        """Yield writable retained windows and final snapshots for the token range."""
        tensors = list(self.caches.values())
        for obj in self.manager.object_groups:
            first = (
                max(start, end - obj.sw_size_chunks * chunk)
                if obj.sw_size_chunks > 0
                else start
            )
            for pos in range(first, end, chunk):
                ids = self.block_ids(total, sequence, pos, pos + chunk)
                for k in obj.kernel_group_indices:
                    count = (
                        self.manager.get_subchunk_sw_size_tokens(k)
                        // self.groups[k].tokens_per_block
                    )
                    for index in self.groups[k].layer_indices:
                        for block in ids[k][-count:]:
                            yield tensors[index][block]

    def checksum(self) -> str:
        """Hash every backing byte, including padding and untouched regions."""
        digest = hashlib.sha256()
        for owner in self.owners.values():
            digest.update(owner.cpu().numpy().tobytes())
        return digest.hexdigest()
