# SPDX-License-Identifier: Apache-2.0
"""
Infer a hybrid model's KV group roles for the SDK pool.
"""

# Future
from __future__ import annotations

# Standard
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any
import dataclasses

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.group_view import EngineGroupInfo

logger = init_logger(__name__)

ATTENTION = "attention"
RECURRENT = "recurrent"

_ATTENTION_LAYER_TYPES = frozenset({"full_attention", "attention"})
_RECURRENT_LAYER_TYPES = frozenset(
    {"linear_attention", "mamba", "mamba2", "gated_deltanet"}
)


class HybridLayoutError(ValueError):
    """Raised when the registered layout cannot be matched to the model."""


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    """How the SDK pool mirrors one of the engine's kernel groups.

    Attributes:
        kernel_group_idx: Server kernel-group index (protocol order).
        engine_group_idx: Engine group (block-id space) of the group.
        object_group_idx: Object group the server stores the group in.
        layer_indices: Registered tensor indices of the group's layers.
        tokens_per_block: Tokens per paged block.
        role: `ATTENTION` or `RECURRENT`.
        sw_size_tokens: Window the engine registered for the group.
        kernel_block_size: Attention only: tokens per physical kernel page
            (``tokens_per_block`` when the page is not sub-paged).
        source: How ``kernel_block_size`` was decided, for the log.
    """

    kernel_group_idx: int
    engine_group_idx: int
    object_group_idx: int
    layer_indices: tuple[int, ...]
    tokens_per_block: int
    role: str
    sw_size_tokens: int
    kernel_block_size: int
    source: str

    @property
    def recurrent(self) -> bool:
        """Whether the group holds recurrent state pages."""
        return self.role == RECURRENT

    def engine_group_info(self) -> EngineGroupInfo:
        """The group as the engine registered it."""
        return EngineGroupInfo(
            engine_group_id=self.engine_group_idx,
            layer_indices=self.layer_indices,
            tokens_per_block=self.tokens_per_block,
            sw_size_tokens=self.sw_size_tokens,
            recurrent_state=self.recurrent,
        )


def text_config(hf_config: Any) -> Any:
    """Return the language-model part of a (possibly multimodal) config."""
    return getattr(hf_config, "text_config", None) or hf_config


def layer_roles(text_cfg: Any) -> list[str]:
    """Return each decoder layer's role from a text config.

    Uses ``layer_types``, else derives it from ``full_attention_interval``
    (every n-th layer is full attention, the rest linear attention).

    Args:
        text_cfg: The model's text config.

    Returns:
        `ATTENTION` or `RECURRENT` per decoder layer.

    Raises:
        HybridLayoutError: If the config names no layer types, or a type the
            SDK cannot serve (e.g. sliding-window attention).
    """
    layer_types = getattr(text_cfg, "layer_types", None)
    if not layer_types:
        interval = getattr(text_cfg, "full_attention_interval", None)
        num_layers = getattr(text_cfg, "num_hidden_layers", None)
        if not interval or not num_layers:
            raise HybridLayoutError(
                "the model config has neither layer_types nor "
                "full_attention_interval, so hybrid layer roles are unknown"
            )
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(num_layers)
        ]
    roles = []
    for layer_type in layer_types:
        if layer_type in _ATTENTION_LAYER_TYPES:
            roles.append(ATTENTION)
        elif layer_type in _RECURRENT_LAYER_TYPES:
            roles.append(RECURRENT)
        else:
            raise HybridLayoutError(
                f"layer type {layer_type!r} is not supported by the SDK's "
                f"hybrid pool (supported: {sorted(_ATTENTION_LAYER_TYPES)} "
                f"and {sorted(_RECURRENT_LAYER_TYPES)})"
            )
    return roles


def _object_group_roles(
    kernel_groups: Sequence[Mapping[str, Any]], roles: Sequence[str]
) -> dict[int, str]:
    """Match each object group to a layer role by its layer count.

    Raises:
        HybridLayoutError: If the counts do not identify every object group.
    """
    layers_per_object: Counter[int] = Counter()
    for group in kernel_groups:
        layers_per_object[int(group["object_group_idx"])] += int(group["num_layers"])
    role_counts = Counter(roles)
    if len(role_counts) != len(layers_per_object):
        raise HybridLayoutError(
            f"the model has {len(role_counts)} layer roles {dict(role_counts)}, "
            f"but the server stores {len(layers_per_object)} object groups "
            f"{dict(layers_per_object)}; start the server with "
            "--separate-object-groups"
        )
    if len(set(role_counts.values())) != len(role_counts):
        raise HybridLayoutError(
            f"layer roles {dict(role_counts)} have equal layer counts, so "
            "their object groups cannot be told apart"
        )
    by_count = {count: role for role, count in role_counts.items()}
    object_roles = {}
    for object_idx, count in layers_per_object.items():
        if count not in by_count:
            raise HybridLayoutError(
                f"object group {object_idx} has {count} layers, matching no "
                f"layer role of the model {dict(role_counts)}"
            )
        object_roles[object_idx] = by_count[count]
    return object_roles


def _registered_heads(group: Mapping[str, Any]) -> tuple[int, int]:
    """Return the registered ``(NH, HS)`` of a ``[NB, 2, BS, NH, HS]`` group.

    Raises:
        HybridLayoutError: If the format is not that one.
    """
    labels = str(group["engine_kv_shape"]).partition("[")[2].rstrip("] ").split(",")
    labels = [label.strip() for label in labels]
    if labels != ["NB", "2", "BS", "NH", "HS"]:
        raise HybridLayoutError(
            f"attention group {group['kernel_group_idx']} has format "
            f"{group['engine_kv_shape']}; the hybrid pool supports "
            "NL x [NB, 2, BS, NH, HS] only"
        )
    dims = str(group["engine_kv_concrete_shape"]).partition("[")[2].rstrip("] ")
    sizes = [int(size) for size in dims.split(",")]
    return sizes[3], sizes[4]


def attention_kernel_block_size(
    group: Mapping[str, Any],
    num_kv_heads: int,
    head_dim: int,
    override: int = 0,
) -> tuple[int, str]:
    """Return the kernel page size of an attention group's registered pages.

    A registered page with the model's real heads is not sub-paged. A page
    with one synthetic head of width ``num_kv_heads * head_dim`` is a
    sub-paged view: vLLM re-pages a hybrid model's inflated attention block
    at the attention backend's kernel block size, which the registration
    does not report, so it must be given (``sdk.kernel_block_size``).

    Args:
        group: The kernel group entry from ``/status``.
        num_kv_heads: KV heads per worker, from the model config.
        head_dim: Head size, from the model config.
        override: The kernel block size set on the server
            (``sdk.kernel_block_size``), or 0 if unset.

    Returns:
        ``(kernel_block_size, source)``, the source for the log.

    Raises:
        HybridLayoutError: If the registered heads match neither view, or
            the pages are sub-paged and ``override`` is unset or does not
            divide the block.
    """
    tokens_per_block = int(group["tokens_per_block"])
    heads, width = _registered_heads(group)
    if (heads, width) == (num_kv_heads, head_dim):
        return tokens_per_block, "real heads registered, not sub-paged"
    if heads != 1 or width != num_kv_heads * head_dim:
        raise HybridLayoutError(
            f"attention group {group['kernel_group_idx']} registers "
            f"{heads} heads x {width}, matching neither the model's "
            f"{num_kv_heads} x {head_dim} nor a sub-paged 1 x "
            f"{num_kv_heads * head_dim}"
        )
    if not override:
        raise HybridLayoutError(
            f"attention group {group['kernel_group_idx']} is sub-paged (one "
            f"synthetic head of {width}) and the registration does not report "
            "its kernel page size. Set vLLM's attention kernel block size on "
            "the server: --runtime-plugin-config "
            "'{\"sdk.kernel_block_size\": 32}' (32 for FlashAttention or "
            f"FlashInfer with a {tokens_per_block}-token block)"
        )
    if override >= tokens_per_block or tokens_per_block % override:
        raise HybridLayoutError(
            f"sdk.kernel_block_size={override} must be a proper divisor of the "
            f"sub-paged block size {tokens_per_block}"
        )
    return override, "server sdk.kernel_block_size"


def plan_hybrid_groups(
    kernel_groups: Sequence[Mapping[str, Any]],
    hf_config: Any,
    world_size: int,
    kernel_block_override: int = 0,
) -> list[GroupPlan]:
    """Plan how the SDK pool mirrors a hybrid model's kernel groups.

    Args:
        kernel_groups: The ``kernel_groups`` of the engine's ``/status``
            layout, in kernel-group order.
        hf_config: The model's Hugging Face config.
        world_size: Tensor-parallel world size of the registration.
        kernel_block_override: The kernel block size set on the server
            (``sdk.kernel_block_size``), or 0 if unset; required when the
            attention pages are sub-paged.

    Returns:
        One :class:`GroupPlan` per kernel group, in kernel-group order.

    Raises:
        HybridLayoutError: If the registration cannot be matched to the
            model config.
    """
    text_cfg = text_config(hf_config)
    roles = layer_roles(text_cfg)
    num_registered = sum(int(g["num_layers"]) for g in kernel_groups)
    if num_registered != len(roles):
        raise HybridLayoutError(
            f"the server registers {num_registered} layers but the model "
            f"config has {len(roles)} decoder layers"
        )
    object_roles = _object_group_roles(kernel_groups, roles)
    num_heads = int(text_cfg.num_attention_heads)
    num_kv_heads = int(getattr(text_cfg, "num_key_value_heads", num_heads))
    head_dim = int(
        getattr(text_cfg, "head_dim", None) or int(text_cfg.hidden_size) // num_heads
    )
    kv_heads_per_worker = max(1, num_kv_heads // world_size)

    plans = []
    for group in kernel_groups:
        object_idx = int(group["object_group_idx"])
        role = object_roles[object_idx]
        tokens_per_block = int(group["tokens_per_block"])
        if int(group["slots_per_block"]) != tokens_per_block:
            raise HybridLayoutError(
                f"kernel group {group['kernel_group_idx']} is compressed "
                "(slots_per_block != tokens_per_block)"
            )
        if role == RECURRENT:
            kernel, source = tokens_per_block, "recurrent state, opaque pages"
            sw_size_tokens = tokens_per_block
        else:
            kernel, source = attention_kernel_block_size(
                group, kv_heads_per_worker, head_dim, kernel_block_override
            )
            sw_size_tokens = -1
        plans.append(
            GroupPlan(
                kernel_group_idx=int(group["kernel_group_idx"]),
                engine_group_idx=int(group["engine_group_idx"]),
                object_group_idx=object_idx,
                layer_indices=tuple(int(i) for i in group["layer_indices"]),
                tokens_per_block=tokens_per_block,
                role=role,
                sw_size_tokens=sw_size_tokens,
                kernel_block_size=kernel,
                source=source,
            )
        )
    _log_plan(plans, text_cfg, roles, kv_heads_per_worker, head_dim, world_size)
    return plans


def _recurrent_state_bytes(text_cfg: Any, world_size: int) -> int | None:
    """Bytes of one GDN layer's conv + SSM state, from ``linear_*`` fields.

    Best effort, for the log: None when the config lacks the fields.
    """
    try:
        key_heads = int(text_cfg.linear_num_key_heads) // world_size
        value_heads = int(text_cfg.linear_num_value_heads) // world_size
        key_dim = int(text_cfg.linear_key_head_dim)
        value_dim = int(text_cfg.linear_value_head_dim)
        conv_kernel = int(text_cfg.linear_conv_kernel_dim)
    except (AttributeError, TypeError):
        return None
    conv_dim = 2 * key_heads * key_dim + value_heads * value_dim
    ssm_elem = 4 if str(getattr(text_cfg, "mamba_ssm_dtype", "")) == "float32" else 2
    conv_bytes = (conv_kernel - 1) * conv_dim * 2
    return conv_bytes + value_heads * key_dim * value_dim * ssm_elem


def _log_plan(
    plans: Sequence[GroupPlan],
    text_cfg: Any,
    roles: Sequence[str],
    kv_heads: int,
    head_dim: int,
    world_size: int,
) -> None:
    """Log the inferred plan, one line per kernel group."""
    counts = Counter(roles)
    logger.info(
        "Hybrid model config (%s): %d decoder layers, %d attention + %d "
        "recurrent; attention %d KV heads x %d per worker (world_size=%d), "
        "partial_rotary_factor=%s; mamba_ssm_dtype=%s",
        type(text_cfg).__name__,
        len(roles),
        counts[ATTENTION],
        counts[RECURRENT],
        kv_heads,
        head_dim,
        world_size,
        getattr(text_cfg, "partial_rotary_factor", None),
        getattr(text_cfg, "mamba_ssm_dtype", None),
    )
    state_bytes = _recurrent_state_bytes(text_cfg, world_size)
    if state_bytes is not None:
        logger.info(
            "Recurrent layer state (conv + ssm) from the config: %d bytes per "
            "layer per snapshot",
            state_bytes,
        )
    for plan in plans:
        logger.info(
            "Hybrid kernel group %d: %s, engine group %d, object group %d, "
            "%d layers %s, %d tokens/block, sw_size_tokens=%d, "
            "kernel_block_size=%d (%s)",
            plan.kernel_group_idx,
            plan.role,
            plan.engine_group_idx,
            plan.object_group_idx,
            len(plan.layer_indices),
            list(plan.layer_indices),
            plan.tokens_per_block,
            plan.sw_size_tokens,
            plan.kernel_block_size,
            plan.source,
        )
