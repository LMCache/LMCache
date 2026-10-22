# SPDX-License-Identifier: Apache-2.0
"""Optional, engine-neutral hints an engine may register with the MP server.

An engine connector sends one :class:`EngineHints` per ``(model_name,
world_size)`` at startup, after the server advertises
:data:`REUSABLE_LOAD_RULE_FEATURE` through ``get_experimental``. The engine
decides what to fill in; the server never infers engine behaviour. Every field
is optional and its default means "no hints", and the structs are msgpack maps,
so decoders ignore fields they do not know.
"""

# Third Party
import msgspec

ENGINE_HINTS_METHOD = "lmcache.engine_hints.register"
"""``server_module_call`` method that registers :class:`EngineHints`."""

REUSABLE_LOAD_RULE_FEATURE = "reusable_load_rule"
"""``get_experimental`` capability: the server handles
:data:`ENGINE_HINTS_METHOD` and applies an accepted rule in its lookups."""

REUSABLE_LOAD_RULE_VERSION = 1

REUSABLE_LOAD_RULE_KILL_SWITCH = "LMCACHE_MAMBA_EAGLE_REUSABLE_LOAD"
"""Environment variable; ``0`` turns the rule off on the side that reads it."""


class ReusableLoadRule(msgspec.Struct, frozen=True, kw_only=True):
    """Where a load may end so that the engine can reuse what it loaded.

    For a lookup whose tokens match ``N`` stored tokens (what the
    full-attention object groups hold), the engine reuses a load ending at
    most at ``reusable_end(N) = max(N // granularity_tokens *
    granularity_tokens - drop_tokens, 0)``. When the engine's own hit ``H``
    (lookup hint ``lmcache.hint.engine_computed_tokens``) is below that and the
    longest servable prefix ``M`` ends above it, the server returns the highest
    servable prefix ``L`` with ``H < L <= reusable_end(N)`` instead of ``M``.
    """

    version: int = REUSABLE_LOAD_RULE_VERSION
    granularity_tokens: int = 0
    drop_tokens: int = 0
    object_group_kinds: tuple[str, ...] = ()
    """Object-group kinds (``"attention"``, ``"recurrent"``, ``"aux"``) whose
    state the rule is about; the rule applies to a model only if one of its
    object groups has a listed kind. Empty means any group."""


class EngineHints(msgspec.Struct, frozen=True, kw_only=True):
    """Optional registration payload of one engine for one model layout."""

    model_name: str = ""
    world_size: int = 0
    engine_type: str = ""
    reusable_load_rule: ReusableLoadRule | None = None


class EngineHintsAck(msgspec.Struct, frozen=True, kw_only=True):
    """The server's answer: whether it applies the rule, and if not, why."""

    accepted: bool = False
    reason: str = ""
