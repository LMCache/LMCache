# SPDX-License-Identifier: Apache-2.0
"""Per-request decision between serving LMCache KV and letting the engine
recompute it.

``LookupModule.lookup`` consults a ``KVLoadPolicy`` before it submits the
prefetch. When the policy declines, the lookup reports zero hit chunks without
touching L1 or L2: no L2 fetch, no L1 read locks. The engine sees a miss and
prefills the prompt itself.

The policy runs before the prefetch, so it does not know how many chunks would
hit. It sees the request (lookup length, model, ``request_configs``) and the
hints the engine connector attaches to ``request_configs``.

The module is pure policy: no I/O. It runs on the server's lookup handler
thread.
"""

# Standard
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import importlib

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

#: Built-in policy name used when no policy is configured.
DEFAULT_POLICY_NAME = "DEFAULT"

#: Request-config key (``kv_transfer_params``) asking the default policy to
#: recompute this request instead of loading it from LMCache.
SKIP_LOAD_REQUEST_KEY = "lmcache.skip_load"

#: Request-config key the engine connector sets to the number of prompt
#: tokens the engine already holds in its own prefix cache. Absent when the
#: connector does not know it yet (e.g. eager prefetch at enqueue time).
ENGINE_COMPUTED_TOKENS_HINT_KEY = "lmcache.hint.engine_computed_tokens"


@dataclass(frozen=True)
class KVLoadContext:
    """What the server knows about one lookup when it decides.

    Attributes:
        request_id: The engine request id.
        model_name: The model the lookup is for.
        chunk_size: Tokens per LMCache chunk.
        num_lookup_tokens: Chunk-aligned tokens the lookup covers, counted
            from token zero. This is an upper bound on the hit, not the hit.
        engine_computed_tokens: Tokens the engine already holds in its own
            prefix cache, from ``ENGINE_COMPUTED_TOKENS_HINT_KEY``; None when
            the connector did not send it.
        request_configs: The request's ``lmcache.*`` entries, hints included,
            or None when it carries none.
    """

    request_id: str
    model_name: str
    chunk_size: int
    num_lookup_tokens: int
    engine_computed_tokens: int | None
    request_configs: Mapping[str, Any] | None


class KVLoadPolicy(ABC):
    """Decides, per lookup, whether to serve KV from LMCache or let the engine
    recompute it.

    Subclasses are constructed with the server's runtime-plugin extra config
    (``--runtime-plugin-config``) and may read their own tunables from it.
    Each lookup is decided once; the engine re-polls the result, not the
    policy.
    """

    def __init__(self, configs: Mapping[str, Any]) -> None:
        """Create the policy.

        Args:
            configs: The server's runtime-plugin extra config, kept as
                ``self.configs`` for subclasses to read tunables from.
        """
        self.configs = configs

    @abstractmethod
    def should_load(self, ctx: KVLoadContext) -> bool:
        """Return whether the server should serve this lookup.

        Args:
            ctx: The lookup being decided.

        Returns:
            True to prefetch and report hits as usual, False to report zero
            hit chunks so the engine recomputes the prompt.
        """


class DefaultKVLoadPolicy(KVLoadPolicy):
    """Serves every lookup unless the request sets ``lmcache.skip_load``.

    Example request body (OpenAI-compatible vLLM server)::

        {"kv_transfer_params": {"lmcache.skip_load": true}, ...}
    """

    def should_load(self, ctx: KVLoadContext) -> bool:
        """Return False only when the request asked to skip loading.

        Args:
            ctx: The lookup being decided.

        Returns:
            Whether the request did not set ``lmcache.skip_load``.
        """
        return not (ctx.request_configs or {}).get(SKIP_LOAD_REQUEST_KEY, False)


_BUILTIN_POLICIES: dict[str, type[KVLoadPolicy]] = {
    DEFAULT_POLICY_NAME: DefaultKVLoadPolicy,
}


def create_kv_load_policy(name: str, configs: Mapping[str, Any]) -> KVLoadPolicy:
    """Build the policy ``name`` refers to.

    ``name`` is either a built-in name (``DEFAULT``) or a
    ``"package.module:ClassName"`` path to a ``KVLoadPolicy`` subclass, e.g.
    ``"my_pkg.policies:ShortPromptsRecompute"``. The module must be importable
    in the server process.

    Args:
        name: Policy name or ``module:ClassName`` path.
        configs: Passed on to the policy's constructor.

    Returns:
        The constructed policy.

    Raises:
        ValueError: If the name is neither a built-in nor a
            ``module:ClassName`` path, or the class is not a
            ``KVLoadPolicy`` subclass.
        ImportError: If the module of a ``module:ClassName`` path cannot be
            imported.
        AttributeError: If the module has no attribute named ``ClassName``.
    """
    policy_cls = _BUILTIN_POLICIES.get(name)
    if policy_cls is None:
        module_path, sep, class_name = name.partition(":")
        if not sep or not module_path or not class_name:
            raise ValueError(
                f"Unknown KV load policy {name!r}: expected one of "
                f"{sorted(_BUILTIN_POLICIES)} or 'package.module:ClassName'"
            )
        policy_cls = getattr(importlib.import_module(module_path), class_name)
        if not (isinstance(policy_cls, type) and issubclass(policy_cls, KVLoadPolicy)):
            raise ValueError(f"{name!r} is not a KVLoadPolicy subclass")
    logger.info("LMCache KV load policy: %s", name)
    return policy_cls(configs)
