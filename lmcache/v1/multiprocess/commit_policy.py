# SPDX-License-Identifier: Apache-2.0
"""
Commit policy: does a finished request's sliding window earn an L2 copy?

A store policy that keeps sliding-window chunks out of L2 on the store path
leaves L1 holding their only copy until an eviction writes them back. That
write happens under memory pressure, which is the wrong moment: the next turn
of the same conversation can arrive while the write-back is still in flight
and miss.

A commit turns that around. When a request finishes at a point a follow-up
will match -- a chat turn boundary -- its final window is copied to L2 right
then, during the seconds or minutes before the next turn arrives. The L1 copy
stays, so the follow-up is still an L1 hit; the L2 copy only has to exist
before L1 gives the window up.

Two decisions, deliberately split:

* **whether** to commit is per request, and this module owns it. It is a
  property of how the conversation ended, so it is decided from
  :class:`~lmcache.v1.multiprocess.custom_types.SessionEndInfo`, the engine's
  report of the finish reason and stop token.
* **where** the window ends is per deployment, not per request: it follows
  from whether the serving frontend echoes the previous turn's reasoning back
  in the next prompt, which is a property of the chat template and the client,
  not of one message. It is server configuration (``--commit-anchor``), and
  :func:`resolve_anchor` turns it into a chunk-aligned token offset.

Nothing here knows about chunk sizes, object groups, or window widths: an
anchor is an *end* offset, and the caller derives each sliding-window group's
``w`` trailing chunks from it. That is why the policy returns a bool rather
than a range -- a range that under-covers a group's window would be written
and never read, and this API cannot express one.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
import enum

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import AttnWindowDesc
from lmcache.v1.multiprocess.custom_types import SessionEndInfo

logger = init_logger(__name__)


class CommitAnchor(str, enum.Enum):
    """Where the committed window ends.

    Subclasses ``str`` so it compares equal to and parses from the bare
    configuration value.

    ``GENERATION_END`` -- the last chunk the request actually stored, i.e. the
    end of the generated answer. Correct when the next turn re-sends the
    answer verbatim (no reasoning stripped, tokenization round-trips).

    ``PROMPT_END`` -- the end of the prompt the engine looked up. Correct when
    the next turn re-renders the assistant message differently from what was
    generated, which is what a reasoning model's chat template does when the
    client does not echo ``reasoning_content`` back.

    Which one is right is a question for measurement, not for configuration
    defaults: it depends on the deployment's clients.
    """

    GENERATION_END = "generation_end"
    PROMPT_END = "prompt_end"


@dataclass(frozen=True)
class CommitContext:
    """Everything a commit policy may look at.

    Assembled by the ``END_SESSION`` handler from the request's session and the
    engine's :class:`SessionEndInfo`. Every field is a plain value: a policy is
    called on the server's CPU pool thread and must not call back into the
    storage manager.
    """

    request_id: str
    """The finished request."""

    end_info: SessionEndInfo
    """How the engine says the request ended."""

    model_name: str
    """Model the session belongs to."""

    prompt_end: int
    """Raw token offset of the end of the last range the engine looked up,
    i.e. the prompt. ``0`` for a session that stored without ever looking
    up."""

    stored_end: int
    """Raw token offset of the end of what the session actually stored --
    prompt plus the generated tail, truncated to the last full chunk. Never
    commit past this: nothing beyond it is in L1."""

    hit_chunks: int
    """Chunks this request's own lookup hit, ``-1`` when its prefetch result
    was never consumed. Positive means the request was itself a follow-up, so
    the conversation is already known to continue."""

    attn_desc: AttnWindowDesc
    """The model's per-object-group attention windows. A policy that wants to
    price a commit can read the sliding-window widths from here."""


class CommitPolicy(ABC):
    """Decides whether a finished request's window is committed to L2.

    Implementations must be thread-safe and side-effect free: several
    ``END_SESSION`` handlers run concurrently on the CPU pool, and a policy
    that raises is treated as "do not commit" by the caller, which leaves the
    window to the eviction write-back.
    """

    @abstractmethod
    def should_commit(self, ctx: CommitContext) -> bool:
        """Return whether to copy this request's final window to L2.

        Args:
            ctx: What the server knows about the finished request.

        Returns:
            True to commit the window now, False to leave it to eviction.
        """


# -----------------------------------------------------------------------------
# Registry: commit policy name -> factory
# -----------------------------------------------------------------------------

CommitPolicyFactory = Callable[["CommitPolicyConfig"], CommitPolicy]
"""Builds a commit policy from the server's commit configuration.

Policies that need no configuration ignore the argument; it is passed to every
factory so ``create_commit_policy`` needs no per-policy knowledge.
"""

_COMMIT_POLICY_REGISTRY: dict[str, CommitPolicyFactory] = {}


@dataclass(frozen=True)
class CommitPolicyConfig:
    """Server-side configuration of the commit path.

    Args:
        policy: Registered commit policy name.
        anchor: Where the committed window ends.
        boundary_token_ids: Token ids that count as a chat turn boundary for
            this deployment. Empty accepts any token a request stopped on.
            The set also chooses *which* boundaries commit on a model that
            ends a tool call and a final answer on different tokens: gpt-oss
            lists both ``<|return|>`` (200002) and ``<|call|>`` (200012) in
            ``eos_token_id``, so ``{200002}`` commits only finished answers,
            ``{200012}`` only tool calls, and empty commits both.
    """

    policy: str = "stop_token"
    anchor: CommitAnchor = CommitAnchor.GENERATION_END
    boundary_token_ids: frozenset[int] = frozenset()


DEFAULT_COMMIT_CONFIG = CommitPolicyConfig()
"""The built-in default: ``stop_token`` with any stop token accepted."""


def register_commit_policy(name: str, policy_cls: type[CommitPolicy]) -> None:
    """Register a commit policy built without configuration.

    Args:
        name: Policy name.
        policy_cls: Concrete subclass with a no-argument constructor.

    Raises:
        ValueError: If the name is already registered.
    """
    register_commit_policy_factory(name, lambda cfg: policy_cls())


def register_commit_policy_factory(name: str, factory: CommitPolicyFactory) -> None:
    """Register a commit policy factory under a name.

    A plugin loaded through ``--runtime-plugin-locations`` registers its own
    policy this way at import time and is then selectable by name.

    Args:
        name: Policy name.
        factory: Callable building a policy from the commit configuration.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in _COMMIT_POLICY_REGISTRY:
        raise ValueError(f"Commit policy already registered: {name!r}")
    _COMMIT_POLICY_REGISTRY[name] = factory


def get_registered_commit_policies() -> list[str]:
    """Return the registered commit policy names."""
    return list(_COMMIT_POLICY_REGISTRY)


def create_commit_policy(config: CommitPolicyConfig) -> CommitPolicy:
    """Create the configured commit policy.

    Args:
        config: The server's commit configuration.

    Returns:
        A new policy instance.

    Raises:
        ValueError: If no policy is registered under ``config.policy``.
    """
    if config.policy not in _COMMIT_POLICY_REGISTRY:
        known = ", ".join(sorted(_COMMIT_POLICY_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown commit policy {config.policy!r}. Known: {known}")
    return _COMMIT_POLICY_REGISTRY[config.policy](config)


def resolve_commit(ctx: CommitContext, policy: CommitPolicy) -> bool:
    """Ask the policy, and treat a failure as "do not commit".

    A policy that raises is reported and refused rather than trusted: the
    window then leaves L1 the ordinary way, through the eviction write-back,
    which loses timeliness but nothing else.

    Args:
        ctx: What the server knows about the finished request.
        policy: The configured policy.

    Returns:
        Whether to commit this request's window.
    """
    try:
        return policy.should_commit(ctx)
    except Exception:
        logger.exception(
            "Commit policy %s raised for request %s; not committing",
            type(policy).__name__,
            ctx.request_id,
        )
        return False


def resolve_anchor(
    anchor: CommitAnchor,
    ctx: CommitContext,
    chunk_size: int,
) -> int:
    """Turn the configured anchor into a chunk-aligned token offset.

    The offset is clipped to ``ctx.stored_end`` -- nothing beyond it is in L1
    to copy -- and rounded down to a chunk boundary, because a partial trailing
    chunk was never stored as one.

    Args:
        anchor: The configured anchor.
        ctx: The finished request's context.
        chunk_size: Tokens per LMCache chunk.

    Returns:
        The chunk-aligned token offset the committed window ends at, or ``0``
        when the session has no full chunk to commit.

    Raises:
        ValueError: If ``chunk_size`` is not positive.
    """
    if chunk_size <= 0:
        raise ValueError(f"resolve_anchor: chunk_size must be > 0, got {chunk_size}")

    raw = ctx.stored_end if anchor is CommitAnchor.GENERATION_END else ctx.prompt_end
    return (min(raw, ctx.stored_end) // chunk_size) * chunk_size


# -----------------------------------------------------------------------------
# Built-in policies
# -----------------------------------------------------------------------------


class StopTokenCommitPolicy(CommitPolicy):
    """Commit when the model stopped on a turn-boundary token.

    A chat model ends its turn by emitting the token that opens the next turn
    (Qwen stops on ``<|im_end|>``, id 151645, because its
    ``generation_config.json`` lists it in ``eos_token_id``). A request that
    reached that token ended where the next prompt will resume, so its final
    window is exactly what the follow-up needs.

    Everything else is refused: an abort, a length cap, an error or a
    repetition stop all leave the sequence mid-turn, and a mid-turn tail is
    re-rendered -- or never sent again -- by the next request, so its window
    would be written and never read.

    A model that ends tool calls and final answers on different tokens (gpt-oss:
    ``<|call|>`` and ``<|return|>``) lets the boundary set pick which of the
    two commits. A model that ends both on the same token (Qwen: ``<|im_end|>``)
    gives the server no way to tell them apart, and both commit. That is the
    safe default either way: a tool result can take an hour to come back, so a
    tool-call window cannot wait in L1 any more than an answer's can.

    Args:
        boundary_token_ids: Ids that count as a turn boundary. Empty accepts
            any stop token, which is right for a model whose only stop token
            is its turn marker.
    """

    def __init__(self, boundary_token_ids: frozenset[int] = frozenset()) -> None:
        self._boundary_token_ids = boundary_token_ids

    def should_commit(self, ctx: CommitContext) -> bool:
        """Return whether the request stopped on a turn boundary.

        Args:
            ctx: The finished request's context.

        Returns:
            True when the finish reason is ``stop`` and the stop token is a
            configured boundary token (or no boundary tokens are configured).
            Whether the session has anything to commit is the caller's
            question, answered by :func:`resolve_anchor`.
        """
        if ctx.end_info.finish_reason != "stop":
            return False
        if not self._boundary_token_ids:
            return True
        return ctx.end_info.stop_token_id in self._boundary_token_ids


register_commit_policy_factory(
    "stop_token",
    lambda cfg: StopTokenCommitPolicy(cfg.boundary_token_ids),
)
