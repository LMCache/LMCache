# SPDX-License-Identifier: Apache-2.0
"""Tests for the sliding-window commit path.

Three layers, all with fakes (no storage, no GPU):

* the built-in policies, which decide *whether* a finished request's window
  is copied to L2;
* ``resolve_commit`` and ``resolve_anchor``, which take the policy's answer
  and turn the configured anchor into a chunk-aligned offset;
* ``LookupModule.handle_end_session``, which turns a decision into the
  sliding-window keys handed to ``StorageManager.flush_l1_keys_to_l2``.

See ``lmcache/v1/multiprocess/commit_policy.py``.
"""

# Standard
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.multiprocess.commit_policy import (
    CommitAnchor,
    CommitContext,
    CommitPolicy,
    CommitPolicyConfig,
    StopTokenCommitPolicy,
    create_commit_policy,
    resolve_anchor,
    resolve_commit,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, SessionEndInfo
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher

CHUNK_SIZE = 4
MODEL_NAME = "model"
REQUEST_ID = "req-1"
BOUNDARY_TOKEN = 151645
"""Qwen's ``<|im_end|>``, the token a chat turn ends on."""

# One sliding-window group of two chunks and one full-attention group, the
# shape of a hybrid model served with --separate-object-groups.
HYBRID_DESC = AttnWindowDesc(num_chunks_in_sw=[2, -1])


def make_context(
    finish_reason: str = "stop",
    stop_token_id: int = BOUNDARY_TOKEN,
    prompt_end: int = 8,
    stored_end: int = 16,
    hit_chunks: int = -1,
    anchor: CommitAnchor = CommitAnchor.GENERATION_END,
) -> CommitContext:
    """Build a commit context with the fields a policy reads.

    Args:
        finish_reason: How the engine says the request finished.
        stop_token_id: Token the generation stopped on.
        prompt_end: Raw token offset of the end of the looked-up prompt.
        stored_end: Raw token offset of the end of what the session stored.
        hit_chunks: Chunks this request's own lookup hit.
        anchor: Where the deployment puts the committed window.

    Returns:
        The context.
    """
    return CommitContext(
        request_id=REQUEST_ID,
        end_info=SessionEndInfo(
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
        ),
        model_name=MODEL_NAME,
        prompt_end=prompt_end,
        stored_end=stored_end,
        hit_chunks=hit_chunks,
        attn_desc=HYBRID_DESC,
        anchor=anchor,
    )


# =============================================================================
# Built-in policies
# =============================================================================


class TestStopTokenCommitPolicy:
    """Tests for the policy that commits on a chat turn boundary."""

    def test_commits_on_configured_boundary_token(self):
        """A turn that stopped on the boundary token earns a commit."""
        policy = StopTokenCommitPolicy(frozenset({BOUNDARY_TOKEN}))

        assert policy.should_commit(make_context()) is True

    def test_refuses_other_stop_token(self):
        """Stopping on some other token is not a turn boundary."""
        policy = StopTokenCommitPolicy(frozenset({BOUNDARY_TOKEN}))

        assert policy.should_commit(make_context(stop_token_id=99)) is False

    def test_accepts_any_stop_token_when_unconfigured(self):
        """With no boundary tokens configured, any clean stop counts."""
        policy = StopTokenCommitPolicy()

        assert policy.should_commit(make_context(stop_token_id=99)) is True

    @pytest.mark.parametrize("reason", ["abort", "length", "error", "repetition", ""])
    def test_refuses_every_non_stop_finish(self, reason: str):
        """A tail no follow-up re-sends is never committed.

        Args:
            reason: The finish reason the engine reported.
        """
        policy = StopTokenCommitPolicy(frozenset({BOUNDARY_TOKEN}))

        assert policy.should_commit(make_context(finish_reason=reason)) is False

    @pytest.mark.parametrize("reason", ["stop", "length", "repetition"])
    def test_prompt_end_commits_whatever_ended_the_generation(self, reason: str):
        """A prompt is re-sent by the follow-up however the answer ended.

        Args:
            reason: The finish reason the engine reported.
        """
        policy = StopTokenCommitPolicy(frozenset({BOUNDARY_TOKEN}))
        ctx = make_context(
            finish_reason=reason, stop_token_id=99, anchor=CommitAnchor.PROMPT_END
        )

        assert policy.should_commit(ctx) is True

    @pytest.mark.parametrize("reason", ["abort", "error", ""])
    def test_prompt_end_refuses_a_conversation_that_may_not_continue(self, reason: str):
        """An abort, an error, or no report at all earns no commit anywhere.

        Args:
            reason: The finish reason the engine reported.
        """
        policy = StopTokenCommitPolicy()
        ctx = make_context(finish_reason=reason, anchor=CommitAnchor.PROMPT_END)

        assert policy.should_commit(ctx) is False


class TestRegistry:
    """Tests for commit policy lookup by name."""

    def test_create_passes_boundary_tokens_to_stop_token(self):
        """``--commit-boundary-tokens`` reaches the policy it configures."""
        config = CommitPolicyConfig(
            policy="stop_token", boundary_token_ids=frozenset({BOUNDARY_TOKEN})
        )

        policy = create_commit_policy(config)

        assert policy.should_commit(make_context(stop_token_id=99)) is False

    def test_unknown_policy_is_rejected(self):
        """An unknown name fails at startup, not at the first request."""
        with pytest.raises(ValueError, match="Unknown commit policy"):
            create_commit_policy(CommitPolicyConfig(policy="no-such-policy"))


# =============================================================================
# Commit resolution and anchors
# =============================================================================


class _RaisingPolicy(CommitPolicy):
    """A policy that fails, to check the caller degrades safely."""

    def should_commit(self, ctx: CommitContext) -> bool:
        """Fail the way a buggy plugin would.

        Args:
            ctx: Not consulted.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("boom")


class TestResolveCommit:
    """Tests for how a policy's answer is taken."""

    def test_raising_policy_does_not_commit(self):
        """A broken policy costs timeliness, not correctness."""
        assert resolve_commit(make_context(), _RaisingPolicy()) is False


class TestResolveAnchor:
    """Tests for turning the configured anchor into a token offset."""

    def test_generation_end_uses_the_stored_end(self):
        """The last chunk the request stored."""
        ctx = make_context(prompt_end=8, stored_end=16)

        assert resolve_anchor(ctx, CHUNK_SIZE) == 16

    def test_prompt_end_uses_the_looked_up_prefix(self):
        """The end of the prompt, for a client that drops the answer."""
        ctx = make_context(prompt_end=8, stored_end=16, anchor=CommitAnchor.PROMPT_END)

        assert resolve_anchor(ctx, CHUNK_SIZE) == 8

    def test_offsets_are_floored_to_chunk_boundaries(self):
        """A partial trailing chunk was never stored as one."""
        assert (
            resolve_anchor(make_context(prompt_end=7, stored_end=17), CHUNK_SIZE) == 16
        )
        assert (
            resolve_anchor(
                make_context(
                    prompt_end=7, stored_end=17, anchor=CommitAnchor.PROMPT_END
                ),
                CHUNK_SIZE,
            )
            == 4
        )

    def test_prompt_end_is_clamped_to_the_stored_end(self):
        """Nothing past what reached L1 can be committed."""
        ctx = make_context(prompt_end=64, stored_end=8, anchor=CommitAnchor.PROMPT_END)

        assert resolve_anchor(ctx, CHUNK_SIZE) == 8


# =============================================================================
# handle_end_session: decision to keys
# =============================================================================


def run_end_session(
    config: CommitPolicyConfig,
    num_chunks: int = 6,
    lookup_chunks: int = 4,
    finish_reason: str = "stop",
    stop_token_id: int = BOUNDARY_TOKEN,
    attn_desc: AttnWindowDesc = HYBRID_DESC,
) -> tuple[MagicMock, list[list[ObjectKey]]]:
    """Drive ``handle_end_session`` over a real session, capture the flush.

    The session is left in the state an ordinary chat turn leaves behind: it
    looked up ``lookup_chunks`` chunks of prompt and then generated up to
    ``num_chunks``, so the two anchors differ.

    Args:
        config: The server's commit configuration.
        num_chunks: Full chunks the session stored.
        lookup_chunks: Leading chunks the engine looked up.
        finish_reason: How the engine says the request finished.
        stop_token_id: Token the generation stopped on.
        attn_desc: Attention windows the layout registry reports.

    Returns:
        ``(ctx, per_group_keys)`` -- the mock context whose ``storage_manager``
        recorded the calls, and the per-object-group key lists (chunk-major /
        rank-minor) for building expectations.
    """
    tokens = list(range(num_chunks * CHUNK_SIZE))
    hasher = TokenHasher(chunk_size=CHUNK_SIZE, hash_algorithm="blake3")
    manager = SessionManager(hasher, ttl=600, cleanup_interval=None)
    session = manager.get_or_create(REQUEST_ID)
    session.set_tokens(tokens)
    looked_up = IPCCacheServerKey.from_token_ids(
        model_name=MODEL_NAME,
        world_size=1,
        worker_id=None,
        token_ids=tokens,
        start=0,
        end=lookup_chunks * CHUNK_SIZE,
        request_id=REQUEST_ID,
    )
    session.begin_lookup(looked_up, tuple(attn_desc.num_chunks_in_sw))
    session.note_resolved(num_chunks * CHUNK_SIZE)

    ctx = MagicMock()
    ctx.chunk_size = CHUNK_SIZE
    ctx.token_hasher = hasher
    ctx.session_manager = manager
    ctx.commit_config = config
    ctx.commit_policy = create_commit_policy(config)
    ctx.layout_desc_registry.find_attn_desc.return_value = attn_desc
    layout = MemoryLayoutDesc(shapes=[torch.Size([2, 2])], dtypes=[torch.bfloat16])
    ctx.layout_desc_registry.find_group_layout_descs.return_value = {
        gid: layout for gid in range(attn_desc.num_object_groups)
    }
    ctx.event_bus.has_subscribers.return_value = False

    LookupModule(ctx).handle_end_session(
        REQUEST_ID,
        SessionEndInfo(
            finish_reason=finish_reason,
            stop_token_id=stop_token_id,
        ),
    )

    chunk_hashes = [
        TokenHasher.hash_to_bytes(h)
        for h in hasher.compute_chunk_hashes(tokens, start=0, end=len(tokens))
    ]
    per_group = ipc_key_to_object_keys(
        IPCCacheServerKey.from_token_ids(
            model_name=MODEL_NAME,
            world_size=1,
            worker_id=None,
            token_ids=tokens,
            start=0,
            end=len(tokens),
            request_id=REQUEST_ID,
        ),
        chunk_hashes,
        list(range(attn_desc.num_object_groups)),
    )
    return ctx, per_group


def flushed_keys(ctx: MagicMock) -> list[ObjectKey]:
    """Return the keys the commit handed to the flush, or an empty list.

    Args:
        ctx: The mock context ``run_end_session`` drove.

    Returns:
        The single flush batch's keys; empty when no flush was issued.
    """
    calls = ctx.storage_manager.flush_l1_keys_to_l2.call_args_list
    if not calls:
        return []
    assert len(calls) == 1, "a commit should issue at most one flush batch"
    return list(calls[0][0][0])


class TestEndSessionCommit:
    """Tests for the keys a committed window actually names."""

    def test_commits_the_trailing_window_of_the_sliding_group_only(self):
        """Under the default config: two chunks of group 0, none of group 1."""
        ctx, per_group = run_end_session(CommitPolicyConfig(), num_chunks=6)

        assert flushed_keys(ctx) == per_group[0][4:6]

    def test_each_sliding_group_takes_its_own_window(self):
        """Groups with different ``w`` end at one anchor and start apart."""
        ctx, per_group = run_end_session(
            CommitPolicyConfig(), num_chunks=6, attn_desc=AttnWindowDesc([2, 4, -1])
        )

        assert flushed_keys(ctx) == per_group[0][4:6] + per_group[1][2:6]

    def test_prompt_end_anchor_commits_an_earlier_window(self):
        """The window ends where the prompt did, not where generation did."""
        config = CommitPolicyConfig(policy="stop_token", anchor=CommitAnchor.PROMPT_END)

        ctx, per_group = run_end_session(config, num_chunks=6, lookup_chunks=4)

        assert flushed_keys(ctx) == per_group[0][2:4]

    def test_length_cap_commits_at_prompt_end_but_not_at_generation_end(self):
        """A length-capped answer is mid-turn, but its prompt is still re-sent."""
        at_prompt = CommitPolicyConfig(
            policy="stop_token", anchor=CommitAnchor.PROMPT_END
        )
        at_generation = CommitPolicyConfig(
            policy="stop_token", anchor=CommitAnchor.GENERATION_END
        )

        ctx, per_group = run_end_session(
            at_prompt, num_chunks=6, lookup_chunks=4, finish_reason="length"
        )
        assert flushed_keys(ctx) == per_group[0][2:4]

        ctx, _ = run_end_session(
            at_generation, num_chunks=6, lookup_chunks=4, finish_reason="length"
        )
        assert flushed_keys(ctx) == []

    def test_abort_does_not_commit(self):
        """An aborted tail is re-rendered or never sent again."""
        config = CommitPolicyConfig(
            policy="stop_token", anchor=CommitAnchor.GENERATION_END
        )

        ctx, _ = run_end_session(config, finish_reason="abort")

        assert flushed_keys(ctx) == []

    def test_session_without_a_full_chunk_commits_nothing(self):
        """A clean stop with nothing stored has no window to name."""
        ctx, _ = run_end_session(CommitPolicyConfig(), num_chunks=0, lookup_chunks=0)

        assert flushed_keys(ctx) == []

    def test_full_attention_only_model_commits_nothing(self):
        """Without a sliding-window group there is no window to commit."""
        config = CommitPolicyConfig(
            policy="stop_token", anchor=CommitAnchor.GENERATION_END
        )

        ctx, _ = run_end_session(config, attn_desc=AttnWindowDesc([-1]))

        assert flushed_keys(ctx) == []

    def test_window_wider_than_the_session_commits_what_exists(self):
        """A short conversation commits its whole prefix, not a negative range."""
        config = CommitPolicyConfig(
            policy="stop_token", anchor=CommitAnchor.GENERATION_END
        )

        ctx, per_group = run_end_session(
            config, num_chunks=1, lookup_chunks=1, attn_desc=AttnWindowDesc([8, -1])
        )

        assert flushed_keys(ctx) == per_group[0][0:1]
