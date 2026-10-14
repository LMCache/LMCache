# SPDX-License-Identifier: Apache-2.0
"""``weka_trace`` loader: provider role framing on shared prefixes, and source tags.

Role framing. A chain whose turn 0 repeats blocks ``[0, L)`` of an earlier
request repeats its role framing too, but the built-in loader always builds
turn 0 as ``[system(tool+system)] + [user(rest)]``, so the prefix cache misses
after the first differing ``<|im_end|>``. This loader copies the role tiling of
``[0, L)`` from the earlier request with the longest such prefix (latest on
ties); chains whose default framing already matches are left unchanged.

Source tags. The built-in loader fills turn ``k``'s output with corpus text,
whose KV never matches the server's. This loader adds
``prt_src = "<session_id>#<k>#<tokens>"`` to every assistant message that
stands for turn ``k``'s full output, and :mod:`aiperf_prt.endpoint` swaps in
the real output when it has it. Tagged at dataset build time:

* An assistant segment created by ``advance_turn`` at turn ``k`` is turn
  ``k-1``'s output; it keeps its tag when a pull-back cuts it.
* Assistant segments copied into a child's turn 0 by the role tiling are
  matched by content within the trace and get the provider's tag.

The reconstructor subclass is swapped into ``weka_synth_buf`` only during
:meth:`PrtWekaTraceLoader.convert_to_conversations` (0.13.0 has no hook).
Parallel workers never see the swap, so the loader always takes the serial
path.
"""

# Future
from __future__ import annotations

# Standard
from collections import defaultdict
from typing import Any
import math

# Third Party
from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.models import Conversation
from aiperf.dataset.loader import weka_synth_buf
from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    RoleSegment,
    compute_asst_block_caps,
    longest_common_prefix,
)
from aiperf.dataset.loader.weka_trace import (
    WekaTraceLoader,
    _child_plans_for_active_subagents,
)

_logger = AIPerfLogger(__name__)

Layout = list[tuple[str, int]]

SRC_KEY = "prt_src"
# Every built-in call site seeds with f"{session_id}:turn_{k}:partial_tail".
_SEED_TURN = ":turn_"
_TURN0_SEED_SUFFIX = ":turn_0:partial_tail"


def _cut_layout(layout: Layout, blocks: int) -> Layout:
    """Return the first ``blocks`` blocks of a role tiling, without empty segments."""
    out: Layout = []
    for role, n in layout:
        take = min(n, blocks)
        if take > 0:
            out.append((role, take))
        blocks -= take
        if blocks == 0:
            break
    return out


def _trace_of(conv: Conversation) -> str:
    """Return the weka trace id of a conversation (child ids are ``{trace}::...``)."""
    return conv.session_id.split("::", 1)[0]


def _tag_inherited(conversations: list[Conversation]) -> int:
    """Tag assistant messages equal to a tagged one in the same trace.

    Args:
        conversations: All built conversations; tagged in place.

    Returns:
        The number of messages newly tagged.
    """
    by_trace: dict[str, dict[str, str]] = {}
    for conv in conversations:
        known = by_trace.setdefault(_trace_of(conv), {})
        for turn in conv.turns:
            for msg in turn.raw_messages or []:
                if SRC_KEY in msg:
                    known.setdefault(msg["content"], msg[SRC_KEY])
    n = 0
    for conv in conversations:
        known = by_trace[_trace_of(conv)]
        for turn in conv.turns:
            for msg in turn.raw_messages or []:
                if msg.get("role") == "assistant" and SRC_KEY not in msg:
                    src = known.get(msg.get("content"))
                    if src is not None:
                        msg[SRC_KEY] = src
                        n += 1
    return n


class PrtReconstructor(ConversationReconstructor):
    """Reconstructor with inherited turn-0 tiling and assistant source tags.

    ``turn0_layouts`` maps a session id to its inherited tiling. It is set on a
    per-call subclass by :class:`PrtWekaTraceLoader`; the base class value
    is empty, which reproduces the built-in framing.
    """

    turn0_layouts: dict[str, Layout] = {}

    def init_turn_0(
        self,
        hash_ids: list[int],
        in_tokens: int,
        tool_tokens: int,
        system_tokens: int,
        seed: str,
        is_tool_result: bool = False,
        *,
        inherit_layout: Layout | None = None,
    ) -> None:
        """Initialize turn 0, following an inherited role tiling when one applies.

        Args:
            hash_ids: Turn-0 hash ids.
            in_tokens: Turn-0 input length.
            tool_tokens: Tool-prefix tokens (ignored when a tiling applies).
            system_tokens: System-prefix tokens (ignored when a tiling applies).
            seed: Partial-tail seed; ``f"{session_id}:turn_0:partial_tail"``.
            is_tool_result: Passed through to the base class.
            inherit_layout: Explicit tiling. When None, it is looked up in
                ``turn0_layouts`` by the session id encoded in ``seed``.

        Raises:
            ValueError: As the base class.
        """
        if inherit_layout is None and seed.endswith(_TURN0_SEED_SUFFIX):
            inherit_layout = self.turn0_layouts.get(seed[: -len(_TURN0_SEED_SUFFIX)])
        if not inherit_layout:
            super().init_turn_0(
                hash_ids, in_tokens, tool_tokens, system_tokens, seed, is_tool_result
            )
            return
        # With no tool/system prefix the base class emits one user segment that
        # holds every covered block plus the synthesized tail; re-tile it.
        super().init_turn_0(hash_ids, in_tokens, 0, 0, seed, is_tool_result)
        (whole,) = self._segments
        bs = self.block_size
        segs: list[RoleSegment] = []
        cursor = 0
        for role, n in inherit_layout:
            tokens = whole.tokens[cursor * bs : (cursor + n) * bs]
            segs.append(
                RoleSegment(
                    role=role,
                    block_start=cursor,
                    block_count=n,
                    tokens=tokens,
                    content=self.decode_tokens_to_text(tokens),
                )
            )
            cursor += n
        rest = whole.tokens[cursor * bs :]
        rest_blocks = whole.block_count - cursor
        if rest and segs[-1].role == "user":
            # Continue the inherited trailing user message rather than opening
            # a second adjacent one; its tokens up to the cut are unchanged.
            last = segs[-1]
            last.block_count += rest_blocks
            last.tokens = last.tokens + rest
            last.content = self.decode_tokens_to_text(last.tokens)
        elif rest:
            segs.append(
                RoleSegment(
                    role="user",
                    block_start=cursor,
                    block_count=rest_blocks,
                    tokens=rest,
                    content=self.decode_tokens_to_text(rest),
                )
            )
        self._segments: list[RoleSegment] = segs

    def advance_turn(self, *args: Any, **kwargs: Any) -> None:
        """Advance as the base class, then record the source of a new assistant segment.

        Args:
            *args: Passed through.
            **kwargs: Passed through; ``seed`` must be
                ``f"{session_id}:turn_{k}:partial_tail"`` for a tag to be set.
        """
        # Hold the old segments: a pull-back frees some, and CPython can give
        # the new assistant segment a freed id, which would then look old.
        before = list(self._segments)
        before_ids = {id(s) for s in before}
        super().advance_turn(*args, **kwargs)
        seed = kwargs.get("seed", args[5] if len(args) > 5 else "")
        if _SEED_TURN not in seed:
            return
        session_id = seed.rsplit(_SEED_TURN, 1)[0]
        for seg in self._segments:
            if seg.role == "assistant" and id(seg) not in before_ids:
                # _turn_index is already k; the segment carries turn k-1's output.
                seg.prt_src = f"{session_id}#{self._turn_index - 1}"

    def turn_delta(self) -> Any:
        """Return the base delta, with ``prt_src`` on tagged assistant messages.

        Returns:
            The base class's delta, its messages tagged in place.
        """
        delta = super().turn_delta()
        msgs = delta.delta_messages
        # Without tool shaping, messages map 1:1 onto the trailing segments.
        segs = self._segments[len(self._segments) - len(msgs) :]
        for msg, seg in zip(msgs, segs, strict=True):
            src = getattr(seg, "prt_src", None)
            if seg.role == "assistant" and src is not None:
                msg[SRC_KEY] = f"{src}#{len(seg.tokens)}"
        return delta

    def segment_layout(self) -> Layout:
        """Return the current ``[(role, block_count), ...]`` tiling, in order."""
        return [(s.role, s.block_count) for s in self._segments]


def _plan_turn0_layouts(
    trace: Any, plan: Any, child_plans: list[Any], flat_plans: list[Any]
) -> dict[str, Layout]:
    """Return ``{session_id: layout}`` for a trace's chains whose framing changes.

    Chains go in turn-0 time order, so a provider that itself inherited is
    replayed (token-free) with its inherited framing.

    Args:
        trace: The trace's first record (tool and system token counts).
        plan: The trace's parent reconstruction plan.
        child_plans: All child plans; only active subagents' are used.
        flat_plans: The trace's flat plans.

    Returns:
        ``{session_id: layout}``, only for non-main chains whose inherited
        framing differs from the default one.
    """
    bs = plan.block_size
    chains: list[tuple[str | None, list[Any], int, int]] = [
        (None, [r for _, r in plan.normals], trace.tool_tokens, trace.system_tokens)
    ]
    chains += [
        (
            fp.session_id,
            [r for _, r in fp.requests],
            fp.init_tool_tokens,
            fp.init_system_tokens,
        )
        for fp in flat_plans
    ]
    chains += [
        (cp.session_id, cp.requests, cp.init_tool_tokens, cp.init_system_tokens)
        for cp in _child_plans_for_active_subagents(plan, child_plans)
    ]
    chains = sorted((c for c in chains if c[1]), key=lambda c: c[1][0].t)
    layouts: dict[str, Layout] = {}
    seen: list[tuple[float, list[int], Layout]] = []
    for session_id, reqs, tool_tokens, system_tokens in chains:
        first = reqs[0]
        covered = min(len(first.hash_ids), first.input_length // bs)
        best_lcp, best_t, best_layout = 0, 0.0, None
        for t, hash_ids, layout in seen:
            if t >= first.t:
                continue
            lcp = min(
                longest_common_prefix(hash_ids, first.hash_ids),
                covered,
                sum(n for _, n in layout),
            )
            if lcp > 0 and (lcp > best_lcp or (lcp == best_lcp and t >= best_t)):
                best_lcp, best_t, best_layout = lcp, t, layout
        inherit: Layout | None = None
        if session_id is not None and best_layout is not None:
            inherit = _cut_layout(best_layout, best_lcp)
            prefix_blocks = (
                min(math.ceil((tool_tokens + system_tokens) / bs), covered, best_lcp)
                if tool_tokens > 0 or system_tokens > 0
                else 0
            )
            default = [("system", prefix_blocks)] if prefix_blocks else []
            if best_lcp > prefix_blocks:
                default.append(("user", best_lcp - prefix_blocks))
            has_tail = first.input_length > best_lcp * bs
            if inherit == default or (inherit[-1][0] != "user" and not has_tail):
                inherit = None
            if inherit is not None:
                layouts[session_id] = inherit
        recon = PrtReconstructor(
            block_size=bs,
            decode_block_tokens=lambda h: [0] * (len(h) * bs),
            sample_partial_tail_tokens=lambda n, seed: [0] * n,
            decode_tokens_to_text=lambda tokens: "",
        )
        caps = compute_asst_block_caps([(r.hash_ids, r.input_length) for r in reqs], bs)
        for k, req in enumerate(reqs):
            if k == 0:
                recon.init_turn_0(
                    hash_ids=req.hash_ids,
                    in_tokens=req.input_length,
                    tool_tokens=tool_tokens,
                    system_tokens=system_tokens,
                    seed="",
                    inherit_layout=inherit,
                )
            else:
                prev = reqs[k - 1]
                recon.advance_turn(
                    prev_hash_ids=prev.hash_ids,
                    prev_in_tokens=prev.input_length,
                    prev_out_tokens=prev.output_length,
                    curr_hash_ids=req.hash_ids,
                    curr_in_tokens=req.input_length,
                    seed="",
                    max_asst_blocks=caps[k],
                )
            seen.append((req.t, req.hash_ids, recon.segment_layout()))
    return layouts


class PrtWekaTraceLoader(WekaTraceLoader):
    """Built-in ``weka_trace`` loader with inherited turn-0 framing and source tags."""

    def convert_to_conversations(
        self, data: dict[str, list[Any]]
    ) -> list[Conversation]:
        """Build conversations as the base class, with framing and source tags.

        Args:
            data: Trace id to parsed traces, as the base class.

        Returns:
            The conversations.
        """
        self._turn0_layouts: dict[str, Layout] = {}
        bound = type(
            "BoundPrtReconstructor",
            (PrtReconstructor,),
            {"turn0_layouts": self._turn0_layouts},
        )
        original = weka_synth_buf.ConversationReconstructor
        weka_synth_buf.ConversationReconstructor = bound
        try:
            conversations = super().convert_to_conversations(data)
        finally:
            weka_synth_buf.ConversationReconstructor = original
        n = _tag_inherited(conversations)
        _logger.info("prt: tagged %d inherited assistant messages", n)
        return conversations

    def _build_reconstruction_plans(self, data: dict[str, list[Any]]) -> Any:
        """Build plans as the base loader, recording each chain's turn-0 layout."""
        plans = super()._build_reconstruction_plans(data)
        flats_by_trace: dict[str, list[Any]] = defaultdict(list)
        for fp in plans.flat_plans:
            flats_by_trace[fp.parent_trace_id].append(fp)
        for plan in plans.parent_plans:
            self._turn0_layouts.update(
                _plan_turn0_layouts(
                    data[plan.trace_id][0],
                    plan,
                    plans.child_plans,
                    flats_by_trace[plan.trace_id],
                )
            )
        return plans

    def _reconstruct_parallel(self, **kwargs: Any) -> list[Conversation]:
        # Parallel workers never see the swap and would silently drop tags.
        kwargs.pop("configured_workers")
        return self._reconstruct_serial(dropped_per_trace={}, **kwargs)
