# Sliding-window commits: `--commit-policy`

A store policy for hybrid-attention models keeps sliding-window chunks out of
L2 on the store path and leaves it to eviction to write back the windows worth
keeping. This design decides that some of those windows should not wait for
eviction.

## Problem

A sliding-window chunk the store path skipped has exactly one copy, in L1. It
reaches L2 only when eviction writes it back, that is, under memory pressure,
which is the worst moment. Nothing below depends on which store policy left
the chunk L1-only: any configuration that does has this problem, and one that
does not has no window to commit and is unaffected.

An E2E run measured the consequence: one GPU serving eight concurrent
conversations with book-length prompts, L1 = 20 GB. Between one session's
turn 0 and its turn 1 the other seven sessions store their own sliding-window
KV, so the first session's window is evicted and written back, and its
follow-up lookup arrives while that write-back is still in flight. p50 was
unaffected; p95 went from 0.5 s to 12.6 s, a full prefill.

The write itself is not the problem; its timing is. A chat turn is followed by
seconds to minutes of nothing while a person reads and types. Writing the
window then costs the same bytes and is finished long before the follow-up.

This matters only while L1 cannot hold every live window. The same run at
L1 = 96 GB writes no sliding-window chunk at all and shows no p95 tail: the
windows stay resident, nothing is evicted, and there is nothing to race. A
commit buys tail latency where L1 is too small for the concurrent
conversations (448 MiB per live window on the 31B hybrid model measured
below) and buys nothing where it is not.

## What earns a commit

A window is worth writing only if some later request will match the prefix it
ends at. For a chat model, how the turn ended decides that.

A chat model ends its turn by emitting its turn-end marker. Qwen emits
`<|im_end|>` (id 151645), which its `generation_config.json` lists in
`eos_token_id` next to `<|endoftext|>`. A request that reached that token
ended exactly where the next prompt resumes. A request that was aborted, hit
its length cap, errored, or tripped repetition detection did not: its tail is
re-rendered or never sent again, and its window would be written and never
read.

The server does not have those facts; vLLM's scheduler does. `Request.status`
gives the finish reason and `Request.stop_reason` the stop token, both still
in hand when the connector's `request_finished` runs. vLLM fills
`stop_reason` only for `stop_token_ids`; a request that stopped on the model's
own EOS leaves it `None` (`check_stop` in `vllm/v1/core/sched/utils.py`), and
that is exactly the chat case, so the connector falls back to the last
generated token.

Those facts travel to the server as `SessionEndInfo`, the second `END_SESSION`
payload. They are observations, not a decision: the rule stays server-side,
where it can be configured and replaced.

### Why the policy returns a bool

The decision splits in two, and only one half varies per request.

*Whether* to commit is a property of how this turn ended. *Where* the committed
window ends is a property of whether the serving frontend re-sends the
generated answer verbatim in the next prompt, which follows from the chat
template and the client, not from any one message. A reasoning model's
template drops the reasoning from earlier assistant turns: Qwen3 re-renders a
past turn as `<|im_start|>assistant\n` followed by the answer alone, while
generation produced `<think>...</think>` before that answer. The re-rendered
prompt diverges from the generated tokens right after the assistant header,
so the matching prefix ends before the answer. A client that echoes
`reasoning_content` back keeps matching through the answer.

That is a deployment constant, so it is the `--commit-anchor` option rather
than a return value. The policy answers the per-request half and returns a
bool. Everything that made a richer return type tempting (clamping a caller's
offset, rounding it to a chunk, capping how much one request may write,
deciding whether to trust a caller with any of it) disappears with it: a bool
can move exactly one window, so none of those guards has anything to guard.

### Why not a range, and why not a timer

An anchor is an *end* offset; each sliding-window object group derives its own
start by subtracting its own `w` (`AttnWindowDesc.num_chunks_in_sw` is a list,
and one model may have several). A caller-supplied `[start, end)` could
under-cover a group's window, and a window missing its oldest chunk is written
and never read. An API that cannot express that mistake is better than one
that validates it away.

An idle timer was the other candidate: commit a window that has sat in L1 for
T seconds, on the theory that a tool loop resumes in milliseconds and a human
does not. It does not survive contact with agents: a tool call can take an
hour. That same fact removes the reason to treat tool boundaries differently
at all. A window nobody will touch for an hour cannot stay in L1 either way,
so it is committed like any other.

## Placement

`END_SESSION` already does its own end-of-request bookkeeping. The commit is a
separate step in front of it, not a step inside it:

```
handle_end_session(request_id, end_info):   # the END_SESSION handler
    maybe commit the window                 # this design
    end_session(request_id)                 # unchanged, and unaware of the above
```

Commit first because `end_session` removes the session the commit reads. The
two share nothing else: the commit derives its keys from the session's own
hash chain and the model's attention layout, and touches neither the
bookkeeping nor its tests.

The commit asks the policy, turns `--commit-anchor` into a chunk-aligned
offset, and names the `w` trailing chunks of every sliding-window object group
ending there. Full-attention groups are skipped: the store path already wrote
them through.

The anchor is clipped to `Session.resolved_end`, the furthest offset the
session resolved object keys for. Nothing past it is in L1 to copy.

## Execution

`StoreController.submit_flush` shares the eviction write-back's queue, lock,
eventfd and processing code. The one difference is at completion:
`StoreMode.WRITEBACK` deletes the key from L1 once every adapter succeeded,
`StoreMode.FLUSH` does not. A commit is a copy. The follow-up turn should
still be an L1 hit; only the window's durability changed.

Both paths bypass the store policy. That is a property of the write-back
machinery rather than a choice the commit makes: nothing on this path consults
the store policy, so a commit writes whatever the anchor names. Under a
write-through policy that is a key L2 already holds; an adapter that skips
keys it already has (the filesystem adapter checks the path before writing)
absorbs the repeat, and one that does not simply overwrites it.

Failure is not costly. A commit that is dropped (no L2 adapter, the key
already evicted, a policy that raised) changes nothing: the window still
reaches L2 through the eviction write-back, as it did before. What is lost is
timeliness, not the window.

## Configuration

```
--commit-policy never|stop_token          # default never = unchanged behaviour
--commit-anchor generation_end|prompt_end
--commit-boundary-tokens 151645           # Qwen: <|im_end|>
```

`never` keeps the previous behaviour exactly, so the commit path is opt-in.

A plugin loaded through `--runtime-plugin-locations` registers its own policy
with `register_commit_policy_factory` at import time and is then selectable by
name, the same way store and eviction policies extend. A deployment whose
serving frontend knows more than the finish reason does (that this
conversation is over, or that a follow-up is certain) expresses that as a
policy. There is deliberately no per-request override: L2 write volume stays
a property of the server, not of its callers.

## Cost

One window per committed turn: `w` chunks, 8 x 56 MiB = 448 MiB on the
model below. Consecutive turns of one conversation overlap in that window and the
chunk hash is content-derived, so an adapter that skips keys it already holds
collapses the repeat: a turn writes only the chunks its window gained.

Measured on a 31B hybrid sliding-window/full-attention model (PLaMo 3, GPTQ
4-bit, `--separate-object-groups`, filesystem
L2, L1 = 16 GB so nothing was ever evicted), three turns of one conversation:

| turn | sliding-window files in L2 | full-attention files in L2 |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 8 (448 MiB) | 32 (256 MiB) |
| 2 | 11 | 35 |
| 3 | 12 | 36 |

Turn 1 commits exactly one window. Turns 2 and 3 add 3 and 1 files, the chunks
their windows gained, against 3 and 1 new full-attention chunks, so every
newly created in-window chunk is committed and nothing else. At the end L2
holds 12 of the 36 sliding-window chunks. Under `--commit-policy never` the
same conversation writes no sliding-window file at all, and the per-turn
latencies are identical (10.0 / 2.5 / 3.3 s), which is the point of copying
rather than moving.

Against `default` write-through, which writes every sliding-window chunk of
every sequence, that is a third of the traffic here and less on longer
conversations. Against eviction-only write-back, which writes nothing while
L1 has room, it is strictly more. The trade is bytes for a bounded tail
latency, and `--commit-policy never` declines it.

## Known limits

* `Request.resumable` (vLLM streaming sessions) keeps a request alive across
  turns, so `END_SESSION` never fires and no commit happens. Such a deployment
  needs a different trigger.
* Eviction still routes a committed window to the write-back path rather
  than discarding it, so the write is attempted a second time. An adapter
  that skips keys it already holds absorbs the write itself, but the L1 read
  that precedes it still happens. Teaching the eviction policy that a
  committed key already has an L2 copy is the obvious next step.
