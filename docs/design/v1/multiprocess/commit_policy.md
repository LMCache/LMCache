# Sliding-window commits: `--commit-policy`

When a request finishes at a chat turn boundary, the server copies its final
sliding window from L1 to L2 right away instead of waiting for eviction to
write it back. The L1 copy stays. The next turn of the same conversation is
still an L1 hit, and if L1 gives the window up in the meantime the follow-up
finds it in L2 instead of recomputing it.

## Problem

Under a store policy that keeps sliding-window chunks out of L2, such a chunk
has exactly one copy, in L1, until eviction writes it back. That write happens
under memory pressure, which is exactly when the next turn of the same
conversation may already be looking the window up.

Measured on one GPU serving eight concurrent conversations with book-length
prompts at L1 = 20 GB: between a session's turn 0 and turn 1 the other seven
sessions push its window out, the write-back is still in flight when the
follow-up arrives, and p95 turn latency goes from 0.5 s to 12.6 s, a full
prefill. p50 is unaffected. At L1 = 96 GB nothing is evicted and there is no
tail.

The bytes are the same either way; the timing is not. A chat turn is followed
by seconds to minutes of idle time while a person reads and types, and a
window written then is in L2 long before the follow-up.

## Design

### Trigger: the request ended on a turn boundary

A window is worth writing only if a later request will match the prefix it
ends at. For a chat model that is decided by how the turn ended: a request
that stopped on the model's turn-end token (Qwen: `<|im_end|>`, id 151645)
ended exactly where the next prompt resumes. A request that was aborted, hit
its length cap, errored, or tripped repetition detection did not; its tail is
re-rendered or never sent again, and its window would be written and never
read.

The vLLM connector observes the finish reason and the stop token in
`request_finished`, while the `Request` is still in hand, and sends them to
the server as `SessionEndInfo`, the second `END_SESSION` payload. vLLM sets
`stop_reason` only for `stop_token_ids`; a stop on the model's own EOS leaves
it `None`, so the connector falls back to the last generated token.

The server-side rule is a `CommitPolicy`, selected by `--commit-policy`. The
built-in `stop_token` policy commits when the finish reason is `stop` and the
stop token is in `--commit-boundary-tokens` (any token if the set is empty).
A policy is a `should_commit(ctx) -> bool`; plugins register more with
`register_commit_policy_factory`.

### Extent: one window, ending at the anchor

A policy answers only *whether*. *Where* the window ends is the
`--commit-anchor` option, a deployment constant:

* `generation_end` (default): the last chunk the request stored. Right when
  the next prompt re-sends the generated answer verbatim.
* `prompt_end`: the end of the looked-up prompt. Right when the next prompt
  re-renders the assistant turn differently from what was generated. Qwen3's
  chat template drops `<think>...</think>` from earlier assistant turns, so a
  client that does not echo `reasoning_content` back diverges from the
  generated tokens right after the assistant header, and only the prompt
  prefix can match.

The anchor is rounded down to a chunk boundary and clipped to
`Session.resolved_end`, the furthest offset the session resolved keys for.
From it, every sliding-window object group takes its own `w` trailing chunks
(`AttnWindowDesc.num_chunks_in_sw`). Full-attention groups are skipped; the
store path already wrote them through.

### Placement: in front of `end_session`

```
handle_end_session(request_id, end_info):   # the END_SESSION handler
    maybe commit the window                 # this design
    end_session(request_id)                 # unchanged
```

The commit runs first because `end_session` removes the session it reads.
The two share nothing else.

### Execution: the write-back path, minus the delete

`StoreController.submit_flush` reuses the eviction write-back's queue, lock,
eventfd and processing. The one difference is at completion:
`StoreMode.WRITEBACK` deletes the key from L1 once every adapter succeeded,
`StoreMode.FLUSH` leaves it. Like the write-back, a flush bypasses the store
policy and writes whatever keys it is given.

A dropped commit (no L2 adapter, key already evicted, policy raised, key
already in flight) changes nothing: the window still reaches L2 through the
eviction write-back, as before. What is lost is timeliness, not the window.

## Configuration

```
--commit-policy stop_token                # default; plugins may add more
--commit-anchor generation_end|prompt_end # default generation_end
--commit-boundary-tokens 151645           # Qwen: <|im_end|>; empty = any
```

The commit path is always live. A full-attention model has no sliding-window
group and commits nothing whatever the setting. There is no "off" policy: the
one reason to want one, an L1 that never evicts, is a sizing fact a
deployment can express as its own policy if the extra L2 bytes matter.

`--commit-boundary-tokens` also selects *which* turn boundaries commit on a
model that ends tool calls and final answers on different tokens. gpt-oss
lists both `<|return|>` (200002) and `<|call|>` (200012) in `eos_token_id`:
`200002` alone commits finished answers, `200012` alone commits tool calls,
empty commits both. Both is the default because a tool result can take an
hour to come back, so a tool-call window cannot wait in L1 any more than an
answer's can. Qwen ends both kinds of turn on `<|im_end|>`; telling them
apart there would need the serving frontend's finish reason, which the
connector does not carry.

There is no per-request override. L2 write volume stays a property of the
server, not of its callers.

## Cost

One window per committed turn: `w` chunks, 8 x 56 MiB = 448 MiB on the model
below. Consecutive turns overlap in that window and chunk hashes are
content-derived, so an L2 adapter that skips keys it already holds (the
filesystem adapter does) writes only the chunks a turn's window gained.

Measured on a 31B hybrid sliding-window/full-attention model (PLaMo 3, GPTQ
4-bit, `--separate-object-groups`, filesystem L2, L1 = 16 GB so nothing was
evicted), three turns of one conversation:

| turn | sliding-window files in L2 | full-attention files in L2 |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 8 (448 MiB) | 32 (256 MiB) |
| 2 | 11 | 35 |
| 3 | 12 | 36 |

Turn 1 commits one window. Turns 2 and 3 add 3 and 1 sliding-window files
against 3 and 1 new full-attention chunks, so every newly created in-window
chunk is committed and nothing else. Per-turn latencies (10.0 / 2.5 / 3.3 s)
are identical with the commit path disabled, which is the point of copying
rather than moving.

Against `default` write-through, which writes every sliding-window chunk of
every sequence, this is a third of the traffic here and less on longer
conversations. Against eviction-only write-back it is strictly more. The
trade is bytes for a bounded tail latency.

## Alternatives considered

* **A policy that returns a range.** Each sliding-window group derives its
  own start from the anchor by subtracting its own `w`, and one model may
  have several. A caller-supplied `[start, end)` could under-cover a group,
  and a window missing its oldest chunk is written and never read. A bool
  cannot express that mistake, and needs none of the clamping, rounding and
  quota guards a range would.
* **An idle timer** (commit a window that has sat in L1 for T seconds, on the
  theory that tool loops resume in milliseconds and humans do not). A tool
  call can take an hour, so the timer would either fire on tool boundaries
  anyway or hold a window L1 cannot afford to keep.
* **A per-request hint from the caller.** Rejected so that the server, not
  its clients, decides how much it writes to L2.

## Known limits

* `Request.resumable` (vLLM streaming sessions) keeps a request alive across
  turns, so `END_SESSION` never fires and no commit happens. Such a
  deployment needs a different trigger.
* Eviction still routes a committed window through the write-back path, so
  the write is attempted a second time. An adapter that skips keys it already
  holds absorbs the write, but the L1 read before it still happens. Teaching
  the eviction policy that a committed key already has an L2 copy is the next
  step.
