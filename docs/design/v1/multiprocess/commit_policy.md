# Sliding-Window Commit on Turn Boundaries (Multiprocess Mode)

## 1. Motivation

In a hybrid-attention model, the KV of a sliding-window layer is reusable
only within the window. Resuming a sequence at token offset `p` needs the
full-attention KV of every chunk in `[0, p)` but the sliding-window KV of
the last `w` chunks only, `[p - w, p)`; every older sliding-window chunk is
dead weight for prefix reuse. Those chunks are also the large ones: per chunk,
a sliding-window object group holds an order of magnitude more bytes than a
full-attention group.

Writing every sliding-window chunk through to L2, as the `default` store
policy does, therefore spends most of the L2 bandwidth and space on KV that
is never read back. The `full_attention_only` store policy
([../distributed/storage_controllers/full_attention_only_store_policy.md](../distributed/storage_controllers/full_attention_only_store_policy.md))
writes full-attention groups through and keeps sliding-window groups in L1
only, which removes that cost and leaves one question the store path cannot
answer on its own: which sliding-window window must still reach L2, and when.

Which window is decided by the resume point. The window worth keeping is the
one ending at the offset `p` where a later request's prefix match will end,
and `p` is not known while the request is running: the sequence is still
growing and nobody knows where it stops or where the next request resumes.
It is known the moment the request finishes. A chat model ends its turn by
emitting its turn-end token, and a request that stopped on it ended exactly
where the next prompt resumes; a request that was aborted, hit its length
cap, errored, or tripped repetition detection ended mid-turn, and no
follow-up resumes there.

So the design commits at `END_SESSION`: when a request ended on a turn
boundary, its final sliding window, `[p - w, p)` for every sliding-window
group, is copied from L1 to L2 right then. It is a copy, not a move; the L1
window stays and the follow-up is still an L1 hit. The write lands during the
idle time between turns, before the follow-up arrives and before L1 pressure
can evict the window. Sliding-window chunks outside a committed window never
leave L1, and eviction simply discards them.

## 2. Design

### 2.1 Overview

```text
vLLM scheduler                         MP server
  request_finished(request)              handle_end_session(request_id, end_info)
    └─ _build_session_end_info             ├─ _maybe_commit_window   (new)
         └─ SessionEndInfo ──END_SESSION──▶│    ├─ CommitPolicy.should_commit
                                           │    ├─ resolve_anchor
                                           │    └─ StorageManager.flush_l1_keys_to_l2
                                           │         └─ StoreController.submit_flush
                                           └─ end_session(request_id)   (unchanged)
```

Two decisions are involved:

* **Where** the committed window ends is per deployment (`--commit-anchor`),
  because it follows from the chat template and the client, not from one
  message.
* **Whether** to commit is per request and is the policy's answer, decided
  from how the engine says the request ended. What counts as evidence
  depends on the anchor: a window at `generation_end` is worth committing
  only if the generation ended where a follow-up resumes, while a window at
  `prompt_end` covers a prompt the follow-up re-sends whatever the
  generation did. The policy therefore sees the anchor in its
  `CommitContext`.

The policy returns a `bool`. A richer return (a token range) would let a
caller under-cover a group's window: each sliding-window object group derives
its own start from the anchor by subtracting its own `w`, and a window
missing its oldest chunk is written and never read. A `bool` cannot express
that mistake and needs none of the clamping, rounding and quota guards a range
would.

### 2.2 Connector Side: `SessionEndInfo`

`request_finished` runs while the vLLM `Request` is still in hand, so the
connector reads the finish reason and stop token there and sends them as the
second `END_SESSION` payload. In lazy-offload mode `END_SESSION` is sent later
from `update_connector_output`, so the value is parked in
`_pending_end_info` in between.

| Field | Source | Meaning |
|---|---|---|
| `finish_reason` | `RequestStatus.get_finished_reason(request.status)` | `"stop"`, `"length"`, `"abort"`, `"error"`, `"repetition"`; `""` when unknown |
| `stop_token_id` | `request.stop_reason` if it is an `int`, else the last of `request.output_token_ids` | Token the generation stopped on; `-1` when unknown |

vLLM sets `stop_reason` only for `stop_token_ids`. A stop on the model's own
EOS leaves it `None` (`check_stop` in `vllm/v1/core/sched/utils.py`), which
is the ordinary chat case, hence the fallback to the last generated token.
`NO_SESSION_END_INFO` (all defaults) is what an engine that observes nothing
sends; every built-in policy reads it as "do not commit".

### 2.3 Server Side: `CommitPolicy`

`handle_end_session` builds a `CommitContext` from the session and the
`SessionEndInfo`, asks the configured policy, and only then calls
`end_session`. The order matters because `end_session` removes the session
the commit reads; the two share nothing else.

| `CommitContext` field | Value |
|---|---|
| `request_id` | the finished request (`Session.request_id`) |
| `model_name` | model the session belongs to (`IPCCacheServerKey.model_name`) |
| `end_info` | the `SessionEndInfo` above |
| `prompt_end` | end of the last range the engine looked up (`Session.lookup_end`) |
| `stored_end` | furthest offset the session resolved keys for (`Session.resolved_end`) |
| `hit_chunks` | chunks the request's own lookup hit; `-1` if never consumed |
| `attn_desc` | the model's `AttnWindowDesc` |
| `anchor` | the configured `--commit-anchor` |

`CommitPolicy.should_commit(ctx) -> bool` runs on the CPU pool thread and
must be side-effect free. A policy that raises is logged and treated as
`False` (`resolve_commit`). Policies are registered by name with
`register_commit_policy_factory`, so a runtime plugin can add its own.

The built-in `stop_token` policy answers by anchor. At `generation_end` it
returns `True` iff `finish_reason == "stop"` and `stop_token_id` is in the
configured boundary set (any token if the set is empty): a request that was
aborted, hit its length cap, errored, or tripped repetition detection ended
mid-turn, and a mid-turn tail may be re-rendered or never sent again, so its
window could be written and never read. At `prompt_end` the generation's
ending is irrelevant, since the prompt is re-sent either way; it returns
`True` for every finish reason except `abort`, `error` and an empty report,
the cases where the conversation itself may not continue.

### 2.4 Anchor and Key Selection

`resolve_anchor` maps the anchor to a token offset, clips it to
`stored_end` (nothing past it is in L1) and rounds down to a chunk boundary.
From that anchor, for every sliding-window object group `g`,
`_maybe_commit_window` takes the `num_chunks_in_sw[g]` chunks ending there and
resolves their `ObjectKey`s through the session's own hash chain.
Full-attention groups are skipped; the store path already wrote them through.

| Anchor | Offset | Right when |
|---|---|---|
| `generation_end` (default) | `stored_end` | the next prompt re-sends the generated answer verbatim |
| `prompt_end` | `prompt_end` | the next prompt re-renders the assistant turn differently from what was generated. Qwen3's template drops `<think>...</think>` from earlier assistant turns, so a client that does not echo `reasoning_content` diverges from the generated tokens right after the assistant header. Also the anchor for a client that caps answers by length and resumes from the capped text: the answer's own chunks are not covered, but the prompt's are, and the commit no longer depends on a stop token |

### 2.5 Execution: `StoreMode.FLUSH`

`StorageManager.flush_l1_keys_to_l2(keys)` hands the keys to
`StoreController.submit_flush`, which enqueues them and wakes the store
loop; nothing runs on the `END_SESSION` handler's thread. The store loop
takes an L1 read lock on each key (`reserve_read`), submits one store task
per active L2 adapter, and releases the lock when that task completes,
exactly as a write-through store does. The task's mode is `StoreMode.FLUSH`,
which differs from a plain store in two ways: it targets every active adapter
without consulting `StorePolicy`, since `full_attention_only` is what kept
these keys out of L2 in the first place, and the policy's L1 deletions are not
applied on completion. A configuration with no active L2 adapter drops the
batch with a warning.

## 3. Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--commit-policy` | `stop_token` | Registered policy name. Unknown names fail at startup. |
| `--commit-anchor` | `generation_end` | `generation_end` or `prompt_end`, see 2.4. |
| `--commit-boundary-tokens` | empty | Token ids the `stop_token` policy accepts as a turn boundary at `generation_end`; empty accepts any stop token. Ignored at `prompt_end`. |

The commit path is always live. A full-attention model has no sliding-window
group and commits nothing whatever the setting; under the `default` store
policy every sliding-window chunk is already in L2 and a commit only re-writes
the window, so the path is meant to be paired with `full_attention_only`. There is no "off" policy: the
one reason to want one, an L1 that never evicts, is a sizing fact a
deployment can express as its own policy if the extra L2 bytes matter. There
is no per-request override either; L2 write volume stays a property of the
server, not of its callers.

`--commit-boundary-tokens` is needed when `eos_token_id` holds more than the
turn marker (Qwen: `<|im_end|>` = 151645 next to `<|endoftext|>` = 151643),
and it selects *which* turn boundaries commit on a model that ends tool calls
and final answers on different tokens. gpt-oss lists both `<|return|>`
(200002) and `<|call|>` (200012): `200002` alone commits finished answers,
`200012` alone commits tool calls, empty commits both. Both is the default
because a tool result can take an hour to come back, so a tool-call window
cannot wait in L1 any more than an answer's can. Qwen ends both kinds of turn
on `<|im_end|>`; telling them apart there would need the serving frontend's
finish reason, which the connector does not carry.

## 4. Cost

One window per committed turn: `w` chunks per sliding-window group. Consecutive
turns of one conversation overlap in that window and chunk hashes are
content-derived, so an L2 adapter that skips keys it already holds (the
filesystem adapter checks the path before writing) writes only the chunks a
turn's window gained. Per-turn latency is unchanged because the L1 copy
stays.

## 5. Failure Modes

* **No L2 adapter, key already evicted, key write-locked.**
  The flush is dropped (a warning for the adapter case) and the window has
  no L2 copy. The next turn is still an L1 hit while the window is resident;
  it pays a prefill only if L1 evicts the window before then, which is the
  situation without this design.
* **Policy raises.** Logged, treated as "do not commit".
* **Engine sends no `SessionEndInfo`.** `NO_SESSION_END_INFO` is the default
  second payload, and the built-in policy refuses it.
* **`Request.resumable` (vLLM streaming sessions).** The request stays alive
  across turns, `END_SESSION` never fires, and no commit happens. Such a
  deployment needs a different trigger.
