# Sliding-Window Commit on Turn Boundaries (Multiprocess Mode)

Covers the `END_SESSION` handler in `lmcache/v1/multiprocess/modules/lookup.py`,
the policy module `lmcache/v1/multiprocess/commit_policy.py`, the
`StoreMode.FLUSH` path in
`lmcache/v1/distributed/storage_controllers/store_controller.py`, and the
finish-reason reporting in `lmcache/integration/vllm/lmcache_mp_connector.py`.

## 1. Motivation

A store policy that keeps sliding-window chunks out of L2 leaves L1 holding
their only copy. Such a chunk reaches L2 only when eviction writes it back,
which happens under memory pressure, the moment the next turn of the same
conversation may already be looking the window up. When the follow-up's
lookup lands while the write-back is still in flight, it misses and pays a
full prefill.

The bytes are the same either way; the timing is not. A chat turn is followed
by idle time while a person reads and types. A window written then is in L2
long before the follow-up, and it only matters at all while L1 cannot hold
every live window.

The design copies a finished request's final sliding window from L1 to L2 at
`END_SESSION` when the request ended on a chat turn boundary. It is a copy:
the L1 window stays, so the follow-up is still an L1 hit and only the
window's durability changes.

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

Two decisions are kept apart:

* **Whether** to commit is per request and is the policy's answer, decided
  from how the engine says the request ended.
* **Where** the committed window ends is per deployment (`--commit-anchor`),
  because it follows from the chat template and the client, not from one
  message.

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
| `end_info` | the `SessionEndInfo` above |
| `prompt_end` | end of the last range the engine looked up (`Session.lookup_end`) |
| `stored_end` | furthest offset the session resolved keys for (`Session.resolved_end`) |
| `hit_chunks` | chunks the request's own lookup hit; `-1` if never consumed |
| `attn_desc` | the model's `AttnWindowDesc` |

`CommitPolicy.should_commit(ctx) -> bool` runs on the CPU pool thread and
must be side-effect free. A policy that raises is logged and treated as
`False` (`resolve_commit`). Policies are registered by name with
`register_commit_policy_factory`, so a runtime plugin can add its own.

The built-in `stop_token` policy returns `True` iff `stored_end > 0`,
`finish_reason == "stop"`, and `stop_token_id` is in the configured boundary
set (any token if the set is empty). A request that was aborted, hit its
length cap, errored, or tripped repetition detection ended mid-turn; its tail
is re-rendered or never sent again, and its window would be written and never
read.

### 2.4 Anchor and Key Selection

`resolve_anchor` maps `--commit-anchor` to a token offset, clips it to
`stored_end` (nothing past it is in L1) and rounds down to a chunk boundary.
From that anchor, for every sliding-window object group `g`,
`_maybe_commit_window` takes the `num_chunks_in_sw[g]` chunks ending there and
resolves their `ObjectKey`s through the session's own hash chain.
Full-attention groups are skipped; the store path already wrote them through.

| Anchor | Offset | Right when |
|---|---|---|
| `generation_end` (default) | `stored_end` | the next prompt re-sends the generated answer verbatim |
| `prompt_end` | `prompt_end` | the next prompt re-renders the assistant turn differently from what was generated. Qwen3's template drops `<think>...</think>` from earlier assistant turns, so a client that does not echo `reasoning_content` diverges from the generated tokens right after the assistant header |

### 2.5 Execution: `StoreMode.FLUSH`

`StoreController.submit_flush(keys)` enqueues on the eviction write-back's
queue and eventfd; the store loop drains both kinds together through the same
`reserve_read` / `submit_store_task` / completion code. The one difference is
at completion: a `WRITEBACK` task deletes the key from L1 once every adapter
succeeded, a `FLUSH` task does not.

Like the write-back, a flush bypasses `StorePolicy`: it writes whatever keys
it is given. Keys already in flight on either path are dropped from the
batch, so a flush racing an eviction write-back of the same key is a no-op
rather than a double write.

## 3. Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--commit-policy` | `stop_token` | Registered policy name. Unknown names fail at startup. |
| `--commit-anchor` | `generation_end` | `generation_end` or `prompt_end`, see 2.4. |
| `--commit-boundary-tokens` | empty | Token ids the `stop_token` policy accepts as a turn boundary; empty accepts any stop token. |

The commit path is always live. A full-attention model has no sliding-window
group and commits nothing whatever the setting. There is no "off" policy: the
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
turn's window gained. Against `default` write-through this is a fraction of
the sliding-window traffic; against eviction-only write-back it is strictly
more. Per-turn latency is unchanged because the L1 copy stays.

## 5. Failure Modes

* **No L2 adapter, key already evicted, key write-locked, key in flight.**
  The flush is dropped (a warning for the adapter case). The window still
  reaches L2 through the eviction write-back, as before; what is lost is
  timeliness, not the window.
* **Policy raises.** Logged, treated as "do not commit".
* **Engine sends no `SessionEndInfo`.** `NO_SESSION_END_INFO` is the default
  second payload, and the built-in policy refuses it.
* **`Request.resumable` (vLLM streaming sessions).** The request stays alive
  across turns, `END_SESSION` never fires, and no commit happens. Such a
  deployment needs a different trigger.
* **Second write on eviction.** Eviction still routes a committed window
  through the write-back path. An adapter that skips keys it already holds
  absorbs the write, but the L1 read before it still happens. Teaching the
  eviction policy that a committed key already has an L2 copy is the next
  step.
