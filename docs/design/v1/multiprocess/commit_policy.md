# Sliding-Window Commit on Turn Boundaries (Multiprocess Mode)

## 1. Motivation

A hybrid-attention model's sliding-window KV is large but short-lived:
only the last `w` chunks matter for prefix reuse, and the rest is dead
weight. This design commits exactly that live window to L2 when a chat
turn ends, instead of writing every sliding-window chunk through.

### Why most sliding-window KV is wasted in L2

Resuming a sequence at token offset `p` requires:

- **Full-attention KV**: every chunk in `[0, p)`.
- **Sliding-window KV**: only the last `w` chunks, `[p - w, p)`.

Every older sliding-window chunk is dead weight for prefix reuse. These
chunks are also the large ones — per chunk, a sliding-window object group
holds an order of magnitude more bytes than a full-attention group.

The `default` store policy writes every chunk through to L2, so most L2
bandwidth and space goes to KV that no future request reads back. The
[`full_attention_only` store policy](../distributed/storage_controllers/full_attention_only_store_policy.md)
fixes this by keeping sliding-window groups in L1 only, but it leaves one
question unanswered: which sliding-window window still needs to reach L2,
and when?

### The right window depends on where the turn ends

The window worth committing ends at offset `p` — wherever the next
request's prefix match will stop. While a request is running, `p` is
unknown: the sequence is still growing. Once the request finishes, `p` is
known.

A chat model ends its turn by emitting a turn-end token. If the request
stopped on that token, it ended exactly where the next prompt resumes. If
the request was aborted, hit its length cap, errored, or tripped repetition
detection, it ended mid-turn — and no follow-up resumes there.

### The commit: copy the live window at `END_SESSION`

When a request ends on a turn boundary, this design copies its final
sliding window — `[p - w, p)` for every sliding-window group — from L1 to
L2. It is a copy, not a move: the L1 window stays, so the follow-up is
still an L1 hit. The write lands during idle time between turns, before the
follow-up arrives and before L1 pressure can evict the window.

Sliding-window chunks outside a committed window never leave L1. Eviction
simply discards them.

## 2. Design

### 2.1 Overview

```text
vLLM scheduler                        
  request_finished(request)            
    └─ _build_session_end_info             
         └─ SessionEndInfo ──END_SESSION──▶  handle_end_session(request_id, end_info)
                                              ├─ _maybe_commit_window   (new)
                                              │    ├─ CommitPolicy.should_commit
                                              │    ├─ resolve_anchor
                                              │    └─ StorageManager.flush_l1_keys_to_l2
                                              │         └─ StoreController.submit_flush
                                              └─ end_session(request_id)   (unchanged)
```

The design splits the commit into two decisions:

* **Where** the window ends (`--commit-anchor`): a per-deployment setting,
  because it follows from the chat template and client, not from one request.
* **Whether** to commit (`CommitPolicy`): a per-request decision based on
  how the engine says the request ended. What counts as evidence depends on
  the anchor — a window at `generation_end` is worth committing only if the
  generation ended where a follow-up resumes, while a window at `prompt_end`
  covers a prompt the follow-up re-sends regardless. The policy therefore
  sees the anchor in its `CommitContext`.

The policy returns a `bool`, not a token range. A range would let callers
under-cover a group's window: each sliding-window object group derives its
own start by subtracting its own `w` from the anchor, and a window missing
its oldest chunk gets written but never read. A `bool` cannot express that
mistake and needs none of the clamping, rounding, or quota guards a range
would.

### 2.2 Connector Side: `SessionEndInfo`

`request_finished` runs while the vLLM `Request` is still alive, so the
connector reads the finish reason and stop token there and packs them into
the `END_SESSION` payload. In lazy-offload mode, `END_SESSION` goes out
later from `update_connector_output`; the value waits in
`_pending_end_info` in between.

| Field | Source | Meaning |
|---|---|---|
| `finish_reason` | `RequestStatus.get_finished_reason(request.status)` | `"stop"`, `"length"`, `"abort"`, `"error"`, `"repetition"`; `""` when unknown |
| `stop_token_id` | `request.stop_reason` if it is an `int`, else the last of `request.output_token_ids` | Token the generation stopped on; `-1` when unknown |

vLLM sets `stop_reason` only for `stop_token_ids`. A stop on the model's
own EOS leaves it `None` (`check_stop` in `vllm/v1/core/sched/utils.py`) —
the ordinary chat case — so the connector falls back to the last generated
token. `NO_SESSION_END_INFO` (all defaults) is what an engine that observes
nothing sends; every built-in policy treats it as "do not commit".

### 2.3 Server Side: `CommitPolicy`

`handle_end_session` builds a `CommitContext` from the session and the
`SessionEndInfo`, asks the policy, and only then calls `end_session`. The
order matters: `end_session` removes the session that the commit reads.

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
must be side-effect-free. If a policy raises, the framework logs the
exception and treats the answer as `False` (`resolve_commit`). Policies
register by name with `register_commit_policy_factory`; a runtime plugin
can add its own.

The built-in `stop_token` policy decides by anchor:

- **`generation_end`**: returns `True` iff `finish_reason == "stop"` and
  `stop_token_id` is in the configured boundary set (empty = any stop
  token). A request that was aborted, hit its length cap, errored, or
  tripped repetition detection ended mid-turn. A mid-turn tail may be
  re-rendered or never sent again, so committing its window risks a write
  that is never read.
- **`prompt_end`**: the generation's ending is irrelevant — the prompt is
  re-sent either way. Returns `True` for every finish reason except
  `abort`, `error`, and an empty report, the cases where the conversation
  itself may not continue.

### 2.4 Anchor and Key Selection

`resolve_anchor` maps the anchor to a token offset, clips it to
`stored_end` (nothing past that is in L1), and rounds down to a chunk
boundary. For each sliding-window object group `g`,
`_maybe_commit_window` takes the `num_chunks_in_sw[g]` chunks ending at
that boundary and resolves their `ObjectKey`s through the session's hash
chain. Full-attention groups are skipped — the store path already wrote
them through.

| Anchor | Offset | Right when |
|---|---|---|
| `generation_end` (default) | `stored_end` | the next prompt re-sends the generated answer verbatim |
| `prompt_end` | `prompt_end` | the next prompt re-renders the assistant turn differently from what was generated. Qwen3's template drops `<think>...</think>` from earlier assistant turns, so a client that does not echo `reasoning_content` diverges from the generated tokens right after the assistant header. Also the anchor for a client that caps answers by length and resumes from the capped text: the answer's own chunks are not covered, but the prompt's are, and the commit no longer depends on a stop token |

### 2.5 Execution: `StoreMode.FLUSH`

`StorageManager.flush_l1_keys_to_l2(keys)` hands the keys to
`StoreController.submit_flush`, which enqueues them and wakes the store
loop. Nothing runs on the `END_SESSION` handler's thread.

The store loop takes an L1 read lock on each key (`reserve_read`), submits
one store task per active L2 adapter, and releases the lock when the task
completes — the same flow as a write-through store. The task's mode is
`StoreMode.FLUSH`, which differs from a plain store in two ways:

1. It targets every active adapter without consulting `StorePolicy`, since
   `full_attention_only` is what kept these keys out of L2 in the first
   place.
2. The policy's L1 deletions are not applied on completion.

A configuration with no active L2 adapter drops the batch with a warning.

## 3. Configuration

| Flag | Default | Meaning |
|---|---|---|
| `--commit-policy` | `stop_token` | Registered policy name. Unknown names fail at startup. |
| `--commit-anchor` | `generation_end` | `generation_end` or `prompt_end`, see 2.4. |
| `--commit-boundary-tokens` | empty | Token ids the `stop_token` policy accepts as a turn boundary at `generation_end`; empty accepts any stop token. Ignored at `prompt_end`. |

The commit path is always live. A full-attention model has no sliding-window
group and commits nothing regardless of the setting. Under the `default`
store policy, every sliding-window chunk is already in L2 and a commit only
re-writes the window, so the path is meant to pair with
`full_attention_only`.

There is no "off" policy. The one reason to want one — an L1 that never
evicts — is a sizing fact a deployment can express as its own policy if the
extra L2 bytes matter. There is no per-request override either: L2 write
volume is a property of the server, not of its callers.

`--commit-boundary-tokens` is needed when `eos_token_id` holds more than
the turn marker (Qwen: `<|im_end|>` = 151645 next to `<|endoftext|>` =
151643). It also selects *which* turn boundaries trigger a commit on a
model that ends tool calls and final answers on different tokens.

Example with gpt-oss, which lists `<|return|>` (200002) and `<|call|>`
(200012):

| Setting | Effect |
|---|---|
| `200002` | commits finished answers only |
| `200012` | commits tool calls only |
| empty (default) | commits both |

Both is the default because a tool result can take an hour to come back —
a tool-call window cannot wait in L1 any more than an answer's can. Qwen
ends both kinds of turn on `<|im_end|>`; distinguishing them would require
the serving frontend's finish reason, which the connector does not carry.

## 4. Cost

Each committed turn writes one window: `w` chunks per sliding-window group.
Consecutive turns of the same conversation overlap in that window, and chunk
hashes are content-derived. An L2 adapter that skips existing keys (the
filesystem adapter checks the path before writing) writes only the chunks
the turn's window gained over the previous one.

Per-turn latency is unchanged because the L1 copy stays.

## 5. Failure Modes

* **No L2 adapter / key already evicted / key write-locked.**
  The flush is dropped (with a warning for the no-adapter case). The window
  has no L2 copy. The next turn is still an L1 hit while the window is
  resident; it pays a prefill only if L1 evicts the window before then —
  the same situation as without this design.
* **Policy raises.** Logged and treated as "do not commit".
* **Engine sends no `SessionEndInfo`.** `NO_SESSION_END_INFO` (all defaults)
  is the default payload, and the built-in policy refuses it.
* **`Request.resumable` (vLLM streaming sessions).** The request stays
  alive across turns, so `END_SESSION` never fires and no commit happens.
  Such a deployment needs a different trigger.
