# PD Disaggregation Debug Logs — Flow & Trace Guide

This guide explains **what each debug log means** and **how they connect** so you can follow the flow and trace the small-prompt bug.

---

## 1. The Big Picture: Who Does What

In **PD (Prefill Disaggregation)** mode you have:

| Component | Role |
|-----------|------|
| **Prefill server** | Runs the prompt (e.g. "Hi"), computes KV cache, **saves** it and **sends ProxyNotif** when done |
| **Proxy** | Forwards requests, waits for **ProxyNotif** from prefill, then tells decode to run |
| **Decode server** | Waits for ProxyNotif, then generates tokens |

**The bug:** For small prompts (e.g. 2 tokens), the prefill server **never sends ProxyNotif**, so the decode server waits forever.

---

## 2. The Call Chain (Where the Logs Live)

When prefill finishes a batch, this is the path that **should** run to send ProxyNotif:

```
vLLM scheduler
    │
    ▼  build_connector_meta()  ← builds ReqMeta for each request
    │
    ▼  [after forward pass] wait_for_save()  ← decides: skip or call store()?
    │
    ▼  lmcache_engine.store(token_ids, ...)  ← copies KV from GPU, builds keys
    │
    ▼  storage_manager.batched_put(keys, memory_objs, transfer_spec)
    │
    ▼  PDBackend.batched_submit_put_task(keys, memory_objs, transfer_spec)
    │
    ▼  proxy_side_channel.send(ProxyNotif)  ← decode server can proceed
```

If any step is skipped or returns early, ProxyNotif is never sent.

---

## 2.1 RequestTracker flow and key values

**RequestTracker** is LMCache’s metadata for one request. It is **not** vLLM’s request object; it’s our own state so we know what we’ve saved and what to save next.

| Concept | Where it lives | Meaning |
|--------|----------------|---------|
| **RequestTracker** | LMCache (`_request_trackers[req_id]`) | Our metadata: `req_id`, `prompt_len`, `token_ids`, `allocated_block_ids`, **`num_saved_tokens`**, `disagg_spec`, etc. |
| **num_computed_tokens** | **vLLM’s request** (we only read it) | Number of tokens vLLM has **already computed** for this request (before this batch). We **do not set** it; vLLM does. Logged in **build_connector_meta**. |
| **num_saved_tokens** | **RequestTracker** | Number of tokens we have **already saved** to LMCache for this request. Starts at 0 for new requests; updated in `from_request_tracker` when we save. |
| **is_last_prefill** | Computed **inside** `from_request_tracker` | **Starts as False.** Set to **True** when `input_token_len >= tracker.prompt_len` (we’ve now computed the full prompt). For a 2-token prompt in one batch, that’s true → **is_last_prefill = True**. |
| **discard_partial_chunks** | Config → passed into `from_request_tracker` | From `_discard_partial_chunks` (config: `discard_partial_chunks` or `save_unfull_chunk`). **Typical value: True.** When True, we round down how many tokens to save to full chunk boundaries, so small prompts can round to 0. |

Flow in code:

1. **New request:** vLLM gives us a request. We create `RequestTracker` via `RequestTracker.from_new_request(...)` and store it in `_request_trackers[req_id]`. We then build `ReqMeta` with `ReqMeta.from_request_tracker(tracker, ..., discard_partial_chunks=self._discard_partial_chunks)`.
2. **Same request scheduled again (e.g. next chunk):** We get the same `RequestTracker` from `_request_trackers[req_id]`, call `tracker.update(...)` with new token/block info, then again `ReqMeta.from_request_tracker(tracker, ...)`.
3. **Inside `from_request_tracker`:** We compute `is_last_prefill = (input_token_len >= tracker.prompt_len)`, then `num_tokens_to_save` (with or without rounding depending on `discard_partial_chunks`). So **num_computed_tokens** is never set here — it’s on vLLM’s request and only appears in the **build_connector_meta** log.

### skip_leading_tokens: when it's set and why it's 0

- **Set in:** `from_request_tracker` at the start: `skip_leading_tokens = tracker.num_saved_tokens`.
- **Stored in:** `SaveSpec(skip_leading_tokens, not skip_save)` and carried on `ReqMeta.save_spec`.
- **Used in:** `wait_for_save()` as `skip_leading_tokens = save_spec.skip_leading_tokens`; then we skip the request when `skip_leading_tokens == len(token_ids)` (nothing new to save).

**Why it's 0 for a new 2-token request:** `num_saved_tokens` is "how many tokens we've already saved to LMCache for this request." For a **new** request it's set in `from_new_request` to `lmcache_cached_tokens` (0 when there's no cache hit). So **skip_leading_tokens = 0** means "we haven't saved any tokens for this request yet." For a request that was already saved in a previous batch, `num_saved_tokens` would be &gt; 0 (e.g. 256), so we'd only save tokens *after* that boundary; `skip_leading_tokens` tells `wait_for_save` how many leading tokens to skip when building the store mask.

---

## 3. Logs in Execution Order (What You See in prefiller.log)

For a **2-token prompt** like "Hi", logs appear in this order. Each line tells you **which step** ran and **what the code decided**.

### Phase A: Request is scheduled (before forward pass)

| Log tag | File:line | What it means |
|---------|-----------|----------------|
| `[DEBUG] build_connector_meta:` | vllm_v1_adapter:1488 | vLLM built metadata. **num_computed_tokens** = from vLLM (tokens already computed; 0 for new request). **num_tokens_to_compute=2, prompt_len=2** → 2 tokens this batch. **discard_partial_chunks** = config value we pass into from_request_tracker (typically True). |
| `[DEBUG] from_new_request:` | vllm_v1_adapter:187 | LMCache created a **new** RequestTracker (not from cache). **token_ids_len=2** → we have 2 token IDs for this prompt. |

**Takeaway:** Request is new, prompt length = 2. num_computed_tokens is set by vLLM; we only log it here.

---

### Phase B: ReqMeta is built (how many tokens to save?)

| Log tag | File:line | What it means |
|---------|-----------|----------------|
| `[DEBUG] from_request_tracker:` | vllm_v1_adapter:362 | **input_token_len=2, prompt_len=2, num_saved_tokens=0** (nothing saved yet). **is_last_prefill=True** (2 ≥ 2 → full prompt computed). **discard_partial_chunks=True** (from config). **num_tokens_to_save=0** (formula `2 // 256 * 256 = 0`). |
| `[DEBUG] from_request_tracker: AFTER SLICE` | vllm_v1_adapter:383 | **token_ids_len=0 (sliced from 2)**. We sliced `token_ids = input_token_ids[:0]` → **empty list**. |

**Takeaway:** Because prompt &lt; chunk size (256) and discard_partial_chunks=True, we end up with **num_tokens_to_save=0** and **token_ids=[]**. This is the root of the bug. is_last_prefill is **True** in this case (not False).

---

### Phase C: After forward pass — should we call store()?

| Log tag | File:line | What it means |
|---------|-----------|----------------|
| `[DEBUG] wait_for_save:` | vllm_v1_adapter:1153 | Before the skip check: **skip_leading_tokens=0, len(token_ids)=0, is_last_prefill=True, kv_role=kv_producer, has_disagg_spec=True**. |
| `[DEBUG] wait_for_save: SKIPPING` | vllm_v1_adapter:1167 | The condition **skip_leading_tokens == len(token_ids)** → **0 == 0** is TRUE, so we **continue** and **never call store()**. |

**Takeaway:** Code skips this request entirely. You will **not** see `wait_for_save: CALLING store()` for this request.

---

### Phase D: These logs do NOT appear (when bug happens)

Because we skipped, the following **never run** for this request:

| Log tag | File:line | When you see it |
|---------|-----------|------------------|
| `[DEBUG] wait_for_save: CALLING store()` | vllm_v1_adapter:1211 | Only when we **don’t** skip → not for small prompts (bug). |
| `[DEBUG] LMCacheEngine.store: EMPTY memory_objs` | cache_engine:477 | Only if store() **was** called with empty tokens (e.g. after a fix). |
| `[DEBUG] cache_engine.store: CALLING batched_put` | cache_engine:495 | Only when store() has non-empty memory_objs and proceeds. |
| `[DEBUG] storage_manager.batched_put: CALLING backend...` | storage_manager:434 | Only when batched_put() runs. |
| `[DEBUG] PDBackend.batched_submit_put_task: ENTERED` | pd_backend:384 | Only when the PD backend receives a put task. |
| `[DEBUG] PDBackend: SENDING ProxyNotif` | pd_backend:453 | Only when we actually send the notification. |

**Takeaway:** Absence of these logs for a small-prompt request **is** the bug: the path from store() → batched_put → PDBackend → ProxyNotif is never taken.

---

### Phase E: Normal case (e.g. 300-token prompt)

For a prompt **≥ chunk size**, you would see:

1. `from_request_tracker:` → **num_tokens_to_save=256** (or more)
2. `from_request_tracker: AFTER SLICE` → **token_ids_len=256**
3. `wait_for_save:` → **len(token_ids)=256** → no skip (0 ≠ 256)
4. `wait_for_save: CALLING store()`
5. `cache_engine.store: CALLING batched_put`
6. `storage_manager.batched_put: CALLING backend...`
7. `PDBackend.batched_submit_put_task: ENTERED`
8. `PDBackend: SENDING ProxyNotif`

That full sequence means ProxyNotif was sent and decode can proceed.

---

## 4. One-Page Trace Cheat Sheet

```
REQUEST "Hi" (2 tokens) — BUG TRACE
===================================

build_connector_meta     → prompt_len=2, num_tokens_to_compute=2
from_new_request         → token_ids_len=2 (new request)
from_request_tracker     → num_tokens_to_save=0  ← rounds down!
from_request_tracker     → AFTER SLICE token_ids_len=0  ← empty list
wait_for_save            → skip_leading_tokens=0, len(token_ids)=0
wait_for_save            → SKIPPING  ← BUG: we skip and never call store()

[NOT REACHED]
  wait_for_save          → CALLING store()
  cache_engine.store     → CALLING batched_put
  storage_manager        → CALLING backend.batched_submit_put_task
  PDBackend              → ENTERED
  PDBackend              → SENDING ProxyNotif

Result: ProxyNotif never sent → decode hangs.
```

---

## 5. Quick Reference: Log → Meaning

| Log contains | Meaning |
|--------------|--------|
| **num_computed_tokens** | From vLLM request (tokens already computed); 0 for new request. Only in build_connector_meta. |
| **num_saved_tokens** | From RequestTracker: how many tokens we’ve already saved for this request. |
| **is_last_prefill** | True when we’ve computed the full prompt (input_token_len ≥ prompt_len); in small-prompt bug it’s True. |
| **discard_partial_chunks** | Config: if True, we round tokens to save down to full chunks; typically True. |
| **num_tokens_to_save=0** | Chunk logic decided to save no tokens (prompt &lt; 256 when discard_partial_chunks=True). |
| **token_ids_len=0 (sliced from N)** | We sliced to 0 tokens; nothing to save in LMCache. |
| **wait_for_save: SKIPPING** | We skipped this request; store() is not called. |
| **CALLING store()** | We did not skip; store() will run. |
| **EMPTY memory_objs** | store() was called but had no KV blocks (e.g. 0 tokens). |
| **CALLING batched_put** | store() is calling the storage layer. |
| **CALLING backend.batched_submit_put_task** | Storage manager is invoking the PD backend. |
| **PDBackend.batched_submit_put_task: ENTERED** | PD backend received the put task. |
| **SENDING ProxyNotif** | Notification sent; decode can proceed. |
| **NOT sending ProxyNotif** | Not last prefill chunk; no notification this time. |

---

## 6. How to Use This When Debugging

1. **Reproduce:** Send a 2-token prompt (e.g. "Hi"), capture prefiller.log.
2. **Confirm bug:** You see SKIPPING and you do **not** see CALLING store(), CALLING batched_put, PDBackend ENTERED, or SENDING ProxyNotif.
3. **After a fix:** For the same 2-token prompt you should see CALLING store(), then either EMPTY memory_objs + batched_put, or batched_put directly, then PDBackend ENTERED and **SENDING ProxyNotif**.
4. **Filter one request:** `grep "req_id=YOUR_REQ_ID" prefiller.log` to see the full trace for that request.

This is the flow and trace the debug logs are designed to show.
