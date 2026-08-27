# Golden KV-event payloads (vLLM wire format)

Each `*.msgpack` file is one `KVEventBatch` payload exactly as
`lmcache.v1.mp_coordinator.kv_event_publisher.ZmqKVEventPublisher` sends it in the
third ZMQ frame (`[topic, seq, payload]`). They are produced and checked by
`tests/v1/mp_coordinator/test_kv_event_publisher.py` (`test_golden_fixture`) and
are meant to be decoded unchanged by a consumer's vLLM adapter (llm-d:
`pkg/kvevents/engineadapter/vllm_adapter.go`, `ParseMessage`).

Layout: `[ts: float, events: [event, ...], data_parallel_rank: nil]` (vLLM `KVEventBatch`, three fields), each event a positional
array with the tag first:

| File | Event | Fields |
|---|---|---|
| `store_offset0.msgpack` | `BlockStored` | `["BlockStored", [hash(0x01)*32], nil, [1,2,3,4], 4, nil, "lmcache-l1"]` |
| `store_offset256_parent.msgpack` | `BlockStored` | same, hash `0x02*32`, `parent_block_hash = 0x01*32`, tokens `[5,6,7,8]` |
| `delete.msgpack` | `BlockRemoved` | `["BlockRemoved", [0x01*32, 0x02*32], "lmcache-l1"]` |
| `store_shared_l2.msgpack` | `BlockStored` | hash `0x03*32`, tokens `[9,10,11,12]`, medium `"lmcache-l2-fs"` |

All batches have `ts = 1700000000.5`; hashes are 32-byte `bin` values
(consumers truncate to the last 8 bytes); `block_size = len(token_ids)`;
`lora_id` is `nil`; no HMA fields (`group_idx`, ...) are present. The
corresponding topics are `kv@node:n1@m` (private) and `kv@pool:fs@m` (shared).

Regenerate after an intended wire change with
`LMCACHE_UPDATE_FIXTURES=1 pytest tests/v1/mp_coordinator/test_kv_event_publisher.py`.
