# Native MP Protocol Schema

Protocol version: `1`.

Version 1 preserves the existing LMCache MP ZMQ/msgpack envelope. The version
constant is additive and does not add a new wire frame.

## Serialization

- Request serialization: msgpack via `msgspec`.
- DEALER client request frames: `request_uid`, `request_type`, then zero or
  more payload frames.
- ROUTER server frames: `identity`, `request_uid`, `request_type`, then zero
  or more payload frames.
- Response frames: original prefix frames plus an optional response payload.
- Error schema: there is no typed wire error frame in version 1. Handler
  exceptions are logged and may omit a response.

## Request Types

| Value | Name | Payload Schema | Response Schema |
|---:|---|---|---|
| 1 | `REGISTER_KV_CACHE` | `int`, `KVCache`, `str`, `int`, `EngineType`, `LayoutHints` | `None` |
| 2 | `UNREGISTER_KV_CACHE` | `int` | `None` |
| 3 | `STORE` | `IPCCacheEngineKey`, `int`, `list[int]`, `bytes` | `tuple[bytes, bool]` |
| 4 | `RETRIEVE` | `IPCCacheEngineKey`, `int`, `list[int]`, `bytes`, `int` | `tuple[bytes, bool]` |
| 5 | `LOOKUP` | `IPCCacheEngineKey`, `int` | `None` |
| 6 | `QUERY_PREFETCH_STATUS` | `str` | `int | None` |
| 7 | `QUERY_PREFETCH_LOOKUP_HITS` | `str` | `int | None` |
| 8 | `FREE_LOOKUP_LOCKS` | `IPCCacheEngineKey`, `int` | `None` |
| 9 | `END_SESSION` | `str` | `None` |
| 10 | `CLEAR` | none | `None` |
| 11 | `GET_CHUNK_SIZE` | none | `int` |
| 12 | `PING` | none | `bool` |
| 13 | `REPORT_BLOCK_ALLOCATION` | `int`, `str`, `list[BlockAllocationRecord]` | `None` |
| 14 | `NOOP` | none | `str` |
| 15 | `CB_REGISTER_KV_CACHE` | `int`, `KVCache`, `str`, `int` | `None` |
| 16 | `CB_UNREGISTER_KV_CACHE` | `int` | `None` |
| 17 | `CB_STORE_PRE_COMPUTED` | `IPCCacheEngineKey`, `int`, `int`, `bytes` | `tuple[bytes, bool]` |
| 18 | `CB_LOOKUP_PRE_COMPUTED` | `IPCCacheEngineKey` | `list[tuple[int, int]]` |
| 19 | `CB_RETRIEVE_PRE_COMPUTED` | `IPCCacheEngineKey`, `list[tuple[int, int]]`, `int`, `int`, `bytes` | `tuple[bytes, bool]` |
| 20 | `CB_STORE_FINAL` | `IPCCacheEngineKey`, `int`, `int`, `bytes` | `tuple[bytes, bool]` |
| 21 | `CB_LOOKUP_PRE_COMPUTED_V2` | `IPCCacheEngineKey` | `list[CBMatchResult]` |
| 22 | `CB_RETRIEVE_PRE_COMPUTED_V2` | `IPCCacheEngineKey`, `list[CBMatchResult]`, `int`, `int`, `bytes` | `tuple[bytes, bool]` |
| 23 | `LOOKUP_WITH_RESULT` | `IPCCacheEngineKey`, `int` | `int | None` |

The authoritative generated JSON form is available with:

```bash
python tools/mp_protocol_schema.py
```
