# Event Metadata Contracts

Each `EventType` has a documented metadata schema.  Producers **must** populate
these keys; subscribers **may** rely on them being present.

For the full list of event types see `event.py`.  For metrics derived from
these events see [METRICS.md](METRICS.md).

---

## L1Manager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `L1_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_READ_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED` | `keys` | `list[ObjectKey]` |
| `L1_WRITE_FINISHED_AND_READ_RESERVED` | `keys` | `list[ObjectKey]` |
| `L1_KEYS_EVICTED` | `keys` | `list[ObjectKey]` |

---

## StorageManager Events

| EventType | Metadata keys | Types |
|---|---|---|
| `SM_READ_PREFETCHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_READ_PREFETCHED_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_RESERVED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |
| `SM_WRITE_FINISHED` | `succeeded_keys`, `failed_keys` | `list[ObjectKey]`, `list[ObjectKey]` |

---

## MP Server Events

These events use `session_id` on the `Event` dataclass (not in `metadata`)
to correlate START/END pairs.

| EventType | Metadata keys | Types |
|---|---|---|
| `MP_STORE_START` | `device` | `str` |
| `MP_STORE_END` | `device`, `stored_count` | `str`, `int` |
| `MP_RETRIEVE_START` | `device` | `str` |
| `MP_RETRIEVE_END` | `device`, `retrieved_count` | `str`, `int` |
| `MP_LOOKUP_PREFETCH_START` | *(none)* | — |
| `MP_LOOKUP_PREFETCH_END` | `found_count` | `int` |
