# Modular Protocol System

This directory contains the modular protocol definitions for the LMCache
multiprocess gRPC transport.

## Directory Structure

```
protocols/
├── README.md           # This file
├── __init__.py         # Collects module definitions and validates them against the proto service
├── base.py             # Common types (HandlerType, ProtocolDefinition, request_name_to_method_name)
├── engine.py           # Engine operations (REGISTER/UNREGISTER, STORE, RETRIEVE, LOOKUP, PREPARE/COMMIT, ...)
├── controller.py       # Controller operations (CLEAR, GET_CHUNK_SIZE, PING)
├── debug.py            # Debug operations (NOOP)
├── blend.py            # CacheBlend v1 operations (CB_STORE/RETRIEVE_PRE_COMPUTED, CB_STORE_FINAL, ...)
├── blend_v2.py         # CacheBlend v2 lookup/retrieve (CB_*_V2)
├── blend_v3.py         # CacheBlend v3 rope + unified lookup (CB_*_V3, CB_UNIFIED_LOOKUP)
├── observability.py    # Observability events (REPORT_BLOCK_ALLOCATION)
└── p2p.py              # Peer-to-peer transfers (P2P_LOOKUP_AND_LOCK, P2P_QUERY_LOOKUP_RESULTS, P2P_UNLOCK_OBJECTS)
```

## Design Overview

The protocol system has two sources of truth:

1. **Python protocol definitions** in this directory.
   Each module declares:
   - `REQUEST_NAMES`: the historical ALL_CAPS operation names used across the
     Python codebase
   - `get_protocol_definitions()`: a dict of request name to `ProtocolDefinition`

2. **The gRPC service descriptor** in
   `transport/grpc_impl/proto/lmcache_mq.proto`.
   It defines the concrete request/response protobuf messages and the unary RPC
   methods exposed by the Engine, Controller, Debug, Blend, Observability, and
   P2P services.

At import time:

1. `protocols.__init__.py` collects all module definitions.
2. It validates that every `REQUEST_NAMES` entry has a definition, that no
   request name is duplicated across modules, and that every definition has a
   matching gRPC service method.
3. `protocol.py` reads the generated gRPC descriptor and builds:
   - `RpcMethod`: descriptor-derived string tokens, including their owning
     gRPC `service_name`
   - `RPC`: a CamelCase namespace for ergonomic attribute access

There is no central request enum anymore. Adding an RPC no longer requires
editing a separate registry or batch transport layer.

## Adding New Protocols

To add a new multiprocess RPC, update both the Python definition and the proto
service.

### Option 1: Add to an Existing Module

If the new operation belongs in an existing category:

1. **Add the request name** to that module's `REQUEST_NAMES` list:
   ```python
   REQUEST_NAMES = [
       "EXISTING_OP",
       "YOUR_NEW_OP",
   ]
   ```

2. **Add the protocol definition**:
   ```python
   def get_protocol_definitions() -> dict[str, ProtocolDefinition]:
       return {
           "YOUR_NEW_OP": ProtocolDefinition(
               payload_classes=[int, str],
               response_class=bool,
               handler_type=HandlerType.SYNC,
           ),
       }
   ```

3. **Add the protobuf request/response messages and unary RPC** to the matching
   service in
   `transport/grpc_impl/proto/lmcache_mq.proto`:
   ```proto
   message YourNewOpRequest {
     int64 first = 1;
     string second = 2;
   }

   message YourNewOpResponse {
     bool ok = 1;
   }

   service EngineService {
     rpc YourNewOp(YourNewOpRequest) returns (YourNewOpResponse);
   }
   ```

4. **Regenerate the stubs**:
   ```bash
   pip install -r requirements/proto.txt
   python -m lmcache.v1.multiprocess.transport.grpc_impl._proto_gen._generate
   ```

### Option 2: Create a New Protocol Module

If the operation belongs in a new category:

1. **Create a new module** (for example `monitoring.py`) with
   `REQUEST_NAMES` and `get_protocol_definitions()`.

2. **Register the module** in `protocols/__init__.py` by importing it and
   appending it to `_PROTOCOL_MODULES`.

3. **Add the protobuf messages and a new gRPC service** to `lmcache_mq.proto`,
   or add the RPC to the existing service that owns that category.

4. **Regenerate the stubs** with `_proto_gen._generate`.

After that, the operation is available as both:

- `RPC.YourNewOp`
- `RpcMethod.YOUR_NEW_OP`

`coerce_rpc_method()` also accepts the string forms `"YourNewOp"` and
`"YOUR_NEW_OP"` when compatibility glue needs them.

## Using the Protocol System

```python
from lmcache.v1.multiprocess.protocol import (
    RPC,
    HandlerType,
    get_payload_classes,
    get_response_class,
    get_handler_type,
)

rpc_method = RPC.Store

payloads = get_payload_classes(rpc_method)
response = get_response_class(rpc_method)
handler = get_handler_type(rpc_method)
```

## Validation

The initialization system validates at startup:

1. Every `REQUEST_NAMES` entry has a matching `ProtocolDefinition`
2. No request name is defined in multiple modules
3. Every Python definition has a matching gRPC method
4. Every gRPC method has a matching Python definition

If validation fails, `ProtocolInitializationError` raises a message that points
to the mismatch.

### Example Error Messages

```python
# If you list a request name but forget the definition:
ProtocolInitializationError: Request name 'YOUR_NEW_OP' in module 'engine' is listed in REQUEST_NAMES but has no protocol definition

# If you add a definition but forget the proto rpc:
ProtocolInitializationError: gRPC services / protocol definition mismatch: missing_methods=['YourNewOp'], missing_definitions=[]
```

## Current Protocol Groups

The authoritative list of request names is `REQUEST_NAMES` in each module; the
list below is a snapshot as of this file's last update.

### Engine Operations (`engine.py`)
Core KV cache operations and their split-phase variants:
- `REGISTER_KV_CACHE` / `UNREGISTER_KV_CACHE`: Register / unregister the GPU KV cache
- `REGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT` / `UNREGISTER_KV_CACHE_ENGINE_DRIVEN_CONTEXT`: Register / unregister the non-GPU (engine-driven) KV cache context
- `STORE` / `RETRIEVE`: Fused store / retrieve
- `PREPARE_STORE` / `COMMIT_STORE` / `PREPARE_RETRIEVE` / `COMMIT_RETRIEVE`: Split-phase store / retrieve (used by the engine-driven path)
- `LOOKUP`: Submit a prefix lookup and return a prefetch job id
- `QUERY_PREFETCH_STATUS` / `WAIT_PREFETCH_STATUS`: Poll / block for a prefetch job's result
- `QUERY_PREFETCH_LOOKUP_HITS` / `FREE_LOOKUP_LOCKS`: Inspect and release the read locks a lookup took on cached chunks
- `END_SESSION`: End a session and clean up associated resources

### Controller Operations (`controller.py`)
Cache management and configuration:
- `CLEAR`: Clear all caches in the server
- `GET_CHUNK_SIZE`: Get the chunk size configuration
- `PING`: Liveness / worker probe (payload: sender's worker instance id or `None`)

### Debug Operations (`debug.py`)
Testing and monitoring:
- `NOOP`: No-operation command for testing / heartbeat

### CacheBlend v1 (`blend.py`)
First-generation CacheBlend operations (pre-computed lookup / retrieve, final store, and blend-scoped KV cache registration):
- `CB_LOOKUP_PRE_COMPUTED`
- `CB_STORE_PRE_COMPUTED`
- `CB_RETRIEVE_PRE_COMPUTED`
- `CB_STORE_FINAL`
- `CB_REGISTER_KV_CACHE`
- `CB_UNREGISTER_KV_CACHE`

### CacheBlend v2 (`blend_v2.py`)
Second-generation CacheBlend lookup + retrieve:
- `CB_LOOKUP_PRE_COMPUTED_V2`
- `CB_RETRIEVE_PRE_COMPUTED_V2`

### CacheBlend v3 (`blend_v3.py`)
Third-generation CacheBlend with rope state and unified lookup:
- `CB_REGISTER_ROPE_V3` / `CB_UNREGISTER_ROPE_V3`
- `CB_RETRIEVE_PRE_COMPUTED_V3`
- `CB_UNIFIED_LOOKUP`

### Observability (`observability.py`)
Server-side observability events:
- `REPORT_BLOCK_ALLOCATION`

### P2P (`p2p.py`)
Peer-to-peer transfer operations:
- `P2P_LOOKUP_AND_LOCK`
- `P2P_QUERY_LOOKUP_RESULTS`
- `P2P_UNLOCK_OBJECTS`

## Handler Types

- `HandlerType.SYNC`: Fast operations that run directly in the main loop
- `HandlerType.BLOCKING`: Operations that may block, run in a thread pool
- `HandlerType.NON_BLOCKING`: Not yet supported (for future async handlers)
