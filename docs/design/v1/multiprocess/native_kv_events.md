# Native KV-event preparation

Native descriptors preserve the engine's block identities, parent identity,
tokens, block size and immutable profile. `kv_event_types.py` owns these values;
`custom_types.py` exposes the STORE metadata types through its existing API.

`IPCCacheServerKey` carries optional descriptors without changing storage-key
equality or hashing. MessagePack and gRPC retain bytes, signed integers and
unsigned integers without converting one identity representation into another.

This prerequisite does not change the advertised capability, legacy event record
or subscription signature. The native schema and its consumers are activated
together in a later change; ordinary event and cache behavior stays available.
