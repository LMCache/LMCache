# Cache identity tags

`ObjectKeyIdentity.tags` holds opaque string dimensions that participate in
cache matching. The mapping is copied, sorted by name, and made immutable so
insertion order and caller mutation cannot change equality or hashing.

Persistent L2 names append a versioned, canonical token:

```text
untagged: <model>@<rank>@<group>@<hash>[@<salt>]
tagged:   <model>@<rank>@<group>@<hash>@<salt>@x1-<token>
```

The token is unpadded base64url of the compact JSON tag map. The explicit empty
salt field prevents an unsalted tagged key from aliasing a legacy salted key.
Existing untagged names remain unchanged. Decoders reject unknown versions,
malformed maps, and noncanonical encodings. Python and native filesystem codecs
preserve the same representation without imposing backend-specific name-length
limits; storage backends report unsupported names through their normal I/O
results.

`EncodedObjectKey` and the internal `ObjectKeyIdentity` protobuf carry the same
mapping. Multiprocess clients select matching tags with string-valued
``lmcache.tag.<name>`` entries in ``IPCCacheServerKey.request_configs``. Object
expansion strips the prefix and preserves the canonical mapping through store,
lookup, and transfer paths. Invalid matching tags reject only their RPC instead
of being dropped or leaving the client waiting for a timeout. Control-plane
behavior belongs in a separate layer.
