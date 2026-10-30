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
preserve the same representation; tagged filenames must fit within the
255-byte filename limit.

`EncodedObjectKey` and the internal `ObjectKeyIdentity` protobuf carry the same
mapping. Multiprocess clients select matching tags with string-valued
``lmcache.tag.<name>`` entries in ``IPCCacheServerKey.request_configs``. Object
expansion strips the prefix and preserves the canonical mapping through store,
lookup, and transfer paths. Control-plane behavior belongs in a separate layer.
