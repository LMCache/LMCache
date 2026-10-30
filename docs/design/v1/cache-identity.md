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
mapping. This layer defines and persists tagged identities but does not choose
tags for requests; request propagation and control-plane behavior belong in
separate changes.
