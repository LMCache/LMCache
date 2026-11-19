# Shared L2 key codec

`lmcache.v1.distributed.key_codec` centralizes storage formatting, filesystem
filenames, hash sharding and recovery. Adapters retain their existing public
interfaces and delegate naming to the shared functions.

The generic storage name is `model@rank@group@hash[@salt]`. Rank is eight
hexadecimal digits. Python/native FS use a `0x` rank prefix, replace model
slashes with `-SEP-` and append `.data`. Raw-block uses URL-escaped model names.
Bigtable retains its configured template and layer suffix. Dynamic NIXL retains
its underscore-separated `.bin` name and two-level hash shard.

Centralization preserves each backend's existing untagged bytes, with and
without salt; no persistent-data migration is needed. Encoding ignores mutable
operational metadata. The identity model itself has no backend formatting
methods.
