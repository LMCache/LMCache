# lmcache-nixl

LMCache's native NIXL connector, used by the `nixl_native` L2 adapter. It
installs the `lmcache_nixl` module and depends on the matching `nixl` release,
which provides `libnixl` and its plugins.

```bash
pip install "lmcache==<version>" "lmcache-nixl==<version>"
```

Install the same version as `lmcache`. See the
[nixl_native documentation](https://docs.lmcache.ai/mp/l2_storage/nixl_native.html).
