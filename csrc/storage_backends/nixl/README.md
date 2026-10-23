# lmcache-nixl

LMCache's native NIXL connector, used by the `nixl_native` L2 adapter. It
installs the `lmcache_nixl` module and depends on `nixl`, which provides
`libnixl` and its plugins. Install it through LMCache's `nixl` extra, which
pins it to the same version as `lmcache`:

```bash
pip install "lmcache[nixl]"
```

See the
[nixl_native documentation](https://docs.lmcache.ai/mp/l2_storage/nixl_native.html).
