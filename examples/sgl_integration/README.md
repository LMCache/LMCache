# SGLang & LMCache Integration

This example shows how to use SGLang & LMCache Integration.

## Install
```bash
pip install --upgrade pip
pip install sglang lmcache
```

## Run

LMCache's MP (multi-process) connector is the default. SGLang dials a
standalone `lmcache server` daemon over ZMQ at the host/port from the YAML.

Create `lmcache_config.yaml` with:
```yaml
# MP mode: SGLang dials the standalone `lmcache server` at this host/port.
mp_host: 127.0.0.1
mp_port: 5556

# Optional vLLM-compatible connector setting. Use `lmcache_driven` or `auto`
# instead when running a hybrid model with multiple KV transfer groups.
extra_config:
  lmcache.mp.mp_transfer_mode: engine_driven
```

When this setting is absent, `LMCACHE_MP_TRANSFER_MODE` may select the mode;
otherwise CUDA defaults to the existing LMCache-driven path.

Terminal 1 — start the LMCache daemon (host/port must match
`mp_host` / `mp_port` in `lmcache_config.yaml`):
```bash
lmcache server --host 127.0.0.1 --port 5556 --l1-size-gb 4 \
  --eviction-policy LRU --supported-transfer-mode engine_driven
```

The client and server must select compatible transfer modes. Engine-driven
transfer currently supports one transfer group, matching the vLLM connector.

Terminal 2 — start SGLang with LMCache enabled:
```bash
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --tp 2 --page-size 32 --enable-lmcache --lmcache-config-file lmcache_config.yaml
```

If you hope to run the benchmark, please refer to `https://github.com/sgl-project/sglang/tree/main/benchmark/hicache`
