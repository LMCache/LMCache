# SGLang & LMCache Integration

This example shows how to use SGLang & LMCache Integration.

## Install
Install sglang through `pip`.
```bash
pip install sglang
```

## Server script
To start SGLang server with LMCache, run
```bash
export LMCACHE_CONFIG_FILE=lmcache_config.yaml
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --tp 2 --page-size 32 --enable-lmcache
```
If you hope to run the benchmark, please refer to `https://github.com/sgl-project/sglang/tree/main/benchmark/hicache`

