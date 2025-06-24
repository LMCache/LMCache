# SGLang & LMCache Integration

This example shows how to use SGLang & LMCache Integration.

## Install
This project depends on a pending pull request in the SGLang repository. Until PR is merged, please use the code from that specific branch instead of the SGLang main branch.
```bash
git clone https://github.com/Oasis-Git/sglang/tree/lmcache
cd sglang

pip install --upgrade pip
pip install -e "python[all]"
```

## Server script
To start SGLang server with LMCache, run
```bash
export LMCACHE_USE_EXPERIMENTAL=True
export LMCACHE_CONFIG_FILE=lmcache_config.yaml
python -m sglang.launch_server --model-path Qwen/Qwen2.5-14B-Instruct --port 30000 --tp 2 --page-size 32 --enable-lmcache-connector
```
If you hope to run the benchmark, please refer to `https://github.com/Oasis-Git/sglang/tree/lmcache/benchmark/benchmark_lmcache`

