# Disaggregated Prefill Setup

## 0. Prerequisites

- Ensure LMCache/NIXL installed

**Tips:** You can use `lmcache/vllm-openai` docker images with dependencies installed then activate the uv env `source /opt/venv/bin/activate` (maybe you need to `uv pip install datasets pandas` additionally)

```bash
# MODEL="Qwen/Qwen2.5-0.5B"
MODEL="/mnt/models/Qwen-2.5-0.5b"
```

---

## 1. xPyD Case (Multiple Prefillers, Multiple Decoders)

### 1.1 (Option 1) Single Machine Multi-GPU Setup (localhost)

```bash
# Use the provided example script for localhost setup
bash disagg_example_xpyd.sh
```

### 1.2 (Option 2) Multi-Host Setup

**Fill your Proxy Node IP here:**

```bash
export YOUR_PROXY_IP_ADDR="......"
configfile=configs/lmcache-prefiller-xpyd-config.yaml
sed -i "s/^nixl_proxy_host: .*/nixl_proxy_host: \"$YOUR_PROXY_IP_ADDR\"/" ${configfile}
```

#### Prefiller Nodes (host1/host2)

```bash
# Launch prefiller on both host1 and host2 with same command:
# the 0 for `lmcache_rpc_port` can be any value.

bash vllm_launcher.sh prefiller 0 xpyd $MODEL
```

#### Decoder Nodes (host3/host4)

```bash
# Launch decoder on host3 and host4 (for ftoken_from_p proxy) with same command:
bash vllm_launcher.sh decoder 0 xpyd $MODEL ftoken_from_p

# Tips:
# Change parameter from `ftoken_from_p` to `ftoken_from_d`,
# to switch the proxy mode from `first-token-from-prefiller` to
# `first-token-from-decoder` (should align with below `python3 disagg_proxy_server*.py`)
```

#### Proxy Server

```bash
# The host1/host2 are IP Address of Prefiller nodes
# The host3/host4 are IP Address of Decoder nodes

python3 disagg_proxy_server_first_token_from_prefiller_xpyd.py \
    --prefiller-hosts host1,host2 \
    --decoder-hosts host3,host4
```

### 1.3 API Access

Now you can reach to proxy node port `8000` for `/v1/completions` or `/v1/chat/completions` API

NOTE: `v1/chat/completions` is not supported for `proxy*_first_token_from_prefiller_*.py` so far.

---

## 2. Xp1D Case (Multiple Prefillers, Single Decoder)

#### Prefiller Nodes (host1/host2)

```bash
# Get your Decoder IP --> example: YOUR_DECODER_IP_ADDR=192.168.1.41
# and fill into the Prefill LMCache config file
export YOUR_DECODER_IP_ADDR=.....
configfile="configs/lmcache-prefiller-xp1d-config.yaml"
sed -i "s/^nixl_receiver_host: .*/nixl_receiver_host: \"$YOUR_DECODER_IP_ADDR\"/" ${configfile}
```

```bash
# Launch prefillers using the script in host1 and host2
bash vllm_launcher.sh prefiller 0 xp1d $MODEL
```

#### Decoder Node (host3)

```bash
# Launch single decoder in host3 (for ftoken_from_p proxy)
bash vllm_launcher.sh decoder 0 xp1d $MODEL ftoken_from_p
```

#### Proxy Server

```bash
# The host1/host2 are IP Address of Prefiller nodes
# The host3 is IP Address of Decoder node
python3 disagg_proxy_server_first_token_from_prefiller_xpyd.py \
    --prefiller-hosts host1,host2 \
    --decoder-hosts host3
```

### 2.1 API Access

Now you can reach to port 8000 for `/v1/chat/completions` or `/v1/chat` API

#### Expected Result

- You can find log from decoder like `LMCache INFO: Reqid: cmpl-xxxxx, Total tokens 36, LMCache hit tokens: 35`
- You can also find log from prefiller like `LMCache INFO: Stored 35 out of total 35 tokens. size: 0.0037 gb, cost 427.9820 ms, throughput: 0.0087 GB/s; offload_time: 0.7341 ms, put_time: 427.2479 ms`
- The prefiller/decoder will be round-robin scheduled among requests.

---

## 3. Notes

### Port Assignment

Controlled by `localhost_mode` parameter (true for single machine, false for multi-host)

#### Single Machine Multi-GPU Mode (localhost_mode=true)

- **Prefillers**: 7100 + (host_id - 1) → host1: 7100, host2: 7101, etc.
- **Decoders**: 7200 + (host_id - 1) → host1: 7200, host2: 7201, etc.

#### Multi-Host Mode (localhost_mode=false, default)

- **Prefillers**: All use port 7100 (different machines)
- **Decoders**: All use port 7200 (different machines)

### Setup Guidelines

- **Single Machine Setup**: Use `disagg_example_xpyd.sh` for localhost multi-GPU setup with automatic CUDA device assignment
- **Multi-Host Setup**: Use individual launch commands without localhost_mode parameter (defaults to false)

### Important Configuration Notes

- The parameter `"lmcache_rpc_port"` is an ID for IPC (part of `socket_path` for ZMQ RPC path), so the value should be mandatory to be different among vllm instance, in single machine use case (but optional for multiple node use case, since vLLM instances do not communicate via IPC)

- For xp1D case, make sure to update the decoder IP address (`nixl_receiver_host`) in the prefiller lmcache config file `lmcache-prefiller-xp1d-config.yaml`

- The `skip_last_n_tokens: 1` parameter is automatically set for decoder nodes when using `ftoken_from_p` (first_token_from_prefiller) proxy code. Otherwise, the prefix caching of decoder vLLM instance will treat the N-1 prompt as a new input and will not reuse the KV-Cache

### Memory Management

- When you are using small GPU partition HBM <=5G, you may see OOM because Qwen2-0.5B requires >=4G HBM, meanwhile, Nixl Buffer requires 1.5~3G HBM for prefiller (according to its *config.yaml) and more for decoder. So if you are using small GPU partition, you can add below argument to VLLM launch command line, to retain enough HBM for Nixl buffer to avoid OOM like `--gpu-memory-utilization 0.6` (it means leaving 40% HBM for nixl buffer and others)

- In `nixl_buffer_size` should be multiple of `align_bytes` (`num_elements * bytes_per_element`), it varies between different models. (`num_elements=[batch_size, num_kv_heads, seq_len, head_dim]`, `bytes_per_element` depends on torch_dtype), adjust the value according to the error log message if any

### Compatibility Notes

- It's not compatible to launch prefiller/decoder in xPyD mode, and start proxy with single `--decoder` host (Prefiller will show error `AttributeError: 'NoneType' object has no attribute 'receiver_info'`)
