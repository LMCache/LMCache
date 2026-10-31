# fp8lin Serde End-to-End Example

The L2 disk adapter stores KV chunks with the **fp8lin** serde: E4M3 codes,
losslessly entropy coded on the GPU against a fitted linear prediction
(`lmcache/v1/fp8lin`). About 1.7–1.9× smaller than the `fp8` serde.

## Run

```bash
# bf16 KV cache (fp8lin rounds to FP8 first, like the fp8 serde)
./run_serde_fp8lin_example.sh
# FP8 KV cache: fp8lin is lossless, so the cold and warm responses must match
KV_CACHE_DTYPE=fp8 ./run_serde_fp8lin_example.sh
# if FlashInfer cannot JIT its FP8-KV kernels on your image, pick another backend
KV_CACHE_DTYPE=fp8 ATTENTION_BACKEND=TRITON_ATTN ./run_serde_fp8lin_example.sh
# the same flow with the fp8 serde, for the size comparison
SERDE=fp8 ./run_serde_fp8lin_example.sh
```

The script fits a coder for the model first if `CODER_PATH` has none (about 3 minutes
for Qwen3-1.7B on one GPU), then:

1. starts `lmcache server` with a disk L2 adapter using the fp8lin serde,
2. starts vLLM with `LMCacheMPConnector`,
3. sends a ~4k-token request (KV goes to L1, then is coded and written to disk),
4. clears L1 and re-sends it (L2 prefetch, fp8lin decode, vLLM resumes from cache),
5. prints the bytes stored per token and whether both responses match.

Measured on an RTX 4500 Ada (vLLM 0.32, Qwen3-1.7B, 6,364-token prompt): `fp8`
stores 57,344 bytes per token, `fp8lin` 33,353 (FP8 KV, warm response identical)
or 33,610 (bf16 KV). See `lmcache/v1/fp8lin/README.md` for prefetch times.

## Quick check without vLLM

```bash
pytest tests/v1/distributed/serde/test_fp8lin.py -xvs
```

## Requirements

vLLM, the `lmcache` CLI, one CUDA GPU with Triton, and a model with `head_dim` 128
(Qwen3, Llama-3 8B+, Mistral).
