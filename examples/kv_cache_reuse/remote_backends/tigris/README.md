## LMCache can use [Tigris](https://www.tigrisdata.com/) as a backend storage.

Tigris is a globally distributed, S3-compatible object storage service with no egress fees. The free tier offers 5 GB of storage, sufficient for evaluating LMCache's remote backend behavior end-to-end.

Tigris uses a single global endpoint (`t3.storage.dev`) and serves requests from the region closest to the caller, so no availability-zone colocation is required between your inference host and the bucket.

## Step 1: Create a Tigris bucket and access key

1. Sign up at the [Tigris Console](https://console.storage.dev/) — free, no credit card required.
2. Create a bucket and note the bucket name.
3. From the sidebar, open **Access Keys** and select **Create New Access Key**. Note the **Access Key ID** (prefixed with `tid_`) and **Secret Access Key** (prefixed with `tsec_`); the secret is only displayed once.

See [Tigris's Manage Access Keys documentation](https://www.tigrisdata.com/docs/iam/manage-access-key/) for details.

## Step 2: Fill out `example.yaml`

Replace `{BUCKET_NAME}`, `{TIGRIS_ACCESS_KEY_ID}`, and `{TIGRIS_SECRET_ACCESS_KEY}` in `example.yaml` with the values from Step 1.

## Step 3: Start a vLLM engine with LMCache

```bash
PYTHONHASHSEED=0 LMCACHE_CONFIG_FILE=example.yaml vllm serve meta-llama/Llama-3.1-8B-Instruct --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' --disable-log-requests --no-enable-prefix-caching
```

## Step 4: Sending requests

You should be able to verify a cache hit on the second request by checking the vLLM/LMCache console logs (which will show cache hit messages) or by observing a significantly lower Time-to-First-Token (TTFT):

```bash
curl -X POST http://localhost:8000/v1/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "'"$(printf 'Elaborate the significance of KV cache in language models. %.0s' {1..1000})"'",
    "max_tokens": 10
  }'
```
