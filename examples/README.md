## Configs in example.yaml
```
chunk_size: an integer (e.g., 256)
local_device: "cuda", "cpu" or "an arbitrary path (local disk)"
remote_url: "remote shared cache server url" (e.g., "lm://localhost:65432")
remote_serde: "torch", "safetensor" or "cachegen"
piplined_backend: True/False
```