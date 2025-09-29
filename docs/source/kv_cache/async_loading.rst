# Async Loading

Storing KV Cache to a backend is always asynchronous while loading KV Cache into paged CPU RAM for transfer back to the GPU is synchronous by default.

In August 2025, LMCache integrated an interface for vLLM to asynchronously load KV Caches for multiple requests in a batch at once and to overlap loading with forward pass. https://github.com/vllm-project/vllm/pull/23620

Image: 

_static/async_loading.png


Some extra protections exist such as protecting the local CPU buffer against race conditions