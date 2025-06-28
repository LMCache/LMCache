# Examples of enabling/disabling `save_decode_cache` in config
This example tests the `save_decode_cache` configuration option in LMCache to verify it correctly saves decode phase KV cache tokens.

The test compares behavior with `save_decode_cache: True` vs `save_decode_cache: False` by analyzing cache store operations and performance patterns.