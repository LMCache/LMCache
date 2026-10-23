# Mixed-length multi-turn KV benchmark

Run from the checkout root with LMCache's CLI dependencies and the serving
model's tokenizer installed:

```bash
PYTHONPATH=. python examples/benchmarks/multi_turn_kv/benchmark.py --help
```

This opt-in example uses the engine-benchmark library without changing
existing `lmcache bench engine` workloads or any serving policy. It targets
vLLM token-ID completions (protocol checked against v0.27.1).

See the [run guide and comparison matrix](../../../docs/source/cli/multi_turn_kv.rst)
for server setup, cold/continuation measurements, actual L2-load checks,
report interpretation, and measurements requiring separate instrumentation.
This runner generates synthetic token continuations; it does not evaluate
the semantic quality of a chatbot's answers.
