# SGLang with the LMCache MP linker

See [the setup and ownership guide](../../../docs/source/mp/sglang_linker.rst)
for the required SGLang interfaces, service commands, configuration, and limits.
Use the exact source revisions and container prerequisites recorded in the
associated draft PR when reproducing its validation results.

After starting both services, run from the LMCache repository root:

```bash
python examples/sglang/kv_linker/verify.py --base-url http://127.0.0.1:30000
```

This checks a cold miss, a native local radix hit, and an external restore after
`/flush_cache`. The restore must have positive external (`host`) hits, zero local
(`device`) hits, the same prefix length as the native hit, identical greedy
output tokens for all three requests, and log-probability differences below
0.05 relative to the native hit. Cold-prefill numerical differences are reported
separately. A fresh salt prevents previous runs from making the first request hit.

For two-GPU tensor parallelism, launch the same SGLang command with `--tp 2`.
For a hybrid full-attention/Gated DeltaNet model, use
`Qwen/Qwen3.5-0.8B` at revision
`2fc06364715b967f1860aea9cf38778875588b17` with `--tp 1` and
`--mem-fraction-static 0.2`. Keep a dedicated MP server with `--chunk-size 1`.
