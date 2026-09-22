# Linker verification

- Verify the SGLang companion interfaces before importing the plugin. Released
  wheels may not contain them; use the pinned checkout on `PYTHONPATH`.
- A local radix hit is not evidence of LMCache restore. Run `verify.py`, require
  a cold miss and then `host > 0`, `device == 0` after `/flush_cache`, and compare
  output tokens and log probabilities.
- Compare restored log probabilities to a native local radix hit with the same
  prefix length. Cold prefill has a different compute shape and can differ even
  with native caching (observed at TP=2). Report its difference separately;
  do not weaken transfer checks by widening tolerances to hide this distinction.
- Keep control-peer unit tests, real MP byte-transfer tests, and model-level
  tests separate in reports. State TP size and model revision explicitly.
- Before building native code, verify `git`, compiler, CUDA headers, and torch.
  On H200, use `TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=1` unless broader code generation
  is required. A CUDA IPC test needs a shared IPC namespace and GPU visibility.
- Regenerate ignored protobuf Python files after changing the schema. Match
  `grpcio-tools` to the installed protobuf runtime using the repository's
  `requirements/proto.txt` pins. A protobuf 7 generator with a protobuf 6
  runtime fails at import in the CUDA 13 / torch 2.13 validation image.
- Start the MP server with explicit `--chunk-size 1 --eviction-policy LRU`.
  One logical chunk represents a full SGLang page or checkpoint.
