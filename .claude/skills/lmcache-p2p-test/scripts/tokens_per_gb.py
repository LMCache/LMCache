#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compute ``--tokens-per-gb-kvcache`` for ``lmcache bench engine`` from config.json.

KV bytes per token = layers x kv_heads x head_dim x 2 (K and V) x dtype bytes. Models
whose attention is not plain GQA/MHA (MLA, hybrid/linear layers, sliding windows) need
an explicit value; pass ``--override``.

usage: tokens_per_gb.py <hf-model-id or path to config.json> [--hf-home DIR]
       [--override N]
prints the integer tokens per GB (base 10, as the bench expects).
"""

# Standard
import argparse
import glob
import json
import os
import sys

DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "fp8": 1,
}


def kv_bytes_per_token(cfg: dict) -> int:
    """Return bytes of K+V per token for a plain-attention transformer config."""
    layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads") or cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"]
    dtype = DTYPE_BYTES.get(str(cfg.get("torch_dtype", "bfloat16")).lower(), 2)
    return layers * kv_heads * head_dim * 2 * dtype


def tokens_per_gb(cfg: dict) -> int:
    """Tokens that fit in one GB (1e9 bytes) of KV cache."""
    return int(1e9 // kv_bytes_per_token(cfg))


def find_config(model: str, hf_home: str) -> str:
    """Resolve an HF id (via the local cache) or a path to its config.json."""
    if os.path.isfile(model):
        return model
    if os.path.isdir(model):
        return os.path.join(model, "config.json")
    hits = glob.glob(
        os.path.join(
            hf_home,
            "hub",
            f"models--{model.replace('/', '--')}",
            "snapshots",
            "*",
            "config.json",
        )
    )
    if not hits:
        raise FileNotFoundError(f"no cached config.json for {model} under {hf_home}")
    return hits[0]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("model")
    ap.add_argument(
        "--hf-home",
        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
    )
    ap.add_argument("--override", type=int)
    a = ap.parse_args()
    if a.override:
        print(a.override)
        return 0
    cfg = json.load(open(find_config(a.model, a.hf_home)))
    if (
        "kv_lora_rank" in cfg
        or cfg.get("layer_types")
        or "linear_attn" in json.dumps(cfg)
    ):
        print(
            f"warning: {a.model} is not plain GQA attention; "
            "the value below assumes it is (use --override)",
            file=sys.stderr,
        )
    print(tokens_per_gb(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
