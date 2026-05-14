# SPDX-License-Identifier: Apache-2.0
"""Example in-memory client for the LMCache KV-cache SDK."""

# Standard
import argparse
import json

# Third Party
import torch

# First Party
import lmcache.sdk as lmc_sdk


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Resolve a small set of dtype names used by the example tensor maker."""
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtypes[dtype_name]


def _tokens(token_start: int, num_tokens: int) -> list[int]:
    """Create the consecutive token IDs used by this example."""
    return list(range(token_start, token_start + num_tokens))


def _make_kv_tensor(args: argparse.Namespace) -> torch.Tensor:
    """Create a toy in-memory KV tensor for SDK calls."""
    num_tokens = args.chunk_size * args.num_chunks
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    return torch.randn(
        (2, args.num_layers, num_tokens, args.hidden_dim),
        dtype=dtype,
    )


def _store_generated(args: argparse.Namespace) -> None:
    """Generate a KV tensor in memory and store it through the SDK."""
    num_tokens = args.chunk_size * args.num_chunks
    result = lmc_sdk.store(
        _make_kv_tensor(args),
        args.url,
        model_name=args.model_name,
        tokens=_tokens(args.token_start, num_tokens),
        cache_salt=args.cache_salt,
        timeout=args.timeout,
    )
    print(json.dumps(result.__dict__, indent=2, default=str))


def _retrieve(args: argparse.Namespace) -> None:
    """Retrieve a cached prefix into memory through the SDK."""
    result = lmc_sdk.retrieve(
        args.url,
        model_name=args.model_name,
        tokens=_tokens(args.token_start, args.num_tokens),
        cache_salt=args.cache_salt,
        timeout=args.timeout,
    )
    if result is None:
        print(json.dumps({"hit_tokens": 0, "kv": None}, indent=2))
        return
    output = {
        "hit_tokens": int(result.shape[2]),
        "kv_shape": tuple(result.shape),
        "kv_dtype": str(result.dtype),
    }
    print(json.dumps(output, indent=2, default=str))


def _add_common_http_args(parser: argparse.ArgumentParser) -> None:
    """Add server URL, cache salt, and timeout arguments."""
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--cache-salt", default="")
    parser.add_argument("--timeout", type=float, default=60.0)


def _add_token_args(parser: argparse.ArgumentParser) -> None:
    """Add consecutive-token range arguments."""
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, required=True)


def _add_tensor_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for constructing a toy KV tensor."""
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the example script."""
    parser = argparse.ArgumentParser(
        description="Store and retrieve KV cache tensors via lmcache.sdk."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    store_generated = subparsers.add_parser("store-generated")
    _add_common_http_args(store_generated)
    _add_tensor_args(store_generated)

    retrieve = subparsers.add_parser("retrieve")
    _add_common_http_args(retrieve)
    _add_token_args(retrieve)
    retrieve.add_argument("--model-name", required=True)

    return parser


def main() -> None:
    """Run the selected example command."""
    args = build_parser().parse_args()
    if args.command == "store-generated":
        _store_generated(args)
    elif args.command == "retrieve":
        _retrieve(args)
    else:
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
