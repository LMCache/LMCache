# SPDX-License-Identifier: Apache-2.0
"""Example client for the LMCache KV-cache SDK."""

# Standard
from pathlib import Path
import argparse
import json

# Third Party
import torch

# First Party
import lmcache.sdk as lmc_sdk


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    """Resolve a small set of dtype names used by the example package maker."""
    dtypes = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtypes[dtype_name]


def _tokens(token_start: int, num_tokens: int) -> list[int]:
    """Create the consecutive token IDs used by this example."""
    return list(range(token_start, token_start + num_tokens))


def _make_package(args: argparse.Namespace) -> None:
    """Create a toy KV package that can be passed to ``lmc_sdk.store``."""
    num_tokens = args.chunk_size * args.num_chunks
    tokens = _tokens(args.token_start, num_tokens)
    dtype = _dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    kv = torch.randn(
        (2, args.num_layers, num_tokens, args.hidden_dim),
        dtype=dtype,
    )
    torch.save(
        {
            "kv": kv,
            "model_name": args.model_name,
            "tokens": tokens,
            "cache_salt": args.cache_salt,
        },
        args.output,
    )
    print(f"wrote {args.output} with shape {tuple(kv.shape)} and dtype {kv.dtype}")


def _store(args: argparse.Namespace) -> None:
    """Store a KV package through the SDK."""
    result = lmc_sdk.store(
        args.input,
        args.url,
        cache_salt=args.cache_salt,
        timeout=args.timeout,
    )
    print(json.dumps(result.__dict__, indent=2, default=str))


def _lookup(args: argparse.Namespace) -> None:
    """Look up a cached prefix through the SDK."""
    result = lmc_sdk.lookup(
        args.url,
        model_name=args.model_name,
        tokens=_tokens(args.token_start, args.num_tokens),
        cache_salt=args.cache_salt,
        timeout=args.timeout,
    )
    print(json.dumps(result.__dict__, indent=2))


def _retrieve(args: argparse.Namespace) -> None:
    """Retrieve a cached prefix through the SDK."""
    result = lmc_sdk.retrieve(
        args.output,
        args.url,
        model_name=args.model_name,
        tokens=_tokens(args.token_start, args.num_tokens),
        cache_salt=args.cache_salt,
        timeout=args.timeout,
    )
    print(json.dumps(result.__dict__, indent=2, default=str))


def _add_common_http_args(parser: argparse.ArgumentParser) -> None:
    """Add server URL, cache salt, and timeout arguments."""
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--cache-salt", default="")
    parser.add_argument("--timeout", type=float, default=60.0)


def _add_token_args(parser: argparse.ArgumentParser) -> None:
    """Add consecutive-token range arguments."""
    parser.add_argument("--token-start", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the example script."""
    parser = argparse.ArgumentParser(
        description="Store, look up, and retrieve KV cache packages via lmcache.sdk."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_package = subparsers.add_parser("make-package")
    make_package.add_argument("--output", type=Path, required=True)
    make_package.add_argument("--model-name", required=True)
    make_package.add_argument("--cache-salt", default="")
    make_package.add_argument("--chunk-size", type=int, required=True)
    make_package.add_argument("--num-chunks", type=int, required=True)
    make_package.add_argument("--num-layers", type=int, required=True)
    make_package.add_argument("--hidden-dim", type=int, required=True)
    make_package.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
    )
    make_package.add_argument("--token-start", type=int, default=0)
    make_package.add_argument("--seed", type=int, default=0)

    store = subparsers.add_parser("store")
    _add_common_http_args(store)
    store.add_argument("--input", type=Path, required=True)

    lookup = subparsers.add_parser("lookup")
    _add_common_http_args(lookup)
    _add_token_args(lookup)
    lookup.add_argument("--model-name", required=True)

    retrieve = subparsers.add_parser("retrieve")
    _add_common_http_args(retrieve)
    _add_token_args(retrieve)
    retrieve.add_argument("--model-name", required=True)
    retrieve.add_argument("--output", type=Path, required=True)

    return parser


def main() -> None:
    """Run the selected example command."""
    args = build_parser().parse_args()
    if args.command == "make-package":
        _make_package(args)
    elif args.command == "store":
        _store(args)
    elif args.command == "lookup":
        _lookup(args)
    elif args.command == "retrieve":
        _retrieve(args)
    else:
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
