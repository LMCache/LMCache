# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import asyncio

# First Party
from lmcache.v1.check import registry

model_name = "/lmcache_test_model/"


def parse_args():
    parser = argparse.ArgumentParser(description="LMCache basic check Tool")
    parser.add_argument(
        "--mode",
        required=True,
        help="Operation mode (e.g. test_remote, test_storage_manager). "
        "Use 'list' to show available modes",
    )
    parser.add_argument("--model", default=model_name, help="model name")
    parser.add_argument(
        "--num-keys",
        type=int,
        default=100,
        help="Number of keys to generate (gen mode only)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Concurrency level for generation (gen mode only)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset for key generation (gen mode only)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # List available modes if requested
    if args.mode == "list":
        registry.load_modes()
        print("Available check modes:")
        for mode_name in registry.modes:
            print(f"  - {mode_name}")
        return

    # Get the requested mode function
    mode_func = registry.get_mode(args.mode)
    if not mode_func:
        print(
            f"Error: Unknown mode '{args.mode}'. "
            "Use '--mode list' to see available modes."
        )
        return

    # Prepare arguments for the mode function
    mode_args = {
        "model": args.model,
        "num_keys": args.num_keys,
        "concurrency": args.concurrency,
        "offset": args.offset,
    }

    # Execute the mode function
    await mode_func(**mode_args)


if __name__ == "__main__":
    asyncio.run(main())
