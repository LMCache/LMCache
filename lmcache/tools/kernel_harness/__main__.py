# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import logging
import os
import sys

# Local
from .benchmark import print_benchmark_table, run_benchmark
from .config import filter_configs, get_all_test_configs
from .correctness import run_correctness_test, run_skip_prefix_test
from .reference import reference_multi_layer_block_kv_transfer

logger = logging.getLogger(__name__)

# Try to import the CUDA kernel module.
# The .so is built in-place in the kernel_harness directory.
_harness_dir = os.path.dirname(os.path.abspath(__file__))
if _harness_dir not in sys.path:
    sys.path.insert(0, _harness_dir)

try:
    # Third Party
    import kernel_harness_ops

    HAS_KERNEL = True
except ImportError:
    HAS_KERNEL = False


def _get_kernel_fn(use_reference: bool):
    """Get the kernel function to use for testing."""
    if use_reference:
        return reference_multi_layer_block_kv_transfer

    if not HAS_KERNEL:
        print(
            "WARNING: kernel_harness_ops not built. "
            "Falling back to Python reference.\n"
            "Build with: cd lmcache/tools/kernel_harness && "
            "python setup.py build_ext --inplace"
        )
        return reference_multi_layer_block_kv_transfer

    # Wrap the C++ kernel to match the Python reference signature
    def cuda_kernel_wrapper(vllm_tensors, memory_objects, block_ids, config, direction):
        # Local
        from .config import Direction

        cpp_direction = (
            kernel_harness_ops.TransferDirection.H2D
            if direction == Direction.H2D
            else kernel_harness_ops.TransferDirection.D2H
        )

        # Map VLLMBufferFormat to GPUKVFormat
        # Local
        from .config import VLLMBufferFormat

        format_map = {
            VLLMBufferFormat.NORMAL: kernel_harness_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # noqa: E501
            VLLMBufferFormat.CROSS_LAYER: kernel_harness_ops.GPUKVFormat.NB_NL_TWO_BS_NH_HS,  # noqa: E501
            VLLMBufferFormat.MLA: kernel_harness_ops.GPUKVFormat.NL_X_NB_BS_HS,
        }
        gpu_kv_format = format_map[config.vllm_format]

        # Third Party
        import torch

        device = torch.device("cuda")
        kernel_harness_ops.multi_layer_block_kv_transfer(
            vllm_tensors,
            memory_objects,
            block_ids,
            device,
            cpp_direction,
            gpu_kv_format,
            config.block_size,
            config.num_blocks,
            config.skip_prefix_n_blocks,
        )

    return cuda_kernel_wrapper


def main():
    parser = argparse.ArgumentParser(
        description="LMCache Block Transfer Kernel Test Harness"
    )
    parser.add_argument(
        "--mode",
        choices=["correctness", "benchmark", "all"],
        default="all",
        help="What to run (default: all)",
    )
    parser.add_argument(
        "--use-reference",
        action="store_true",
        help="Use Python reference implementation instead of CUDA kernel",
    )
    parser.add_argument(
        "--format",
        choices=["normal", "cross_layer", "mla", "all"],
        default="all",
        help="Which vLLM format to test (default: all)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp8", "all"],
        default="all",
        help="Which dtype to test (default: all)",
    )
    parser.add_argument(
        "--num-bench-iters",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--num-warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--skip-prefix-n-blocks",
        type=int,
        default=0,
        help="Number of prefix blocks to skip (default: 0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s: %(message)s",
    )

    configs = get_all_test_configs(
        num_warmup=args.num_warmup_iters,
        num_bench=args.num_bench_iters,
        skip_prefix_n_blocks=args.skip_prefix_n_blocks,
    )
    configs = filter_configs(configs, args.format, args.dtype)

    if not configs:
        print("No test configurations match the given filters.")
        return

    kernel_fn = _get_kernel_fn(args.use_reference)
    mode_label = "reference" if args.use_reference else "CUDA kernel"
    print(f"Using: {mode_label}")
    print(f"Test configurations: {len(configs)}")
    print()

    # Correctness tests
    if args.mode in ("correctness", "all"):
        print("=" * 60)
        print("Correctness Tests")
        print("=" * 60)

        all_passed = True
        for config in configs:
            result = run_correctness_test(config, kernel_fn)
            status = "PASS" if result else "FAIL"
            print(f"  [{status}] {config.name}")
            if not result:
                all_passed = False

            # Also run skip prefix test
            skip_result = run_skip_prefix_test(config, kernel_fn)
            skip_status = "PASS" if skip_result else "FAIL"
            print(f"  [{skip_status}] {config.name} (skip_prefix=4)")
            if not skip_result:
                all_passed = False

        print()
        if all_passed:
            print("All correctness tests PASSED.")
        else:
            print("Some correctness tests FAILED.")
        print()

    # Benchmark
    if args.mode in ("benchmark", "all"):
        print("=" * 60)
        print("Benchmark")
        print("=" * 60)

        results = []
        for config in configs:
            print(f"  Benchmarking {config.name}...")
            result = run_benchmark(config, kernel_fn)
            results.append(result)

        print_benchmark_table(results)


if __name__ == "__main__":
    main()
