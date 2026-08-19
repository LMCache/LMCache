# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench engine`` subcommand implementation.

This module provides argument registration via :func:`add_engine_arguments`
and the execution orchestrator :func:`run_engine_bench` for the inference
engine benchmark.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import os
import sys

# First Party
from lmcache.cli.commands.bench.engine_bench.config import (
    EngineBenchConfig,
    parse_args_to_config,
)
from lmcache.cli.commands.bench.engine_bench.interactive import run_interactive
from lmcache.cli.commands.bench.engine_bench.interactive.state import (
    InteractiveState,
)
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.quality.dataset import (
    describe_hub_datasets,
)
from lmcache.cli.commands.bench.engine_bench.request_sender import (
    RequestSender,
)
from lmcache.cli.commands.bench.engine_bench.stats import (
    FinalStats,
    StatsCollector,
)
from lmcache.cli.commands.bench.engine_bench.workloads import (
    DEFAULT_DOC_ALIGN_TOKENS,
    create_workload,
    parse_template_kwargs,
    validate_max_output_length_supported,
)
from lmcache.cli.commands.bench.engine_bench.workloads.base import BaseWorkload
from lmcache.logging import init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand

logger = init_logger(__name__)

# Default for --ldqa-max-output-length; centralized so the "max output length
# explicitly set" check stays in sync with the parser.
_LDQA_MAX_OUTPUT_LENGTH_DEFAULT = 128

# Workload-specific arguments that have no default, as
# ``{workload: ((namespace attr, CLI flag), ...)}``. Routing them through the
# general missing-argument path gives --no-interactive, the TUI, and --config
# replay consistent handling.
_REQUIRED_WORKLOAD_ARGS: dict[str, tuple[tuple[str, str], ...]] = {
    "rag-qa-quality": (("rag_dataset", "--rag-dataset"),),
}


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_engine_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench engine`` arguments to *parser*.

    Args:
        parser: The ``ArgumentParser`` for the engine bench subcommand.
    """

    # --- Config file ---
    parser.add_argument(
        "--config",
        default=None,
        metavar="FILE",
        help="Load configuration from a JSON file (skips interactive mode).",
    )

    # --- General args ---
    parser.add_argument(
        "--engine-url",
        default=None,
        help=(
            "Inference engine URL (e.g., http://localhost:8000). "
            "Set OPENAI_API_KEY env var if authentication is needed."
        ),
    )
    parser.add_argument(
        "--lmcache-url",
        default=None,
        help="LMCache MP server URL for auto-detecting tokens per GB.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (auto-detected from engine if omitted).",
    )
    parser.add_argument(
        "--workload",
        default=None,
        choices=[
            "long-doc-permutator",
            "long-doc-qa",
            "multi-round-chat",
            "prefix-suffix-tuner",
            "rag-qa-quality",
            "random-prefill",
        ],
        help="Workload type.",
    )
    parser.add_argument(
        "--kv-cache-volume",
        type=float,
        default=100.0,
        help="Target active KV cache in GB (default: 100).",
    )
    parser.add_argument(
        "--tokens-per-gb-kvcache",
        type=int,
        default=None,
        help=("Tokens per GB of KV cache (required if --lmcache-url is not set)."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory for output files (default: current).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip CSV export.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export JSON summary.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress real-time progress display.",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Force generation to run for the full output length by ignoring "
            "the model's EOS token (vLLM sampling extension). Makes decode "
            "throughput reproducible regardless of when the model would stop."
        ),
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help=("Disable interactive mode. Errors if required arguments are missing."),
    )
    parser.add_argument(
        "--export-config",
        default=None,
        metavar="FILE",
        help=(
            "Export resolved configuration to a JSON file and exit. "
            "Does not run the benchmark or enter interactive mode."
        ),
    )

    # --- Long-doc-permutator workload args ---
    ldp_group = parser.add_argument_group("long-doc-permutator workload options")
    ldp_group.add_argument(
        "--ldp-num-contexts",
        type=int,
        default=5,
        help="Number of unique context documents (default: 5).",
    )
    ldp_group.add_argument(
        "--ldp-context-length",
        type=int,
        default=5000,
        help="Exact token length of each context (default: 5000). Requires a "
        "loadable tokenizer; pass --model when the engine reports a name that "
        "is not a HuggingFace repo ID or local path.",
    )
    ldp_group.add_argument(
        "--ldp-system-prompt-length",
        type=int,
        default=1000,
        help="Exact token length of the shared system prompt (default: 1000). "
        "Use 0 for no system prompt.",
    )
    ldp_group.add_argument(
        "--ldp-num-permutations",
        type=int,
        default=10,
        help="Number of distinct permutations to send (default: 10). "
        "Capped at N! where N = --ldp-num-contexts.",
    )
    ldp_group.add_argument(
        "--ldp-num-inflight-requests",
        type=int,
        default=1,
        help="Max concurrent in-flight requests (default: 1).",
    )
    ldp_group.add_argument(
        "--ldp-max-output-length",
        type=int,
        default=128,
        help="Max tokens to generate per permutation request (default: 128). "
        "Use 1 to measure prefill alone; combine larger values with "
        "--ignore-eos for a reproducible decode phase.",
    )

    # --- Long-doc-qa workload args ---
    group = parser.add_argument_group("long-doc-qa workload options")
    group.add_argument(
        "--ldqa-document-length",
        type=int,
        default=10000,
        help="Token length per document (default: 10000).",
    )
    group.add_argument(
        "--ldqa-query-per-document",
        type=int,
        default=2,
        help="Questions per document (default: 2).",
    )
    group.add_argument(
        "--ldqa-shuffle-policy",
        default="random",
        choices=["random", "tile"],
        help="Request ordering (default: random).",
    )
    group.add_argument(
        "--ldqa-num-inflight-requests",
        type=int,
        default=3,
        help="Max concurrent in-flight requests (default: 3).",
    )
    group.add_argument(
        "--ldqa-max-output-length",
        type=int,
        default=_LDQA_MAX_OUTPUT_LENGTH_DEFAULT,
        help=(
            f"Max tokens to generate per benchmark query "
            f"(default: {_LDQA_MAX_OUTPUT_LENGTH_DEFAULT}). Combine with "
            "--ignore-eos for a reproducible decode phase."
        ),
    )

    # --- Multi-round-chat workload args ---
    mrc_group = parser.add_argument_group(
        "multi-round-chat workload options",
    )
    mrc_group.add_argument(
        "--mrc-shared-prompt-length",
        type=int,
        default=2000,
        help="System prompt token length (default: 2000).",
    )
    mrc_group.add_argument(
        "--mrc-chat-history-length",
        type=int,
        default=10000,
        help="Pre-filled chat history token length (default: 10000).",
    )
    mrc_group.add_argument(
        "--mrc-user-input-length",
        type=int,
        default=50,
        help="Tokens per user query (default: 50).",
    )
    mrc_group.add_argument(
        "--mrc-output-length",
        type=int,
        default=200,
        help="Max tokens to generate per response (default: 200).",
    )
    mrc_group.add_argument(
        "--mrc-qps",
        type=float,
        default=1.0,
        help="Queries per second (default: 1.0).",
    )
    mrc_group.add_argument(
        "--mrc-duration",
        type=float,
        default=60.0,
        help="Benchmark duration in seconds (default: 60).",
    )

    # --- Prefix-suffix-tuner workload args ---
    psf_group = parser.add_argument_group(
        "prefix-suffix-tuner workload options",
    )
    psf_group.add_argument(
        "--psf-context-length",
        type=int,
        default=8000,
        help="Total tokens per request (prefix + breaker + suffix) (default: 8000).",
    )
    psf_group.add_argument(
        "--psf-prefix-ratio",
        type=float,
        default=0.8,
        help="Fraction of context-length used by the prefix (default: 0.8). "
        "Must be in (0.0, 1.0). The remainder (minus a 32-token breaker) is "
        "the shared suffix.",
    )
    psf_group.add_argument(
        "--psf-thrash",
        type=float,
        default=20.0,
        help="Size in GB of the KV-cache tier to overflow (default: 20.0). "
        "The workload sizes its prefix pool to slightly more than this, "
        "so every pass-2 request misses that tier and falls through to "
        "the next one. Use the L0 (HBM) size for vanilla vLLM baselines, "
        "or the L1 (LMCache DRAM) size for tiered baselines.",
    )

    # --- Rag-qa-quality workload args ---
    rag_group = parser.add_argument_group("rag-qa-quality workload options")
    rag_group.add_argument(
        "--rag-dataset",
        default=None,
        help=(
            "Required for this workload. A known dataset name or a path to a "
            f"local QA file. Known names -- {describe_hub_datasets()}. Named "
            "datasets download from the HuggingFace Hub on first use."
        ),
    )
    rag_group.add_argument(
        "--rag-num-samples",
        type=int,
        default=50,
        help="Questions to measure, taken in dataset order (default: 50).",
    )
    rag_group.add_argument(
        "--rag-max-output-length",
        type=int,
        default=1024,
        help=(
            "Token budget per answer (default: 1024). Must fit a reasoning "
            "model's thinking block as well as the <final_answer> tags, or "
            "samples fail to parse and drop out of the score."
        ),
    )
    rag_group.add_argument(
        "--rag-doc-align-tokens",
        type=int,
        default=DEFAULT_DOC_ALIGN_TOKENS,
        help=(
            f"Pad documents and the system block to a multiple of this many "
            f"tokens (default: {DEFAULT_DOC_ALIGN_TOKENS}, LMCache's own "
            "default chunk size). Set it to the deployment's chunk size: a "
            "mismatch leaves documents off-phase and reuse partial. Both runs "
            "being compared must use the same value."
        ),
    )
    rag_group.add_argument(
        "--rag-template-kwargs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Chat-template variables, repeatable. Left unset, the model's own "
            "template default applies. Use it to bound a runaway thinking "
            "block (reasoning_effort=high) or to enable one "
            "(thinking_mode=enabled) -- the right value is model-specific, and "
            "turning thinking off outright can lower multi-hop answer quality."
        ),
    )
    rag_group.add_argument(
        "--rag-output",
        default=None,
        metavar="FILE",
        help=(
            "Per-sample results JSON (default: <output-dir>/rag_qa_quality.json). "
            "Name the two runs apart to diff them by sample id."
        ),
    )

    # --- Random-prefill workload args ---
    rp_group = parser.add_argument_group(
        "random-prefill workload options",
    )
    rp_group.add_argument(
        "--rp-request-length",
        type=int,
        default=10000,
        help="Token length per request (default: 10000).",
    )
    rp_group.add_argument(
        "--rp-num-requests",
        type=int,
        default=50,
        help="Number of requests to send (default: 50).",
    )


# ---------------------------------------------------------------------------
# Argument resolution helpers
# ---------------------------------------------------------------------------


def _get_missing_args(args: argparse.Namespace) -> list[str]:
    """Return list of missing required CLI flags."""
    missing: list[str] = []
    if args.engine_url is None:
        missing.append("--engine-url")
    if args.workload is None:
        missing.append("--workload")
    if (
        args.tokens_per_gb_kvcache is None
        and getattr(args, "lmcache_url", None) is None
    ):
        missing.append("--tokens-per-gb-kvcache or --lmcache-url")
    for attr, flag in _REQUIRED_WORKLOAD_ARGS.get(args.workload, ()):
        if getattr(args, attr, None) is None:
            missing.append(flag)
    return missing


def _needs_interactive(args: argparse.Namespace) -> bool:
    """Check whether interactive mode should be triggered."""
    if getattr(args, "config", None):
        return False
    return len(_get_missing_args(args)) > 0


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve args via config file, interactive mode, or pass through."""
    # Case 1: --config file
    config_path = getattr(args, "config", None)
    if config_path:
        state = InteractiveState.load_json(config_path)
        state.merge_cli_args(args)
        resolved = state.to_namespace()
        # Carry over output flags from CLI
        for attr in (
            "output_dir",
            "seed",
            "no_csv",
            "json",
            "quiet",
            "format",
            "output",
        ):
            cli_val = getattr(args, attr, None)
            if cli_val is not None:
                setattr(resolved, attr, cli_val)
        return resolved

    # Case 2: --no-interactive or --export-config — error if missing
    no_interactive = getattr(args, "no_interactive", False)
    export_config = getattr(args, "export_config", None)
    if no_interactive or export_config:
        missing = _get_missing_args(args)
        if missing:
            flag = "--export-config" if export_config else "--no-interactive"
            raise SystemExit(
                "Missing required arguments: "
                + ", ".join(missing)
                + f". Provide them or remove {flag} "
                "for guided setup."
            )
        return args

    # Case 3: Interactive mode
    if _needs_interactive(args):
        return run_interactive(args)

    # Case 4: All required args present — run directly
    return args


def _export_config(
    config: EngineBenchConfig,
    args: argparse.Namespace,
    path: str,
) -> None:
    """Export resolved config to JSON and exit.

    Builds a standalone config dict from the resolved
    ``EngineBenchConfig`` and workload-specific CLI args.
    Environment-specific keys (``engine_url``, ``lmcache_url``)
    are excluded by ``InteractiveState.to_json()`` so the exported
    config is portable.
    """
    # Standard
    import json as json_mod

    state = InteractiveState()
    state.set("engine_url", config.engine_url)
    state.set("model", config.model)
    state.set("workload", config.workload)
    state.set("kv_cache_volume", config.kv_cache_volume_gb)
    state.set("tokens_per_gb_kvcache", config.tokens_per_gb_kvcache)
    state.set("ignore_eos", config.ignore_eos)

    # Workload-specific args from namespace
    for item in state.get_workload_items():
        value = getattr(args, item.key, item.default)
        if value is not None:
            state.set(item.key, value)

    # Required workload args live in the required phase, so the loop above
    # does not see them; without this the export would drop them.
    for attr, _flag in _REQUIRED_WORKLOAD_ARGS.get(config.workload, ()):
        value = getattr(args, attr, None)
        if value is not None:
            state.set(attr, value)

    # to_json() handles filtering out engine_url, lmcache_url, etc.
    data = state.to_json()

    with open(path, "w") as f:
        json_mod.dump(data, f, indent=2)
        f.write("\n")

    print(f"Configuration exported to {path}")
    print(
        f"\033[1mReplay with:\033[0m \033[96mlmcache bench engine "
        f"--engine-url <URL> --config {path}\033[0m"
    )


# ---------------------------------------------------------------------------
# Final metrics emission
# ---------------------------------------------------------------------------


def _resolve_extra_body(args: argparse.Namespace) -> dict[str, object]:
    """Build the request ``extra_body`` implied by the CLI arguments.

    Args:
        args: Resolved CLI arguments.

    Returns:
        Fields to merge into every request body; empty when none apply.

    Raises:
        ValueError: If a template kwarg is not in ``KEY=VALUE`` form.
    """
    template_kwargs = parse_template_kwargs(getattr(args, "rag_template_kwargs", []))
    if not template_kwargs:
        return {}
    return {"chat_template_kwargs": template_kwargs}


def _emit_final_metrics(
    command: "BaseCommand",
    config: EngineBenchConfig,
    final: FinalStats,
    args: argparse.Namespace,
    workload: BaseWorkload,
) -> None:
    """Emit final benchmark summary using the CLI metrics system."""
    title = f"Engine Benchmark Result ({config.workload})"
    metrics = command.create_metrics(title, args, width=56)

    cfg_section = metrics.add_section("config", "Configuration")
    cfg_section.add("engine_url", "Engine URL", config.engine_url)
    cfg_section.add("model", "Model", config.model)
    cfg_section.add("workload", "Workload", config.workload)

    results = metrics.add_section("results", "Results")
    results.add(
        "successful",
        "Successful requests",
        final.successful_requests,
    )
    results.add("failed", "Failed requests", final.failed_requests)
    results.add(
        "duration",
        "Benchmark duration (s)",
        round(final.elapsed_time, 2),
    )
    results.add(
        "input_tokens",
        "Total input tokens",
        final.total_input_tokens,
    )
    results.add(
        "output_tokens",
        "Total output tokens",
        final.total_output_tokens,
    )
    results.add(
        "input_tput",
        "Input throughput (tok/s)",
        round(final.input_throughput, 2),
    )
    results.add(
        "output_tput",
        "Output throughput (tok/s)",
        round(final.output_throughput, 2),
    )

    ttft = metrics.add_section("ttft", "Time to First Token")
    ttft.add("mean", "Mean TTFT (ms)", round(final.mean_ttft_ms, 2))
    ttft.add("p50", "P50 TTFT (ms)", round(final.p50_ttft_ms, 2))
    ttft.add("p90", "P90 TTFT (ms)", round(final.p90_ttft_ms, 2))
    ttft.add("p99", "P99 TTFT (ms)", round(final.p99_ttft_ms, 2))

    decode = metrics.add_section("decode", "Decoding Speed")
    decode.add(
        "mean",
        "Mean decode (tok/s)",
        round(final.mean_decode_speed, 2),
    )
    decode.add(
        "p99",
        "P99 decode (tok/s)",
        round(final.p99_decode_speed, 2),
    )

    for extra in workload.extra_metric_sections():
        section = metrics.add_section(extra.key, extra.label)
        for key, label, value in extra.entries:
            section.add(key, label, value)

    metrics.emit()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_engine_bench(command: "BaseCommand", args: argparse.Namespace) -> None:
    """Centralized orchestrator: create all modules and run the engine bench.

    Args:
        command: The outer ``BenchCommand`` instance, used for
            ``create_metrics`` (inherited from ``BaseCommand``).
        args: Parsed CLI arguments for ``lmcache bench engine``.
    """
    # 0. Resolve args (config file / interactive / pass-through)
    args = _resolve_args(args)

    # 1. Parse config
    config = parse_args_to_config(args)

    # 1a. A max output length can only be set for workloads that have a
    # max-output-length parameter; reject it for any other workload.
    if args.ldqa_max_output_length != _LDQA_MAX_OUTPUT_LENGTH_DEFAULT:
        validate_max_output_length_supported(config.workload)

    # 1b. --export-config: save resolved config and exit
    export_path = getattr(args, "export_config", None)
    if export_path:
        _export_config(config, args, export_path)
        return

    logger.info(
        "Benchmark config: workload=%s, model=%s, kv_cache=%.1f GB, tokens_per_gb=%d",
        config.workload,
        config.model,
        config.kv_cache_volume_gb,
        config.tokens_per_gb_kvcache,
    )

    # 2. Create shared modules
    stats_collector = StatsCollector()
    progress_monitor = ProgressMonitor(
        stats_collector,
        quiet=config.quiet,
    )

    # 3. Create request sender (callbacks wired after workload creation)
    request_sender = RequestSender(
        config.engine_url,
        config.model,
        ignore_eos=config.ignore_eos,
        extra_body=_resolve_extra_body(args),
    )

    # 4. Create workload
    workload = create_workload(
        config,
        args,
        request_sender,
        stats_collector,
        progress_monitor,
    )

    # 5. Wire callbacks on sender
    request_sender.add_on_finished_callback(
        lambda result, _text: stats_collector.on_request_finished(result),
    )
    request_sender.add_on_finished_callback(
        lambda result, _text: progress_monitor.on_request_finished(
            result.request_id,
            result.successful,
        ),
    )
    request_sender.add_on_finished_callback(workload.request_finished)

    # 6. Log config and run benchmark
    workload.log_config()
    progress_monitor.start()
    try:
        workload.run()
    finally:
        progress_monitor.stop()

    # 7. Final metrics
    final = stats_collector.get_final_stats()
    _emit_final_metrics(command, config, final, args, workload)

    # 8. Export
    if config.export_csv:
        csv_path = os.path.join(config.output_dir, "bench_results.csv")
        stats_collector.export_csv(csv_path)
        logger.info("CSV results written to %s", csv_path)
    if config.export_json:
        json_path = os.path.join(
            config.output_dir,
            "bench_summary.json",
        )
        stats_collector.export_json(json_path, config)
        logger.info("JSON summary written to %s", json_path)

    # 9. Exit code
    if final.failed_requests > 0:
        sys.exit(1)
