# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench`` command — sustained performance benchmarking."""

# Standard
import argparse
import asyncio
import os
import sys

# First Party
from lmcache.cli.commands.base import BaseCommand
from lmcache.cli.commands.bench.engine_bench.config import (
    EngineBenchConfig,
    parse_args_to_config,
)
from lmcache.cli.commands.bench.engine_bench.interactive import run_interactive
from lmcache.cli.commands.bench.engine_bench.interactive.state import (
    InteractiveState,
)
from lmcache.cli.commands.bench.engine_bench.progress import ProgressMonitor
from lmcache.cli.commands.bench.engine_bench.request_sender import (
    RequestSender,
)
from lmcache.cli.commands.bench.engine_bench.stats import (
    FinalStats,
    StatsCollector,
)
from lmcache.cli.commands.bench.engine_bench.workloads import create_workload
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BenchCommand(BaseCommand):
    """CLI command for sustained performance benchmarking."""

    def name(self) -> str:
        return "bench"

    def help(self) -> str:
        return "Run sustained performance benchmarks."

    def add_arguments(self, _parser: argparse.ArgumentParser) -> None:
        pass  # args registered in register() via subparsers

    def register(self, subparsers: argparse._SubParsersAction) -> None:
        parser = subparsers.add_parser(
            self.name(),
            help=self.help(),
            description="Run sustained performance benchmarks.",
        )
        inner = parser.add_subparsers(
            dest="bench_target",
            required=True,
            metavar="{engine}",
        )
        self._register_engine(inner)

    def _register_engine(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "engine",
            help="Benchmark an inference engine.",
        )

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
            "--no-interactive",
            action="store_true",
            help=(
                "Disable interactive mode. Errors if required arguments are missing."
            ),
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
            help="Token length of each context (default: 5000).",
        )
        ldp_group.add_argument(
            "--ldp-system-prompt-length",
            type=int,
            default=1000,
            help="Token length of the shared system prompt (default: 1000). "
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

        parser.set_defaults(func=self.execute)

    def execute(self, args: argparse.Namespace) -> None:
        handlers = {"engine": self._bench_engine}
        handler = handlers.get(args.bench_target)
        if handler is None:
            print(
                f"Unknown bench target: {args.bench_target}",
                file=sys.stderr,
            )
            sys.exit(1)
        handler(args)

    # ------------------------------------------------------------------
    # Engine benchmark orchestrator
    # ------------------------------------------------------------------

    def _get_missing_args(self, args: argparse.Namespace) -> list[str]:
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
        return missing

    def _needs_interactive(self, args: argparse.Namespace) -> bool:
        """Check whether interactive mode should be triggered."""
        if getattr(args, "config", None):
            return False
        return len(self._get_missing_args(args)) > 0

    def _resolve_args(self, args: argparse.Namespace) -> argparse.Namespace:
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
            missing = self._get_missing_args(args)
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
        if self._needs_interactive(args):
            return run_interactive(args)

        # Case 4: All required args present — run directly
        return args

    def _export_config(
        self,
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

        # Workload-specific args from namespace
        for item in state.get_workload_items():
            value = getattr(args, item.key, item.default)
            if value is not None:
                state.set(item.key, value)

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

    def _bench_engine(self, args: argparse.Namespace) -> None:
        """Centralized orchestrator: create all modules and run benchmark."""
        # 0. Resolve args (config file / interactive / pass-through)
        args = self._resolve_args(args)

        # 1. Parse config
        config = parse_args_to_config(args)

        # 1b. --export-config: save resolved config and exit
        export_path = getattr(args, "export_config", None)
        if export_path:
            self._export_config(config, args, export_path)
            return

        logger.info(
            "Benchmark config: workload=%s, model=%s, "
            "kv_cache=%.1f GB, tokens_per_gb=%d",
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
        request_sender = RequestSender(config.engine_url, config.model)

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
            asyncio.run(request_sender.close())

        # 7. Final metrics
        final = stats_collector.get_final_stats()
        self._emit_final_metrics(config, final, args)

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

    def _emit_final_metrics(
        self,
        config: EngineBenchConfig,
        final: FinalStats,
        args: argparse.Namespace,
    ) -> None:
        """Emit final benchmark summary using the CLI metrics system."""
        title = f"Engine Benchmark Result ({config.workload})"
        metrics = self.create_metrics(title, args, width=56)

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

        metrics.emit()
