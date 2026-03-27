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

        # --- General args ---
        parser.add_argument(
            "--engine-url",
            required=True,
            help=(
                "Inference engine URL (e.g., http://localhost:8000). "
                "Set OPENAI_API_KEY env var if authentication is needed."
            ),
        )
        parser.add_argument(
            "--lmcache-url",
            default=None,
            help=("LMCache MP server URL for autoconfig (not yet implemented)."),
        )
        parser.add_argument(
            "--model",
            default=None,
            help="Model name (auto-detected from engine if omitted).",
        )
        parser.add_argument(
            "--workload",
            required=True,
            choices=["long-doc-qa"],
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

        # --- Long-doc-qa workload args ---
        group = parser.add_argument_group("long-doc-qa workload options")
        group.add_argument(
            "--document-length",
            type=int,
            default=10000,
            help="Token length per document (default: 10000).",
        )
        group.add_argument(
            "--query-per-document",
            type=int,
            default=2,
            help="Questions per document (default: 2).",
        )
        group.add_argument(
            "--shuffle-policy",
            default="random",
            choices=["random", "tile"],
            help="Request ordering (default: random).",
        )
        group.add_argument(
            "--num-inflight-requests",
            type=int,
            default=3,
            help="Max concurrent in-flight requests (default: 3).",
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

    def _bench_engine(self, args: argparse.Namespace) -> None:
        """Centralized orchestrator: create all modules and run benchmark."""
        # 1. Parse config
        config = parse_args_to_config(args)
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
        request_sender._on_finished.extend(
            [
                lambda result, _text: stats_collector.on_request_finished(result),
                lambda result, _text: progress_monitor.on_request_finished(
                    result.request_id,
                    result.successful,
                ),
                workload.request_finished,
            ]
        )

        # 6. Run benchmark
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
            "p50",
            "P50 decode (tok/s)",
            round(final.p50_decode_speed, 2),
        )
        decode.add(
            "p90",
            "P90 decode (tok/s)",
            round(final.p90_decode_speed, 2),
        )
        decode.add(
            "p99",
            "P99 decode (tok/s)",
            round(final.p99_decode_speed, 2),
        )

        metrics.emit()
