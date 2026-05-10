# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench`` command — sustained performance benchmarking."""

# Standard
import argparse
import asyncio
import os
import sys

# First Party
from lmcache.cli.commands import test_cache as _test_cache_mod
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
from lmcache.cli.commands.test_cache import TestCacheCommand
from lmcache.logging import init_logger

logger = init_logger(__name__)


class BenchCommand(BaseCommand):
    """CLI command for sustained performance benchmarking."""

    def __init__(self) -> None:
        super().__init__()
        # None on slim install; _register_kvcache registers a stub instead.
        self._kvcache_delegate = (
            TestCacheCommand() if _test_cache_mod._IMPORT_ERROR is None else None
        )

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
            metavar="{engine,kvcache,l2-adapter}",
        )
        self._register_engine(inner)
        self._register_kvcache(inner)
        self._register_l2_adapter(inner)

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
                "prefix-suffix-tuner",
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

        # --- Prefix-suffix-tuner workload args ---
        psf_group = parser.add_argument_group(
            "prefix-suffix-tuner workload options",
        )
        psf_group.add_argument(
            "--psf-context-length",
            type=int,
            default=8000,
            help="Total tokens per request (prefix + breaker + suffix) "
            "(default: 8000).",
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

    # ------------------------------------------------------------------
    # kvcache bench target — end-to-end MP cache sanity test
    # ------------------------------------------------------------------

    def _register_kvcache(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        """Register ``lmcache bench kvcache``. Delegates to
        :class:`TestCacheCommand`, or registers a stub on slim install.
        """
        if _test_cache_mod._IMPORT_ERROR is not None:
            subparsers.add_parser(
                "kvcache",
                help="(requires full lmcache install)",
                description=(
                    "End-to-end sanity test for the LMCache MP cache server. "
                    "Requires the full `lmcache` package; not available in "
                    "the `lmcache-cli` install."
                ),
            ).set_defaults(func=self.execute)
            return
        assert self._kvcache_delegate is not None
        parser = subparsers.add_parser(
            "kvcache",
            help=self._kvcache_delegate.help(),
            description=(
                "End-to-end sanity test for the LMCache MP cache server: "
                "runs LOOKUP / STORE / RETRIEVE against a live MP server "
                "and verifies KV cache checksums."
            ),
        )
        assert self._kvcache_delegate is not None
        self._kvcache_delegate.add_arguments(parser)
        parser.set_defaults(func=self.execute)

    def _bench_kvcache(self, args: argparse.Namespace) -> None:
        """Dispatch ``lmcache bench kvcache`` to ``TestCacheCommand``."""
        if _test_cache_mod._IMPORT_ERROR is not None:
            print(
                "ERROR: `lmcache bench kvcache` needs the full LMCache "
                "package (torch, zmq, MP runtime), but only the "
                "`lmcache-cli` shell appears to be installed.\n"
                "  Install the full package with `pip install lmcache` "
                "and try again.\n"
                f"  Original import error: {_test_cache_mod._IMPORT_ERROR}",
                file=sys.stderr,
            )
            sys.exit(1)
        assert self._kvcache_delegate is not None
        self._kvcache_delegate.execute(args)

    def execute(self, args: argparse.Namespace) -> None:
        handlers = {
            "engine": self._bench_engine,
            "kvcache": self._bench_kvcache,
            "l2-adapter": self._bench_l2_adapter,
        }
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

    # ------------------------------------------------------------------
    # L2 adapter benchmark
    # ------------------------------------------------------------------

    def _register_l2_adapter(
        self,
        subparsers: argparse._SubParsersAction,
    ) -> None:
        parser = subparsers.add_parser(
            "l2-adapter",
            help="Benchmark an L2 adapter (store / lookup / load).",
            description=(
                "Benchmark L2 adapters using the standard LMCache adapter "
                "configuration mechanism (parse_args_to_l2_adapters_config "
                "+ create_l2_adapter). Any registered adapter type can be "
                "tested without code changes."
            ),
        )

        parser.add_argument(
            "--l2-adapter",
            dest="l2_adapter",
            action="append",
            default=None,
            type=str,
            metavar="JSON",
            help=(
                'L2 adapter spec as JSON with a "type" field and adapter-'
                'specific configs, e.g. \'{"type":"fs","base_path":"/tmp/'
                "bench\"}'. If not provided, falls back to L2_ADAPTER_JSON "
                "environment variable."
            ),
        )
        parser.add_argument(
            "--num-keys",
            type=int,
            default=32,
            help="Keys per submit (default: 32).",
        )
        parser.add_argument(
            "--in-flight",
            type=int,
            default=1,
            help=(
                "In-flight submits per round. Each round issues this many "
                "submits sequentially from a single producer thread, then "
                "waits for all of them (default: 1)."
            ),
        )
        parser.add_argument(
            "--data-size-kb",
            type=int,
            default=256,
            help="Data size per key in KB (default: 256).",
        )
        parser.add_argument(
            "--rounds",
            type=int,
            default=1,
            help="Measurement rounds per operation (default: 1).",
        )
        parser.add_argument(
            "--warmup-rounds",
            type=int,
            default=1,
            help="Warmup rounds before measurement (default: 1).",
        )
        parser.add_argument(
            "--lookup-max-hit-rate",
            type=float,
            default=0.0,
            help=(
                "Upper bound on the lookup hit rate, in [0, 1]. The "
                "benchmark will request floor(N * rate) keys from the "
                "potentially-existing range and (N - hit) keys from a "
                "guaranteed-non-existent range, where N is the total "
                "number of lookup keys (rounds * in_flight * num_keys). "
                "The actual hit rate may be lower if those keys were "
                "never stored. Default: 0.0."
            ),
        )
        # Round-trip verification is OFF by default (cheaper memory
        # footprint: see make_memory_objects' share_buffer layout).
        # Use --no-skip-verify to enable verification.
        parser.add_argument(
            "--skip-verify",
            action=argparse.BooleanOptionalAction,
            default=True,
            help=(
                "Skip round-trip data verification (default). "
                "Pass --no-skip-verify to enable verification."
            ),
        )
        parser.add_argument(
            "--only",
            choices=["lookup", "store", "load"],
            default=None,
            help="Run only the specified operation (default: run all).",
        )

        parser.set_defaults(func=self.execute)

    def _bench_l2_adapter(self, args: argparse.Namespace) -> None:
        """Run the L2 adapter benchmark."""
        # Lazy imports: keep CLI loadable without torch / native deps.
        # Third Party
        import torch

        # First Party
        from lmcache.cli.commands.bench.l2_adapter_bench.data import (
            create_l1_memory_desc,
            make_memory_objects,
            make_object_keys,
            verify_round_trip,
        )
        from lmcache.cli.commands.bench.l2_adapter_bench.runner import (
            bench_load,
            bench_lookup,
            bench_store,
        )
        from lmcache.v1.distributed.l2_adapters import create_l2_adapter
        from lmcache.v1.distributed.l2_adapters.config import (
            parse_args_to_l2_adapters_config,
        )

        kb = 1024
        mb = 1024 * 1024
        data_size = args.data_size_kb * kb
        in_flight = args.in_flight
        num_keys = args.num_keys
        rounds = args.rounds
        warmup = args.warmup_rounds
        total_rounds = warmup + rounds
        max_hit_rate = max(0.0, min(1.0, args.lookup_max_hit_rate))
        quiet = getattr(args, "quiet", False)

        # Keys per round (one in-flight wave) and total measured keys per
        # operation. Warmup rounds extend the consumed idx range.
        keys_per_round = in_flight * num_keys
        total_run_keys = total_rounds * keys_per_round  # warmup + measured

        def log(msg: str) -> None:
            # Per-round progress log; suppressed by --quiet.
            if not quiet:
                print(msg)

        # Resolve L2 adapter JSON: CLI arg takes priority, then env var
        l2_adapter_specs = args.l2_adapter
        if not l2_adapter_specs:
            env_json = os.environ.get("L2_ADAPTER_JSON")
            if env_json:
                l2_adapter_specs = [env_json]
            else:
                print(
                    "Error: No L2 adapter configuration provided.\n"
                    "Use --l2-adapter JSON or set L2_ADAPTER_JSON "
                    "environment variable.",
                    file=sys.stderr,
                )
                sys.exit(2)

        # Parse adapter config using the standard LMCache mechanism
        ns = argparse.Namespace(l2_adapter=l2_adapter_specs)
        try:
            l2_cfg = parse_args_to_l2_adapters_config(ns)
        except (ValueError, KeyError) as e:
            print(f"Error parsing L2 adapter config: {e}", file=sys.stderr)
            sys.exit(2)

        if not l2_cfg.adapters:
            print("Error: no L2 adapter configs parsed", file=sys.stderr)
            sys.exit(2)

        # Use the first adapter config for benchmarking
        adapter_cfg = l2_cfg.adapters[0]
        adapter_type_name = type(adapter_cfg).__name__

        log("=" * 60)
        log("L2 Adapter Benchmark")
        log("=" * 60)
        log(f"  Adapter config         : {adapter_type_name}")
        log(f"  L2 adapter JSON        : {l2_adapter_specs[0]}")
        log(f"  Keys / submit          : {num_keys}")
        log(f"  In-flight / round      : {in_flight}")
        log(f"  Keys / round           : {keys_per_round}")
        log(f"  Data size / key        : {args.data_size_kb} KB")
        log(f"  Data / round           : {(keys_per_round * data_size) / mb:.2f} MB")
        log(f"  Rounds                 : {rounds} (+ {warmup} warmup)")
        if args.only is None or args.only == "lookup":
            log(f"  Lookup max hit rate    : {max_hit_rate:.2%}")
        log("=" * 60)

        # Backing L1 memory buffer for adapters that need an L1 desc.
        # Sized for one in-flight wave of store + load buffers.
        l1_buffer = torch.empty(2 * keys_per_round * data_size, dtype=torch.uint8)
        l1_memory_desc = create_l1_memory_desc(l1_buffer)

        log("\n[Init] Creating adapter...")
        try:
            adapter = create_l2_adapter(adapter_cfg, l1_memory_desc=l1_memory_desc)
            log(f"[Init] Adapter created successfully ({type(adapter).__name__}).\n")
        except Exception as e:
            print(f"[Init] Failed to create adapter: {e}", file=sys.stderr)
            sys.exit(1)

        # ------------------------------------------------------------------
        # Idx layout
        # ------------------------------------------------------------------
        # All ops live in the same idx universe so that ``--only store``
        # followed by ``--only load`` (or lookup) with the same flags hits
        # the exact same keys.
        #
        # Round r (0-indexed, warmup rounds first) consumes the idx slice
        #   [r * keys_per_round, (r+1) * keys_per_round)
        # split into ``in_flight`` contiguous batches of ``num_keys`` each.
        #
        # Lookup additionally splits each round into a hit-portion (drawn
        # from the same idx range as store/load) and a miss-portion drawn
        # from a guaranteed-non-existent range starting at
        # ``total_run_keys``.
        # ------------------------------------------------------------------

        def _build_round_keys(r: int) -> list[list]:
            """Build per-submit key batches for round *r* (store/load)."""
            base = r * keys_per_round
            return [
                make_object_keys(num_keys, key_offset=base + i * num_keys)
                for i in range(in_flight)
            ]

        def _build_round_objs() -> list[list]:
            """Allocate per-submit object batches for one round.

            Every key in every batch gets its OWN ``data_size`` tensor,
            pre-filled with a distinguishing byte pattern. This keeps
            ``verify_round_trip`` meaningful (it can detect cross-key
            corruption after a store -> load cycle) and keeps the
            memory layout consistent regardless of whether verify is
            actually run.

            Per-round (per direction) memory:
            ``in_flight * num_keys * data_size`` bytes.
            """
            return [make_memory_objects(num_keys, data_size) for _ in range(in_flight)]

        # Lookup hit/miss split per round.
        per_round_hit = int(keys_per_round * max_hit_rate)
        per_round_miss = keys_per_round - per_round_hit
        # Total expected hit count over measured rounds only.
        expected_hit_count = per_round_hit * rounds
        # Origin of the guaranteed-miss idx range.
        miss_origin = total_run_keys

        def _build_lookup_round_keys(r: int) -> list[list]:
            """Build per-submit lookup key batches for round *r*.

            Hit slice for round r:
              [r * per_round_hit, (r+1) * per_round_hit)
            Miss slice for round r (disjoint from any store/load idx):
              [miss_origin + r * per_round_miss,
               miss_origin + (r+1) * per_round_miss)

            The combined ``keys_per_round`` keys are concatenated then
            split into ``in_flight`` chunks of ``num_keys`` each.
            """
            hit_base = r * per_round_hit
            miss_base = miss_origin + r * per_round_miss
            keys_round: list = []
            keys_round.extend(make_object_keys(per_round_hit, key_offset=hit_base))
            keys_round.extend(make_object_keys(per_round_miss, key_offset=miss_base))
            # Split into in_flight equal-sized batches of num_keys.
            return [
                keys_round[i * num_keys : (i + 1) * num_keys] for i in range(in_flight)
            ]

        # Per-direction object batches for store / load.
        #
        # Allocation strategy:
        # * Lazy: only allocate when the corresponding direction is
        #   actually exercised. With ``--only store`` we never touch
        #   load buffers (and vice versa), saving
        #   ``in_flight * num_keys * data_size`` bytes of host memory.
        # * Cross-round reuse: once allocated, the same batches are
        #   fed into every round; only the keys change per round. The
        #   L2 adapter does not care about object identity across
        #   rounds, and re-allocating these buffers each round would
        #   just be wasted work.
        store_obj_batches: list[list] | None = None
        load_obj_batches: list[list] | None = None

        def _store_objs(_r: int) -> list[list]:
            nonlocal store_obj_batches
            if store_obj_batches is None:
                store_obj_batches = _build_round_objs()
            return store_obj_batches

        def _load_objs(_r: int) -> list[list]:
            nonlocal load_obj_batches
            if load_obj_batches is None:
                load_obj_batches = _build_round_objs()
            return load_obj_batches

        results: list = []
        failed = False

        # Track the very last measured store round so we can verify it
        # against the matching load round (round-trip integrity check).
        last_store_round_keys: list[list] | None = None
        last_load_round_keys: list[list] | None = None

        try:
            # ---- Store ----
            if args.only is None or args.only == "store":
                log(f"[Store] Running {warmup} warmup + {rounds} measurement rounds...")
                all_store = bench_store(
                    adapter,
                    in_flight=in_flight,
                    num_keys=num_keys,
                    data_size=data_size,
                    rounds=total_rounds,
                    keys_for_round=_build_round_keys,
                    objs_for_round=_store_objs,
                    log=log,
                )
                results.append(self._strip_warmup(all_store, warmup))
                # Last measured store round is total_rounds - 1.
                last_store_round_keys = _build_round_keys(total_rounds - 1)
                log("")

            # ---- Lookup ----
            if args.only is None or args.only == "lookup":
                log(
                    f"[Lookup] Running {warmup} warmup + {rounds} measurement rounds..."
                )
                all_lookup = bench_lookup(
                    adapter,
                    in_flight=in_flight,
                    num_keys=num_keys,
                    rounds=total_rounds,
                    keys_for_round=_build_lookup_round_keys,
                    log=log,
                    expected_max_hit_rate=max_hit_rate,
                    expected_hit_count=expected_hit_count,
                )
                results.append(self._strip_warmup(all_lookup, warmup))
                log("")

            # ---- Load ----
            if args.only is None or args.only == "load":
                log(f"[Load] Running {warmup} warmup + {rounds} measurement rounds...")
                all_load = bench_load(
                    adapter,
                    in_flight=in_flight,
                    num_keys=num_keys,
                    data_size=data_size,
                    rounds=total_rounds,
                    keys_for_round=_build_round_keys,
                    objs_for_round=_load_objs,
                    log=log,
                )
                results.append(self._strip_warmup(all_load, warmup))
                last_load_round_keys = _build_round_keys(total_rounds - 1)
                log("")

            # ---- Round-trip verification (last measured round only) ----
            if (
                not args.skip_verify
                and store_obj_batches is not None
                and load_obj_batches is not None
                and last_store_round_keys is not None
                and last_load_round_keys is not None
            ):
                # Sanity: store and load used the same key idx range for
                # the last measured round, and load buffers now hold what
                # the adapter returned. Compare against the byte pattern
                # written by store (i & 0xFF, where i is position within
                # the batch — see make_memory_objects).
                log(
                    "[Verify] Checking store -> load data integrity for last "
                    "measured round..."
                )
                flat_keys = [k for kl in last_load_round_keys for k in kl]
                flat_store = [o for ol in store_obj_batches for o in ol]
                flat_load = [o for ol in load_obj_batches for o in ol]
                ok = verify_round_trip(flat_keys, flat_store, flat_load, log)
                if not ok:
                    failed = True
                log("")

            # ---- Summary via metrics system ----
            self._emit_l2_adapter_metrics(args, adapter_type_name, results)
        finally:
            log("[Cleanup] Closing adapter...")
            try:
                adapter.close()
            except Exception as e:
                print(f"[Cleanup] adapter.close() failed: {e}", file=sys.stderr)
            log("[Cleanup] Done.")

        if failed:
            sys.exit(1)

    @staticmethod
    def _strip_warmup(result, warmup: int):
        """Drop the leading *warmup* rounds from a BenchResult."""
        # First Party
        from lmcache.cli.commands.bench.l2_adapter_bench.result import BenchResult

        # Adjust the expected hit count proportionally for the kept rounds.
        kept_rounds = max(0, len(result.round_durations) - warmup)
        total_rounds = max(1, len(result.round_durations))
        scaled_expected_hit = int(
            result.expected_hit_count * kept_rounds / total_rounds
        )

        return BenchResult(
            operation=result.operation,
            in_flight=result.in_flight,
            num_keys=result.num_keys,
            data_size_bytes=result.data_size_bytes,
            round_durations=result.round_durations[warmup:],
            success_counts=result.success_counts[warmup:],
            expected_max_hit_rate=result.expected_max_hit_rate,
            expected_hit_count=scaled_expected_hit,
        )

    def _emit_l2_adapter_metrics(
        self,
        args: argparse.Namespace,
        adapter_type_name: str,
        results: list,
    ) -> None:
        """Emit L2 adapter benchmark summary using the CLI metrics system."""
        title = f"L2 Adapter Benchmark Result ({adapter_type_name})"
        metrics = self.create_metrics(title, args, width=64)

        cfg_section = metrics.add_section("config", "Configuration")
        cfg_section.add("adapter", "Adapter", adapter_type_name)
        cfg_section.add("num_keys", "Keys / submit", args.num_keys)
        cfg_section.add("in_flight", "In-flight / round", args.in_flight)
        cfg_section.add(
            "data_size_kb",
            "Data size / key (KB)",
            args.data_size_kb,
        )
        cfg_section.add("rounds", "Measurement rounds", args.rounds)
        cfg_section.add("warmup_rounds", "Warmup rounds", args.warmup_rounds)
        cfg_section.add(
            "lookup_max_hit_rate",
            "Lookup max hit rate",
            round(args.lookup_max_hit_rate, 4),
        )

        for idx, r in enumerate(results):
            section_id = f"op_{idx}"
            section = metrics.add_section(section_id, r.operation)
            section.add("operation", "Operation", r.operation)
            section.add("rounds", "Rounds", len(r.round_durations))
            section.add("keys_per_round", "Keys / round", r.keys_per_round)
            section.add("total_keys", "Total keys", r.total_keys)
            section.add("total_success", "Total success", r.total_success)
            section.add(
                "duration_avg_ms",
                "Duration avg (ms)",
                round(r.avg_duration * 1000, 2),
            )
            section.add(
                "duration_min_ms",
                "Duration min (ms)",
                round(r.min_duration * 1000, 2),
            )
            section.add(
                "duration_max_ms",
                "Duration max (ms)",
                round(r.max_duration * 1000, 2),
            )
            section.add(
                "duration_p50_ms",
                "Duration p50 (ms)",
                round(r.p50_duration * 1000, 2),
            )
            section.add(
                "duration_p99_ms",
                "Duration p99 (ms)",
                round(r.p99_duration * 1000, 2),
            )
            section.add(
                "duration_std_ms",
                "Duration std (ms)",
                round(r.std_duration * 1000, 2),
            )
            section.add(
                "throughput_avg_mbps",
                "Throughput avg (MB/s)",
                round(r.avg_throughput_mbps, 2),
            )
            section.add(
                "throughput_min_mbps",
                "Throughput min (MB/s)",
                round(r.min_throughput_mbps, 2),
            )
            section.add(
                "throughput_max_mbps",
                "Throughput max (MB/s)",
                round(r.max_throughput_mbps, 2),
            )
            section.add(
                "ops_per_sec_avg",
                "Avg ops/s",
                round(r.avg_ops_per_sec, 2),
            )
            section.add(
                "latency_per_key_ms",
                "Avg latency / key (ms)",
                round(r.avg_latency_per_key_ms, 3),
            )
            if r.expected_max_hit_rate > 0 or r.expected_hit_count > 0:
                section.add(
                    "expected_max_hit_rate",
                    "Expected max hit rate",
                    round(r.expected_max_hit_rate, 4),
                )
                section.add(
                    "expected_hit_count",
                    "Expected hit keys",
                    r.expected_hit_count,
                )
                section.add(
                    "actual_hit_rate",
                    "Actual hit rate",
                    round(r.actual_hit_rate, 4),
                )

        metrics.emit()
