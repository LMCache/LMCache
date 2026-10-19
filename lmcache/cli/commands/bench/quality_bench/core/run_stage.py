# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench quality run``: send the plan to arms and save the answers.

One invocation runs one or more arms against live engines and saves each
pass to a results directory as soon as it finishes. Arms can be run one at a
time -- each on its own GPU server, in separate CI jobs -- into the same
directory; ``bench quality analyze`` then reaches the verdict from the files.
"""

# Standard
from collections.abc import Callable
from datetime import datetime
import argparse
import asyncio
import json
import urllib.request

# First Party
from lmcache.cli.commands.bench.engine_bench.quality.alignment import ChunkAligner
from lmcache.cli.commands.bench.engine_bench.quality.dataset import (
    describe_hub_datasets,
    load_samples,
    resolve_dataset_path,
)
from lmcache.cli.commands.bench.engine_bench.request_sender import RequestSender
from lmcache.cli.commands.bench.engine_bench.tokenizers import try_load_tokenizer
from lmcache.cli.commands.bench.engine_bench.workloads.rag_qa_quality import (
    parse_template_kwargs,
)
from lmcache.cli.commands.bench.quality_bench.core.blend_counters import (
    BlendCounterSource,
    PrometheusBlendCounters,
    UnmeasuredBlendCounters,
)
from lmcache.cli.commands.bench.quality_bench.core.config import (
    ArmEndpoint,
    ArmRole,
    QualityRunConfig,
    ReuseScenario,
    TaskType,
)
from lmcache.cli.commands.bench.quality_bench.core.plan import QualityPlan, build_plan
from lmcache.cli.commands.bench.quality_bench.core.report import arm_counts
from lmcache.cli.commands.bench.quality_bench.core.run_files import (
    ResultsDirError,
    create_run_dir,
    ensure_plan,
    exit_on_results_dir_error,
    remove_role,
    write_arm,
)
from lmcache.cli.commands.bench.quality_bench.core.runner import (
    ArmRun,
    ArmRunner,
    StorePhase,
)
from lmcache.cli.commands.bench.quality_bench.core.task import check_task
from lmcache.cli.commands.bench.quality_bench.core.verdict import run_problems
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Parent of the results directory created when --output-dir is omitted.
DEFAULT_RESULTS_ROOT = "quality_results"

_DEFAULT_REFERENCE_REPEATS = 2

_VERSION_TIMEOUT_S = 5.0

# Called with each finished pass and when it started and finished.
OnRunFinished = Callable[[ArmRun, datetime, datetime], None]


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench quality run`` arguments to *parser*.

    Args:
        parser: The subcommand's parser.
    """
    arms = parser.add_argument_group("arms (at least one)")
    arms.add_argument(
        "--reference-url",
        default="",
        help="Engine serving full recompute (vLLM, prefix caching off).",
    )
    arms.add_argument(
        "--blend-url",
        default="",
        help="Engine serving CacheBlend (vLLM + CBKVConnector).",
    )
    arms.add_argument(
        "--apc-url",
        default="",
        help="Engine serving vLLM prefix caching only. Its gap to the "
        "reference is reported as a noise floor.",
    )
    arms.add_argument(
        "--blend-metrics-url",
        default="",
        help="The blend server's Prometheus /metrics URL (needs --blend-url). "
        "Without it, engagement is not checked and cannot make the verdict "
        "INVALID.",
    )
    arms.add_argument(
        "--reference-repeats",
        type=int,
        default=None,
        help="Reference passes (needs --reference-url; default "
        f"{_DEFAULT_REFERENCE_REPEATS}). Two or more report a flip-rate noise "
        "floor.",
    )

    workload = parser.add_argument_group("workload (same on every run into a dir)")
    workload.add_argument(
        "--model",
        required=True,
        help="Served model name. Its tokenizer aligns passages to chunks.",
    )
    workload.add_argument(
        "--task",
        choices=[t.value for t in TaskType],
        default=TaskType.QA.value,
        help="The dataset's task type. Only short-answer QA is supported. A "
        "dataset with long references is refused.",
    )
    workload.add_argument(
        "--dataset",
        required=True,
        help=f"Dataset name or local QA file. Names: {describe_hub_datasets()}",
    )
    workload.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Questions to measure, in dataset order (0 = all).",
    )
    workload.add_argument(
        "--scenario",
        choices=[s.value for s in ReuseScenario],
        default=ReuseScenario.INDEPENDENT.value,
        help="independent: each passage cached alone; reordered: each "
        "question's passages cached together, then prompted in a new order.",
    )
    workload.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="LMCache chunk size. Must match the blend server's --chunk-size.",
    )
    workload.add_argument(
        "--max-output-tokens",
        type=int,
        default=2048,
        help="Token budget per answer, reasoning included.",
    )
    workload.add_argument(
        "--template-kwargs",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Chat-template variable (repeatable), e.g. reasoning_effort=low.",
    )
    workload.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the chunk-alignment filler and the reordered "
        "permutation (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Results directory, used as given; runs of different arms into "
        "the same directory are analyzed together. Default: a new "
        f"timestamped directory under {DEFAULT_RESULTS_ROOT}/.",
    )


def run_config_from_args(args: argparse.Namespace) -> QualityRunConfig:
    """Build the validated config from parsed arguments.

    Args:
        args: Namespace produced by :func:`add_run_arguments`.

    Returns:
        The config.

    Raises:
        ValueError: If an argument is invalid.
    """
    if args.reference_repeats is not None and not args.reference_url:
        raise ValueError("--reference-repeats needs --reference-url")
    arms = [
        ArmEndpoint(role, url)
        for role, url in (
            (ArmRole.REFERENCE, args.reference_url),
            (ArmRole.PREFIX_CACHE, args.apc_url),
            (ArmRole.BLEND, args.blend_url),
        )
        if url
    ]
    return QualityRunConfig(
        model=args.model,
        dataset=args.dataset,
        arms=arms,
        scenario=ReuseScenario(args.scenario),
        num_samples=args.limit,
        chunk_size=args.chunk_size,
        reference_repeats=(
            _DEFAULT_REFERENCE_REPEATS
            if args.reference_repeats is None
            else args.reference_repeats
        ),
        max_output_tokens=args.max_output_tokens,
        template_kwargs=parse_template_kwargs(args.template_kwargs),
        task=TaskType(args.task),
        blend_metrics_url=args.blend_metrics_url,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def build_quality_plan(config: QualityRunConfig) -> QualityPlan:
    """Load the dataset and build the plan every arm will receive.

    Args:
        config: The run's config.

    Returns:
        The plan.

    Raises:
        RuntimeError: If the model's tokenizer cannot be loaded.
        ValueError: If the dataset yields no usable questions, or does not
            fit the configured task type.
    """
    tokenizer = try_load_tokenizer(config.model)
    if tokenizer is None:
        raise RuntimeError(
            f"bench quality needs the tokenizer for {config.model!r} to align "
            f"passages to cache chunks. Install transformers and pass --model "
            f"with a HuggingFace repo ID or a local path."
        )
    samples = load_samples(resolve_dataset_path(config.dataset))
    if config.num_samples:
        samples = samples[: config.num_samples]
    check_task(samples, config.task)
    aligner = ChunkAligner(tokenizer, config.model, config.chunk_size, config.seed)
    return build_plan(samples, aligner, config.scenario, config.chunk_size, config.seed)


async def run_arms(
    config: QualityRunConfig,
    plan: QualityPlan,
    senders: dict[ArmRole, RequestSender],
    blend_counters: BlendCounterSource,
    on_run: OnRunFinished,
) -> list[ArmRun]:
    """Run every configured arm over *plan*, one pass at a time.

    Order: every reference pass, then apc, then blend, skipping arms that
    are not configured. The caching arms store the plan's passages first --
    the reference does not. Each arm must start from an empty cache: restart
    its engine (and the blend server) before the run.

    Args:
        config: The run's config.
        plan: The prompts to send.
        senders: One request sender per configured arm role. All are closed
            when the last pass finishes.
        blend_counters: Engagement counters for the blend arm.
        on_run: Called with each pass as soon as it finishes, so a failure
            in a later pass does not lose it.

    Returns:
        Every arm run, in the order run.
    """
    unmeasured = UnmeasuredBlendCounters()
    schedule: list[tuple[ArmRole, int, StorePhase, BlendCounterSource]] = []
    if config.has_arm(ArmRole.REFERENCE):
        schedule += [
            (ArmRole.REFERENCE, k, StorePhase.SKIP_STORE, unmeasured)
            for k in range(config.reference_repeats)
        ]
    if config.has_arm(ArmRole.PREFIX_CACHE):
        schedule.append((ArmRole.PREFIX_CACHE, 0, StorePhase.STORE_FIRST, unmeasured))
    if config.has_arm(ArmRole.BLEND):
        schedule.append((ArmRole.BLEND, 0, StorePhase.STORE_FIRST, blend_counters))

    runs: list[ArmRun] = []
    runners: dict[ArmRole, ArmRunner] = {}
    try:
        for role, run_index, store_phase, counters in schedule:
            if role not in runners:
                runners[role] = ArmRunner(
                    senders[role], config.max_output_tokens, counters
                )
            logger.info("Running %s pass %d", role.value, run_index + 1)
            started = _now()
            run = await runners[role].run(plan, role, run_index, store_phase)
            on_run(run, started, _now())
            runs.append(run)
    finally:
        for sender in senders.values():
            await sender.close()
    return runs


def engine_version(url: str) -> str | None:
    """Ask an engine for its version, best-effort.

    vLLM serves ``GET /version`` at the server root. Recorded with each arm
    so that a stale or mismatched engine is visible when results are
    compared or a saved reference is reused.

    Args:
        url: The engine's base URL, with or without ``/v1``.

    Returns:
        The reported version, or ``None`` if the engine reports none.
    """
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    try:
        with urllib.request.urlopen(
            f"{base}/version", timeout=_VERSION_TIMEOUT_S
        ) as response:
            version = json.loads(response.read()).get("version")
    except (OSError, ValueError, AttributeError):
        # Unreachable, not JSON, or not an object: the engine reports none.
        return None
    return None if version is None else str(version)


def run_quality_run(args: argparse.Namespace) -> None:
    """Entry point for ``lmcache bench quality run``.

    Builds the plan, records it in the results directory (or checks it
    matches the one there), runs each configured arm and saves every pass
    as it finishes. Exits non-zero when a pass already guarantees an
    INVALID verdict, so CI flags the job that broke; its files are still
    saved for ``analyze`` to report.

    Args:
        args: Parsed arguments.

    Raises:
        SystemExit: With status 1 when a pass had no successful request or
            failed store requests, or 2 if the plan differs from the one
            already in the results directory (a setup problem, checked
            before any request).
        ValueError: If an argument is invalid or the dataset is refused.
        RuntimeError: If the blend-server metrics cannot be read.
    """
    started = _now()
    config = run_config_from_args(args)
    plan = build_quality_plan(config)
    results_dir = config.output_dir or create_run_dir(DEFAULT_RESULTS_ROOT, started)
    try:
        fingerprint = ensure_plan(results_dir, plan, config.plan_settings())
    except ResultsDirError as e:
        exit_on_results_dir_error("run", e)
    arm_names = ", ".join(arm.role.value for arm in config.arms)
    print(
        f"bench quality run: {len(plan.samples)} questions, "
        f"{len(plan.store_requests)} store requests, "
        f"scenario={plan.scenario.value}, arms: {arm_names}"
    )
    print(f"results directory: {results_dir}")

    blend_counters: BlendCounterSource = (
        PrometheusBlendCounters(config.blend_metrics_url)
        if config.blend_metrics_url
        else UnmeasuredBlendCounters()
    )
    # The counters are first needed when the blend pass starts, after every
    # other pass. Read them now so a bad --blend-metrics-url fails before
    # any arm time is spent.
    blend_counters.read()

    versions = {arm.role: engine_version(arm.url) for arm in config.arms}
    extra_body: dict[str, object] = {}
    if config.template_kwargs:
        extra_body["chat_template_kwargs"] = dict(config.template_kwargs)
    senders = {
        arm.role: RequestSender(arm.url, config.model, extra_body=extra_body)
        for arm in config.arms
    }

    replaced: set[ArmRole] = set()

    def save(run: ArmRun, pass_started: datetime, pass_finished: datetime) -> None:
        if run.role not in replaced:
            # A role's passes all come from this run: never mixed with
            # passes from an earlier engine.
            removed = remove_role(results_dir, run.role)
            if removed:
                logger.warning("Replaced earlier results: %s", ", ".join(removed))
            replaced.add(run.role)
        write_arm(
            results_dir,
            run,
            fingerprint,
            url=config.arm(run.role).url,
            blend_metrics_url=(
                config.blend_metrics_url if run.role is ArmRole.BLEND else ""
            ),
            engine_version=versions[run.role],
            started_at=pass_started,
            finished_at=pass_finished,
        )

    runs = asyncio.run(run_arms(config, plan, senders, blend_counters, on_run=save))

    for run in runs:
        counts = arm_counts(run)
        print(
            f"{run.label}: {counts['answered']} answered, {counts['empty']} "
            f"empty, {run.store_failures} store failures"
        )
    problems = [problem for run in runs for problem in run_problems(run)]
    print(f"next: lmcache bench quality analyze {results_dir}")
    if problems:
        print(f"these passes will make the verdict INVALID: {'; '.join(problems)}")
        raise SystemExit(1)


def _now() -> datetime:
    """Return the current local time, with its UTC offset."""
    return datetime.now().astimezone()
