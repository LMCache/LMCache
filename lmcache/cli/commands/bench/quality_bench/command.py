# SPDX-License-Identifier: Apache-2.0
"""CLI arguments and orchestration for ``lmcache bench quality``."""

# Standard
from datetime import datetime
import argparse
import asyncio
import os

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
from lmcache.cli.commands.bench.quality_bench.blend_counters import (
    BlendCounterSource,
    PrometheusBlendCounters,
    UnmeasuredBlendCounters,
)
from lmcache.cli.commands.bench.quality_bench.config import (
    ArmEndpoint,
    ArmRole,
    JudgeSettings,
    QualityBenchConfig,
    ReuseScenario,
    TaskType,
    VerdictMetric,
)
from lmcache.cli.commands.bench.quality_bench.judge import (
    JUDGE_METRIC,
    AnswerAccuracyJudge,
    OpenAIJudgeClient,
)
from lmcache.cli.commands.bench.quality_bench.plan import QualityPlan, build_plan
from lmcache.cli.commands.bench.quality_bench.report import (
    create_run_dir,
    format_summary,
    write_answers,
    write_results,
)
from lmcache.cli.commands.bench.quality_bench.runner import (
    ArmRun,
    ArmRunner,
    StorePhase,
)
from lmcache.cli.commands.bench.quality_bench.task import check_task
from lmcache.cli.commands.bench.quality_bench.verdict import (
    ScoreTable,
    Verdict,
    build_rows,
    decide,
    lexical_scores,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Judgments are cached here, directly under --output-dir rather than in a run's
# subdirectory, so every later run for the same --output-dir reuses them.
JUDGE_CACHE_FILE = "judge_cache.jsonl"


def add_quality_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench quality`` arguments to *parser*.

    Args:
        parser: The subcommand's parser.
    """
    arms = parser.add_argument_group("arms")
    arms.add_argument(
        "--reference-url",
        required=True,
        help="Engine serving full recompute (vLLM, prefix caching off).",
    )
    arms.add_argument(
        "--blend-url",
        required=True,
        help="Engine serving CacheBlend (vLLM + CBKVConnector).",
    )
    arms.add_argument(
        "--apc-url",
        default="",
        help="Optional engine serving vLLM prefix caching only. Its gap to "
        "the reference is reported as a noise floor.",
    )
    arms.add_argument(
        "--blend-metrics-url",
        default="",
        help="The blend server's Prometheus /metrics URL. Without it, "
        "engagement is not checked and cannot make the verdict INVALID.",
    )

    workload = parser.add_argument_group("workload")
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

    verdict = parser.add_argument_group("verdict")
    verdict.add_argument(
        "--reference-repeats",
        type=int,
        default=2,
        help="Reference runs. Two or more report a flip-rate noise floor.",
    )
    verdict.add_argument(
        "--verdict-metric",
        choices=[m.value for m in VerdictMetric],
        default=None,
        help="judge: LLM-judge answer accuracy (RAGAS AnswerAccuracy), the "
        "reference metric; match: answer states a gold answer, a judge-free "
        "screen; f1: token F1. Default: judge with --judge-url, else match.",
    )
    verdict.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Acceptable blend-vs-reference score difference, in either "
        "direction, in score units (0.05 = 5 points).",
    )
    verdict.add_argument(
        "--min-blend-share",
        type=float,
        default=0.3,
        help="Minimum share of looked-up tokens served by blending, overall "
        "and per window of questions, for a valid verdict.",
    )
    judge = parser.add_argument_group("judge")
    judge.add_argument(
        "--judge-url",
        default="",
        help="OpenAI-compatible endpoint serving the judge model. Use a "
        "separate deployment from the arms under test.",
    )
    judge.add_argument(
        "--judge-model",
        default="",
        help="Judge model name (default: --model).",
    )
    judge.add_argument(
        "--judge-api-key",
        default=os.environ.get("LMCACHE_JUDGE_API_KEY", "EMPTY"),
        help="API key for the judge endpoint (default: $LMCACHE_JUDGE_API_KEY).",
    )
    judge.add_argument(
        "--judge-max-tokens",
        type=int,
        default=4096,
        help="Completion budget per judge call, reasoning included.",
    )
    judge.add_argument(
        "--judge-concurrency",
        type=int,
        default=8,
        help="Answers judged at once.",
    )
    parser.add_argument(
        "--output-dir",
        default="quality_results",
        help="Results root. Each run writes into a new timestamped "
        "subdirectory. The judge cache is shared at the top level.",
    )


def config_from_args(args: argparse.Namespace) -> QualityBenchConfig:
    """Build the validated config from parsed arguments.

    Args:
        args: Namespace produced by :func:`add_quality_arguments`.

    Returns:
        The config.

    Raises:
        ValueError: If an argument is invalid.
    """
    arms = [
        ArmEndpoint(ArmRole.REFERENCE, args.reference_url),
        ArmEndpoint(ArmRole.BLEND, args.blend_url),
    ]
    if args.apc_url:
        arms.append(ArmEndpoint(ArmRole.PREFIX_CACHE, args.apc_url))
    if args.verdict_metric:
        metric = VerdictMetric(args.verdict_metric)
    else:
        metric = VerdictMetric.JUDGE if args.judge_url else VerdictMetric.MATCH
    judge = None
    if args.judge_url:
        judge = JudgeSettings(
            url=args.judge_url,
            model=args.judge_model or args.model,
            api_key=args.judge_api_key,
            max_tokens=args.judge_max_tokens,
            concurrency=args.judge_concurrency,
        )
    return QualityBenchConfig(
        model=args.model,
        dataset=args.dataset,
        arms=arms,
        scenario=ReuseScenario(args.scenario),
        num_samples=args.limit,
        chunk_size=args.chunk_size,
        reference_repeats=args.reference_repeats,
        max_output_tokens=args.max_output_tokens,
        template_kwargs=parse_template_kwargs(args.template_kwargs),
        verdict_metric=metric,
        judge=judge,
        task=TaskType(args.task),
        tolerance=args.tolerance,
        min_blend_share=args.min_blend_share,
        blend_metrics_url=args.blend_metrics_url,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def build_quality_plan(config: QualityBenchConfig) -> QualityPlan:
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
    config: QualityBenchConfig,
    plan: QualityPlan,
    senders: dict[ArmRole, RequestSender],
    blend_counters: BlendCounterSource,
) -> list[ArmRun]:
    """Run every configured arm over *plan*, one pass at a time.

    Order: every reference pass, then apc (if configured), then blend. The
    caching arms store the plan's passages first -- the reference does not.
    Each arm must start from an empty cache: restart its engine (and the
    blend server) before the run.

    Args:
        config: The run's config.
        plan: The prompts to send.
        senders: One request sender per configured arm role. Each is closed
            when its arm finishes.
        blend_counters: Engagement counters for the blend arm.

    Returns:
        Every arm run, in the order run.
    """
    runs: list[ArmRun] = []
    unmeasured = UnmeasuredBlendCounters()
    schedule: list[tuple[ArmRole, int, StorePhase, BlendCounterSource]] = [
        (ArmRole.REFERENCE, k, StorePhase.SKIP_STORE, unmeasured)
        for k in range(config.reference_repeats)
    ]
    if config.has_arm(ArmRole.PREFIX_CACHE):
        schedule.append((ArmRole.PREFIX_CACHE, 0, StorePhase.STORE_FIRST, unmeasured))
    schedule.append((ArmRole.BLEND, 0, StorePhase.STORE_FIRST, blend_counters))

    runners: dict[ArmRole, ArmRunner] = {}
    try:
        for role, run_index, store_phase, counters in schedule:
            if role not in runners:
                runners[role] = ArmRunner(
                    senders[role], config.max_output_tokens, counters
                )
            logger.info("Running %s pass %d", role.value, run_index + 1)
            runs.append(await runners[role].run(plan, role, run_index, store_phase))
    finally:
        for sender in senders.values():
            await sender.close()
    return runs


async def check_judge(settings: JudgeSettings) -> None:
    """Check the judge before the arms run, so a bad judge costs no arm time.

    Beyond reachability, the judge must rate a sanity-check question correctly
    (see :meth:`AnswerAccuracyJudge.check_ratings`): a judge whose replies
    cannot be parsed would otherwise turn the run into a silent PASS.

    Args:
        settings: The judge to check.

    Raises:
        RuntimeError: If the judge cannot be reached or misrates the sanity check.
    """
    client = OpenAIJudgeClient(settings)
    try:
        await AnswerAccuracyJudge(client, settings.model, 1).check_ratings()
    finally:
        await client.close()


async def score_answers(
    config: QualityBenchConfig, plan: QualityPlan, runs: list[ArmRun], run_dir: str
) -> ScoreTable:
    """Score every run with the configured verdict metric.

    Args:
        config: The run's config.
        plan: The plan the runs measured.
        runs: Every arm run.
        run_dir: The run's directory, holding its answers. Named in the
            error if the judge fails.

    Returns:
        Per run label, one score per question.

    Raises:
        RuntimeError: If the judge fails mid-run.
    """
    if config.verdict_metric is not VerdictMetric.JUDGE or config.judge is None:
        return lexical_scores(plan, runs, config.verdict_metric)
    cache_path = os.path.join(config.output_dir, JUDGE_CACHE_FILE)
    client = OpenAIJudgeClient(config.judge)
    try:
        judge = AnswerAccuracyJudge(
            client,
            config.judge.model,
            config.judge.concurrency,
            cache_path=cache_path,
        )
        return await judge.score_runs(plan, runs)
    except RuntimeError as e:
        raise RuntimeError(
            f"The judge failed mid-run ({e}). The arms' answers are in "
            f"{run_dir}. Judgments made so far are cached in "
            f"{cache_path}, so a rerun into the same --output-dir re-runs the "
            f"arms but only judges what is missing."
        ) from e
    finally:
        await client.close()


def run_quality_bench(args: argparse.Namespace) -> None:
    """Entry point for ``lmcache bench quality``.

    Builds the plan, runs every arm, writes the answers, scores them,
    writes the results, prints the summary, and exits non-zero on a FAIL or
    INVALID verdict so CI can gate on it. Files go in a new subdirectory of
    ``--output-dir`` named for the run's start time.

    Args:
        args: Parsed arguments.

    Raises:
        SystemExit: With status 1 on a FAIL or INVALID verdict.
        RuntimeError: If the judge fails, before or during scoring, or the
            blend-server metrics cannot be read.
    """
    started = datetime.now()
    config = config_from_args(args)
    plan = build_quality_plan(config)
    print(
        f"bench quality: {len(plan.samples)} questions, "
        f"{len(plan.store_requests)} store requests, "
        f"scenario={plan.scenario.value}, metric={config.verdict_metric.value}"
    )
    if config.verdict_metric is VerdictMetric.JUDGE and config.judge is not None:
        asyncio.run(check_judge(config.judge))
        print(f"judge: {config.judge.model} at {config.judge.url} ({JUDGE_METRIC})")
    blend_counters: BlendCounterSource = (
        PrometheusBlendCounters(config.blend_metrics_url)
        if config.blend_metrics_url
        else UnmeasuredBlendCounters()
    )
    # The counters are first needed when the blend pass starts, after every
    # reference pass. Read them now so a bad --blend-metrics-url fails before
    # any arm time is spent / wasted.
    blend_counters.read()

    extra_body: dict[str, object] = {}
    if config.template_kwargs:
        extra_body["chat_template_kwargs"] = dict(config.template_kwargs)
    senders = {
        arm.role: RequestSender(arm.url, config.model, extra_body=extra_body)
        for arm in config.arms
    }
    runs = asyncio.run(run_arms(config, plan, senders, blend_counters))
    run_dir = create_run_dir(config.output_dir, started)
    write_answers(run_dir, plan, runs)

    scores = asyncio.run(score_answers(config, plan, runs, run_dir))
    verdict = decide(
        plan,
        runs,
        config.verdict_metric,
        config.tolerance,
        config.min_blend_share,
        scores=scores,
    )
    rows = build_rows(plan, runs, config.verdict_metric, scores=scores)
    summary_path = write_results(run_dir, config, plan, runs, verdict, rows)
    print(format_summary(plan, runs, verdict))
    print(f"Results written to {summary_path}")
    if verdict.verdict in (Verdict.FAIL, Verdict.INVALID):
        raise SystemExit(1)
