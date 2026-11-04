# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench quality analyze``: score saved answers and reach a verdict.

Reads only a results directory written by ``bench quality run``: no engine,
tokenizer or dataset is needed, so it runs on any machine -- including after
the arms' GPU servers are gone -- and can be rerun with another metric or
tolerance without rerunning an arm.
"""

# Standard
import argparse
import asyncio
import os

# First Party
from lmcache.cli.commands.bench.quality_bench.core.config import (
    DEFAULT_MIN_REUSE_RATIO,
    JudgeSettings,
    QualityAnalyzeConfig,
    VerdictMetric,
)
from lmcache.cli.commands.bench.quality_bench.core.judge import (
    JUDGE_METRIC,
    AnswerAccuracyJudge,
    OpenAIJudgeClient,
)
from lmcache.cli.commands.bench.quality_bench.core.plan import QualityPlan
from lmcache.cli.commands.bench.quality_bench.core.report import (
    format_summary,
    write_results,
)
from lmcache.cli.commands.bench.quality_bench.core.run_files import (
    JUDGE_CACHE_FILE,
    ResultsDirError,
    exit_on_results_dir_error,
    load_arms,
    load_plan,
)
from lmcache.cli.commands.bench.quality_bench.core.runner import ArmRun
from lmcache.cli.commands.bench.quality_bench.core.verdict import (
    ScoreTable,
    Verdict,
    build_rows,
    decide,
    lexical_scores,
)
from lmcache.logging import init_logger

logger = init_logger(__name__)


def add_analyze_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench quality analyze`` arguments to *parser*.

    Args:
        parser: The subcommand's parser.
    """
    parser.add_argument(
        "results_dir",
        metavar="RESULTS_DIR",
        help="Directory that `lmcache bench quality run` wrote the arms into.",
    )
    verdict = parser.add_argument_group("verdict")
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
        help="Acceptable candidate-vs-reference score difference, in either "
        "direction, in score units (0.05 = 5 points).",
    )
    verdict.add_argument(
        "--min-blend-share",
        type=float,
        default=0.3,
        help="When blend metrics were captured (run --capture-blend-metrics): "
        "minimum share of looked-up tokens served by blending, overall and per "
        "window of questions, for a valid verdict. Not applied to --scenario "
        "concatenated (see --min-reuse-ratio).",
    )
    verdict.add_argument(
        "--min-reuse-ratio",
        type=float,
        default=DEFAULT_MIN_REUSE_RATIO,
        help="For --scenario concatenated, when blend metrics were captured: "
        "minimum ratio of observed reused tokens (prefix and blended) to the "
        "plan's expected reuse, overall and per window of questions, for a "
        "valid verdict.",
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
        help="Judge model name (default: the model the arms served).",
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


def analyze_config_from_args(
    args: argparse.Namespace, served_model: str
) -> QualityAnalyzeConfig:
    """Build the validated config from parsed arguments.

    Args:
        args: Namespace produced by :func:`add_analyze_arguments`.
        served_model: The model the arms served, recorded in ``plan.json``;
            the default judge model.

    Returns:
        The config.

    Raises:
        ValueError: If an argument is invalid.
    """
    if args.verdict_metric:
        metric = VerdictMetric(args.verdict_metric)
    else:
        metric = VerdictMetric.JUDGE if args.judge_url else VerdictMetric.MATCH
    judge = None
    if args.judge_url:
        judge = JudgeSettings(
            url=args.judge_url,
            model=args.judge_model or served_model,
            api_key=args.judge_api_key,
            max_tokens=args.judge_max_tokens,
            concurrency=args.judge_concurrency,
        )
    return QualityAnalyzeConfig(
        results_dir=args.results_dir,
        verdict_metric=metric,
        judge=judge,
        tolerance=args.tolerance,
        min_blend_share=args.min_blend_share,
        min_reuse_ratio=args.min_reuse_ratio,
    )


async def check_judge(settings: JudgeSettings) -> None:
    """Check the judge before scoring, so a bad judge fails fast.

    Beyond reachability, the judge must rate a sanity-check question correctly
    (see :meth:`AnswerAccuracyJudge.check_ratings`): a judge whose replies
    cannot be parsed would otherwise turn the analysis into a silent PASS.

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
    config: QualityAnalyzeConfig, plan: QualityPlan, runs: list[ArmRun]
) -> ScoreTable:
    """Score every run with the configured verdict metric.

    Judgments are cached in the results directory's ``judge_cache.jsonl``,
    so rerunning ``analyze`` judges only what is missing.

    Args:
        config: The analysis's config.
        plan: The plan the runs measured.
        runs: Every arm run.

    Returns:
        Per run label, one score per question.

    Raises:
        RuntimeError: If the judge fails mid-scoring.
    """
    if config.verdict_metric is not VerdictMetric.JUDGE or config.judge is None:
        return lexical_scores(plan, runs, config.verdict_metric)
    cache_path = os.path.join(config.results_dir, JUDGE_CACHE_FILE)
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
            f"The judge failed mid-scoring ({e}). Judgments made so far are "
            f"cached in {cache_path}; rerun `lmcache bench quality analyze` "
            f"to judge only what is missing."
        ) from e
    finally:
        await client.close()


def run_quality_analyze(args: argparse.Namespace) -> None:
    """Entry point for ``lmcache bench quality analyze``.

    Loads the plan and every saved arm pass, scores the answers, reaches
    the verdict, writes ``rows.jsonl`` and ``summary.json`` into the results
    directory, prints the summary, and exits non-zero on a FAIL or INVALID
    verdict so CI can gate on it.

    Args:
        args: Parsed arguments.

    Raises:
        SystemExit: With status 1 on a FAIL or INVALID verdict, or 2 if the
            results directory lacks the plan, the reference arm or the
            candidate arm, or holds passes from another plan (a setup problem, not a
            quality result).
        ValueError: If an argument is invalid.
        RuntimeError: If the judge fails, before or during scoring.
    """
    try:
        plan_record = load_plan(args.results_dir)
        arms = load_arms(args.results_dir, plan_record)
    except ResultsDirError as e:
        exit_on_results_dir_error("analyze", e)
    config = analyze_config_from_args(
        args, served_model=str(plan_record.settings["model"])
    )
    plan = plan_record.plan
    runs = [arm.run for arm in arms]
    print(
        f"bench quality analyze: {len(plan.samples)} questions, "
        f"{len(runs)} passes, metric={config.verdict_metric.value}"
    )
    if config.verdict_metric is VerdictMetric.JUDGE and config.judge is not None:
        asyncio.run(check_judge(config.judge))
        print(f"judge: {config.judge.model} at {config.judge.url} ({JUDGE_METRIC})")

    scores = asyncio.run(score_answers(config, plan, runs))
    verdict = decide(
        plan,
        runs,
        config.verdict_metric,
        config.tolerance,
        config.min_blend_share,
        scores=scores,
        min_reuse_ratio=config.min_reuse_ratio,
    )
    rows = build_rows(plan, runs, config.verdict_metric, scores=scores)
    summary_path = write_results(config, plan_record, arms, verdict, rows)
    for arm in arms:
        print(
            f"{arm.run.label}: {arm.url} (engine {arm.engine_version or 'unknown'}, "
            f"lmcache {arm.lmcache_version}, finished {arm.finished_at})"
        )
    print(format_summary(plan, runs, verdict))
    print(f"Results written to {summary_path}")
    if verdict.verdict in (Verdict.FAIL, Verdict.INVALID):
        raise SystemExit(1)
