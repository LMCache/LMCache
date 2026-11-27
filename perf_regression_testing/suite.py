# SPDX-License-Identifier: Apache-2.0
"""Run a whole test suite: images, every (workload, deployment) run, report.

Usage (repo root, Python >= 3.11, docker on the host, env as in config.toml)::

    python -m perf_regression_testing.suite --suite nightly [--out <dir>]
        [--deployments vllm_simple,...]

Flow::

    lock <host.root>/prt-suite.lock (suites share the L2 dir)
    -> LMCache under test: this checkout (Buildkite's commit)
    -> tonight's images (runner/images.py), mooncake if needed, AIPerf
    -> any model missing from the HF cache, downloaded (runner/hf.py)
    -> runner/run.py for each pair of [suites.<name>], one after another
    -> each valid run vs its own history (runner/baseline.py), then into history
    -> <out>/report.json and report.md

Exit codes: 0 every pair valid and none regressed; 1 a gated pair regressed;
2 a pair was invalid, no usable images, or the host lacks an env var the
suite needs; 75 another suite held the lock past --lock-wait-s.
A non-gating suite returns 0 whenever every pair produced a valid result.
"""

# Standard
from pathlib import Path
from typing import IO, Any
import argparse
import fcntl
import json
import logging
import signal
import sys
import time
import traceback

# First Party
from perf_regression_testing.runner import baseline, docker, hf, images
from perf_regression_testing.runner.config import (
    PKG_DIR,
    HostSetupError,
    check_host,
    display,
    load,
    missing_models,
    resolve,
)
from perf_regression_testing.runner.run import EXIT_VALID, Cancelled, run

# stdlib logging: PRT runs on the host, where LMCache is not installed.
logger = logging.getLogger(__name__)

EXIT_SUITE_FAIL, EXIT_SUITE_INVALID, EXIT_LOCKED = 1, 2, 75
REPORT_COLS = [
    ("decode_p10", "decode p10 tok/s"),
    ("ttft_p95", "TTFT p95 s"),
    ("preemptions", "preemptions"),
    ("spec_accept_len", "spec decode accept length"),
    ("external_token_share", "cache-layer token share"),
    ("cache_hit_share", "cache hit share"),
    ("ideal_hit_share", "ideal hit share"),
    ("l2_disk_used_gb", "L2 disk used GB"),
    ("disk_write_gb", "disk written GB"),
    ("disk_read_gb", "disk read GB"),
    ("disk_write_gb_s_steady", "disk write GB/s, steady"),
    ("disk_write_gb_s", "disk write GB/s, whole run"),
    ("disk_read_per_write", "disk read / written"),
    ("kv_cache_tokens", "GPU KV cache tokens"),
]


def _fmt(v: Any) -> str:
    return f"{v:.4g}" if isinstance(v, float) else "" if v is None else str(v)


def _vs_history(entry: dict[str, Any]) -> str:
    """One cell: each metric against this pair's own regime, shifts spelled out."""
    b = entry.get("vs_history")
    if b is None:
        return ""
    cell = _changes(b["metrics"])
    n = max((c.get("n") or 0 for c in b["metrics"].values()), default=0)
    if n < b["min_runs"]:
        cell += f" (n={n}, gate can fire from n={b['min_runs']})"
    else:
        cell += f" (n={n})"
    return cell


def _changes(metrics: dict[str, Any]) -> str:
    """Each metric's change vs its regime median; shifts marked."""
    parts = []
    for m, c in metrics.items():
        if c.get("change") is None:
            continue
        part = f"{m} {c['change']:+.1%}"
        if c["shift"]:
            part += f" [{c['shift'].upper()}{'!' if c['regressed'] else ''}]"
        parts.append(part)
    return ", ".join(parts)


def report_md(rep: dict[str, Any]) -> str:
    """Render the suite report as a Markdown table.

    Args:
        rep: The ``report.json`` content.

    Returns:
        Markdown text.
    """
    im = rep["images"]
    vllm = f"vLLM {(im.get('vllm_commit') or '?')[:9]}"
    if im.get("error"):
        vllm += f" (NOT RUN: {im['error']})"
    elif im.get("vllm_source") == "pinned":
        since = time.strftime("%Y-%m-%d", time.localtime(im["pinned_since"]))
        vllm += f" (pinned since {since})"
    elif im.get("vllm_source") == "nightly":
        vllm += " (tonight's nightly, NEW vLLM)"
    lines = [
        f"# PRT {rep['suite']} {rep['date']}",
        "",
        vllm
        + (f"; LMCache {im['lmcache_commit'][:9]}" if im.get("lmcache") else "")
        + (
            f"; Mooncake {im['mooncake_commit'][:9]}"
            f" (nightly built {im['mooncake_built']})"
            if im.get("mooncake_commit")
            else ""
        ),
        "",
        "| workload | deployment | verdict | "
        + " | ".join(c[1] for c in REPORT_COLS)
        + " | vs own history |",
        "|---|---|---|" + "---|" * len(REPORT_COLS) + "---|",
    ]
    for r in rep["runs"]:
        s = r["summary"]
        worse = [
            m
            for m, c in r.get("vs_history", {}).get("metrics", {}).items()
            if c["regressed"]
        ]
        if s["invalid"]:
            verdict = "INVALID: " + "; ".join(s["invalid"])
        elif worse:
            verdict = "REGRESSED: " + ", ".join(worse)
        else:
            verdict = "pass"
        cells = [_fmt(s.get(k)) for k, _ in REPORT_COLS] + [_vs_history(r)]
        lines.append(
            f"| {r['workload']} | {r['deployment']} | {verdict} | "
            + " | ".join(cells)
            + " |"
        )
    return "\n".join(lines) + "\n"


def suite(name: str, out: Path, deployments: list[str] | None = None) -> int:
    """Run suite ``name`` into ``out``; see the module docstring.

    Args:
        name: Key under ``[suites]``.
        out: Output dir; must not exist.
        deployments: Run only these of the suite's deployments, in this
            order; None runs all of them. The caller checks they belong to
            the suite.

    Returns:
        Exit code.
    """
    cfg = load()
    plan = cfg["suites"][name]
    deps = deployments or plan["deployments"]
    out.mkdir(parents=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rep: dict[str, Any] = {"suite": name, "date": stamp, "runs": []}

    logger.info("suite %s, output %s", name, out)

    def finish(code: int) -> int:
        if code >= 0:
            logger.info("suite %s done, exit %d", name, code)
        rep["exit_code"] = code
        (out / "report.json").write_text(json.dumps(rep, indent=1))
        (out / "report.md").write_text(report_md(rep))
        return code

    try:
        check_host(cfg, deps)
    except HostSetupError as e:
        rep["images"] = {"usable": False, "error": str(e)}
        return finish(EXIT_SUITE_INVALID)
    free_gb, min_gb = docker.root_free_gb(), cfg["host"]["min_docker_free_gb"]
    if free_gb < min_gb:
        rep["images"] = {
            "usable": False,
            "error": f"docker root fs {free_gb:.1f} GB free < {min_gb} GB",
        }
        return finish(EXIT_SUITE_INVALID)

    targets = {cfg["deployments"][d]["image_target"] for d in deps}
    root = Path(cfg["host"]["root"])
    rep["images"] = im = images.tonight(
        root, PKG_DIR.parent, out / "images", need_lmcache="lmcache" in targets
    )
    if not im["usable"]:
        return finish(EXIT_SUITE_INVALID)
    by_target = {"serving": im["serving"]}
    if im.get("lmcache"):
        by_target["lmcache"] = im["lmcache"]
    # A broken competitor image invalidates only its own pairs.
    mooncake_error = None
    if "mooncake" in targets:
        try:
            by_target["mooncake"] = mc = images.build(
                ["mooncake", im["serving"]], out / "build_mooncake.log"
            )
            for k in ("commit", "built"):
                im[f"mooncake_{k}"] = docker.image_label(mc, f"prt.mooncake.{k}")
        except RuntimeError as e:
            mooncake_error = str(e)
    try:
        aiperf = images.build(["aiperf"], out / "build_aiperf.log")
    except RuntimeError as e:
        im["usable"] = False
        im["error"] = f"aiperf image: {e}"
        return finish(EXIT_SUITE_INVALID)
    # Before downloading, so a replaced model frees its disk first. Failing to
    # delete one costs only disk, so the runs go ahead.
    try:
        rep["pruned_models"] = hf.prune(
            cfg["host"]["hf_cache"],
            cfg["host"]["root"],
            list(cfg["models"].values()),
        )
    except OSError as e:
        logger.exception("pruning old models failed")
        rep["prune_error"] = str(e)
    # A failed download invalidates only the pairs of that model.
    model_errors = {}
    for model in missing_models(cfg, deps):
        try:
            hf.fetch(
                cfg["models"][model],
                cfg["host"]["hf_cache"],
                im["serving"],
                cfg["container"],
                out / f"download_{model}.log",
            )
        except RuntimeError as e:
            model_errors[model] = str(e)
    # Likewise a trace dir that was never built.
    trace_errors = {}
    traces = {resolve(cfg, w, d).trace_dir for w in plan["workloads"] for d in deps}
    for trace_dir in sorted(t for t in traces if not t.is_dir()):
        try:
            hf.fetch_trace(
                trace_dir,
                cfg["host"]["hf_cache"],
                im["serving"],
                cfg["container"],
                out / f"download_{trace_dir.name}.log",
            )
        except RuntimeError as e:
            trace_errors[trace_dir] = str(e)
    reg = cfg["regression"]
    alpha = reg["false_alarm_rate"] / len(reg["metrics"])
    hist_path = root / "prt-results" / "history.jsonl"
    history = baseline.load_history(hist_path)
    codes = []
    regressed = False
    for w in plan["workloads"]:
        for d in deps:
            spec = resolve(cfg, w, d)
            run_dir = out / f"{stamp}-{w}-{d}"
            model = cfg["deployments"][d]["model"]
            shown = display(cfg, d, plan["gate"])
            skip = None
            if spec.image_target == "mooncake" and mooncake_error:
                skip = f"no mooncake image: {mooncake_error}"
            elif model in model_errors:
                skip = f"no {model} snapshot: {model_errors[model]}"
            elif spec.trace_dir in trace_errors:
                skip = f"no {spec.trace_dir.name}: {trace_errors[spec.trace_dir]}"
            if skip:
                # The page keys a run's setup on these, as run() records them.
                summary = {
                    "model": spec.served_name,
                    "concurrency": spec.aiperf_args["concurrency"],
                    "invalid": [skip],
                }
                rep["runs"].append(
                    {
                        "workload": w,
                        "deployment": d,
                        "summary": summary,
                        "display": shown,
                    }
                )
                codes.append(EXIT_SUITE_INVALID)
                continue
            logger.info("run %s/%s", w, d)
            code = run(spec, run_dir, by_target[spec.image_target], aiperf)
            summary = json.loads((run_dir / "summary.json").read_text())
            entry = {
                "workload": w,
                "deployment": d,
                "summary": summary,
                "display": shown,
            }
            if not summary["invalid"]:
                record = {
                    "run_id": run_dir.name,
                    "date": stamp,
                    "suite": name,
                    "workload": w,
                    "deployment": d,
                    "vllm_commit": summary["vllm_commit"],
                    "lmcache_commit": summary["lmcache_commit"],
                    "config_hash": baseline.config_hash(spec),
                    "metrics": {m: summary.get(m) for m in reg["metrics"]},
                }
                entry["vs_history"] = res = baseline.check(
                    history, record, reg["metrics"], alpha
                )
                worse = [m for m, c in res["metrics"].items() if c["regressed"]]
                logger.info("%s/%s vs history: %s", w, d, worse or "no regression")
                regressed |= bool(worse)
                baseline.append(hist_path, {**record, "regimes": res["regimes"]})
            rep["runs"].append(entry)
            codes.append(code)
            # Partial reports survive a crash of a later pair.
            finish(-1)
    rep["pruned_images"] = images.prune(list(by_target.values()))
    if any(c != EXIT_VALID for c in codes):
        return finish(EXIT_SUITE_INVALID)
    # Only a vLLM that LMCache ran on is pinned for the week.
    if "lmcache" in im:
        images.pin(root, im)
    if plan["gate"] and regressed:
        return finish(EXIT_SUITE_FAIL)
    return finish(EXIT_VALID)


def lock(path: Path, wait_s: float) -> IO[str] | None:
    """Take the suite lock, waiting up to ``wait_s``.

    Args:
        path: Lock file; created if missing.
        wait_s: Longest wait for another suite to finish.

    Returns:
        The open lock file (held until closed or the process exits), or None
        when the wait timed out.
    """
    f = open(path, "w")
    deadline = time.monotonic() + wait_s
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except BlockingIOError:
            if time.monotonic() >= deadline:
                f.close()
                return None
            time.sleep(10)


def main() -> None:
    """Parse arguments, take the suite lock and run the suite."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--suite", required=True, help="key under [suites]")
    ap.add_argument(
        "--out",
        type=Path,
        help="must not exist; default <host.root>/prt-suites/<suite>_<UTC time>",
    )
    ap.add_argument(
        "--deployments",
        help="comma-separated subset of the suite's deployments; default all",
    )
    ap.add_argument("--lock-wait-s", type=float, default=3600)
    a = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [prt] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Buildkite stops a timed-out or cancelled step with SIGTERM, whose default
    # action skips the finally blocks that remove containers and shm regions.
    def cancel(*_: Any) -> None:
        raise Cancelled(EXIT_SUITE_INVALID)

    signal.signal(signal.SIGTERM, cancel)
    try:
        cfg = load()
        if a.suite not in cfg["suites"]:
            # Exits 2, a soft fail like any other unusable suite.
            ap.error(f"--suite {a.suite!r}: choose from {', '.join(cfg['suites'])}")
        known = cfg["suites"][a.suite]["deployments"]
        deployments = a.deployments.split(",") if a.deployments else None
        if deployments is not None and not set(deployments) <= set(known):
            ap.error(
                f"--deployments {a.deployments!r}: {a.suite} runs {', '.join(known)}"
            )
        root = Path(cfg["host"]["root"])
        held = lock(root / "prt-suite.lock", a.lock_wait_s)
        if held is None:
            print(f"another suite still holds {root}/prt-suite.lock", file=sys.stderr)
            sys.exit(EXIT_LOCKED)
        # With the lock held no other suite runs, so any PRT container is left
        # by a suite killed before its teardown (a job lost after SIGTERM), and
        # still holds GPUs and pinned memory.
        stale = docker.prt_containers()
        if stale:
            logger.warning("removing containers of a dead suite: %s", stale)
            docker.remove(stale, stop_s=0)
        out = a.out or root / "prt-suites" / time.strftime(
            f"{a.suite}_%Y%m%d-%H%M%S", time.gmtime()
        )
        code = suite(a.suite, out, deployments)
        print(f"report: {out}/report.json (exit {code})")
    except Exception:
        # A crash would exit 1, which means "regressed".
        traceback.print_exc()
        code = EXIT_SUITE_INVALID
    sys.exit(code)


if __name__ == "__main__":
    main()
