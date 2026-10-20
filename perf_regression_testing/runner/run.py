# SPDX-License-Identifier: Apache-2.0
"""Run one (workload, deployment) pair end to end and write its summary.

Flow::

    pick idle GPUs -> trace pinned? -> wipe scratch -> start services -> ready?
    -> start vLLM -> ready? -> sampler on -> AIPerf
    -> services still up? -> stop all -> record scratch usage -> wipe scratch
    -> validity + score

Output in the run dir: ``meta.json`` (images, commits, resolved settings),
``<service>.log``, ``vllm.log``, ``aiperf.log``, ``aiperf/``,
``samples.jsonl``, ``summary.json`` (metrics, validity).

Exit codes: 0 valid run, 2 invalid run, 3 GPUs busy, 4 a server did not
come up, 5 trace or scratch check failed.
"""

# Standard
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging
import re
import shutil
import subprocess
import time

# First Party
from perf_regression_testing.runner import docker
from perf_regression_testing.runner.config import (
    PKG_DIR,
    RunSpec,
    cli_args,
    substitute,
)
from perf_regression_testing.runner.disk_cap import (
    DiskCap,
    allocated_bytes,
    set_dirs_writable,
)
from perf_regression_testing.runner.sampler import Sampler
from perf_regression_testing.runner.score import score
from perf_regression_testing.workloads.agentic.fetch_trace import dir_digest

# stdlib logging: PRT runs on the host, where LMCache is not installed.
logger = logging.getLogger(__name__)

EXIT_VALID, EXIT_INVALID, EXIT_GPU_BUSY, EXIT_NOT_UP, EXIT_INPUT = 0, 2, 3, 4, 5
SHM_DIR = Path("/dev/shm")
# AIPerf runs for up to hours; log that it is alive this often.
_HEARTBEAT_S = 300.0
SHM_REGION = re.compile(r"Created mmap file /dev/shm/(vllm_offload_[0-9a-f-]+\.mmap)")
# "<HH:MM:SS>.<ms> <LEVEL> Phase <name> [(<kind>)] <event> | <fields> (<file>:<line>)"
AIPERF_PHASE = re.compile(
    r"(\d\d):(\d\d):(\d\d)\.\d+ +[A-Z]+ +"
    r"Phase (\S+) (?:\(\S+\) )?([^(\r\n]+?) \(\w+\.py:\d+\)"
)
AIPERF_DURATION = re.compile(r"target: ([\d.]+)s duration")


class Cancelled(SystemExit):
    """The CI job is being cancelled or timed out (SIGTERM).

    Buildkite kills the job about 10 s after its SIGTERM, too soon for a
    graceful container stop, so a run interrupted by this kills its containers
    at once rather than leave them holding the GPUs.
    """


def _idle_gpus() -> list[int]:
    """Return the indices of the GPUs with no compute process.

    Returns:
        GPU indices, ascending.
    """

    def query(arg: str) -> list[str]:
        return subprocess.run(
            ["nvidia-smi", arg, "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()

    busy = {u.strip() for u in query("--query-compute-apps=gpu_uuid")}
    idle = []
    for row in query("--query-gpu=index,uuid"):
        index, uuid = row.split(",")
        if uuid.strip() not in busy:
            idle.append(int(index))
    return sorted(idle)


def pick_gpus(count: int, wait_s: float) -> list[int] | None:
    """Wait until ``count`` GPUs have no compute process and return them.

    The same GPUs must be idle on two polls 10 s apart, so a GPU another job
    is still loading onto is not taken.

    Args:
        count: GPUs needed.
        wait_s: Give up after this many seconds.

    Returns:
        The first ``count`` idle GPU indices, or None on timeout.
    """
    deadline = time.time() + wait_s
    prev: list[int] = []
    while True:
        idle = _idle_gpus()
        if len(idle) >= count and idle[:count] == prev[:count]:
            return idle[:count]
        prev = idle
        if time.time() > deadline:
            return None
        time.sleep(10)


def _trace_ok(spec: RunSpec) -> str | None:
    """Return None when the trace matches its pin, else the reason."""
    got = dir_digest(str(spec.trace_dir))
    want = spec.trace["dir_digest"]
    return None if got == want else f"trace digest {got} != pinned {want}"


def _wipe(path: Path) -> None:
    # A DiskCap freeze leaves the dirs read-only, and rmtree would skip them.
    set_dirs_writable(path, True)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)


def _scratch_dirs(spec: RunSpec) -> list[dict[str, Any]]:
    return [*spec.vllm_scratch, *(d for s in spec.services for d in s.scratch)]


def prepare_scratch(spec: RunSpec) -> str | None:
    """Wipe every scratch dir (services' and vLLM's) and check its free space.

    Args:
        spec: Resolved run settings.

    Returns:
        None when all dirs are ready, else the reason.
    """
    for d in _scratch_dirs(spec):
        p = Path(d["path"])
        _wipe(p)
        free_gb = shutil.disk_usage(p).free / 2**30
        if free_gb < d["min_free_gb"]:
            return f"{p}: {free_gb:.0f} GiB free"
    return None


def wipe_scratch(spec: RunSpec) -> None:
    """Wipe every scratch dir (after a run).

    Args:
        spec: Resolved run settings.
    """
    for d in _scratch_dirs(spec):
        _wipe(Path(d["path"]))


def remove_shm_regions(out: Path) -> list[str]:
    """Unlink the CPU offload regions this run's vLLM created in /dev/shm.

    vLLM unlinks them only on a clean exit, and a leftover region (800 GiB
    for vllm_native) makes the next run fail to allocate. Only paths named
    in this run's vllm.log are removed: other vLLMs share the host.

    Args:
        out: Run dir holding ``vllm.log``.

    Returns:
        The file names unlinked.
    """
    log = out / "vllm.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    names = sorted(set(SHM_REGION.findall(text)))
    for n in names:
        (SHM_DIR / n).unlink(missing_ok=True)
    return names


def aiperf_phase(log: Path, now: datetime) -> str:
    """Return AIPerf's latest phase event, for the heartbeat.

    AIPerf logs each phase's start and end, and warmup progress every 30 s.
    Profiling logs nothing until it ends, so while a fixed-duration phase runs,
    the time since its start line is appended.

    Args:
        log: AIPerf's stdout log; its times are UTC wall clock.
        now: The current time, timezone-aware.

    Returns:
        E.g. ``"warmup progress | returned=41/207 | ... | errors=0"``,
        ``"profiling started | ... target: 960.0s duration (6 of 16 min)"``, or
        ``"loading the trace"`` before the first phase starts.
    """
    hits = AIPERF_PHASE.findall(log.read_text(errors="replace"))
    if not hits:
        return "loading the trace"
    h, m, sec, name, event = hits[-1]
    text = f"{name} {event}"
    target = AIPERF_DURATION.search(event)
    if event.startswith("started") and target:
        utc = now.astimezone(timezone.utc)
        start = utc.replace(hour=int(h), minute=int(m), second=int(sec), microsecond=0)
        # The log has no date: a start "after" now was yesterday.
        elapsed = (utc - start).total_seconds() % 86400
        text += f" ({elapsed / 60:.0f} of {float(target[1]) / 60:.0f} min)"
    return text


def start_servers(
    spec: RunSpec, cmds: dict[str, list[str]], out: Path, run_id: str
) -> str | None:
    """Write the deployment's files, start the services, then vLLM.

    Args:
        spec: Resolved run settings.
        cmds: Output of :func:`build_commands`.
        out: Run dir; each container logs to ``<role>.log`` there.
        run_id: Run id (container names).

    Returns:
        None when all are up, else the reason.
    """
    for name, content in spec.files.items():
        (out / name).write_text(json.dumps(content, indent=1))
    for s in spec.services:
        logger.info("starting %s", s.name)
        docker.start(cmds[s.name], out / f"{s.name}.log")
        if not docker.wait_http(
            s.ready_url, s.ready_timeout_s, f"prt-{run_id}-{s.name}"
        ):
            return f"{s.name} not up"
    logger.info("starting vLLM (log: %s)", out / "vllm.log")
    t0 = time.monotonic()
    docker.start(cmds["vllm"], out / "vllm.log")
    if not docker.wait_http(
        f"http://127.0.0.1:{spec.vllm_port}/v1/models",
        spec.vllm_startup_timeout_s,
        f"prt-{run_id}-vllm",
    ):
        return "vllm not up"
    logger.info("vLLM up after %.0f s", time.monotonic() - t0)
    return None


def _git_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(PKG_DIR), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def build_commands(
    spec: RunSpec,
    gpus: list[int],
    run_id: str,
    out: Path,
    image: str,
    aiperf_image: str,
) -> dict[str, list[str]]:
    """Build the ``docker run`` command of every container of the run.

    Args:
        spec: Resolved run settings.
        gpus: GPU indices for the GPU containers (:func:`pick_gpus`).
        run_id: Run id (names and labels the containers).
        out: Run output dir (mounted at the same path).
        image: Deployment image (vLLM + cache layer).
        aiperf_image: AIPerf image.

    Returns:
        Container role -> full ``docker run ... image cmd`` argument list, in
        start order: services, ``vllm``, ``aiperf``. ``<artifact_dir>`` in
        env values becomes ``out``.
    """
    subst = {"<artifact_dir>": str(out), "<trace_dir>": str(spec.trace_dir)}
    mounts = [
        {**m, "path": subst.get(m["path"], m["path"])} for m in spec.container["mounts"]
    ]
    # Docker would create a missing one owned by root, unwritable for us.
    for m in mounts:
        if not m.get("ro"):
            Path(m["path"]).mkdir(parents=True, exist_ok=True)
    cmds = {}
    for s in spec.services:
        scratch = [{"path": d["path"]} for d in s.scratch]
        cmds[s.name] = (
            docker.run_args(
                f"prt-{run_id}-{s.name}",
                run_id,
                image,
                spec.container,
                mounts + scratch,
                substitute(s.env, subst),
                gpus,
                spec.devices,
                spec.cap_add,
                s.seccomp,
            )
            + s.command
            + cli_args(s.args)
        )
    cmds["vllm"] = (
        docker.run_args(
            f"prt-{run_id}-vllm",
            run_id,
            image,
            spec.container,
            mounts + [{"path": d["path"]} for d in spec.vllm_scratch],
            substitute(spec.vllm_env, subst),
            gpus,
            spec.devices,
            spec.cap_add,
            # Without it a root container ignores DiskCap's read-only dirs.
            cap_drop=(
                ("DAC_OVERRIDE",)
                if any("cap_gb" in d for d in spec.vllm_scratch)
                else ()
            ),
            memory_gb=spec.vllm_memory_gb,
        )
        + [
            "vllm",
            "serve",
            spec.model_path,
            "--served-model-name",
            spec.served_name,
            "--port",
            str(spec.vllm_port),
        ]
        + cli_args(spec.vllm_args)
    )
    aiperf_out = out / "aiperf"
    env = {
        **spec.aiperf_env,
        "AIPERF_DATASET_MMAP_BASE_PATH": str(out / "aiperf_mmap" / "base"),
        "AIPERF_DATASET_MMAP_CACHE_DIR": str(out / "aiperf_mmap" / "cache"),
        "PRT_LIVE_HISTORY_LOG_DIR": str(aiperf_out / "prt_live_history"),
    }
    cmds["aiperf"] = docker.run_args(
        f"prt-{run_id}-aiperf",
        run_id,
        aiperf_image,
        spec.container,
        mounts,
        env,
        None,
        [],
        [],
    ) + [
        "aiperf",
        "profile",
        *cli_args(spec.aiperf_args),
        "--artifact-dir",
        str(aiperf_out),
    ]
    return cmds


def run(spec: RunSpec, out: Path, image: str, aiperf_image: str) -> int:
    """Execute the run; see the module docstring for the flow and outputs.

    Args:
        spec: Resolved run settings.
        out: Run output dir; must not exist.
        image: Deployment image.
        aiperf_image: AIPerf image.

    Returns:
        Exit code (``EXIT_*``).
    """
    run_id = out.name
    out.mkdir(parents=True)
    summary: dict[str, Any] = {
        "workload": spec.workload,
        "deployment": spec.deployment,
        "invalid": [],
    }

    def finish(code: int) -> int:
        summary["exit_code"] = code
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        return code

    logger.info("%s: waiting for %d idle GPUs", run_id, spec.gpu_count)
    gpus = pick_gpus(spec.gpu_count, spec.gpu_busy_wait_s)
    if gpus is None:
        # Without a reason the suite would record a run that never ran.
        summary["invalid"].append(f"GPUs busy: fewer than {spec.gpu_count} free")
        return finish(EXIT_GPU_BUSY)
    summary["gpus"] = gpus
    reason = _trace_ok(spec)
    if reason:
        summary["invalid"].append(reason)
        return finish(EXIT_INPUT)
    reason = prepare_scratch(spec)
    if reason:
        summary["invalid"].append(reason)
        return finish(EXIT_INPUT)

    cmds = build_commands(spec, gpus, run_id, out, image, aiperf_image)
    # Created here so they belong to the host user, not to the container.
    for sub in ("aiperf", "aiperf_mmap/base", "aiperf_mmap/cache"):
        (out / sub).mkdir(parents=True)
    meta = {
        "run_id": run_id,
        "prt_commit": _git_commit(),
        "image": image,
        "image_id": docker.image_id(image),
        "vllm_commit": docker.image_label(image, "prt.vllm.commit"),
        "lmcache_commit": docker.image_label(image, "prt.lmcache.commit"),
        "aiperf_image": aiperf_image,
        "aiperf_image_id": docker.image_id(aiperf_image),
        "commands": cmds,
        "spec": asdict(spec),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    summary["vllm_commit"] = meta["vllm_commit"]
    summary["lmcache_commit"] = meta["lmcache_commit"]
    names = [f"prt-{run_id}-{r}" for r in cmds]
    sampler = None
    caps = {
        d["path"]: DiskCap(Path(d["path"]), int(d["cap_gb"] * 2**30))
        for d in spec.vllm_scratch
        if "cap_gb" in d
    }
    for c in caps.values():
        c.start()
    stop_s = docker.STOP_S
    try:
        reason = start_servers(spec, cmds, out, run_id)
        if reason:
            summary["invalid"].append(reason)
            return finish(EXIT_NOT_UP)
        sampler = Sampler(
            out / "samples.jsonl",
            f"http://127.0.0.1:{spec.vllm_port}/metrics",
            # Only a capped run needs it, and only cgroup v2 has one dir to read.
            cgroup=docker.cgroup_dir(f"prt-{run_id}-vllm")
            if spec.vllm_memory_gb is not None
            else None,
        )
        sampler.start()
        logger.info("AIPerf running (log: %s)", out / "aiperf.log")
        aiperf = docker.start(cmds["aiperf"], out / "aiperf.log")
        t0 = time.monotonic()
        deadline = t0 + spec.aiperf_timeout_s
        rc = None
        while rc is None and (left := deadline - time.monotonic()) > 0:
            try:
                rc = aiperf.wait(timeout=min(_HEARTBEAT_S, left))
            except subprocess.TimeoutExpired:
                mins = (time.monotonic() - t0) / 60
                phase = aiperf_phase(out / "aiperf.log", datetime.now(timezone.utc))
                logger.info("AIPerf %.0f min: %s", mins, phase)
        summary["aiperf_rc"] = rc
        if rc != 0:
            summary["invalid"].append(f"aiperf rc={rc}")
        for s in spec.services:
            if docker.status(f"prt-{run_id}-{s.name}") != "running":
                summary["invalid"].append(f"{s.name} exited during the run")
    except Cancelled:
        stop_s = 0
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        # Not invalid: the measurement is done; the next run fails to start instead.
        summary["containers_left"] = docker.remove(names, stop_s)
        remove_shm_regions(out)
        summary["disk_cap"] = {p: c.stop() for p, c in caps.items()}
        for p, c in summary["disk_cap"].items():
            if c["error"]:
                summary["invalid"].append(f"{p}: cap watch died: {c['error']}")
            if c["peak_bytes"] > c["cap_bytes"]:
                summary["invalid"].append(f"{p}: {c['peak_bytes']} B over its cap")
        dirs = _scratch_dirs(spec)
        summary["l2_disk_used_gb"] = (
            sum(allocated_bytes(Path(d["path"])) for d in dirs) / 1e9 if dirs else None
        )
        wipe_scratch(spec)

    logger.info("%s: torn down", run_id)
    log = (out / "vllm.log").read_text(errors="replace")
    for pat in spec.invalid_log_patterns:
        if pat in log:
            summary["invalid"].append(f"vllm.log contains {pat!r}")
    if not summary["invalid"]:
        try:
            summary.update(score(out, spec.window_s))
        except (KeyError, ValueError) as e:
            summary["invalid"].append(f"score failed: {e!r}")
    if summary["invalid"]:
        logger.info("%s: INVALID: %s", run_id, "; ".join(summary["invalid"]))
        return finish(EXIT_INVALID)
    logger.info(
        "%s: valid, ttft_p50 %.2f s, decode_p50 %.1f tok/s",
        run_id,
        summary["ttft_p50"],
        summary["decode_p50"],
    )
    return finish(EXIT_VALID)
