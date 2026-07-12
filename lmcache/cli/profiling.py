# SPDX-License-Identifier: Apache-2.0
"""Flame-graph profiling shared by the LMCache CLI.

Three callers use it. ``lmcache bench l2 --flamegraph on`` profiles the
benchmark's own process, and ``lmcache bench server --flamegraph on``
profiles the MP server it drives load into; both wrap a process that is
already running the workload. ``lmcache tool flamegraph --pid`` instead
attaches to a process it did not start -- an MP cache server, a vLLM
worker, or any Python process -- for a set duration (or until
interrupted).

Six modes are supported, in two families. Pick by what you want to see:

* To profile **Python execution or GIL contention**, use the *Python
  modes* ``gil`` / ``wall`` (``py-spy``). They give one root frame per
  thread and need no change to the target, but work only on a CPython
  process:

  - ``gil``  -- samples only threads holding the interpreter lock, so a
    Python adapter's GIL contention is directly visible.
  - ``wall`` -- wall-clock time per thread, blocked threads included.
    Separates a worker pool that ``on-cpu`` would superimpose.

* To look at **CPU/IO time, kernel frames, or a non-Python process**,
  use the *whole-process modes* (``perf`` / bcc). They see kernel frames
  and profile any process, but merge every thread into one chart:

  - ``on-cpu``  -- ``perf record``; where CPU cycles go (serialization,
    copies, hashing).
  - ``off-cpu`` -- ``offcputime-bpfcc``; time blocked off-CPU (I/O,
    locks, eventfds), usually the more informative view for I/O-bound
    L2 adapters.
  - ``offwake`` -- ``offwaketime-bpfcc``; like ``off-cpu`` but each
    blocked stack also carries the stack of whoever woke it -- answers
    "what unblocked me?".
  - ``wakeup``  -- ``wakeuptime-bpfcc``; the reverse view, the stacks
    that spend time *doing* the waking.

Rendering: the whole-process modes use Brendan Gregg's FlameGraph scripts
(``flamegraph.pl`` and, for on-CPU, ``stackcollapse-perf.pl``); ``py-spy``
emits its SVG directly. ``off-cpu`` requires ``sudo`` because
``offcputime-bpfcc`` loads a BPF program.

Python-implemented adapters need CPython's perf trampolines to appear in
the whole-process charts; see :func:`activate_python_frames`. Trampolines
can only be installed from inside the target interpreter, so when
attaching to another process the whole-process modes need that process to
have been launched with ``PYTHONPERFSUPPORT=1``. ``py-spy`` reads
interpreter state directly and needs no trampoline at all, which makes
``wall`` and ``gil`` the modes that work against an unmodified server.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
from typing import IO
import os
import shutil
import signal
import subprocess
import sys
import time

# Sampling frequency for ``perf record`` (on-CPU), in Hz. 99 is the
# flame-graph convention: enough samples to see proportions, ~10x less
# overhead and perf.data than 1 kHz, and off-100 to avoid beating against
# periodic (timer-driven) activity.
_PERF_FREQ_HZ = 99
# Seconds to wait for a recorder to flush and exit after SIGINT.
_STOP_TIMEOUT_SEC = 60
# Seconds to let a bcc BPF program load before the work starts.
_OFFCPU_SETTLE_SEC = 1.0
# Rendered flame-graph width, in pixels.
_FLAME_WIDTH_PX = 1600
# Highest ``kernel.perf_event_paranoid`` that still allows a non-root
# user to sample its own process with ``perf record``. Level 2 only
# withholds kernel-symbol resolution; level 3 (a Debian addition)
# rejects ``perf_event_open`` outright.
_MAX_PERF_PARANOID = 2
# Path holding the kernel's perf sampling restriction level.
_PERF_PARANOID_PATH = "/proc/sys/kernel/perf_event_paranoid"
# Path holding the Yama ptrace restriction level. Above 0, a process may
# only trace its own descendants -- and the recorder is a *child* of the
# process it profiles, so anything above 0 denies py-spy.
_YAMA_PTRACE_PATH = "/proc/sys/kernel/yama/ptrace_scope"
_MAX_YAMA_PTRACE_SCOPE = 0
# py-spy sampling frequency, in Hz.
_PY_SPY_RATE_HZ = 200
# Poll interval while recording an attached process until interrupted.
_ATTACH_POLL_SEC = 0.5
# CPython writes its perf trampoline map here (one per pid) when the
# process runs with PYTHONPERFSUPPORT=1. Its presence is how we tell
# whether an attached target's Python frames will resolve.
_PERF_MAP_DIR = "/tmp"
# Fallback advice when a recording captured nothing. Callers pass their
# own, naming the knob that lengthens *their* recording window.
_DEFAULT_SHORT_RUN_HINT = "record for longer"

# Per-mode tag used in default output filenames.
_MODE_TAG = {
    "on-cpu": "oncpu",
    "off-cpu": "offcpu",
    "offwake": "offwake",
    "wakeup": "wakeup",
    "wall": "wall",
    "gil": "gil",
}
# Modes recorded by py-spy, which writes its own SVG and needs no
# FlameGraph scripts, no trampoline, and no perf.
PY_SPY_MODES = frozenset({"wall", "gil"})
# bcc mode -> (tool binary, argv flags before ``-p <pid>``). All three
# emit folded stacks (``-f``) rendered through ``flamegraph.pl``, load a
# BPF program (so need ``sudo``), and consult ``/tmp/perf-<pid>.map`` for
# the target's Python frames just as perf does.
#
# * off-cpu  -- where a thread was blocked.
# * offwake  -- where it was blocked *and* the stack of whoever woke it.
# * wakeup   -- the stacks doing the waking (no ``-d``: it has none).
_BCC_MODES = {
    "off-cpu": ("offcputime-bpfcc", ["-df"]),
    "offwake": ("offwaketime-bpfcc", ["-df"]),
    "wakeup": ("wakeuptime-bpfcc", ["-f"]),
}
# flamegraph.pl ``--colors`` palette per bcc mode, so the three read
# apart at a glance and offwake's two halves are distinguishable:
#   off-cpu -> io    : one blue tower, purely "where it blocked".
#   offwake -> chain : blocked stack blue, waker stack (past the ``;--;``
#                      delimiter, tagged ``_[w]``) aqua -- two colors.
#   wakeup  -> wakeup: the wakers, aqua.
_BCC_PALETTE = {"off-cpu": "io", "offwake": "chain", "wakeup": "wakeup"}


class ProfileError(RuntimeError):
    """Raised when the profiling toolchain is missing or misconfigured."""


def _perf_map_path(pid: int) -> str:
    """Return the path CPython uses for *pid*'s perf trampoline map."""
    return os.path.join(_PERF_MAP_DIR, f"perf-{pid}.map")


def _warn(message: str) -> None:
    """Print a profiling warning to stderr, bypassing a quiet ``log``.

    Some warnings change how the resulting chart must be read (e.g.
    unresolvable Python frames). Those must reach the operator even when
    the caller silenced progress output with ``--quiet``, so they go
    straight to stderr rather than through the caller's logger.

    Args:
        message: The warning text, printed with a ``[Profile]`` prefix.
    """
    print(f"[Profile] WARNING: {message}", file=sys.stderr, flush=True)


def _check_perf_paranoid() -> None:
    """Validate that the kernel allows unprivileged ``perf record`` sampling.

    ``perf record`` silently produces an empty ``perf.data`` when
    ``kernel.perf_event_paranoid`` is above
    :data:`_MAX_PERF_PARANOID`, which surfaces much later as an empty
    flame graph. Fail fast instead.

    Raises:
        ProfileError: If the paranoid level forbids sampling. The message
            names the sysctl to lower.
    """
    try:
        with open(_PERF_PARANOID_PATH, encoding="utf-8") as fh:
            level = int(fh.read().strip())
    except (OSError, ValueError):
        # Non-Linux or an unreadable procfs: let perf itself report.
        return
    if level > _MAX_PERF_PARANOID:
        raise ProfileError(
            f"kernel.perf_event_paranoid is {level}; perf cannot sample "
            f"(needs <= {_MAX_PERF_PARANOID}). Lower it with "
            f"'sudo sysctl -w kernel.perf_event_paranoid={_MAX_PERF_PARANOID}' "
            "or pick the off-cpu mode."
        )


def _check_ptrace_scope() -> None:
    """Validate that py-spy may attach to its target.

    Under Yama ``ptrace_scope`` 1 (the Ubuntu default) a process may only
    trace its own descendants, and py-spy's target never is one: when
    self-profiling it is a *child* of the target, and when attaching it
    aims at an unrelated pid. Either way, scope 1 denies it -- unless we
    are root, for which Yama's restrictions do not apply.

    Raises:
        ProfileError: If the scope forbids the attach. The message names
            the sysctl to lower.
    """
    # Root can ptrace regardless of the Yama scope, so the check would be
    # a false failure. (``geteuid`` is Unix-only; treat its absence as
    # non-root and fall through to the scope check.)
    if getattr(os, "geteuid", lambda: 1)() == 0:
        return
    try:
        with open(_YAMA_PTRACE_PATH, encoding="utf-8") as fh:
            scope = int(fh.read().strip())
    except (OSError, ValueError):
        # No Yama LSM: ptrace is unrestricted for same-uid processes.
        return
    if scope > _MAX_YAMA_PTRACE_SCOPE:
        raise ProfileError(
            f"kernel.yama.ptrace_scope is {scope}; py-spy may only trace its "
            "own descendants, and its target never is one "
            f"(needs {_MAX_YAMA_PTRACE_SCOPE}). Lower it with "
            f"'sudo sysctl -w kernel.yama.ptrace_scope={_MAX_YAMA_PTRACE_SCOPE}', "
            "run as root, or pick the on-cpu / off-cpu mode."
        )


def activate_python_frames(log: Callable[[str], None]) -> None:
    """Emit a perf map so Python frames resolve in the flame graph.

    Python calls create no native stack frame, so both recorders render
    Python-implemented adapters as ``[unknown]``. CPython 3.12+ fixes
    this by emitting trampolines and a ``/tmp/perf-<pid>.map`` symbol
    file that ``perf`` and ``offcputime-bpfcc`` both consult.

    Args:
        log: Sink for a one-line status message.
    """
    if not hasattr(sys, "activate_stack_trampoline"):
        log(
            "[Profile] python frames unavailable "
            f"(python {sys.version_info.major}.{sys.version_info.minor} "
            "lacks perf trampolines; needs 3.12+)"
        )
        return
    try:
        sys.activate_stack_trampoline("perf")
    except (ValueError, RuntimeError) as e:
        log(f"[Profile] python frames unavailable: {e}")
        return
    log("[Profile] python frames enabled (perf trampoline)")


def deactivate_python_frames() -> None:
    """Turn off the perf trampoline installed by :func:`activate_python_frames`.

    The ``/tmp/perf-<pid>.map`` file outlives deactivation, so rendering
    can still resolve the recorded samples afterwards.
    """
    if hasattr(sys, "deactivate_stack_trampoline"):
        sys.deactivate_stack_trampoline()


def check_profiling_deps(mode: str) -> None:
    """Validate that the external tools required for *mode* are present.

    This is a runtime check meant to run *before* any recording starts --
    before the benchmark spins up an adapter, or before the CLI attaches
    to a live process -- so a missing dependency fails fast with an
    actionable message instead of producing an empty flame graph.

    Args:
        mode: One of ``"on-cpu"``, ``"off-cpu"``, ``"offwake"``,
            ``"wakeup"``, ``"wall"``, ``"gil"``.

    Raises:
        ProfileError: If ``mode`` is invalid or a required tool is
            missing. The message names the missing tool and how to
            install it or which mode to use instead.
    """
    if mode not in _MODE_TAG:
        raise ProfileError(
            f"invalid flame-graph mode: {mode!r} "
            f"(expected one of {', '.join(sorted(_MODE_TAG))})"
        )
    if mode in PY_SPY_MODES:
        if shutil.which("py-spy") is None:
            raise ProfileError(
                f"py-spy not found on PATH (needed for the '{mode}' flame "
                "graph). Install it ('pip install py-spy') or pick the "
                "on-cpu mode."
            )
        _check_ptrace_scope()
        return
    if mode == "on-cpu":
        if shutil.which("perf") is None:
            raise ProfileError(
                "perf not found on PATH (needed for on-cpu flame graphs). "
                "Install it (e.g. 'apt install linux-perf' or the linux-tools "
                "package matching your kernel) or pick the off-cpu mode."
            )
        _check_perf_paranoid()
    if mode in _BCC_MODES:
        tool = _BCC_MODES[mode][0]
        if shutil.which("sudo") is None:
            raise ProfileError(
                f"sudo not found on PATH (needed for the '{mode}' flame "
                "graph, which loads a BPF program). Install sudo or pick "
                "the on-cpu mode."
            )
        # The bcc tools usually live in /usr/sbin, which is not on the
        # user's PATH but is reachable under sudo; check both so a missing
        # tool surfaces here instead of as an empty flame graph.
        if shutil.which(tool) is None and not os.path.exists(f"/usr/sbin/{tool}"):
            raise ProfileError(
                f"{tool} not found (needed for the '{mode}' flame graph). "
                "Install bcc / bpfcc-tools (e.g. 'apt install bpfcc-tools') "
                "or pick the on-cpu mode."
            )


# Upstream FlameGraph repo, shallow-cloned on demand when the scripts
# are not already present locally.
_FLAMEGRAPH_REPO = "https://github.com/brendangregg/FlameGraph"


def _managed_flamegraph_dir() -> str:
    """Return the temp path where FlameGraph is auto-cloned."""
    return "/tmp/lmcache_flamegraph/FlameGraph"


def _has_flamegraph(flamegraph_dir: str) -> bool:
    """Return True if ``flamegraph.pl`` exists under *flamegraph_dir*."""
    return os.path.isfile(os.path.join(flamegraph_dir, "flamegraph.pl"))


def _clone_flamegraph(dest: str, log: Callable[[str], None]) -> None:
    """Shallow-clone the FlameGraph repo into *dest*."""
    if shutil.which("git") is None:
        raise ProfileError(
            "git not found; cannot auto-clone FlameGraph. Clone "
            f"{_FLAMEGRAPH_REPO} manually and pass --flamegraph-scripts-dir "
            "or set FLAMEGRAPH_DIR."
        )
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    log(f"[Profile] FlameGraph not found; cloning {_FLAMEGRAPH_REPO} -> {dest}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", _FLAMEGRAPH_REPO, dest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        detail = e.stderr.decode(errors="replace").strip() if e.stderr else str(e)
        raise ProfileError(f"git clone of FlameGraph failed: {detail}") from e
    if not _has_flamegraph(dest):
        raise ProfileError(f"clone completed but flamegraph.pl missing in {dest}")


def resolve_flamegraph_dir(explicit: str, log: Callable[[str], None]) -> str:
    """Resolve the FlameGraph scripts directory, auto-cloning if needed.

    Resolution order: the explicit ``--flamegraph-scripts-dir`` value,
    then ``$FLAMEGRAPH_DIR``, then ``~/FlameGraph``, then the managed temp
    clone under ``/tmp/lmcache_flamegraph/FlameGraph``. When none of these
    contains ``flamegraph.pl`` and no directory was given explicitly, the
    repo is shallow-cloned into the managed temp location.

    Args:
        explicit: Directory passed via ``--flamegraph-scripts-dir``; may
            be an empty string when the flag was not given.
        log: Progress logger (suppressed under ``--quiet``).

    Returns:
        A directory that contains ``flamegraph.pl``.

    Raises:
        ProfileError: If an explicit directory lacks the scripts, or the
            scripts are missing and the auto-clone fails.
    """
    if explicit:
        if not _has_flamegraph(explicit):
            raise ProfileError(
                f"flamegraph.pl not found under --flamegraph-scripts-dir {explicit}"
            )
        return explicit

    for candidate in (
        os.environ.get("FLAMEGRAPH_DIR", ""),
        os.path.expanduser("~/FlameGraph"),
        _managed_flamegraph_dir(),
    ):
        if candidate and _has_flamegraph(candidate):
            return candidate

    # Not found anywhere: auto-clone into the managed cache location.
    dest = _managed_flamegraph_dir()
    _clone_flamegraph(dest, log)
    return dest


def default_output_path(label: str, mode: str) -> str:
    """Build the default SVG output path for a profiling run.

    Args:
        label: Subject of the profile (an adapter class name, or the
            target pid), used in the filename.
        mode: One of ``"on-cpu"``, ``"off-cpu"``, ``"offwake"``, ``"wakeup"``,
            ``"wall"``, ``"gil"``.

    Returns:
        An absolute path under ``/tmp/lmcache_bench_flames``.
    """
    tag = _MODE_TAG.get(mode, mode)
    return f"/tmp/lmcache_bench_flames/{label}.{tag}.svg"


class FlameProfiler:
    """Struct wrapper for managing flame-chart profiling.

    It starts a separate process to profile the target process, and
    renders the captured data to a flame chart.
    """

    def __init__(
        self,
        *,
        mode: str,
        output: str,
        flamegraph_dir: str,
        pid: int,
        title: str,
        short_run_hint: str = _DEFAULT_SHORT_RUN_HINT,
    ) -> None:
        """Validate the toolchain and prepare output paths.

        Args:
            mode: One of ``"on-cpu"``, ``"off-cpu"``, ``"offwake"``, ``"wakeup"``,
            ``"wall"``, ``"gil"``.
            output: SVG output path.
            flamegraph_dir: Directory holding the FlameGraph scripts.
                Unused by the py-spy modes, which render their own SVG.
            pid: Process id to profile. Equal to this process when the
                caller is profiling itself.
            title: Flame-graph title.
            short_run_hint: Imperative clause naming the caller's own
                knob for recording longer, used when nothing was sampled.

        Raises:
            ProfileError: If ``mode`` is invalid or a required tool is
                unavailable.
        """
        check_profiling_deps(mode)

        self._mode = mode
        self._output = output
        self._flamegraph_dir = flamegraph_dir
        self._pid = pid
        self._title = title
        self._short_run_hint = short_run_hint
        self._proc: subprocess.Popen[bytes] | None = None
        self._raw_fh: object = None
        self._err_fh: object = None
        # Trampolines can only be installed from inside the interpreter
        # being profiled, so they are available only when self-profiling.
        self._self_profiling = pid == os.getpid()
        self._trampoline_on = False
        # Intermediate capture file (perf.data for on-cpu, folded stacks
        # for off-cpu); removed after a successful render. py-spy writes
        # the SVG itself, so its capture *is* the output.
        if mode in PY_SPY_MODES:
            self._raw_path = output
        else:
            suffix = ".perf.data" if mode == "on-cpu" else ".folded"
            self._raw_path = output + suffix
        # Recorder stderr, surfaced only when the recorder fails.
        self._err_path = output + ".recorder.err"
        self._stopped = False

        out_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(out_dir, exist_ok=True)
        # A failed recorder leaves any previous SVG in place, which would
        # read as a successful run. Start from a clean slate.
        for stale in (self._output, self._raw_path):
            try:
                os.remove(stale)
            except OSError:
                pass

    def _start_py_spy(self, log: Callable[[str], None]) -> None:
        """Start ``py-spy`` recording straight to the output SVG.

        ``--idle`` keeps blocked threads in the chart -- without it a
        worker parked on I/O simply vanishes, which is precisely what the
        per-thread view is for. ``--gil`` narrows sampling to threads
        holding the interpreter lock. ``--nonblocking`` samples without
        pausing the target: the profiled process is typically a live
        server -- and, when driven by ``bench server``, one the benchmark
        is blocked on -- so stopping it every sample would throttle the
        very workload being measured (and can stall the run past its
        timeout). The trade is that a stack is occasionally read mid-
        unwind and dropped, which a flame graph absorbs.

        Args:
            log: Sink for a one-line status message.
        """
        self._err_fh = open(self._err_path, "wb")
        argv = [
            "py-spy",
            "record",
            "--pid",
            str(self._pid),
            "--rate",
            str(_PY_SPY_RATE_HZ),
            "--format",
            "flamegraph",
            "--threads",
            "--idle",
            "--nonblocking",
            "--output",
            self._output,
        ]
        if self._mode == "gil":
            argv.append("--gil")
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=self._err_fh,
        )
        log(
            f"[Profile] {self._mode} recording started (pid={self._pid}) "
            f"-> {self._output}"
        )

    def start(self, log: Callable[[str], None]) -> None:
        """Start the background recorder targeting the profiled process."""
        if self._mode in PY_SPY_MODES:
            # py-spy reads interpreter state directly; trampolines would
            # only add overhead.
            self._start_py_spy(log)
            return
        if self._self_profiling:
            activate_python_frames(log)
            self._trampoline_on = True
        elif os.path.exists(_perf_map_path(self._pid)):
            # The target was launched with PYTHONPERFSUPPORT=1, so its own
            # trampoline map already lets perf resolve Python frames.
            log(
                f"[Profile] target pid {self._pid} has a perf map; "
                "python frames will resolve"
            )
        else:
            # An attached process's trampoline can only be installed from
            # inside it, so without that map the interpreter frames render
            # as [unknown]. Warn loudly enough to survive --quiet, since it
            # changes how the whole chart should be read.
            _warn(
                f"python frames will be [unknown]: pid {self._pid} has no "
                "perf map. Use --mode wall / gil (py-spy needs no map and "
                "adds no standing overhead), or -- only for a session you "
                "dedicate to profiling, since it slows every call -- "
                "relaunch the target with PYTHONPERFSUPPORT=1."
            )
        if self._mode == "on-cpu":
            self._err_fh = open(self._err_path, "wb")
            self._proc = subprocess.Popen(
                [
                    "perf",
                    "record",
                    "-F",
                    str(_PERF_FREQ_HZ),
                    "-g",
                    "-p",
                    str(self._pid),
                    "-o",
                    self._raw_path,
                ],
                stdout=subprocess.DEVNULL,
                stderr=self._err_fh,
            )
        else:
            tool, flags = _BCC_MODES[self._mode]
            self._raw_fh = open(self._raw_path, "wb")
            self._proc = subprocess.Popen(
                ["sudo", tool, *flags, "-p", str(self._pid)],
                stdout=self._raw_fh,  # type: ignore[arg-type]
                stderr=subprocess.DEVNULL,
            )
            # The BPF program takes a moment to load; give it time so the
            # first stretch of the measured phase is not missed.
            time.sleep(_OFFCPU_SETTLE_SEC)
        log(
            f"[Profile] {self._mode} recording started (pid={self._pid}) "
            f"-> {self._output}"
        )

    def stop(self, log: Callable[[str], None]) -> None:
        """Stop the recorder and render the SVG. Idempotent."""
        if self._stopped:
            return
        self._stopped = True
        if self._proc is None:
            return

        # SIGINT makes perf, the bcc tools and py-spy all flush their
        # output and exit cleanly (sudo forwards the signal to offcputime).
        try:
            self._proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=_STOP_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        if self._raw_fh is not None:
            self._raw_fh.close()  # type: ignore[attr-defined]
            self._raw_fh = None
        if self._err_fh is not None:
            self._err_fh.close()  # type: ignore[attr-defined]
            self._err_fh = None

        # py-spy renders the SVG itself and, unlike perf, reports failure
        # through its exit status rather than an empty capture.
        if self._mode in PY_SPY_MODES:
            try:
                self._check_py_spy_succeeded(log)
            except ProfileError as e:
                log(f"[Profile] recording failed: {e}")
                return
            log(f"[Profile] wrote {self._output}")
            return

        # The trampoline is no longer needed once sampling has stopped;
        # /tmp/perf-<pid>.map outlives it, so rendering still resolves.
        if self._trampoline_on:
            deactivate_python_frames()
            self._trampoline_on = False

        log("[Profile] rendering flame graph...")
        try:
            self._check_samples_captured(log)
            self._render()
        except ProfileError as e:
            log(f"[Profile] render failed: {e}")
            return
        finally:
            self._remove_intermediate_files()
        log(f"[Profile] wrote {self._output}")

    def _remove_intermediate_files(self) -> None:
        """Delete this run's capture, stderr, and self-profiling perf map.

        The perf map is CPython's ``/tmp/perf-<pid>.map``, which the
        interpreter never cleans up. When self-profiling, that pid is our
        own, and rendering has already consulted it, so removing it here
        keeps repeated runs from littering ``/tmp``. In attach mode the
        map belongs to the target process and is left untouched.
        """
        leftovers = [self._raw_path, self._raw_path + ".old", self._err_path]
        if self._self_profiling:
            leftovers.append(_perf_map_path(self._pid))
        for leftover in leftovers:
            try:
                os.remove(leftover)
            except OSError:
                pass

    def _check_py_spy_succeeded(self, log: Callable[[str], None]) -> None:
        """Confirm py-spy attached, sampled, and wrote its SVG.

        py-spy reports an attach failure through its exit status while
        leaving any earlier SVG untouched, so the status is what settles
        it. A clean exit with no SVG means the measured phase produced no
        samples.

        Args:
            log: Sink for py-spy's stderr, echoed as context.

        Raises:
            ProfileError: If py-spy failed or captured nothing.
        """
        proc = self._proc
        if proc is None:
            raise ProfileError("py-spy was never started")

        if proc.returncode != 0:
            try:
                with open(self._err_path, encoding="utf-8", errors="replace") as fh:
                    for line in fh.read().strip().splitlines():
                        log(f"[Profile]   {line}")
            except OSError:
                pass
            raise ProfileError(f"py-spy exited with status {proc.returncode}")
        if not (os.path.isfile(self._output) and os.path.getsize(self._output) > 0):
            raise ProfileError(
                "py-spy wrote no flame graph. The recording window is likely "
                f"too short to sample at {_PY_SPY_RATE_HZ} Hz; "
                f"{self._short_run_hint}."
            )
        try:
            os.remove(self._err_path)
        except OSError:
            pass

    def _check_samples_captured(self, log: Callable[[str], None]) -> None:
        """Fail with an actionable message when nothing was sampled.

        The recorder exits 0 after ``SIGINT`` even when it sampled
        nothing. A bcc capture (off-cpu / offwake / wakeup) is folded text,
        the question; a ``perf.data`` always carries a header, so only
        decoding it does.

        Args:
            log: Sink for the recorder's stderr, echoed as context.

        Raises:
            ProfileError: If the capture holds no stacks.
        """
        if self._mode == "on-cpu":
            captured = self._perf_capture_has_stacks()
        else:
            captured = os.path.isfile(self._raw_path) and (
                os.path.getsize(self._raw_path) > 0
            )
        if captured:
            return

        try:
            with open(self._err_path, encoding="utf-8", errors="replace") as fh:
                for line in fh.read().strip().splitlines():
                    log(f"[Profile]   {line}")
        except OSError:
            pass
        raise ProfileError(
            f"the {self._mode} recorder captured no samples. The recording "
            f"window is likely too short to sample; {self._short_run_hint}."
        )

    def _perf_capture_has_stacks(self) -> bool:
        """Return whether ``perf.data`` decodes to at least one stack.

        A ``perf.data`` always carries a header, so its size proves
        nothing; it has to be decoded. Only the first decoded line is
        read -- the full decode (potentially hundreds of MB) is left to
        :meth:`_render` -- so the pipe is drained and closed here rather
        than buffered.

        Returns:
            True if ``perf script`` produced any output.
        """
        proc = subprocess.Popen(
            ["perf", "script", "-i", self._raw_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            first_line = proc.stdout.readline() if proc.stdout else b""
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            proc.wait()
        return bool(first_line)

    def _render(self) -> None:
        """Render the captured stacks into the SVG at ``self._output``.

        Raises:
            ProfileError: If a rendering step fails or ``flamegraph.pl``
                exits non-zero.
        """
        flamegraph_pl = os.path.join(self._flamegraph_dir, "flamegraph.pl")
        try:
            with open(self._output, "wb") as svg:
                if self._mode == "on-cpu":
                    self._render_on_cpu(flamegraph_pl, svg)
                else:
                    self._render_folded(flamegraph_pl, svg)
        except OSError as e:
            raise ProfileError(str(e)) from e

    def _render_on_cpu(self, flamegraph_pl: str, svg: IO[bytes]) -> None:
        """Decode ``perf.data`` and pipe it through the FlameGraph scripts.

        The three stages run as a pipeline; each parent-held read end is
        closed after being handed downstream so a stage that dies early
        propagates ``SIGPIPE`` upstream, and every stage is waited on so
        none is left unreaped.

        Args:
            flamegraph_pl: Path to ``flamegraph.pl``.
            svg: Open binary sink for the rendered SVG.

        Raises:
            ProfileError: If a stage fails to start or exits non-zero.
        """
        collapse_pl = os.path.join(self._flamegraph_dir, "stackcollapse-perf.pl")
        started: list[tuple[str, subprocess.Popen[bytes]]] = []
        try:
            script = subprocess.Popen(
                ["perf", "script", "-i", self._raw_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            started.append(("perf script", script))
            collapse = subprocess.Popen(
                [collapse_pl], stdin=script.stdout, stdout=subprocess.PIPE
            )
            started.append(("stackcollapse", collapse))
            flame = subprocess.Popen(
                [
                    flamegraph_pl,
                    "--title",
                    self._title,
                    "--width",
                    str(_FLAME_WIDTH_PX),
                ],
                stdin=collapse.stdout,
                stdout=svg,
            )
            started.append(("flamegraph.pl", flame))
        except OSError as e:
            # A later stage failed to spawn; tear down any that did so
            # they are not orphaned, then surface the error.
            for _name, proc in started:
                if proc.stdout is not None:
                    proc.stdout.close()
                proc.kill()
                proc.wait()
            raise ProfileError(f"failed to start render pipeline: {e}") from e

        # Drop the parent's copies so EOF/SIGPIPE flows through the pipe.
        for pipe in (script.stdout, collapse.stdout):
            if pipe is not None:
                pipe.close()
        for _name, proc in reversed(started):
            proc.wait()
        # Check every stage, including ``perf script``: a silent decode
        # failure there would otherwise pass unnoticed.
        for name, proc in started:
            if proc.returncode != 0:
                raise ProfileError(f"{name} exited with code {proc.returncode}")

    def _render_folded(self, flamegraph_pl: str, svg: IO[bytes]) -> None:
        """Render folded bcc stacks straight through ``flamegraph.pl``.

        The ``--colors`` palette is chosen per mode (see
        :data:`_BCC_PALETTE`) so off-cpu / offwake / wakeup are visually
        distinct and offwake's blocked and waker halves get two colors.

        Args:
            flamegraph_pl: Path to ``flamegraph.pl``.
            svg: Open binary sink for the rendered SVG.

        Raises:
            ProfileError: If ``flamegraph.pl`` exits non-zero.
        """
        palette = _BCC_PALETTE.get(self._mode, "io")
        with open(self._raw_path, "rb") as folded:
            flame = subprocess.Popen(
                [
                    flamegraph_pl,
                    "--colors",
                    palette,
                    "--countname",
                    "us",
                    "--title",
                    self._title,
                    "--width",
                    str(_FLAME_WIDTH_PX),
                ],
                stdin=folded,
                stdout=svg,
            )
            flame.wait()
        if flame.returncode != 0:
            raise ProfileError(f"flamegraph.pl exited with code {flame.returncode}")


def record_attached(
    *,
    pid: int,
    mode: str,
    output: str,
    flamegraph_dir: str,
    duration: float,
    log: Callable[[str], None],
) -> None:
    """Record a flame graph of an already-running process.

    Records for *duration* seconds, or until the operator interrupts,
    whichever comes first. The interrupt is absorbed here so that the
    partial capture is still rendered rather than discarded.

    Args:
        pid: Process to profile.
        mode: One of ``"on-cpu"``, ``"off-cpu"``, ``"offwake"``, ``"wakeup"``,
            ``"wall"``, ``"gil"``.
        output: SVG output path.
        flamegraph_dir: Directory holding the FlameGraph scripts. Unused
            by the py-spy modes.
        duration: Seconds to record. Non-positive records until
            interrupted.
        log: Progress logger.

    Raises:
        ProfileError: If the target does not exist, or the toolchain is
            unavailable.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError as e:
        raise ProfileError(f"no such process: pid {pid}") from e
    except PermissionError as e:
        raise ProfileError(
            f"pid {pid} belongs to another user; profiling it needs root"
        ) from e

    profiler = FlameProfiler(
        mode=mode,
        output=output,
        flamegraph_dir=flamegraph_dir,
        pid=pid,
        title=f"{mode} (pid {pid})",
        short_run_hint="raise --duration",
    )
    profiler.start(log)
    try:
        if duration > 0:
            log(f"[Profile] recording for {duration:g}s (Ctrl-C to stop early)")
            time.sleep(duration)
        else:
            log("[Profile] recording until interrupted (Ctrl-C to stop)")
            while True:
                time.sleep(_ATTACH_POLL_SEC)
    except KeyboardInterrupt:
        log("[Profile] interrupted; rendering what was captured")
    finally:
        profiler.stop(log)
