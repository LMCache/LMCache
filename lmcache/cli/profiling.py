# SPDX-License-Identifier: Apache-2.0
"""Flame-graph profiling shared by the LMCache CLI.

Dependency: py-spy (``gil`` / ``wall``), Linux perf (``on-cpu``), or the bcc
``*-bpfcc`` tools (``off-cpu`` / ``offwake`` / ``wakeup``), validated at
runtime by :func:`check_profiling_deps`.

Usage: construct a :class:`FlameProfiler` and start/stop it around a
workload, or call :func:`record_attached` to profile a pid for a duration.

Principle: py-spy reads the interpreter's memory from outside the target;
perf and bcc sample or hook the kernel, and name Python frames only when the
target ran with ``PYTHONPERFSUPPORT=1`` (its perf trampoline map).

Cost: py-spy is lightest (its work runs off the target's CPUs), perf is a
bounded 99 Hz sample, bcc scales with the target's event rate. The ``lmcache
tool flamegraph`` CLI docs cover per-mode detail.
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

# --- Recording tunables: sampling rates, timeouts, render size ---
# Sampling frequency for ``perf record`` (on-CPU), in Hz. 99 is the
# flame-graph convention: enough samples to see proportions, ~10x less
# overhead and perf.data than 1 kHz, and off-100 to avoid beating against
# periodic (timer-driven) activity.
_PERF_FREQ_HZ = 99
# py-spy sampling frequency, in Hz.
_PY_SPY_RATE_HZ = 200
# Seconds to wait for a recorder to flush and exit after SIGINT.
_STOP_TIMEOUT_SEC = 60
# Seconds to let a bcc BPF program load before the work starts.
_OFFCPU_SETTLE_SEC = 1.0
# Poll interval while recording an attached process until interrupted.
_ATTACH_POLL_SEC = 0.5
# Rendered flame-graph width, in pixels.
_FLAME_WIDTH_PX = 1600

# --- Kernel permission gates (checked before recording) ---
# Highest ``kernel.perf_event_paranoid`` that still allows a non-root
# user to sample its own process with ``perf record``. Level 2 only
# withholds kernel-symbol resolution; level 3 (a Debian addition)
# rejects ``perf_event_open`` outright.
_MAX_PERF_PARANOID = 2
# Path holding the kernel's perf sampling restriction level.
_PERF_PARANOID_PATH = "/proc/sys/kernel/perf_event_paranoid"
# Yama ptrace restriction level, and the highest value that still lets
# py-spy attach. Above 0 a process may trace only its descendants; the full
# rationale and the CAP_SYS_PTRACE bypass live in _check_ptrace_scope.
_YAMA_PTRACE_PATH = "/proc/sys/kernel/yama/ptrace_scope"
_MAX_YAMA_PTRACE_SCOPE = 0
# Effective-capability set (the "CapEff:" line of /proc/self/status) and the
# bit for CAP_SYS_PTRACE. Holding it (not being uid 0) bypasses the ptrace
# scope; see _has_cap_sys_ptrace.
_PROC_STATUS_PATH = "/proc/self/status"
_CAP_SYS_PTRACE_BIT = 19

# --- Python-frame resolution: the CPython perf trampoline map ---
# CPython writes its perf trampoline map here (one per pid) when the
# process runs with PYTHONPERFSUPPORT=1. Its presence is how we tell
# whether an attached target's Python frames will resolve.
_PERF_MAP_DIR = "/tmp"

# --- Mode registry: the six modes and their per-mode tool config ---
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
# bcc mode -> (tool binary, argv flags before ``-p <pid>``). All emit folded
# stacks and load a BPF program (so need ``sudo``); wakeup has no delimiter,
# hence ``-f`` rather than ``-df``.
_BCC_MODES = {
    "off-cpu": ("offcputime-bpfcc", ["-df"]),
    "offwake": ("offwaketime-bpfcc", ["-df"]),
    "wakeup": ("wakeuptime-bpfcc", ["-f"]),
}
# flamegraph.pl ``--colors`` palette per bcc mode, so the three read apart
# and offwake's blocked (blue) and waker (aqua, past the ``;--;`` delimiter)
# halves are distinguishable.
_BCC_PALETTE = {"off-cpu": "io", "offwake": "chain", "wakeup": "wakeup"}


class ProfileError(RuntimeError):
    """Raised when the profiling toolchain is missing or misconfigured."""


def _perf_map_path(pid: int) -> str:
    """Return the path CPython uses for *pid*'s perf trampoline map."""
    return os.path.join(_PERF_MAP_DIR, f"perf-{pid}.map")


def _check_perf_paranoid() -> None:
    """Fail fast if ``kernel.perf_event_paranoid`` forbids perf sampling.

    Above :data:`_MAX_PERF_PARANOID`, ``perf record`` silently writes an
    empty ``perf.data`` that only surfaces later as an empty graph; raise
    instead, naming the sysctl to lower.
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


def _has_cap_sys_ptrace() -> bool:
    """Return whether the effective set holds CAP_SYS_PTRACE.

    That capability (not being uid 0) is what bypasses the Yama ptrace
    scope, and a container can run as root while dropping it, so this reads
    ``CapEff`` from ``/proc/self/status`` rather than checking the uid.
    Returns False when it is absent or unreadable (e.g. non-Linux).
    """
    try:
        with open(_PROC_STATUS_PATH, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("CapEff:"):
                    caps = int(line.split()[1], 16)
                    return bool(caps >> _CAP_SYS_PTRACE_BIT & 1)
    except (OSError, ValueError, IndexError):
        pass
    return False


def _check_ptrace_scope() -> None:
    """Fail fast if the Yama ptrace scope forbids py-spy's attach.

    Above :data:`_MAX_YAMA_PTRACE_SCOPE` a process may trace only its
    descendants, and py-spy's target never is one; CAP_SYS_PTRACE bypasses
    this, so the check is skipped when the process holds it (uid 0 alone is
    not enough, since a container can be root without the capability).
    """
    if _has_cap_sys_ptrace():
        return
    try:
        with open(_YAMA_PTRACE_PATH, encoding="utf-8") as fh:
            scope = int(fh.read().strip())
    except (OSError, ValueError):
        # No Yama LSM: ptrace is unrestricted for same-uid processes.
        return
    if scope > _MAX_YAMA_PTRACE_SCOPE:
        raise ProfileError(
            f"kernel.yama.ptrace_scope is {scope} and this process lacks "
            "CAP_SYS_PTRACE; py-spy may only trace its own descendants, and "
            f"its target never is one (needs scope {_MAX_YAMA_PTRACE_SCOPE} or "
            "CAP_SYS_PTRACE). Lower the sysctl with "
            f"'sudo sysctl -w kernel.yama.ptrace_scope={_MAX_YAMA_PTRACE_SCOPE}', "
            "grant the capability (in a container, launch it with "
            "'--cap-add SYS_PTRACE'), or pick the on-cpu / off-cpu mode."
        )


def _activate_python_frames(log: Callable[[str], None]) -> None:
    """Emit a perf map so Python frames resolve in the flame graph.

    Python calls make no native frame, so ``perf`` / bcc render adapters as
    ``[unknown]`` without this; CPython 3.12+ fixes it with a trampoline map
    (``/tmp/perf-<pid>.map``) both recorders consult.
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


def check_profiling_deps(mode: str) -> None:
    """Validate *mode*'s external tools before any recording starts.

    Runs before the adapter spins up (or the CLI attaches) so a missing
    tool or restrictive sysctl fails fast with an actionable message
    instead of an empty flame graph.
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


# Upstream FlameGraph repo, and the managed temp location it is
# shallow-cloned into on demand when the scripts are not found locally.
_FLAMEGRAPH_REPO = "https://github.com/brendangregg/FlameGraph"
_MANAGED_FLAMEGRAPH_DIR = "/tmp/lmcache_flamegraph/FlameGraph"


def _has_flamegraph(flamegraph_dir: str) -> bool:
    """Return True if ``flamegraph.pl`` exists under *flamegraph_dir*."""
    return os.path.isfile(os.path.join(flamegraph_dir, "flamegraph.pl"))


def _clone_flamegraph(log: Callable[[str], None]) -> None:
    """Shallow-clone the FlameGraph repo into :data:`_MANAGED_FLAMEGRAPH_DIR`."""
    dest = _MANAGED_FLAMEGRAPH_DIR
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

    Tries ``--flamegraph-scripts-dir``, ``$FLAMEGRAPH_DIR``, ``~/FlameGraph``,
    then a managed ``/tmp`` clone; shallow-clones the repo there when nothing
    else has ``flamegraph.pl``, and raises if an explicit dir lacks it.
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
        _MANAGED_FLAMEGRAPH_DIR,
    ):
        if candidate and _has_flamegraph(candidate):
            return candidate

    # Not found anywhere: auto-clone into the managed cache location.
    _clone_flamegraph(log)
    return _MANAGED_FLAMEGRAPH_DIR


def _bcc_capture_has_stacks(path: str) -> bool:
    """Return whether *path* holds a folded-stack line (``frame;... <count>``).

    A non-empty file is not proof of samples: the ``*-bpfcc`` tools also
    write diagnostics (e.g. "Unable to find kernel headers") to the same
    stdout, so only a line ending in a space and an integer counts.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stack, _sep, count = line.rstrip("\n").rpartition(" ")
                if stack and count.isdigit():
                    return True
    except OSError:
        return False
    return False


def default_output_path(label: str, mode: str) -> str:
    """Build the default ``/tmp/lmcache_bench_flames/<label>.<tag>.svg`` path."""
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
    ) -> None:
        """Validate the toolchain and prepare output paths.

        *pid* equal to this process selects self-profiling. Raises
        ProfileError on an invalid mode or missing tool.
        """
        check_profiling_deps(mode)

        self._mode = mode
        self._output = output
        self._flamegraph_dir = flamegraph_dir
        self._pid = pid
        self._title = title
        self._proc: subprocess.Popen[bytes] | None = None
        self._raw_fh: object = None
        self._err_fh: object = None
        # True when the target is this process (``bench l2``). Only then can
        # we activate CPython's perf trampolines (they install from inside the
        # target interpreter) and own the resulting /tmp/perf-<pid>.map.
        self._self_profiling = pid == os.getpid()
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

        ``--idle`` keeps blocked threads visible, ``--gil`` narrows to
        GIL-holders, and ``--nonblocking`` samples without pausing the target
        (else a live server, or one ``bench server`` is blocked on, gets
        throttled), trading the occasional dropped mid-unwind stack.
        """
        self._err_fh = open(self._err_path, "wb")
        # Equivalent to: py-spy record --pid P --rate 200 --format flamegraph
        #   --threads --idle --nonblocking [--gil] --output OUT
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
            _activate_python_frames(log)
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
            # as [unknown]. Print straight to stderr, not through ``log``, so
            # it survives a --quiet caller: it changes how the chart reads.
            print(
                "[Profile] WARNING: python frames will be [unknown]: pid "
                f"{self._pid} has no perf map. Use --mode wall / gil (py-spy "
                "needs no map and adds no standing overhead), or (only for a "
                "session you dedicate to profiling, since it slows every "
                "call) relaunch the target with PYTHONPERFSUPPORT=1.",
                file=sys.stderr,
                flush=True,
            )
        if self._mode == "on-cpu":
            self._err_fh = open(self._err_path, "wb")
            # Equivalent to: perf record -F 99 -g -p P -o <perf.data>
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
            # Capture stderr, not discard: a failed BPF load prints its reason
            # (missing kernel headers, no privilege) here, surfaced on error.
            self._err_fh = open(self._err_path, "wb")
            # Equivalent to: sudo <tool> <flags> -p P
            self._proc = subprocess.Popen(
                ["sudo", tool, *flags, "-p", str(self._pid)],
                stdout=self._raw_fh,  # type: ignore[arg-type]
                stderr=self._err_fh,  # type: ignore[arg-type]
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

        # We only activated a trampoline when self-profiling (py-spy modes
        # returned above), so only then is there one to turn off. Its
        # /tmp/perf-<pid>.map outlives it, so rendering still resolves.
        if self._self_profiling and hasattr(sys, "deactivate_stack_trampoline"):
            sys.deactivate_stack_trampoline()

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
        """Delete this run's capture, stderr, and (when self-profiling) perf map.

        CPython never cleans up ``/tmp/perf-<pid>.map``; we remove it only
        when the pid is our own, leaving an attach target's map untouched.
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

        py-spy signals an attach failure through its exit status (leaving any
        old SVG in place), so the status settles it; a clean exit with no SVG
        means the window produced no samples.
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
                f"too short to sample at {_PY_SPY_RATE_HZ} Hz; record for longer."
            )
        try:
            os.remove(self._err_path)
        except OSError:
            pass

    def _check_samples_captured(self, log: Callable[[str], None]) -> None:
        """Fail with an actionable message when nothing usable was captured.

        The recorder exits 0 after ``SIGINT`` even when empty, so the capture
        is inspected: on-cpu by decoding its ``perf.data`` header, bcc by
        :func:`_bcc_capture_has_stacks` (a non-empty folded file is not proof).
        """
        if self._mode == "on-cpu":
            captured = self._perf_capture_has_stacks()
        else:
            captured = _bcc_capture_has_stacks(self._raw_path)
        if captured:
            return

        # Surface the recorder's diagnostics so the failure names its cause:
        # stderr for every mode, plus (for bcc) the folded stdout capture --
        # the *-bpfcc tools misroute some errors (missing kernel headers) there
        # rather than to stderr.
        sources = [self._err_path]
        if self._mode in _BCC_MODES:
            sources.append(self._raw_path)
        for source in sources:
            try:
                with open(source, encoding="utf-8", errors="replace") as fh:
                    for line in fh.read().strip().splitlines():
                        log(f"[Profile]   {line}")
            except OSError:
                pass

        if self._mode in _BCC_MODES:
            raise ProfileError(
                f"the {self._mode} recorder produced no stacks. Either the "
                "recording window was too short (record for longer), or "
                "the bcc program failed to load; see the recorder output "
                "above (commonly missing kernel headers or insufficient "
                "privileges)."
            )
        raise ProfileError(
            f"the {self._mode} recorder captured no samples. The recording "
            "window is likely too short to sample; record for longer."
        )

    def _perf_capture_has_stacks(self) -> bool:
        """Return whether ``perf.data`` decodes to at least one stack.

        Its header makes size meaningless, so it must be decoded; only the
        first line is read (the full decode is left to :meth:`_render`).
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
        """Render the captured stacks into the SVG at ``self._output``."""
        flamegraph_pl = os.path.join(self._flamegraph_dir, "flamegraph.pl")
        try:
            with open(self._output, "wb") as svg:
                # on-cpu captures a perf.data (decode + collapse + render);
                # the bcc modes capture pre-folded stacks (render directly).
                if self._mode == "on-cpu":
                    self._render_perf_data(flamegraph_pl, svg)
                else:
                    self._render_folded(flamegraph_pl, svg)
        except OSError as e:
            raise ProfileError(str(e)) from e

    def _render_perf_data(self, flamegraph_pl: str, svg: IO[bytes]) -> None:
        """Decode ``perf.data`` and pipe it through the FlameGraph scripts.

        Three stages piped together: ``perf script`` | ``stackcollapse-perf.pl``
        | ``flamegraph.pl``. Each read end is closed once handed downstream so
        an early death propagates ``SIGPIPE`` upstream, and every stage's exit
        code is checked, since a silent ``perf script`` decode failure would
        otherwise pass unnoticed.
        """
        collapse_pl = os.path.join(self._flamegraph_dir, "stackcollapse-perf.pl")
        script = subprocess.Popen(
            ["perf", "script", "-i", self._raw_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        collapse = subprocess.Popen(
            [collapse_pl], stdin=script.stdout, stdout=subprocess.PIPE
        )
        flame = subprocess.Popen(
            [flamegraph_pl, "--title", self._title, "--width", str(_FLAME_WIDTH_PX)],
            stdin=collapse.stdout,
            stdout=svg,
        )
        # Drop the parent's copies so EOF/SIGPIPE flows down the chain.
        for pipe in (script.stdout, collapse.stdout):
            if pipe is not None:
                pipe.close()
        stages = (
            ("perf script", script),
            ("stackcollapse", collapse),
            ("flamegraph.pl", flame),
        )
        for _name, proc in reversed(stages):
            proc.wait()
        for name, proc in stages:
            if proc.returncode != 0:
                raise ProfileError(f"{name} exited with code {proc.returncode}")

    def _render_folded(self, flamegraph_pl: str, svg: IO[bytes]) -> None:
        """Render folded bcc stacks through ``flamegraph.pl``.

        The ``--colors`` palette is per mode (see :data:`_BCC_PALETTE`) so
        off-cpu / offwake / wakeup read apart and offwake's blocked and waker
        halves get distinct colors.
        """
        palette = _BCC_PALETTE[self._mode]
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

    Records for *duration* seconds (non-positive = until Ctrl-C); the
    interrupt is absorbed so the partial capture is still rendered. Raises
    ProfileError if the target is gone or the toolchain is unavailable.
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
