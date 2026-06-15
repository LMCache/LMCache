# SPDX-License-Identifier: Apache-2.0
"""Profiling tools for benchmarking lmcache performance.

When ``--flamegraph on`` is set, the benchmark records a flame graph of
its own measured phases and renders it to an SVG, so no second terminal
or externally attached profiler is needed.

Two modes are supported:

* ``on-cpu``  -- ``perf record`` sampling; shows where CPU time goes
  (serialization, copies, hashing).
* ``off-cpu`` -- ``offcputime-bpfcc`` (bcc); shows time spent blocked
  off-CPU (waiting on I/O, locks, eventfds), usually the more
  informative view for I/O-bound L2 adapters.

Rendering uses Brendan Gregg's FlameGraph scripts (``flamegraph.pl`` and,
for on-CPU, ``stackcollapse-perf.pl``). ``off-cpu`` additionally requires
``sudo`` because ``offcputime-bpfcc`` loads a BPF program.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Callable
import os
import shutil
import signal
import subprocess
import time

# Sampling frequency for ``perf record`` (on-CPU), in Hz.
_PERF_FREQ_HZ = 999
# Seconds to wait for a recorder to flush and exit after SIGINT.
_STOP_TIMEOUT_SEC = 60
# Seconds to let the off-CPU BPF program load before the work starts.
_OFFCPU_SETTLE_SEC = 1.0
# Rendered flame-graph width, in pixels.
_FLAME_WIDTH_PX = 1600

# Per-mode tag used in default output filenames.
_MODE_TAG = {"on-cpu": "oncpu", "off-cpu": "offcpu"}


class ProfileError(RuntimeError):
    """Raised when the profiling toolchain is missing or misconfigured."""


def check_profiling_deps(mode: str) -> None:
    """Validate that the external tools required for *mode* are present.

    This is a runtime check meant to run *before* the benchmark spins up
    any adapter, so a missing dependency fails fast with an actionable
    message instead of producing an empty flame graph midway through a
    run.

    Args:
        mode: ``"on-cpu"`` or ``"off-cpu"``.

    Raises:
        ProfileError: If ``mode`` is invalid or a required tool is
            missing. The message names the missing tool and how to
            install it or which mode to use instead.
    """
    if mode not in _MODE_TAG:
        raise ProfileError(
            f"invalid flame-graph mode: {mode!r} (expected 'on-cpu' or 'off-cpu')"
        )
    if mode == "on-cpu" and shutil.which("perf") is None:
        raise ProfileError(
            "perf not found on PATH (needed for on-cpu flame graphs). "
            "Install it (e.g. 'apt install linux-perf' or the linux-tools "
            "package matching your kernel) or use --flamegraph-mode off-cpu."
        )
    if mode == "off-cpu":
        if shutil.which("sudo") is None:
            raise ProfileError(
                "sudo not found on PATH (needed for off-cpu flame graphs, "
                "which load a BPF program). Install sudo or use "
                "--flamegraph-mode on-cpu."
            )
        # offcputime-bpfcc usually lives in /usr/sbin, which is not on the
        # user's PATH but is reachable under sudo; check both so a missing
        # tool surfaces here instead of as an empty flame graph.
        if shutil.which("offcputime-bpfcc") is None and not os.path.exists(
            "/usr/sbin/offcputime-bpfcc"
        ):
            raise ProfileError(
                "offcputime-bpfcc not found (needed for off-cpu flame "
                "graphs). Install bcc / bpfcc-tools (e.g. 'apt install "
                "bpfcc-tools') or use --flamegraph-mode on-cpu."
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
            f"{_FLAMEGRAPH_REPO} manually and pass --flamegraph-dir "
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

    Resolution order: the explicit ``--flamegraph-dir`` value,
    then ``$FLAMEGRAPH_DIR``, then ``~/FlameGraph``, then the managed temp
    clone under ``/tmp/lmcache_flamegraph/FlameGraph``. When none of these
    contains ``flamegraph.pl`` and no directory was given explicitly, the
    repo is shallow-cloned into the managed temp location.

    Args:
        explicit: Directory passed via ``--flamegraph-dir``; may
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
                f"flamegraph.pl not found under --flamegraph-dir {explicit}"
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


def default_output_path(adapter_name: str, mode: str) -> str:
    """Build the default SVG output path for a profiling run.

    Args:
        adapter_name: Adapter class name, used in the filename.
        mode: ``"on-cpu"`` or ``"off-cpu"``.

    Returns:
        An absolute path under ``/tmp/lmcache_bench_flames``.
    """
    tag = _MODE_TAG.get(mode, mode)
    return f"/tmp/lmcache_bench_flames/{adapter_name}.{tag}.svg"


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

        Args:
            mode: ``"on-cpu"`` or ``"off-cpu"``.
            output: SVG output path.
            flamegraph_dir: Directory holding the FlameGraph scripts.
            pid: Process id to profile (the benchmark process itself).
            title: Flame-graph title.

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
        self._proc: subprocess.Popen[bytes] | None = None
        self._raw_fh: object = None
        # Intermediate capture file (perf.data for on-cpu, folded
        # stacks for off-cpu); removed after a successful render.
        suffix = ".perf.data" if mode == "on-cpu" else ".folded"
        self._raw_path = output + suffix
        self._stopped = False

        out_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(out_dir, exist_ok=True)

    def start(self, log: Callable[[str], None]) -> None:
        """Start the background recorder targeting the profiled process."""
        if self._mode == "on-cpu":
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
                stderr=subprocess.DEVNULL,
            )
        else:
            self._raw_fh = open(self._raw_path, "wb")
            self._proc = subprocess.Popen(
                ["sudo", "offcputime-bpfcc", "-df", "-p", str(self._pid)],
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

        # SIGINT makes both perf and offcputime-bpfcc flush their output
        # and exit cleanly (sudo forwards the signal to offcputime).
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

        log("[Profile] rendering flame graph...")
        try:
            self._render()
        except ProfileError as e:
            log(f"[Profile] render failed: {e}")
            return
        for leftover in (self._raw_path, self._raw_path + ".old"):
            try:
                os.remove(leftover)
            except OSError:
                pass
        log(f"[Profile] wrote {self._output}")

    def _render(self) -> None:
        """Render the captured stacks into viewable flamechart at path at self._path"""
        flamegraph_pl = os.path.join(self._flamegraph_dir, "flamegraph.pl")
        try:
            with open(self._output, "wb") as svg:
                if self._mode == "on-cpu":
                    collapse_pl = os.path.join(
                        self._flamegraph_dir, "stackcollapse-perf.pl"
                    )
                    script = subprocess.Popen(
                        ["perf", "script", "-i", self._raw_path],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                    )
                    collapse = subprocess.Popen(
                        [collapse_pl],
                        stdin=script.stdout,
                        stdout=subprocess.PIPE,
                    )
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
                    flame.communicate()
                    rc = flame.returncode
                else:
                    with open(self._raw_path, "rb") as folded:
                        flame = subprocess.Popen(
                            [
                                flamegraph_pl,
                                "--color=io",
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
                        flame.communicate()
                        rc = flame.returncode
        except OSError as e:
            raise ProfileError(str(e)) from e
        if rc != 0:
            raise ProfileError(f"flamegraph.pl exited with code {rc}")
