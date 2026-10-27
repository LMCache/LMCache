# SPDX-License-Identifier: Apache-2.0
"""Background sampler: server counters, one JSON line per tick.

Each line holds ``t`` (unix seconds), every :data:`METRICS` sample summed over
labels and URLs, and ``vllm:prompt_tokens_by_source/<source>``. With a cgroup,
also ``mem_anon`` and ``mem_file`` (bytes): the container's own memory and the
page cache charged to it. With an ``io`` reader, also ``io_rbytes`` and
``io_wbytes``: the cache tiers' block I/O so far.
"""

# Standard
from collections.abc import Callable
from pathlib import Path
from typing import Any
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

# The counters runner/score.py reads, besides the prompt tokens by source.
METRICS = frozenset(
    {
        "vllm:num_preemptions_total",
        "vllm:generation_tokens_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    }
)
BY_SOURCE = ("vllm:prompt_tokens_by_source", "vllm:prompt_tokens_by_source_total")
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def _scrape(url: str, rec: dict[str, Any]) -> None:
    """Add the counters :data:`METRICS` and :data:`BY_SOURCE` from ``url``.

    Values are summed over labels; by-source counters are keyed
    ``vllm:prompt_tokens_by_source/<source>``.

    Args:
        url: A Prometheus text endpoint.
        rec: The tick's record, updated in place.
    """
    with urllib.request.urlopen(url, timeout=4) as r:
        text = r.read().decode()
    for line in text.splitlines():
        m = _LINE.match(line)
        if not m:
            continue
        name, labels, value = m.groups()
        if name in BY_SOURCE:
            src = re.search(r'source="([^"]+)"', labels or "")
            if src is None:
                continue
            key = f"vllm:prompt_tokens_by_source/{src.group(1)}"
        elif name in METRICS:
            key = name
        else:
            continue
        rec[key] = rec.get(key, 0.0) + float(value)


def _memory(cgroup: Path, rec: dict[str, Any]) -> None:
    """Add ``mem_anon`` and ``mem_file`` from ``cgroup``'s ``memory.stat``.

    Args:
        cgroup: A cgroup v2 dir.
        rec: The tick's record, updated in place.
    """
    for line in (cgroup / "memory.stat").read_text().splitlines():
        key, value = line.split()
        if key in ("anon", "file"):
            rec[f"mem_{key}"] = int(value)


class Sampler:
    """Samples metrics URLs into a JSONL file."""

    def __init__(
        self,
        out: Path,
        url: str,
        interval_s: float = 5.0,
        cgroup: Path | None = None,
        io: Callable[[], dict[str, int]] | None = None,
    ) -> None:
        """Prepare a sampler; call :meth:`start` to begin.

        Args:
            out: JSONL file to append to.
            url: Prometheus endpoint to scrape.
            interval_s: Seconds between ticks.
            cgroup: cgroup v2 dir whose memory to sample, or None.
            io: Returns block I/O as ``rbytes`` and ``wbytes``, or None.
        """
        self.out = out
        self.url = url
        self.interval_s = interval_s
        self.cgroup = cgroup
        self.io = io
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start sampling in a background thread."""
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling and wait for the last line to be written."""
        self._stop.set()
        self._thread.join()

    def _loop(self) -> None:
        """Write one record per tick until stopped; a failed scrape is recorded."""
        with open(self.out, "a") as f:
            while not self._stop.is_set():
                t = time.time()
                rec: dict[str, Any] = {"t": t}
                try:
                    _scrape(self.url, rec)
                except (
                    urllib.error.URLError,
                    ConnectionError,
                    TimeoutError,
                    ValueError,
                ) as e:
                    rec["err"] = str(e)
                if self.cgroup is not None:
                    try:
                        _memory(self.cgroup, rec)
                    except OSError as e:
                        rec["mem_err"] = str(e)
                if self.io is not None:
                    try:
                        rec |= {f"io_{k}": v for k, v in self.io().items()}
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        rec["io_err"] = str(e)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                self._stop.wait(max(0.0, self.interval_s - (time.time() - t)))
