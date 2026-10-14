# SPDX-License-Identifier: Apache-2.0
"""Background sampler: server counters and process I/O, one JSON line per tick.

Each line holds ``t`` (unix seconds), every :data:`METRICS` sample summed over
labels and URLs, ``vllm:prompt_tokens_by_source/<source>``, and
``io_rbytes/<container>`` (cgroup ``io.stat`` read bytes from one device).
"""

# Standard
from pathlib import Path
from typing import Any
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

# The counters runner/score.py reads, besides the prompt tokens by source.
METRICS = frozenset({"vllm:num_preemptions_total"})
BY_SOURCE = ("vllm:prompt_tokens_by_source", "vllm:prompt_tokens_by_source_total")
_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def _scrape(url: str, rec: dict[str, Any]) -> None:
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


def _rbytes(io_stat: Path, dev: int) -> int:
    key = f"{os.major(dev)}:{os.minor(dev)} "
    with open(io_stat) as f:
        for line in f:
            if line.startswith(key):
                return int(line.split("rbytes=")[1].split()[0])
    # A cgroup lists a device only after its first I/O to it.
    return 0


class Sampler:
    """Samples metrics URLs and cgroup read bytes into a JSONL file."""

    def __init__(
        self,
        out: Path,
        urls: list[str],
        io: dict[str, tuple[Path, int]],
        interval_s: float = 5.0,
    ) -> None:
        """Prepare a sampler; call :meth:`start` to begin.

        Args:
            out: JSONL file to append to.
            urls: Prometheus endpoints to scrape.
            io: Container role -> (its cgroup ``io.stat``, device number
                ``st_dev``) whose read bytes to record.
            interval_s: Seconds between ticks.
        """
        self.out = out
        self.urls = urls
        self.io = io
        self.interval_s = interval_s
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
        with open(self.out, "a") as f:
            while not self._stop.is_set():
                t = time.time()
                rec: dict[str, Any] = {"t": t}
                for url in self.urls:
                    try:
                        _scrape(url, rec)
                    except (
                        urllib.error.URLError,
                        ConnectionError,
                        TimeoutError,
                        ValueError,
                    ) as e:
                        rec[f"err {url}"] = str(e)
                for name, (io_stat, dev) in self.io.items():
                    try:
                        rec[f"io_rbytes/{name}"] = _rbytes(io_stat, dev)
                    except OSError as e:
                        rec[f"err io_rbytes/{name}"] = str(e)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                self._stop.wait(max(0.0, self.interval_s - (time.time() - t)))
