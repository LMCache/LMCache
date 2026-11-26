#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live per-node P2P dashboard from Prometheus.

Every ``--interval`` seconds, queries the expressions in ``promql.EXPRS`` (lookup/hit
token rates, active P2P lookup jobs, prefetch chunks and failures, per-load throughput,
L1 usage and evictions, store/retrieve rates, vLLM queue depth, TTFT) and prints one row
per node; rows are appended to ``--csv``.

usage: monitor.py [--prom http://127.0.0.1:9710] [--interval 5] [--csv FILE]
       [--duration S]
"""

# Standard
import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request

# Third Party
import p2penv
import promql


def query(prom: str, expr: str) -> dict[str, float]:
    with urllib.request.urlopen(
        f"{prom}/api/v1/query?" + urllib.parse.urlencode({"query": expr}), timeout=10
    ) as r:
        return promql.parse_vector(json.load(r))


def main() -> None:
    env = p2penv.load()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--prom", default=os.environ.get("PROM_URL", "http://127.0.0.1:9710")
    )
    ap.add_argument("--interval", type=float, default=5)
    ap.add_argument(
        "--csv",
        default=os.path.join(env.get("P2P_HOME", "."), "results", "monitor.csv"),
    )
    ap.add_argument("--duration", type=float, default=0)
    a = ap.parse_args()
    cols = list(promql.EXPRS)
    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    new = not os.path.exists(a.csv)
    f = open(a.csv, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["ts", "node"] + cols)
    hdr = f"{'time':8} {'node':14}" + "".join(f"{c:>13}" for c in cols)
    print(hdr, flush=True)
    t_end = time.time() + a.duration if a.duration else float("inf")
    i = 0
    while time.time() < t_end:
        vals: dict[str, dict[str, float]] = {}
        for col, (expr, kind) in promql.EXPRS.items():
            try:
                res = query(a.prom, expr)
            except (OSError, ValueError, KeyError) as e:
                print("query error", col, e, file=sys.stderr)
                res = {}
            for node, v in res.items():
                vals.setdefault(node, {})[col] = v * 100 if kind == "gauge_pct" else v
        ts = time.strftime("%H:%M:%S")
        for node in sorted(vals):
            row = [vals[node].get(c) for c in cols]
            print(
                f"{ts:8} {node:14}"
                + "".join(f"{(v if v is not None else 0):>13.2f}" for v in row),
                flush=True,
            )
            w.writerow([ts, node] + [("" if v is None else round(v, 3)) for v in row])
        f.flush()
        i += 1
        if i % 15 == 0:
            print(hdr, flush=True)
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
