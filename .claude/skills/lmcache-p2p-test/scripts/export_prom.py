#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export Prometheus range data for a time window to CSV (one row per step per node).

Accepts explicit timestamps or a stress_<tag>.json (uses its phases' start/end, padded).
usage: export_prom.py (<start_ts> <end_ts> | --from-json stress_x.json) <out.csv>
       [--prom URL]
"""

# Standard
import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request

# Third Party
import promql


def window_from_json(path: str, pad: float = 10.0) -> tuple[float, float]:
    """Return (start, end) covering every phase of a stress JSON, plus ``pad`` s."""
    d = json.load(open(path))
    return min(p["start_ts"] for p in d["phases"]) - pad, max(
        p["end_ts"] for p in d["phases"]
    ) + pad


def export(prom: str, start: float, end: float, out: str, step: int = 5) -> int:
    table: dict[tuple[float, str], dict[str, float]] = {}
    for col, (expr, _) in promql.EXPRS.items():
        qs = urllib.parse.urlencode(
            {"query": expr, "start": start, "end": end, "step": step}
        )
        with urllib.request.urlopen(f"{prom}/api/v1/query_range?{qs}", timeout=30) as r:
            for key, v in promql.parse_matrix(json.load(r)).items():
                table.setdefault(key, {})[col] = v
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ts", "node"] + list(promql.EXPRS))
        for key in sorted(table):
            w.writerow([key[0], key[1]] + [table[key].get(c) for c in promql.EXPRS])
    return len(table)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("args", nargs="+")
    ap.add_argument("--from-json")
    ap.add_argument(
        "--prom", default=os.environ.get("PROM_URL", "http://127.0.0.1:9710")
    )
    a = ap.parse_args()
    if a.from_json:
        start, end = window_from_json(a.from_json)
        out = a.args[0]
    else:
        start, end, out = float(a.args[0]), float(a.args[1]), a.args[2]
    n = export(a.prom, start, end, out)
    print(f"wrote {out}: {n} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
