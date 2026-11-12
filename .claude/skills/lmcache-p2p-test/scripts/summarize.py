#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Turn stress JSONs, bench directories and Prometheus exports into Markdown tables.

usage: summarize.py [--results DIR] [--out REPORT.md]
Scans DIR (default $P2P_HOME/results) for stress_*.json,
bench_*/{populate_A,query_B,query_A_local}/bench_summary.json and prom_*.csv, and
prints (or writes) the tables.
"""

# Standard
import argparse
import csv
import glob
import json
import os
import sys

# Third Party
import p2penv


def stress_rows(path: str) -> list[dict]:
    """Rows for one stress JSON: tag, phase, requests, errors, TTFT, tok/s, hit."""
    d = json.load(open(path))
    tag = os.path.basename(path)[len("stress_") : -len(".json")]
    return [
        dict(
            tag=tag,
            phase=p["phase"],
            requests=p["requests"],
            errors=p["errors"],
            ttft_p50=p["ttft_p50"],
            ttft_p99=p["ttft_p99"],
            prefill_tok_s=p["prompt_tok_per_s"],
            hit=p["lmcache_hit_ratio"],
        )
        for p in d["phases"]
    ]


def bench_rows(path: str) -> list[dict]:
    """Rows for one bench directory (one per pass that exists)."""
    rows = []
    for name in ("populate_A", "query_B", "query_A_local"):
        f = os.path.join(path, name, "bench_summary.json")
        if os.path.isfile(f):
            r = json.load(open(f))["results"]
            rows.append(
                dict(
                    bench=os.path.basename(path),
                    pass_=name,
                    requests=r["total_requests"],
                    failed=r["failed_requests"],
                    ttft_p50_ms=round(r["p50_ttft_ms"]),
                    ttft_p99_ms=round(r["p99_ttft_ms"]),
                    input_tok_s=round(r["input_throughput"]),
                )
            )
    return rows


def prom_peaks(path: str) -> list[dict]:
    """Per-node peaks of the key series in one Prometheus export."""
    rows = list(csv.DictReader(open(path)))
    out = []
    for node in sorted({r["node"] for r in rows}):
        sub = [r for r in rows if r["node"] == node]

        def mx(c: str, sub: list[dict] = sub) -> float:
            vals = [float(r[c]) for r in sub if r.get(c) not in ("", None, "None")]
            return max(vals) if vals else 0.0

        out.append(
            dict(
                export=os.path.basename(path)[len("prom_") : -len(".csv")],
                node=node,
                load_GB_s=round(mx("load_GB/s"), 1),
                hit_l2_tok_s=round(mx("hit_l2_tok/s")),
                pf_fail_s=round(mx("pf_fail/s"), 1),
                l1_use_pct=round(mx("l1_use%"), 1),
                ttft_mean_s=round(mx("ttft_mean_s"), 2),
            )
        )
    return out


def md_table(rows: list[dict]) -> str:
    """Render dict rows as a Markdown table (empty string for no rows)."""
    if not rows:
        return ""
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> int:
    env = p2penv.load()
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", default=os.path.join(env.get("P2P_HOME", "."), "results")
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    parts = []
    stress = [
        r
        for f in sorted(glob.glob(os.path.join(a.results, "stress_*.json")))
        for r in stress_rows(f)
    ]
    if stress:
        parts.append("## Stress runs\n\n" + md_table(stress))
    bench = [
        r
        for d in sorted(glob.glob(os.path.join(a.results, "bench_*")))
        if os.path.isdir(d)
        for r in bench_rows(d)
    ]
    if bench:
        parts.append("## lmcache bench long-doc-qa\n\n" + md_table(bench))
    peaks = [
        r
        for f in sorted(glob.glob(os.path.join(a.results, "prom_*.csv")))
        for r in prom_peaks(f)
    ]
    if peaks:
        parts.append("## Prometheus peaks per run\n\n" + md_table(peaks))
    text = "\n\n".join(parts) or "(no results found)"
    if a.out:
        open(a.out, "w").write(text + "\n")
        print("wrote", a.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
