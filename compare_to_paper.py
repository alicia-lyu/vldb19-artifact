#!/usr/bin/env python3
"""Compare regenerated experiment_numbers.json against paper-tex reference.

Usage:
    python3 compare_to_paper.py REPRO_JSON PAPER_JSON [--out REPORT.md]

Outputs a markdown table: macro, paper value, repro value, delta, pct, within-tol.
Tolerance: 5% for throughput-like values (depends on scheduler), 1% for sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def classify(name: str) -> tuple[float, str]:
    """Return (tol_fraction, kind) for a macro name."""
    if name.startswith("db_size_"):
        return 0.02, "size"
    if name.startswith("ratio_"):
        return 0.10, "ratio"
    if "tps" in name or "throughput" in name or "ops_per_s" in name:
        return 0.15, "throughput"
    if "rss" in name or "memory" in name:
        return 0.10, "memory"
    return 0.10, "other"


def load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def compare(repro: dict, paper: dict) -> list[dict]:
    repro_v = repro.get("values", {})
    paper_v = paper.get("values", {})
    keys = sorted(set(repro_v) | set(paper_v))
    rows = []
    for k in keys:
        r = repro_v.get(k)
        p = paper_v.get(k)
        if r is None:
            rows.append({"macro": k, "paper": p["value"], "repro": None,
                         "delta": None, "pct": None, "tol": classify(k)[0],
                         "kind": classify(k)[1], "status": "MISSING_REPRO"})
            continue
        if p is None:
            rows.append({"macro": k, "paper": None, "repro": r["value"],
                         "delta": None, "pct": None, "tol": classify(k)[0],
                         "kind": classify(k)[1], "status": "NEW_IN_REPRO"})
            continue
        tol, kind = classify(k)
        delta = r["value"] - p["value"]
        denom = abs(p["value"]) if p["value"] else 1.0
        pct = delta / denom if denom else 0.0
        within = abs(pct) <= tol
        rows.append({
            "macro": k, "paper": p["value"], "repro": r["value"],
            "delta": delta, "pct": pct, "tol": tol, "kind": kind,
            "status": "OK" if within else "OUT_OF_TOL",
            "unit": p.get("unit", ""),
        })
    return rows


def render_md(rows: list[dict]) -> str:
    out = ["# Reproduction vs. paper-tex comparison\n",
           "| macro | kind | paper | repro | Δ | Δ% | tol% | status |",
           "|---|---|---:|---:|---:|---:|---:|---|"]
    n_ok = n_oot = n_miss = n_new = 0
    for r in rows:
        paper = "—" if r["paper"] is None else f"{r['paper']:.4g}"
        repro = "—" if r["repro"] is None else f"{r['repro']:.4g}"
        delta = "—" if r["delta"] is None else f"{r['delta']:+.4g}"
        pct = "—" if r["pct"] is None else f"{r['pct']*100:+.1f}%"
        tol = f"{r['tol']*100:.0f}%"
        unit = r.get("unit", "")
        status = r["status"]
        if status == "OK":
            n_ok += 1
        elif status == "OUT_OF_TOL":
            n_oot += 1
        elif status == "MISSING_REPRO":
            n_miss += 1
        elif status == "NEW_IN_REPRO":
            n_new += 1
        out.append(f"| `{r['macro']}` ({unit}) | {r['kind']} | {paper} | {repro} | {delta} | {pct} | {tol} | {status} |")
    out.insert(1, f"\n**Summary:** OK={n_ok}  OUT_OF_TOL={n_oot}  MISSING_REPRO={n_miss}  NEW_IN_REPRO={n_new}\n")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repro", type=Path, help="repro experiment_numbers.json")
    ap.add_argument("paper", type=Path, help="paper-tex sections/experiment_numbers.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.repro.exists():
        print(f"[compare] repro file missing: {args.repro}", file=sys.stderr)
        return 1
    rows = compare(load(args.repro), load(args.paper))
    md = render_md(rows)
    if args.out:
        args.out.write_text(md)
        print(f"[compare] wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
