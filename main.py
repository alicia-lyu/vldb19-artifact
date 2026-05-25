"""Plot orchestrator for the VLDB 2026 merged-index artifact.

CSV-in, PDF-out. Does not build LeanStore, does not run the sweep, does
not pull docker images — that is `docker_run.sh`'s job. This script
consumes the paper-data-shaped result tree produced there and shells out
to the figure builders under ``leanstore/paper-data/scripts/``.

Expected layout under ``--results``:

    results/
      headline-ssd/       -> paper-data tag for SSD headline (btree + lsm, all 6 queries, S1..S4 @ 1.0 GiB)
        manifest.yaml
        summary/
      headline-hdd/       -> LSM HDD subset for {q3,q3i,q5,q5i}
      refresh-5L/         -> RF1+RF2 sweep at 1.0 GiB (beyond-memory)
      refresh-5H/         -> RF1+RF2 sweep at 9 GiB (DBToaster point)
      refresh-5HH/        -> RF1+RF2 sweep at 0.1 GiB (in-memory stress)
      dbtoaster/
        update_times.csv  -> emitted by the in-tree dbtoaster image

The mapping from result subtree to tex-referenced figure filename is in
``FIGURES`` below. To add a figure, append a row; to swap the underlying
sweep, change the subtree name.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional


SCRIPTS_DIR_DEFAULT = Path(
    os.environ.get(
        "LEANSTORE_SCRIPTS",
        Path.home() / "Local" / "leanstore" / "paper-data" / "scripts",
    )
)


@dataclass
class Figure:
    """One tex-referenced PDF and the recipe to produce it."""
    out_name: str                     # filename as cited by experiments_revised.tex
    builder: Callable[["Ctx", "Figure"], Optional[Path]]
    subtree: str                      # results/<subtree>/ (paper-data layout)
    extra: dict                       # builder-specific knobs


@dataclass
class Ctx:
    results: Path
    out: Path
    scripts: Path

    def script(self, name: str) -> Path:
        return self.scripts / name


# ---------------------------------------------------------------------------
# Builders. Each shells out to a leanstore/paper-data/scripts/ entry point
# with --root=results/<subtree>/.. and --tag=<subtree>. Returns the
# absolute path to the produced PDF under figures/paper/ (or None on miss).
# ---------------------------------------------------------------------------

def _run(cmd: List[str]) -> None:
    print("[main] $", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _ensure_summary(ctx: Ctx, subtree: str) -> Path:
    """Run analyze_paper_sweep.py if summary/ is missing. Idempotent."""
    tree = ctx.results / subtree
    summary = tree / "summary"
    if not summary.exists():
        _run(["python3", str(ctx.script("analyze_paper_sweep.py")),
              "--tag", subtree, "--root", str(ctx.results)])
    return tree


def build_paper_sweep(ctx: Ctx, fig: Figure) -> Optional[Path]:
    tree = _ensure_summary(ctx, fig.subtree)
    _run(["python3", str(ctx.script("plot_paper_sweep.py")),
          "--tag", fig.subtree, "--root", str(ctx.results),
          "--figures", fig.extra["figure_name"]])
    # plot_paper_sweep auto-suffixes _ssd / _hdd via SweepData.paper_name().
    paper_dir = tree / "figures" / "paper"
    candidates = sorted(paper_dir.glob(fig.extra["figure_name"] + "*.pdf"))
    return candidates[0] if candidates else None


def build_refresh_sales(ctx: Ctx, fig: Figure) -> Optional[Path]:
    tree = _ensure_summary(ctx, fig.subtree)
    cmd = ["python3", str(ctx.script("plot_refresh_sales.py")),
           "--tag", fig.subtree, "--root", str(ctx.results)]
    dbt = ctx.results / "dbtoaster" / "update_times.csv"
    if dbt.exists():
        cmd += ["--dbtoaster", str(dbt)]
    _run(cmd)
    paper_dir = tree / "figures" / "paper"
    candidates = sorted(paper_dir.glob("refresh_5L_pair_latency*.pdf"))
    return candidates[0] if candidates else None


def build_refresh_lsm_vs_btree(ctx: Ctx, fig: Figure) -> Optional[Path]:
    _ensure_summary(ctx, "refresh-5L")
    _ensure_summary(ctx, "refresh-5H")
    _run(["python3", str(ctx.script("plot_refresh_lsm_vs_btree.py")),
          "--tag-5L", "refresh-5L", "--tag-5H", "refresh-5H",
          "--root", str(ctx.results),
          "--out-tag", "refresh-5L",
          "--basename", "refresh_lsm_vs_btree_5L_5H"])
    paper_dir = ctx.results / "refresh-5L" / "figures" / "paper"
    candidates = sorted(paper_dir.glob("refresh_lsm_vs_btree_5L_5H*.pdf"))
    return candidates[0] if candidates else None


def build_lsm_diagnostics(ctx: Ctx, fig: Figure) -> Optional[Path]:
    tree = _ensure_summary(ctx, fig.subtree)
    _run(["python3", str(ctx.script("plot_lsm_s3_vs_s2_diagnostics.py")),
          "--tag", fig.subtree, "--root", str(ctx.results)])
    # script writes to figures/diagnostics/ or figures/paper/.
    for sub in ("paper", "diagnostics"):
        paper_dir = tree / "figures" / sub
        if paper_dir.exists():
            for pat in ("diag_ssd_lsm_sst_path*.pdf",
                        "lsm_s3_vs_s2_diagnostics*.pdf"):
                hits = sorted(paper_dir.glob(pat))
                if hits:
                    return hits[0]
    return None


def build_q10(ctx: Ctx, fig: Figure) -> Optional[Path]:
    tree = _ensure_summary(ctx, fig.subtree)
    _run(["python3", str(ctx.script("plot_paper_sweep.py")),
          "--tag", fig.subtree, "--root", str(ctx.results),
          "--figures", "paper_q10"])
    paper_dir = tree / "figures" / "paper"
    candidates = sorted(paper_dir.glob("paper_q10*.pdf"))
    return candidates[0] if candidates else None


# ---------------------------------------------------------------------------
# Figure catalog — one row per tex-referenced PDF.
# ---------------------------------------------------------------------------

FIGURES: List[Figure] = [
    Figure("tpch_btree_headline.pdf", build_paper_sweep, "headline-ssd",
           {"figure_name": "paper_tpch_btree"}),
    Figure("tpch_lsm_headline.pdf",   build_paper_sweep, "headline-ssd",
           {"figure_name": "paper_tpch_lsm"}),
    Figure("tpch_lsm_headline_hdd.pdf", build_paper_sweep, "headline-hdd",
           {"figure_name": "paper_tpch_lsm"}),
    Figure("q10.pdf", build_q10, "headline-ssd", {}),
    Figure("refresh_5L_pair_latency.pdf", build_refresh_sales, "refresh-5L", {}),
    Figure("refresh_lsm_vs_btree_5L_5H.pdf", build_refresh_lsm_vs_btree,
           "refresh-5L", {}),
    Figure("diag_ssd_lsm_sst_path_ssd.pdf", build_lsm_diagnostics,
           "headline-ssd", {}),
]


def build_space_table(ctx: Ctx) -> Optional[Path]:
    """Table 3 (DB sizes). Output is .tex; copied alongside the PDFs."""
    _ensure_summary(ctx, "headline-ssd")
    out_tex = ctx.out / "exp-baselines.tex"
    _run(["python3", str(ctx.script("space_table.py")),
          "--tag", "headline-ssd", "--root", str(ctx.results),
          "--out", str(out_tex)])
    return out_tex if out_tex.exists() else None


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("results"),
                    help="results tree produced by docker_run.sh")
    ap.add_argument("--out", type=Path, default=Path("paper-ready"),
                    help="destination for tex-referenced PDFs")
    ap.add_argument("--scripts", type=Path, default=SCRIPTS_DIR_DEFAULT,
                    help="leanstore/paper-data/scripts/ directory "
                         "(or set $LEANSTORE_SCRIPTS)")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"[main] missing results tree: {args.results}", file=sys.stderr)
        return 2
    if not args.scripts.exists():
        print(f"[main] missing scripts dir: {args.scripts} "
              "(set --scripts or $LEANSTORE_SCRIPTS)", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(results=args.results.resolve(),
              out=args.out.resolve(),
              scripts=args.scripts.resolve())

    failures: List[str] = []
    for fig in FIGURES:
        try:
            src = fig.builder(ctx, fig)
        except subprocess.CalledProcessError as e:
            print(f"[main] {fig.out_name}: builder failed ({e})", file=sys.stderr)
            failures.append(fig.out_name)
            continue
        if src is None:
            print(f"[main] {fig.out_name}: builder produced no PDF",
                  file=sys.stderr)
            failures.append(fig.out_name)
            continue
        dest = ctx.out / fig.out_name
        shutil.copyfile(src, dest)
        print(f"[main] {fig.out_name} <- {src}")

    try:
        build_space_table(ctx)
    except subprocess.CalledProcessError as e:
        print(f"[main] exp-baselines.tex: builder failed ({e})", file=sys.stderr)
        failures.append("exp-baselines.tex")

    if failures:
        print(f"[main] {len(failures)} figure(s) missing: {failures}",
              file=sys.stderr)
        return 1
    print(f"[main] wrote {len(FIGURES)} figures to {ctx.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
