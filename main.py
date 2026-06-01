"""Host-side artifact wrapper for the VLDB 2026 merged-index artifact.

This script performs host sanity-checks and copies the paper-ready outputs
produced by ``docker_run.sh`` (specifically the ``plots`` cell) into the
local ``paper-ready/`` directory for easy inspection.

The sweep, analysis, and plotting all happen inside the container.
``docker_run.sh`` is the entry point for running the full artifact;
this script is a thin convenience wrapper around its outputs.

Expected layout after ``docker_run.sh`` completes:

    $RESULTS/paper-ready/
      tpch_btree_headline.pdf     (Fig. 4a)
      tpch_lsm_headline.pdf       (Fig. 4b)
      q10.pdf                     (Fig. 5)
      refresh_5L_pair_latency.pdf (Fig. 6)  [absent when refresh cell was skipped]
      refresh_lsm_vs_btree.pdf    (Fig. 7)  [absent when refresh cell was skipped]
      tpch_lsm_headline_hdd.pdf   (supplementary)
      paper_lsm_sst_path.pdf      (supplementary)
      experiment_numbers.json
      experiment_numbers.tex
      space_table.txt
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _check_docker() -> bool:
    try:
        subprocess.run(["docker", "info"], check=True,
                       capture_output=True, timeout=10)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_perf_event_paranoid() -> bool:
    p = Path("/proc/sys/kernel/perf_event_paranoid")
    if not p.exists():
        return True   # not Linux — skip check
    try:
        val = int(p.read_text().strip())
        return val <= 0
    except ValueError:
        return True


def _check_mount(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir()) if path.exists() else False


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results", type=Path, default=Path("results"),
                    help="results directory written by docker_run.sh (default: ./results)")
    ap.add_argument("--out", type=Path, default=Path("paper-ready"),
                    help="destination for paper-ready outputs (default: ./paper-ready)")
    ap.add_argument("--skip-checks", action="store_true",
                    help="skip host sanity checks (Docker, perf_event_paranoid, mounts)")
    args = ap.parse_args()

    # ------------------------------------------------------------------ checks
    if not args.skip_checks:
        ok = True

        if not _check_docker():
            print("[main] ERROR: Docker is not running or not installed.", file=sys.stderr)
            print("[main]        Install Docker Engine >= 24 and start the daemon.",
                  file=sys.stderr)
            ok = False

        if not _check_perf_event_paranoid():
            print("[main] ERROR: kernel.perf_event_paranoid > 0.", file=sys.stderr)
            print("[main]        Run: sudo sysctl -w kernel.perf_event_paranoid=0",
                  file=sys.stderr)
            ok = False

        if not ok:
            print("[main] Host pre-check failed. Fix the above before running docker_run.sh.",
                  file=sys.stderr)
            return 2

    # ------------------------------------------------------------------ copy
    paper_ready_src = args.results / "paper-ready"
    if not paper_ready_src.exists():
        print(f"[main] paper-ready dir not found: {paper_ready_src}", file=sys.stderr)
        print("[main] Run ./docker_run.sh first to produce the artifact outputs.",
              file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    copied = []
    for src in sorted(paper_ready_src.iterdir()):
        dest = args.out / src.name
        shutil.copyfile(src, dest)
        copied.append(src.name)
        print(f"[main] {src.name} -> {dest}")

    if not copied:
        print(f"[main] WARNING: paper-ready dir is empty: {paper_ready_src}",
              file=sys.stderr)
        return 1

    print(f"\n[main] Copied {len(copied)} file(s) to {args.out}/")
    print("[main] To verify numbers against the paper source:")
    print("  diff -u sections/experiment_numbers.json "
          f"{args.out}/experiment_numbers.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
