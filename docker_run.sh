#!/usr/bin/env bash
# docker_run.sh — run the VLDB 2026 merged-index artifact end-to-end.
#
# Pulls the pre-built image, runs each sweep cell in sequence, and
# produces paper-ready PDFs and macros under $RESULTS/paper-ready/.
#
# Prerequisites (see README.md §Host prerequisites for details):
#   - Docker Engine >= 24
#   - SSD (ROTA=0) mounted at $SSD_MOUNT (default: /mnt/ssd)
#   - HDD mounted at $HDD_MOUNT (default: /mnt/hdd) -- only for tpch-headline-hdd
#   - ~10 GiB free RAM; ~30 GiB free disk for result CSVs
#
# Note: kernel.perf_event_paranoid is NOT required. Binaries run regardless;
#       only perf-counter columns in raw CSVs come out blank when the sysctl
#       is non-zero, and no paper figure or \auto* macro depends on them.
#
# Usage:
#   ./docker_run.sh [--results DIR] [--ssd /mnt/ssd] [--hdd /mnt/hdd]
#                  [--reps N] [--smoke] [--skip-hdd] [--skip-refresh] [--skip-dbtoaster]
#
#   --smoke   Fast end-to-end validation (SMOKE=1): each cell runs its smallest
#             configuration (c2 / 10HH, all structures, 1 rep) so the whole
#             pipeline — pull → all cells → plots — completes in minutes.
#             Figures come out sparse but complete; use it to sanity-check the
#             host + image before committing to the multi-hour full sweep.
#
# Env overrides (alternative to flags):
#   RESULTS       output directory (default: ./results)
#   SSD_MOUNT     SSD mount for LeanStore image files
#   HDD_MOUNT     HDD mount for the supplementary HDD cell
#   REPS          repetitions per run (default: 5)

set -euo pipefail

IMAGE="ghcr.io/alicia-lyu/leanstore:vldb26"

RESULTS="${RESULTS:-$(pwd)/results}"
SSD_MOUNT="${SSD_MOUNT:-/mnt/ssd}"
HDD_MOUNT="${HDD_MOUNT:-/mnt/hdd}"
REPS="${REPS:-5}"
SMOKE=0
SKIP_HDD=0
SKIP_REFRESH=0
SKIP_DBTOASTER=0

usage() {
    grep '^#' "$0" | grep -v '^#!/' | sed 's/^# *//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results)        RESULTS="$2";     shift 2 ;;
        --ssd)            SSD_MOUNT="$2";   shift 2 ;;
        --hdd)            HDD_MOUNT="$2";   shift 2 ;;
        --reps)           REPS="$2";        shift 2 ;;
        --smoke)          SMOKE=1;          shift   ;;
        --skip-hdd)       SKIP_HDD=1;       shift   ;;
        --skip-refresh)   SKIP_REFRESH=1;   shift   ;;
        --skip-dbtoaster) SKIP_DBTOASTER=1; shift   ;;
        -h|--help)        usage             ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$RESULTS"

log() { echo "[docker_run] $(date '+%H:%M:%S') $*"; }

# SMOKE=1 is forwarded into every cell for a fast small-scale validation run.
SMOKE_ENV=()
[[ "$SMOKE" -eq 1 ]] && SMOKE_ENV=(-e SMOKE=1)

# Host-side pre-check: the SSD mount must be a real bind-mount (the in-container
# entrypoint fails fast otherwise; this just gives a clearer message up front).
if ! mountpoint -q "$SSD_MOUNT"; then
    echo "[docker_run] ERROR: SSD mount '$SSD_MOUNT' is not a mounted filesystem." >&2
    echo "[docker_run]        Mount an SSD there (see README §Host prerequisites)" >&2
    echo "[docker_run]        or pass --ssd <path>." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Pull the image once.
# ---------------------------------------------------------------------------
log "pulling $IMAGE ..."
docker pull "$IMAGE"

# ---------------------------------------------------------------------------
# Helper: run one cell. Stamps on success so re-runs skip completed cells.
# ---------------------------------------------------------------------------
run_cell() {
    local cell="$1"; shift
    local extra_mounts=("$@")

    local stamp_name="${cell//-/_}"
    local stamp="$RESULTS/.stamp_${stamp_name}"
    if [[ -f "$stamp" ]]; then
        log "cell '$cell': already complete (stamp found) -- skipping"
        return 0
    fi

    log "starting cell '$cell' ..."
    docker run --rm \
        -e CELL="$cell" \
        -e REPS="$REPS" \
        "${SMOKE_ENV[@]}" \
        -v "$RESULTS":/results \
        -v "$SSD_MOUNT":/mnt/ssd \
        "${extra_mounts[@]}" \
        "$IMAGE"

    touch "$stamp"
    log "cell '$cell': done"
}

# ---------------------------------------------------------------------------
# Cell 1: SSD headline sweep (Fig. 4a/4b, SST diagnostics).
# sstables.csv is captured automatically by run_paper_sweep.sh for every
# LSM run; no separate sst-diagnostics cell is needed.
# (Fig. 5 / q10 comes from a dedicated q10 cell — see the TODO below.)
# ---------------------------------------------------------------------------
run_cell "tpch-headline"

# TODO(q10): once the leanstore entrypoint exposes a q10 cell (Q10/Q10i
# families, structures incl. S5/S7), run it here so q10.pdf (Fig. 5) gets data:
#   run_cell "tpch-q10"
# Until then q10.pdf renders empty.

# ---------------------------------------------------------------------------
# Cell 2: HDD LSM subset (supplementary tpch_lsm_headline_hdd figure).
# ---------------------------------------------------------------------------
if [[ "$SKIP_HDD" -eq 0 ]]; then
    if mountpoint -q "$HDD_MOUNT"; then
        run_cell "tpch-headline-hdd" \
            -v "$HDD_MOUNT":/mnt/hdd
    else
        log "WARN: HDD mount '$HDD_MOUNT' is not a mounted filesystem -- skipping tpch-headline-hdd."
        log "      Mount a rotational HDD there and re-run (stamp logic skips completed cells)."
        log "      Or pass --skip-hdd to suppress this warning and continue."
    fi
else
    log "cell 'tpch-headline-hdd': skipped (--skip-hdd)"
fi

# ---------------------------------------------------------------------------
# Cell 3: refresh sweep (Fig. 7, refresh_lsm_vs_btree).
# RF1/RF2 update throughput across the 10HH/10H/10L cells, both backends.
# ---------------------------------------------------------------------------
if [[ "$SKIP_REFRESH" -eq 0 ]]; then
    run_cell "refresh"
else
    log "cell 'refresh': skipped (--skip-refresh)"
fi

# ---------------------------------------------------------------------------
# Cell 4: DBToaster baseline (refresh_sales throughput CSV).
# ---------------------------------------------------------------------------
if [[ "$SKIP_DBTOASTER" -eq 0 ]]; then
    run_cell "dbtoaster"
else
    log "cell 'dbtoaster': skipped (--skip-dbtoaster)"
fi

# ---------------------------------------------------------------------------
# Cell 5: plots -- reads all result dirs, writes /results/paper-ready/.
# The plots cell uses --tag-map to redirect the plotter from the authored
# dated tag names in diagrams.yaml to the neutral cell dirs under /results/.
# ---------------------------------------------------------------------------
log "running plots cell ..."
docker run --rm \
    -e CELL=plots \
    "${SMOKE_ENV[@]}" \
    -v "$RESULTS":/results \
    "$IMAGE"
log "plots: done"

log ""
log "Artifact complete. Paper-ready outputs in: $RESULTS/paper-ready/"
log "  PDFs:           $RESULTS/paper-ready/*.pdf"
log "  Macro numbers:  $RESULTS/paper-ready/experiment_numbers.{json,tex}"
log "  Space table:    $RESULTS/paper-ready/space_table.txt"
log ""
log "To verify numbers against the paper source:"
log "  diff -u sections/experiment_numbers.json $RESULTS/paper-ready/experiment_numbers.json"
