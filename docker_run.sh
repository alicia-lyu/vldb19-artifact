#!/usr/bin/env bash
# docker_run.sh — run the VLDB 2026 merged-index artifact end-to-end.
#
# Pulls the pre-built image, runs each sweep cell in sequence, and
# produces paper-ready PDFs and macros under $RESULTS/paper-ready/.
#
# Prerequisites (see REPRODUCE.md §Host prerequisites for details):
#   - Docker Engine >= 24
#   - kernel.perf_event_paranoid = 0  (sudo sysctl -w kernel.perf_event_paranoid=0)
#   - SSD mounted at $SSD_MOUNT (default: /mnt/nvme/leanstore)
#   - HDD mounted at $HDD_MOUNT (default: /mnt/hdd/leanstore) -- only for tpch-headline-hdd
#   - ~10 GiB free RAM; ~30 GiB free disk for result CSVs
#
# Usage:
#   ./docker_run.sh [--results DIR] [--ssd /mnt/nvme/leanstore]
#                  [--hdd /mnt/hdd/leanstore] [--reps N] [--skip-hdd]
#                  [--skip-refresh] [--skip-dbtoaster]
#
# Env overrides (alternative to flags):
#   RESULTS       output directory (default: ./results)
#   SSD_MOUNT     SSD mount for LeanStore image files
#   HDD_MOUNT     HDD mount for the supplementary HDD cell
#   REPS          repetitions per run (default: 5)

set -euo pipefail

IMAGE="ghcr.io/alicia-lyu/leanstore:vldb26"

RESULTS="${RESULTS:-$(pwd)/results}"
SSD_MOUNT="${SSD_MOUNT:-/mnt/nvme/leanstore}"
HDD_MOUNT="${HDD_MOUNT:-/mnt/hdd/leanstore}"
REPS="${REPS:-5}"
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
        --skip-hdd)       SKIP_HDD=1;       shift   ;;
        --skip-refresh)   SKIP_REFRESH=1;   shift   ;;
        --skip-dbtoaster) SKIP_DBTOASTER=1; shift   ;;
        -h|--help)        usage             ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$RESULTS"

log() { echo "[docker_run] $(date '+%H:%M:%S') $*"; }

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
        -v "$RESULTS":/results \
        -v "$SSD_MOUNT":/mnt/nvme/leanstore \
        "${extra_mounts[@]}" \
        "$IMAGE"

    touch "$stamp"
    log "cell '$cell': done"
}

# ---------------------------------------------------------------------------
# Cell 1: SSD headline sweep (Fig. 4, Fig. 5, SST diagnostics).
# sstables.csv is captured automatically by run_paper_sweep.sh for every
# LSM run; no separate sst-diagnostics cell is needed.
# ---------------------------------------------------------------------------
run_cell "tpch-headline"

# ---------------------------------------------------------------------------
# Cell 2: HDD LSM subset (supplementary tpch_lsm_headline_hdd figure).
# ---------------------------------------------------------------------------
if [[ "$SKIP_HDD" -eq 0 ]]; then
    if [[ -d "$HDD_MOUNT" ]]; then
        run_cell "tpch-headline-hdd" \
            -v "$HDD_MOUNT":/mnt/hdd/leanstore
    else
        log "WARN: HDD mount '$HDD_MOUNT' not found -- skipping tpch-headline-hdd."
        log "      Mount the HDD and re-run (stamp logic will skip completed cells)."
        log "      Or pass --skip-hdd to suppress this warning and continue."
    fi
else
    log "cell 'tpch-headline-hdd': skipped (--skip-hdd)"
fi

# ---------------------------------------------------------------------------
# Cell 3: refresh sweep (Fig. 6, Fig. 7) -- KNOWN GAP.
# The dedicated refresh runner was not committed to the repo.
# See REPRODUCE.md section "Known gaps" for details and the workaround.
# ---------------------------------------------------------------------------
if [[ "$SKIP_REFRESH" -eq 0 ]]; then
    log "WARN: refresh cell is not yet implemented (known gap)."
    log "      Figs. 6+7 will show 'no data' panels."
    log "      See REPRODUCE.md section 'Known gaps' for details."
    log "      Pass --skip-refresh to suppress this warning."
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
