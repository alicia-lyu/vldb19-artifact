#!/usr/bin/env bash
# docker_run.sh — run the VLDB 2026 merged-index artifact end-to-end.
#
# Thin host-side driver: pull the pre-built image and run each sweep cell in
# sequence. All experiment logic lives in the image's entrypoint
# (experiments/docker_entrypoint.sh); this script just sequences the cells and
# wires up the mounts. Outputs land in $RESULTS/paper-ready/.
#
# Prerequisites (see README.md §Host prerequisites for details):
#   - Docker Engine >= 24
#   - SSD (ROTA=0) mounted at $SSD_MOUNT (default: /mnt/ssd)
#   - HDD (rotational) at $HDD_MOUNT (default: /mnt/hdd) -- only for the
#     supplementary tpch-headline-hdd cell; skipped automatically if absent.
#   - ~10 GiB free RAM; ~230 GiB free on the SSD for the full sweep
#     (per-structure images: 2 families x 2 backends x S1-S4). Use --smoke for
#     a few-GiB run.
#
# kernel.perf_event_paranoid is NOT required; only perf-counter CSV columns come
# out blank when it is non-zero, and no figure or \auto* macro depends on them.
#
# Usage:
#   ./docker_run.sh [--results DIR] [--ssd /mnt/ssd] [--hdd /mnt/hdd]
#                   [--reps N] [--smoke]
#
#   --smoke   Fast end-to-end validation (SMOKE=1): each cell runs its smallest
#             configuration (SF=15, all structures, 1 rep) so pull -> all cells
#             -> plots completes in minutes. Use it to sanity-check the host +
#             image before the multi-hour full sweep.
#
# Env overrides (alternative to flags): RESULTS, SSD_MOUNT, HDD_MOUNT, REPS.

set -euo pipefail

IMAGE="ghcr.io/alicia-lyu/leanstore:vldb26"

RESULTS="${RESULTS:-$(pwd)/results}"
SSD_MOUNT="${SSD_MOUNT:-/mnt/ssd}"
HDD_MOUNT="${HDD_MOUNT:-/mnt/hdd}"
REPS="${REPS:-5}"
SMOKE=0

usage() { grep '^#' "$0" | grep -v '^#!/' | sed 's/^# *//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results) RESULTS="$2"; shift 2 ;;
        --ssd)     SSD_MOUNT="$2"; shift 2 ;;
        --hdd)     HDD_MOUNT="$2"; shift 2 ;;
        --reps)    REPS="$2"; shift 2 ;;
        --smoke)   SMOKE=1; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$RESULTS"
log() { echo "[docker_run] $(date '+%H:%M:%S') $*"; }

SMOKE_ENV=()
[[ "$SMOKE" -eq 1 ]] && SMOKE_ENV=(-e SMOKE=1)

# The SSD mount must be a real bind-mount (the in-container entrypoint fails fast
# otherwise; this gives a clearer message up front).
if ! mountpoint -q "$SSD_MOUNT"; then
    echo "[docker_run] ERROR: SSD mount '$SSD_MOUNT' is not a mounted filesystem." >&2
    echo "[docker_run]        Mount an SSD there (README §Host prerequisites) or pass --ssd." >&2
    exit 1
fi

log "pulling $IMAGE ..."
docker pull "$IMAGE"

# Run one cell. The container entrypoint (CELL=<name>) owns all experiment logic.
run_cell() {
    local cell="$1"; shift
    log "cell '$cell' ..."
    docker run --rm \
        -e CELL="$cell" -e REPS="$REPS" "${SMOKE_ENV[@]}" \
        -v "$RESULTS":/results -v "$SSD_MOUNT":/mnt/ssd \
        "$@" \
        "$IMAGE"
}

# Cells, in order. tpch-headline drives Fig. 4a/4b + Fig. 5 (q10); refresh drives
# Fig. 7; plots reads all result dirs and writes $RESULTS/paper-ready/.
run_cell tpch-headline

# Supplementary HDD figure -- only when a real HDD is mounted (rotational media
# is enforced in-container by require_hdd).
if mountpoint -q "$HDD_MOUNT"; then
    run_cell tpch-headline-hdd -v "$HDD_MOUNT":/mnt/hdd
else
    log "HDD mount '$HDD_MOUNT' absent -- skipping tpch-headline-hdd (supplementary figure)."
fi

run_cell refresh
run_cell dbtoaster
run_cell plots

log ""
log "Artifact complete. Paper-ready outputs in: $RESULTS/paper-ready/"
log "  PDFs:          $RESULTS/paper-ready/*.pdf"
log "  Macro numbers: $RESULTS/paper-ready/experiment_numbers.{json,tex}"
log "  Space table:   $RESULTS/paper-ready/space_table.txt"
