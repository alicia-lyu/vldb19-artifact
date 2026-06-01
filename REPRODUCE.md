# Reproducing the VLDB 2026 Merged-Index Paper Results

This document describes how to reproduce the experimental results in the paper
using the pre-built artifact image. All sweep logic, binaries, and plotting
scripts are bundled in a single Docker image.

Do not edit `README.md` — that file is maintained separately.

## Host prerequisites

1. **Docker Engine >= 24**

   ```bash
   docker --version   # must be >= 24.0
   sudo systemctl start docker
   ```

2. **Performance counters enabled**

   ```bash
   sudo sysctl -w kernel.perf_event_paranoid=0
   ```

   This must be set before each host reboot. Add to `/etc/sysctl.d/99-perf.conf`
   to make it persistent.

3. **SSD mount** at `/mnt/nvme/leanstore` (or set `SSD_MOUNT`):

   ```bash
   sudo mount /dev/nvme0n1p1 /mnt/nvme/leanstore
   ```

   The sweep writes LeanStore image files here (~5 GiB at the default SF).

4. **HDD mount** at `/mnt/hdd/leanstore` (or set `HDD_MOUNT`) — required only
   for the supplementary `tpch-headline-hdd` cell (Fig. tpch_lsm_headline_hdd).
   Use `--skip-hdd` to skip that cell.

5. **Disk space**: ~30 GiB free on the SSD for image files + result CSVs.

6. **RAM**: ~10 GiB free (LeanStore runs with `dram_gib=1.0` by default).

7. **ISA floor**: AVX2. The image is compiled with `-march=haswell`; no
   AVX-512 is baked in. Any post-2013 Intel/AMD CPU works. The CloudLab
   `c220g2` node (Haswell-EP) is the target platform.

## Pulling the image

```bash
docker pull ghcr.io/alicia-lyu/leanstore:vldb26
```

Pinned digest (recorded after last push — verify with `docker inspect`):

```
ghcr.io/alicia-lyu/leanstore:vldb26@sha256:<digest-here>
```

## Running the full artifact

```bash
cd /path/to/vldb19-artifact
./docker_run.sh [--results ./results] [--ssd /mnt/nvme/leanstore]
```

`docker_run.sh` runs each cell in order, stamps completed cells, and skips
them on re-run. The final `plots` cell writes all paper-ready outputs to
`$RESULTS/paper-ready/`.

### Sweep matrix

| Cell | Figures | Approx. walltime | Notes |
|------|---------|-----------------|-------|
| `tpch-headline` | Fig. 4a, 4b, 5, diag_ssd_lsm_sst_path | ~4 h (5 reps, c220g2) | SSD; btree + LSM |
| `tpch-headline-hdd` | tpch_lsm_headline_hdd (suppl.) | ~2 h | HDD; LSM only |
| `refresh` | Fig. 6, Fig. 7 | — | **Known gap** — see below |
| `dbtoaster` | refresh overlay | ~30 min | In-memory DBToaster baseline |
| `plots` | all PDFs + macros | ~5 min | Pure post-processing |

Env knobs accepted by all cells:

| Variable | Default | Description |
|----------|---------|-------------|
| `REPS` | `5` | Repetitions per (binary, cell, structure, bg) |
| `SSD_MOUNT` | `/mnt/nvme/leanstore` | SSD bind-mount point |
| `HDD_MOUNT` | `/mnt/hdd/leanstore` | HDD bind-mount point |

### Skipping cells

```bash
./docker_run.sh --skip-hdd         # skip supplementary HDD figure
./docker_run.sh --skip-refresh     # skip refresh (already a no-op gap)
./docker_run.sh --skip-dbtoaster   # skip DBToaster baseline
```

## Expected outputs

After `docker_run.sh` completes, `$RESULTS/paper-ready/` contains:

- `tpch_btree_headline.pdf` — Fig. 4a (B-tree headline throughput)
- `tpch_lsm_headline.pdf` — Fig. 4b (LSM headline throughput)
- `q10.pdf` — Fig. 5 (Q10 / Q10i breakdown)
- `refresh_5L_pair_latency.pdf` — Fig. 6 (RF pair latency; absent if refresh skipped)
- `refresh_lsm_vs_btree.pdf` — Fig. 7 (LSM vs. B-tree refresh; absent if refresh skipped)
- `tpch_lsm_headline_hdd.pdf` — supplementary HDD figure (absent if --skip-hdd)
- `paper_lsm_sst_path.pdf` — supplementary SST diagnostics
- `experiment_numbers.json` — all `\auto*` macro values
- `experiment_numbers.tex` — LaTeX macro definitions
- `space_table.txt` — Table 3 (storage sizes)

## Verifying numbers

```bash
diff -u sections/experiment_numbers.json \
        results/paper-ready/experiment_numbers.json
```

Only the `_meta.*_tag` provenance fields and any newly-dated values should
differ. Numeric `value` fields must match within a small tolerance (~5%
due to non-deterministic scheduling on shared hardware).

To run `main.py` for a host-side sanity check and copy:

```bash
python3 main.py --results results --out paper-ready-local
```

## Makefile targets

```bash
make paper-ready   # docker_run.sh + main.py (full end-to-end)
make plots         # main.py only (skip sweep if stamp exists)
make clean         # remove paper-ready/ and results/
```

## Known gaps

### Refresh figures (Fig. 6 and Fig. 7) cannot be automatically reproduced

The refresh sweep requires a dedicated runner that invokes the LeanStore
binary in RF1+RF2 update mode, recovers from per-structure image copies,
drops OS page caches between runs, and emits the
`raw/<cell>/<backend>.s<N>.csv` layout consumed by
`paper-data/scripts/summarize_refresh_10L.py`.

This runner (`build/scratch/run_refresh_10L_*.sh` on the author's Linux
machine) was not committed to the repository. The `refresh` cell in
`docker_entrypoint.sh` exits with an error when invoked.

**Impact**: `refresh_5L_pair_latency.pdf` (Fig. 6) and
`refresh_lsm_vs_btree.pdf` (Fig. 7) will be absent from `paper-ready/`.
All other figures and all `experiment_numbers.json` macro values are
unaffected.

**Workaround for committee members**: The authoring-run CSVs and figures
are included in the supplementary material at `paper-data/2026-05-30-refresh-10L/`.

**Linux follow-up**: See `LINUX_PENDING.md` in the leanstore repo for the
tracked item to commit `experiments/run_refresh_sweep.sh`.

## Troubleshooting

**`perf_event_paranoid` error in sweep log**:

```bash
sudo sysctl -w kernel.perf_event_paranoid=0
```

**`SIGILL` inside the container**:
The image requires AVX2. Check `grep avx2 /proc/cpuinfo`. On CloudLab
`c220g2` this is always present; on older hardware it may not be.

**Sweep stalls at load phase**:
The load phase writes LeanStore image files to the SSD mount. Confirm
the SSD is mounted and has at least 10 GiB free.

**Re-running a single cell after failure**:
Delete the corresponding stamp file and re-run `docker_run.sh`:

```bash
rm results/.stamp_tpch_headline
./docker_run.sh
```
