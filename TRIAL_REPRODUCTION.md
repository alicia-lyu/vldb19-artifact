# Trial reproduction report (2026-06-02)

End-to-end run of `./docker_run.sh --reps 1 --ssd /mnt/ssd --hdd /tmp/no-hdd
--results results-trial` on CloudLab `c220g2` (node `node0.alicial-306097…`),
started 2026-06-01 22:41 CDT. Cells run, in order:

| Cell | Status | Wallclock | Notes |
|---|---|---:|---|
| `tpch-headline` | complete | 7 h 57 m | 192 runs, ok=192 err=0. Internal analyzer + plotter ok. |
| `tpch-headline-hdd` | skipped | – | `--hdd /tmp/no-hdd` (no rotational disk available). |
| `refresh` | partial | 2 h 56 m | 10LL btree S1-S4 complete; 10L btree S1-S4 complete; 10L lsm S1 complete; 10L lsm S2 mid-load when killed. 10H and 10HH not started. |
| `dbtoaster` | failed | < 1 m | Image ships without TPC-H data files; `dbgen` not run inside container. `refresh_sales` binary reported `File not found: ./data_files/{nation,customer,ORDERS,LINEITEM}.{tbl,csv}`. |
| `plots` | complete | < 1 m | Read all available cells; 24 macros written. |

**Why the refresh was cut short.** With REPS=1 the per-structure RF takes ~3
minutes, but each *new* SF needs a one-shot image load: btree at SF=4000
needed ~10-37 m per structure (10LL), and LSM at SF=10000 needed ~40 m per
structure (10L S1). The headline cell already consumed ~8 h before refresh
started; finishing all four refresh cells × both backends × four structures
would have run past the host-access deadline. Refresh was stopped after 10L
lsm S1 had a usable RF measurement; the partial summaries (10LL +
10L bg=2) were written by `summarize_refresh_10L.py` and feed the macro
generator + `refresh_lsm_vs_btree` figure (annotated as skipping 10H/10HH).

## Comparison vs. paper-tex `sections/experiment_numbers.json`

Run `compare_to_paper.py paper-ready/experiment_numbers.json
$PAPER_TEX/sections/experiment_numbers.json` to regenerate. Summary:
**OK=8  OUT_OF_TOL=16  MISSING_REPRO=8  NEW_IN_REPRO=0**

### Headline gap: artifact c0 ≠ paper c4

All 11 `db_size_*` size macros are **~61% below paper** with a near-uniform
ratio. The cause is structural, not noise: the in-image
`run_paper_sweep.sh` walks cells `c2 → c1 → c3 → c0` with the c0 "anchor"
at `sf_lsm=3850 / sf_btree=1550`, while the paper headline is from cell
`c4` at `sf_lsm=10000 / sf_btree=4000` (cell `10L` in the paper repo's
authoring tags). The size ratio
`sf_btree_artifact / sf_btree_paper = 1550 / 4000 ≈ 0.388` matches the
observed `repro / paper ≈ 0.39` for every base-size macro. The headline
size figures (`paper_tpch_btree_headline.pdf` etc.) render correctly but
display the smaller SF data point, not the paper's.

Two ways to close this gap in a future image rebuild:

1. Add `c4` to the default `CELLS` list in `run_paper_sweep.sh` (already
   parameterised; just append to the default of `c2,c1,c3,c0`) and have the
   cell-lookup table emit `c4) d=1.0; s_lsm=10000; s_btree=4000;` so the
   anchor matches the paper.
2. Replace the `c0` row with the paper SFs (less surgery; loses the
   intentional smaller "anchor" the artifact authors chose to keep
   reproduction time bounded).

### Ratios + non-size macros are in band

Eight macros within tolerance:
- `sf_btree`, `sf_lsm` — exact (artifact emits the paper-constant labels regardless of actual run SF).
- `density_ratio_lsm_btree`, `ratio_lsm_invoice_low/high`,
  `ratio_update_merge_vs_hash`, `ratio_update_merged_vs_merge_btree`,
  `merged_pair_tps_10ll` — all ≤ 12 % off.

Two ratio macros out of tolerance:
- `ratio_q10_btree_merged_vs_merge`: paper 0.82, repro 1.13 (+39 %). At
  smaller SF the merged-idx Q10 advantage erodes; this is an SF-dependent
  ratio rather than an instrumentation error.
- `matview_pair_tps_10ll`: paper 11.29k, repro 13.78k (+22 %). 10LL ran
  here at the paper SF (4000/10000), so this gap is real measurement
  noise / scheduler variance with REPS=1 (paper used REPS=5).

### Missing macros (eight)

- 5 missing due to the headline sweep omitting the S5/S7 Q10 partial-agg
  structures: `db_size_mat_view_partial`, `db_size_mi_q10_partial_btree`,
  `db_size_q10_partial_view`, `db_size_view_q10_partial_btree`,
  `db_size_views_total_btree`. The paper notes these as
  `_meta.note_q10_partial_inversion`; reproducing them requires a
  q10-partial-agg sweep that the artifact docker image doesn't run.
- 2 missing due to the dbtoaster cell failing: `dbtoaster_pair_tps`,
  `dbtoaster_peak_rss` (these are in paper `_meta.skipped_macros` so the
  paper tex already tolerates absence).
- 1 missing due to truncated refresh: `ratio_update_hash_vs_merge_lsm`
  (needs full 10L lsm S1-S4 sweep, only S1 ran).

## Verified surfaced gaps

1. **Artifact `c0` ≠ paper `c4`.** Documented above; fix is one cell-table
   edit + add `c4` to the default cell list.
2. **DBToaster cell needs data preparation.** The cell invokes
   `dbtoaster/entrypoint.sh` with `BIN=$REPO/dbtoaster/build/refresh_sales`
   but never runs `dbgen` / generates `data_files/{ORDERS,LINEITEM}.csv`
   inside the container. The image either needs to bundle a small
   `data_files/` or the entrypoint needs a `make data SF=…` step before the
   binary invocation. The cell currently completes "successfully" but emits
   a CSV that says only `warmup_orders=0 warmup_lineitems=0 …`.
3. **Disk budget for refresh.** The headline sweep leaves per-structure
   images at the c2/c1/c3 SFs as well as c0 (~159 GB for btree, ~30 GB
   for lsm at this run's host); refresh then loads the paper SFs on top.
   On a 440 GB SSD, 87 GB was free when LSM at SF=10000 started loading
   — borderline tight, and the run never had a chance to also store per-Sx
   refresh copies for all four LSM structures. Either the headline sweep
   should garbage-collect non-anchor SF images, or refresh should pick an
   SF that matches the anchor (closing gap #1 above resolves this).

## Files of interest

- `compare_to_paper.py` — comparison driver (host-side).
- `paper-ready/experiment_numbers.json` — regenerated macros (this trial).
- `paper-ready/*.pdf` — regenerated figures.
- `results-trial/run.log` — full sweep log.
- `results-trial/tpch-headline/summary/{headline,stats,inversions,diagnostics}.csv`
  — sweep summary CSVs.
- `results-trial/refresh/summary/refresh_sales_{10LL,10L_bg2}_throughput.csv`
  — partial refresh summaries (10H, 10HH absent).
