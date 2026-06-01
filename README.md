# Artifact for "Storing and Indexing Multiple Tables by Interesting Orderings"

This artifact reproduces the experiments in the VLDB 2026 paper
*Storing and Indexing Multiple Tables by Interesting Orderings* by
Wenhui Lyu and Goetz Graefe. The paper evaluates **multi-table merged
indexes** on TPC-H order-sharing pipelines against hash-based and
merge-based query-time computation and against parameter-reusable
materialized views. This document is the **specification** the
reproducible package targets: schema, queries, parameters, storage
structures under test, and the labels needed to decode the numeric
macros reported in the paper.

## How to reproduce the experiments

Execution is wrapped in two pre-built Docker images (the main LeanStore
sweep image and a DBToaster image). The host runs Docker plus a thin
Python harness; no bare-metal LeanStore build is required.

### Host prerequisites

These do not fit inside a container and must be set on the host:

1. **Kernel** — `sudo sysctl kernel.perf_event_paranoid=0` (LeanStore
   uses `perf_event_open` for its counters).
2. **SSD / HDD** — a dedicated NVMe partition mounted at the path the
   image expects, plus the optional SAS HDD for the `subsec:btree_vs_lsm`
   results. The full partitioning + mount recipe is in the source
   repo's [`LINUX_SETUP.md`](https://github.com/alicia-lyu/leanstore/blob/main/LINUX_SETUP.md);
   follow it verbatim per machine.
3. **Docker** ≥ 24, **Python** ≥ 3.10.
4. `pip install -r requirements.txt`.

### Run

```bash
docker pull ghcr.io/alicia-lyu/leanstore:vldb26
docker pull ghcr.io/alicia-lyu/dbtoaster:vldb26
make paper-ready
```

`make paper-ready` invokes `docker_run.sh` (which runs the full sweep
matrix and writes a paper-data-shaped tree under `results/`) and then
`main.py` (which calls the figure builders from
`leanstore/paper-data/scripts/` and copies the tex-referenced PDFs into
`paper-ready/`). Expected end-to-end wall-clock is *TBD — flag for
confirmation post-sweep*.

### Sweep matrix

`docker_run.sh` populates `results/` with six subtrees, one per sweep
cell:

| Subtree            | Purpose                                              | Backends   | DRAM    | Queries                       | Disk |
|--------------------|------------------------------------------------------|------------|---------|-------------------------------|------|
| `headline-ssd/`    | SSD headline (Fig. tpch\_{btree,lsm}\_headline, q10) | btree, lsm | 1.0 GiB | Q3, Q3i, Q5, Q5i, Q10, Q10i   | SSD  |
| `headline-hdd/`    | LSM HDD subset (Fig. tpch\_lsm\_headline\_hdd)       | lsm        | 1.0 GiB | Q3, Q3i, Q5, Q5i              | HDD  |
| `refresh-5L/`      | RF1+RF2 beyond-memory                                | btree, lsm | 1.0 GiB | all                           | SSD  |
| `refresh-5H/`      | RF1+RF2 with DBToaster comparison                    | btree, lsm | 9 GiB   | all                           | SSD  |
| `refresh-5HH/`     | RF1+RF2 in-memory stress                             | btree, lsm | 0.1 GiB | all                           | SSD  |
| `dbtoaster/`       | DBToaster update CSV (5H column)                     | dbtoaster  | 9 GiB   | all                           | SSD  |

Each non-dbtoaster subtree mirrors the `paper-data/<tag>/` layout
(`manifest.yaml` + `summary/`) so the existing figure builders run
unchanged. Three reps per cell; the median per-query latency is
reported.


## Workload Specification

### Dataset

Experiments run on **TPC-H**. The reported sweeps use two scale factors:
**SF 4000** on the B-tree backend (LeanStore) and **SF 10000** on the
LSM-tree backend (RocksDB). The per-backend on-disk density of the base
tables differs by ≈2.7×, so the two scale factors land at comparable
absolute footprints rather than comparable row counts; see §"Scale
factors and DBToaster scoping" below.

In addition to the standard schema, the paper defines an
**Invoice-extended** schema. It adds one new table, `Invoice`, and one
foreign-key column, `l_invoicekey`, on `Lineitem`. The two families
live in separate database images so that an Invoice-extended run never
shares state with a vanilla TPC-H run.

#### Invoice-extended schema (inline DDL)

```sql
CREATE TABLE invoice (
    i_invoicekey  INTEGER       NOT NULL PRIMARY KEY,
    i_custkey     INTEGER       NOT NULL,   -- FK → customer.c_custkey
    i_invoicedate TIMESTAMP     NOT NULL,
    i_totaldue    NUMERIC       NOT NULL,   -- Σ l_extendedprice * (1 - l_discount) * (1 + l_tax)
                                            -- over the lineitems bundled into this invoice
    i_status      CHAR(1)       NOT NULL,   -- 'P' paid, 'O' open, 'L' late
    i_paymentterm VARCHAR(25)   NOT NULL,
    i_comment     VARCHAR(79)   NOT NULL
);

ALTER TABLE lineitem
    ADD COLUMN l_invoicekey INTEGER NOT NULL;   -- FK → invoice.i_invoicekey
```

`i_totaldue` is back-filled at load time over the lineitems bundled
into the invoice (loader: `loadInvoiceAndLinkLineitem` in
`frontend/tpch/tpchi_family/`). Each lineitem's `l_invoicekey` is
assigned by the same loader.

The Invoice-extended query texts (Q3i, Q5i, Q10i) are reproduced
verbatim in Appendix A of the paper's response document
(`response.tex:276-343` in the source repo).

### Queries

| Query | Pipeline | Merged index used |
|---|---|---|
| Q3, Q5, Q10 | Customer–Orders–Lineitem | `MI_B` |
| Q3i, Q5i, Q10i | Customer–Orders–Invoice–Lineitem | `MI_C` |

Each query is a single order-sharing pipeline: it joins a customer record with
its orders (and, for the Invoice-extended variants, its invoices) and then
with the corresponding line items. The workload is restricted to the
order-sharing-pipeline component the paper targets.

#### Q3, Q5, Q3i, Q5i

Standard TPC-H aggregations over the order-sharing pipeline. The four
approaches in §"Approaches under test" (S1–S4) are reported for each.

#### Q10 and Q10i

TPC-H Q10 reports each customer's lost revenue from returned line items
in a given quarter. `orderdate` is a substitution parameter, so a
per-customer pre-aggregate is not reusable across parameter values; the
naive materialized view at this grain stores the full three-way
Customer–Orders–Lineitem join.

For these two queries the paper additionally reports two per-order
partial-aggregation variants:

- **S5** — per-order partial-aggregation variant of `Merged-Idx`
  (aCOL for Q10, aCOLI for Q10i). The orderdate-independent per-order
  aggregate is baked into the merged-index record.
- **S7** — per-order partial-aggregation variant of `Mat-View`. The
  view stores one row per order with the same baked aggregate as S5;
  query-time code filters by `orderdate` and sums per customer.

The baked per-order aggregates are:

- **Q10:** `returned_revenue = Σ l_extendedprice * (1 - l_discount)`
  filtered by `l_returnflag = 'R'`.
- **Q10i:** three columns split by invoice status —
  `paid_returns` (`i_status = 'P'`), `open_returns` (`i_status = 'O'`),
  `late_returns` (`i_status = 'L'`).

Per-order is the smallest grain at which the aggregate remains sound
for these predicates (`l_returnflag = 'R'` is a spec constant; the
parameterised `orderdate` filter is applied at the order grain above
the aggregate).

### Update workload

Update throughput is measured against the TPC-H refresh functions **RF1 and
RF2**, run at a steady rate after a warm-up phase with:

- `update_size = 1`
- `refresh_seconds = 90`

A separate sweep at a 9 GiB DRAM budget is used only when comparing against
DBToaster, which will not run below that point.

### Background load

Every measured query runs against a live background workload of two TPC-H
worker threads plus a uniform-random point-lookup stream over all base tables.
Each query is repeated three times and the median per-query latency is
reported.

### Per-query parameters

Substitution parameters and the validation seed used for parity checking
are recorded per query. The sweep rotates `--param_seed ∈ {0, 1, 2}`
across the three repetitions, picking a different parameter combination
per rep but holding the combination identical across structures within
a rep.

| Query | Parameter ranges | Validation seed |
|---|---|---|
| Q3 | `SEGMENT ∈ {BUILDING, AUTOMOBILE, FURNITURE, HOUSEHOLD, MACHINERY}`; `DATE ∈ [1995-03-01, 1995-03-31]` | `SEGMENT = BUILDING`, `DATE = 1995-03-15` |
| Q3i | Q3 ranges + `THRESHOLD ≥ 0` | `BUILDING`, `1995-03-15`, `THRESHOLD = 0` |
| Q5 | `REGION ∈ {AFRICA, AMERICA, ASIA, EUROPE, MIDDLE EAST}`; `DATE` = start of a 1-year window in `[1993, 1997]` | `REGION = ASIA`, `DATE = 1994-01-01` |
| Q5i | Same as Q5 | `REGION = ASIA`, `DATE = 1994-01-01` |
| Q10 | `DATE` monthly in `[1993-02-01, 1995-01-01]` (24 values); `LIMIT 20` | `DATE = 1993-10-01` |
| Q10i | Same as Q10 | `DATE = 1993-10-01` |

### Correctness

All evaluated structures (S1–S4 for every query; S5 and S7 additionally
for Q10 and Q10i) return identical answers for each query under the
validation seed, verified by an XOR parity check over the result sets.

## Approaches under test

The structures differ only in the physical execution of the order-sharing
pipeline; everything outside the pipeline is held constant. Each row is
selected via the `--storage_structure` flag.

| `--storage_structure` | Label | Scope | Physical execution | Extra storage |
|---|---|---|---|---|
| 1 | `Base-Merge` | all queries | Order-based execution (e.g., merge joins) over per-table secondary indexes | Secondary indexes equivalent to `MI_B` / `MI_C`, used as pre-sorted inputs |
| 2 | `Mat-View` | all queries | Pre-computed indexed join view per query, scanned at query time | One indexed join view per query plus the secondary indexes required to maintain it |
| 3 | `Merged-Idx` | all queries | Order-based execution over a multi-table merged index | `MI_B` (vanilla) or `MI_C` (Invoice-extended) |
| 4 | `Base-Hash` | all queries | Hash-based execution | None |
| 5 | `Merged-Idx`, per-order partial aggregate (aCOL / aCOLI) | Q10 / Q10i only | Order-based execution over the merged index with the per-order aggregate baked into the record | `MI_B` (Q10) or `MI_C` (Q10i) extended with the baked aggregate column(s) |
| 7 | `Mat-View`, per-order preagg | Q10 / Q10i only | Per-order partial-aggregation view; query-time code filters by `orderdate` and sums per customer | Per-order view (shares the S2 disk image; see `q10/load.tpp:122-125`, `q10i/load.tpp:337-340`) |

`Mat-View` (S2 and S7) is a parameter-reusable indexed join view, not a
per-query result cache.

S5 uses a hand-rolled walker (`acol_group_walk` for Q10,
`acoli_group_walk` for Q10i) rather than the generic `std::visit`
merged-index traversal. S7 is selected via `--storage_structure=7`,
which is equivalent to `--storage_structure=2 --q10_view_variant=preagg`
(Q10) or `--q10i_view_variant=preagg` (Q10i).

Within an image, `Base-Merge` (S1) and `Merged-Idx` (S3) share their
custkey-sorted secondaries / merged index across the queries in the
family; `Mat-View` (S2) stores one view per query.

### Joint database sizes (B-tree, SF 4000)

The numbers below are the LeanStore on-disk footprint in GiB for the
joint image at SF 4000 with `bg=2,c4` (see
`sections/experiment_numbers.json`).

| Approach | Joint DB size (GiB) | `experiment_numbers.json` macro |
|---|---|---|
| `Base-Hash` (S4 = base tables only) | 8.57 | `db_size_base_hash` |
| `Base-Merge` (S1 shared secondary indexes) | 10.81 | `db_size_base_merge` |
| `Merged-Idx` (S3 shared `MI_B`) | 11.06 | `db_size_merged_idx` |
| `Mat-View` (S2 joint, naive Q10 view) | 31.48 | `db_size_mat_view` |
| `Mat-View, partial` (S2 for Q3/Q5 + S7 for Q10) | 15.84 | `db_size_mat_view_partial` |

`Base-Hash` is base tables only; the four other rows are the joint size
when the structure jointly supports Q3, Q5, and Q10 in the vanilla
family (or Q3i, Q5i, Q10i in the Invoice-extended family — the joint
image always groups three queries together).

### Per-query storage deltas

The macros below report the per-query disk delta over the SF 4000 base
tables. `S5.q10` is reported as an incremental size on top of the shared
`MI_B` baseline.

| Per-query delta | B-tree (GiB) | LSM (GiB) | B-tree macro | LSM macro |
|---|---|---|---|---|
| S1 secondary indexes — Q3 | 2.23 | — | `db_size_secondary_btree` | — |
| S2 view — Q3 | 2.83 | 1.74 | `db_size_view_q3_btree` | `db_size_view_q3_lsm` |
| S2 view — Q5 | 3.20 | 1.68 | `db_size_view_q5_btree` | `db_size_view_q5_lsm` |
| S2 view — Q10 (naive, per-customer grain) | 16.88 | — | `db_size_q10_naive_view` | — |
| S7 view — Q10 (per-order partial aggregate) | 1.24 | — | `db_size_q10_partial_view` / `db_size_view_q10_partial_btree` | — |
| S3 merged index `MI_B` — Q3 | 2.49 | 1.81 | `db_size_mi_btree` | `db_size_mi_lsm` |
| S5 merged index — Q10 (incremental over `MI_B`) | 0.29 | — | `db_size_mi_q10_partial_btree` | — |
| S2 views joint (Q3 + Q5 + Q10 partial) | 7.26 | — | `db_size_views_total_btree` | — |
| Base tables only (S4) | 8.57 | 7.88 | `base_size_btree` | `base_size_lsm` |

At SF 4000 on the B-tree backend, the Q10 partial-aggregate variants
land at `S5.q10 = 0.29 GiB` (incremental over the shared `MI_B`) and
`S7.q10 = 1.24 GiB`.

### Scale factors and DBToaster scoping

The B-tree and LSM backends are reported at different TPC-H scale
factors so that their absolute disk footprints are comparable.

| Macro | Value | Source filter |
|---|---|---|
| `sf_btree` | 4000 | `SF_TPCH[btree][c4]` |
| `sf_lsm` | 10000 | `SF_TPCH[lsm][c4]` |
| `base_size_btree` | 8.57 GiB | base tables only, B-tree |
| `base_size_lsm` | 7.88 GiB | base tables only, LSM |
| `density_ratio_lsm_btree` | 2.7× | `(btree GiB / btree SF) / (lsm GiB / lsm SF)` |

DBToaster is reported at its largest viable scale factor (SF ≈ 0.36,
peak RSS ≈ 8.2 GiB; macros `dbtoaster_pair_tps`, `dbtoaster_peak_rss`).
The two `Mat-View` and `Merged-Idx` update-side comparisons against it
(macros `matview_pair_tps_10ll`, `merged_pair_tps_10ll`) run at SF 4000
with `dram_gib = 20` and `bg = 0`. The scale factors differ across the
two columns of that comparison.

### Query plans (Q5)

Q5 is the most demanding (six-table) query in the workload. The four
LeanStore plans — one per structure label (`Base-Hash`, `Base-Merge`,
`Mat-View`, `Merged-Idx`) — are reproduced in the `q5-plans/` directory
of this repository in LeanStore plan-dump format. They are referenced
from the response document at `response.tex:231` and `response.tex:272`.

### Calcite optimizer prototype (external)

The optimizer-side prototype described in `response.tex:251` lives in a
separate repository at <https://github.com/alicia-lyu/calcite>. It is
not part of this artifact's reproduction path; this is a pointer only.

## Environment

| Component | Specification |
|---|---|
| Machine | CloudLab `c220g2` |
| CPU | 2× Intel Xeon E5-2660 v3 @ 2.60 GHz |
| RAM | 160 GB DDR4 |
| SSD (default) | Intel DC S3500 480 GB SATA |
| HDD (where noted) | 2× 1.2 TB 10K RPM SAS |
| Engine DRAM budget | **1.0 GiB** (beyond-memory operating point; OS page cache bypassed) |
| Extended DRAM budget | 9 GiB, used only when comparing against DBToaster |
| Backends | LeanStore (B-tree), RocksDB (LSM-tree) |

Results use the SSD by default. HDD results are reported only where the paper
explicitly notes them (e.g., the LSM-tree HDD comparison in
`subsec:btree_vs_lsm`).

## Backend configuration

Flag / option values below are the ones the camera-ready sweep actually
used. They are extracted from `frontend/shared/config_standalone.cpp`,
`frontend/tpch/tpch_flags.hpp`, and `frontend/shared/RocksDB.cpp` in the
source repo; the per-cell overrides (e.g. `dram_gib` per sweep subtree)
are recorded in each `results/<subtree>/manifest.yaml`.

### LeanStore flags

| Flag                       | Value                              | Description                                                              |
|----------------------------|------------------------------------|--------------------------------------------------------------------------|
| `--tpch_scale_factor`      | 1550                               | TPC-H scale (≈5 GiB base tables). Per `manifest.yaml: sf`.               |
| `--storage_structure`      | 1 / 2 / 3 / 4                      | 1=`Base-Merge`, 2=`Mat-View`, 3=`Merged-Idx`, 4=`Base-Hash`. Swept.      |
| `--tx_seconds`             | 15                                 | Seconds per measured transaction type.                                   |
| `--warmup_seconds`         | 5                                  | Warm-up before measurement.                                              |
| `--param_seed`             | 0..2 (per rep)                     | Substitution-parameter rotation; identical across structures within a rep. |
| `--dram_gib`               | 0.1 / 1.0 / 9.0                    | Engine DRAM budget. 1.0 is the headline; 9.0 is the DBToaster point; 0.1 is the in-memory stress cell. |
| `--ssd_path`               | `/mnt/nvme/leanstore`              | Mounted per `LINUX_SETUP.md`.                                            |
| `--isolation_level`        | `si`                               | Snapshot isolation.                                                      |
| `--worker_threads`         | 4                                  | Foreground workers.                                                      |
| `--pp_threads`             | 1                                  | Page-provider threads.                                                   |
| `--tentative_skip_bytes`   | 4096 (B-tree) / 12288 (LSM)        | Per-backend default; set by each per-query executable.                   |
| `--bg_query_thread`        | true                               | Background TPC-H query thread.                                           |
| `--bg_point_lookups`       | true                               | Point-lookup noisy-neighbor stream.                                      |
| `--coli_walker_variant`    | `fused_emit`                       | Post-A2c default (q3i/PERFORMANCE.md §2 H4).                             |
| `--use_seek_skip`          | -1 (trait default)                 | Backend trait decides; -1 keeps it.                                      |

### RocksDB options

Set in `frontend/shared/RocksDB.cpp::set_options()`.

| Option                                       | Value                            | Description                                                |
|----------------------------------------------|----------------------------------|------------------------------------------------------------|
| `use_direct_reads`                           | true                             | Bypass the OS page cache (so `--dram_gib` is authoritative). |
| `use_direct_io_for_flush_and_compaction`     | true                             | Same — bypass page cache for background IO.                |
| `max_background_jobs`                        | 1                                | Single background thread (one compaction or one flush) for transparency. |
| `compression`                                | `kNoCompression`                 | Disabled — we measure the raw scan path.                   |
| `compaction_style`                           | `kCompactionStyleLevel`          | Plus `OptimizeLevelStyleCompaction()`.                     |
| `target_file_size_base`                      | 1 MiB                            | Dataset is smaller than RocksDB's 64 MiB default.          |
| `target_file_size_multiplier`                | 2                                |                                                            |
| `block_cache` (LRU)                          | `dram_gib × block_share`         | `strict_capacity_limit=true`; charged via `cache_usage_options`. |
| `table.metadata_block_size`                  | 64 KiB                           | 4 KiB default is too small for modern hardware.            |
| `table.filter_policy`                        | Bloom(bits=10, use_block_based=false) | Full filter.                                          |
| `table.index_type`                           | `kTwoLevelIndexSearch`           | Partitioned index.                                         |
| `table.partition_filters`                    | true                             | Partitioned filter blocks.                                 |
| `table.cache_index_and_filter_blocks`        | true                             | Index/filter charged to the block cache.                   |
| `table.pin_l0_filter_and_index_blocks_in_cache` | true                          |                                                            |
| `max_total_wal_size`                         | `memtable_budget × 0.1`          |                                                            |
| `write_buffer_manager`                       | `memtable_budget × 0.9`          | Shared across CFs.                                         |
| `max_write_buffer_number`                    | 10                               |                                                            |
| `rate_limiter` (recovery only)               | 10 MiB/s                         | Active during `--recover`; bulk load is unthrottled.       |

## Source repositories

The Docker images are built upstream from the LeanStore source repo;
the reproducer only needs to `docker pull` them. The source trees are
listed here for inspection and rebuild.

| Image                                       | Source                                                                                  | Commit                           |
|---------------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------|
| `ghcr.io/alicia-lyu/leanstore:vldb26`       | <https://github.com/alicia-lyu/leanstore>                                               | `<TODO: pin before camera-ready>` |
| `ghcr.io/alicia-lyu/dbtoaster:vldb26`       | same repo, subdir `dbtoaster/` (in-tree `CMakeLists.txt` + `Dockerfile` + `entrypoint.sh` + `data_files/`) | `<TODO: pin before camera-ready>` |

DBToaster is no longer a separate repository — it lives in
`leanstore/dbtoaster/` and is built as a standalone CMake project. It
only contributes the 9 GiB column of `refresh_5L_pair_latency`
(`results/dbtoaster/update_times.csv`).
