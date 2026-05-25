# Artifact for "Storing and Indexing Multiple Tables by Interesting Orderings"

This artifact reproduces the experiments in the VLDB 2026 paper
*Storing and Indexing Multiple Tables by Interesting Orderings* by
Wenhui Lyu and Goetz Graefe. The paper evaluates **multi-table merged
indexes** on TPC-H order-sharing pipelines, comparing them against
hash-based and merge-based query-time computation and against parameter-reusable
materialized views. The headline claim is that merged indexes
closely trail `Mat-View` on query latency (and match or surpass it on several
queries) while keeping the update throughput and space footprint of traditional
secondary indexes.

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

Experiments run on **TPC-H** at a scale where the base tables occupy
approximately 5 GiB of external storage. In addition to the standard schema,
the paper defines an **Invoice-extended** schema that adds an `Invoice` table
joined to each order on `orderkey`. The full Invoice-extended specification is
available at the paper's supplementary URL (see `\vldbavailabilityurl` in the
paper source).

The two families live in separate database images so that an Invoice-extended
run never shares state with a vanilla TPC-H run.

### Queries

| Query | Pipeline | Merged index used |
|---|---|---|
| Q3, Q5, Q10 | Customer–Orders–Lineitem | `MI_B` |
| Q3i, Q5i, Q10i | Customer–Orders–Invoice–Lineitem | `MI_C` |

Each query is a single order-sharing pipeline: it joins a customer record with
its orders (and, for the Invoice-extended variants, its invoices) and then
with the corresponding line items. We deliberately restrict the workload to
the components this paper targets, so the signal is not diluted by full
benchmark suites.

#### Q3, Q5, Q3i, Q5i

Standard TPC-H aggregations over the order-sharing pipeline. These four
queries follow the general approach ordering (`Base-Hash > Base-Merge >
Mat-View` on latency) on both backends.

#### Q10 and Q10i

TPC-H Q10 reports each customer's "lost revenue" from returned line items in
a given quarter. Because `orderdate` is a substitution parameter, per-customer
pre-aggregation is not possible, so the naive materialized view degenerates to
the full three-way Customer–Orders–Lineitem join (≈5.1 GiB).

Both `Mat-View` and `Merged-Idx` are therefore also evaluated in a
**partial-pre-aggregation** variant: lost revenue is materialized per order
(not per customer); at query time, partial aggregates are filtered by
`orderdate` and summed per customer. With this variant the general ordering is
restored.

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

### Correctness

All four approaches return identical answers for each query, verified by an
XOR parity check over the result sets.

## Approaches under test

The artifact evaluates four approaches. They differ only in the physical
execution of the order-sharing pipeline; everything outside the pipeline is
held constant across approaches.

| Label | Physical execution | Extra storage | DB size (GiB) |
|---|---|---|---|
| `Base-Hash` | Hash-based | None | 3.32 |
| `Base-Merge` | Order-based (e.g., merge joins) | Secondary indexes equivalent to `MI_B`, used as pre-sorted inputs | 4.18 |
| `Mat-View` | Pre-computed and materialized | One materialized view per query, plus secondary indexes required for view maintenance | 5.65 |
| `Merged-Idx` | Order-based over the merged index | `MI_B` | 4.28 |

`DB size` is the LeanStore (B-tree) on-disk footprint in GiB for one image
that jointly supports both queries in its family (Q3+Q5 or Q3i+Q5i). The two
families live in separate images. Within an image, `Base-Merge` and
`Merged-Idx` share their custkey-sorted secondaries / merged index across the
two queries, while `Mat-View` stores one view per query.

The Invoice-extended approaches mirror these designs, with `Merged-Idx`
requiring only `MI_C`.

> **Note on `Mat-View`.** Throughout this artifact, `Mat-View` refers to a
> **parameter-reusable indexed join view**, not a per-query result cache. A
> fully pre-computed result would dominate any index on query latency but is
> infeasible across the substitution parameters that TPC-H (and most analytic
> workloads) require. Within this scope, `Mat-View` serves order-sharing
> pipelines directly from pre-computed join output and thus sets the
> practical upper bound on query latency.

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
