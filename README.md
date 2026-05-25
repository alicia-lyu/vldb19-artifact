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

<!-- TODO(reproduction agent): fill in VM prerequisites, docker pull,
     `make paper-ready`, expected runtime, and where the resulting plots land. -->

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

<!-- TODO(reproduction agent): copy the current LeanStore flags and RocksDB
     options from the experiments code repo. The previous artifact's tables
     belong to the old workload (e.g., `dram_gib = 0.1`) and were intentionally
     dropped — do not re-import them without checking the current values. -->

### LeanStore flags

<!-- TODO(reproduction agent): table of flag / value / description -->

### RocksDB options

<!-- TODO(reproduction agent): table of option / value / description -->

## Source repositories

<!-- TODO(reproduction agent): confirm branch / tag used for the camera-ready run. -->

- LeanStore implementation (main repository): <https://github.com/alicia-lyu/leanstore>
- DBToaster implementation: <https://github.com/alicia-lyu/geodb-dbtoaster>
