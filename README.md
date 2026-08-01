# Artifact for "Storing and Indexing Multiple Tables by Interesting Orderings"

This artifact reproduces the experiments in the VLDB 2026 paper
*Storing and Indexing Multiple Tables by Interesting Orderings* by
Wenhui Lyu and Goetz Graefe. The paper evaluates **multi-table merged
indexes** on TPC-H order-sharing pipelines against hash-based and
merge-based query-time computation and against parameter-reusable
materialized views.

The reproducible package below targets a single pre-built Docker image
that bundles the LeanStore sweep binaries, the DBToaster baseline, and
the plotting scripts. The remainder of this document covers (1) how to
run it end-to-end and (2) the workload specification it implements.

## How to reproduce

All sweep logic, binaries, and plotting scripts are bundled in
`ghcr.io/alicia-lyu/leanstore:vldb26`. The host only needs Docker, a
mounted SSD (and HDD for the supplementary figure), and `make`.

The Docker entrypoint invokes binaries directly and does **not**
depend on `kernel.perf_event_paranoid`. If the host has the default
non-zero value, the binaries still run; only perf-counter columns in
raw CSVs come out blank, and no paper figure or macro depends on them.

### Host prerequisites

**Operating system: Linux.** The mount commands, device paths, and
`systemctl` invocations below are Linux-specific. The authors used
Ubuntu 20.04 on CloudLab `c220g2`.

1. **Docker Engine ≥ 24** and **`make`**. Python ≥ 3.10 is used only by the
   thin `main.py` copy wrapper (standard library only — no `pip install`
   needed; all analysis and plotting run inside the image).

   ```bash
   docker --version
   sudo systemctl start docker
   ```

2. **SSD mount** at `/mnt/ssd` (or set `SSD_MOUNT`).

   **Critical: the device must be non-rotational.** On CloudLab `c220g2`,
   both the 480 GB SATA SSD and the two 1.2 TB SAS HDDs show up as
   `sda` / `sdb` / `sdc` in non-deterministic order; verify before formatting.

   Verify, format, mount:

   ```bash
   lsblk -d -o NAME,ROTA,SIZE,MODEL          # the SSD has ROTA=0
   cat /sys/block/<dev>/queue/rotational     # must print 0

   sudo mkfs.ext4 -F -L leanstore-ssd /dev/<rota0-dev>
   sudo mkdir -p /mnt/ssd
   echo 'LABEL=leanstore-ssd /mnt/ssd ext4 defaults,noatime 0 2' \
     | sudo tee -a /etc/fstab
   sudo mount /mnt/ssd
   sudo chown $USER:$(id -gn) /mnt/ssd
   ```

   Full partitioning / mount recipe per machine is in the source repo's
   [`LINUX_SETUP.md`](https://github.com/alicia-lyu/leanstore/blob/main/LINUX_SETUP.md).

3. **HDD mount** at `/mnt/hdd` (or set `HDD_MOUNT`) — required only for
   the supplementary `tpch-headline-hdd` cell (Fig. `paper_tpch_lsm_headline_hdd`).
   Use `LABEL=leanstore-hdd` with the same fstab pattern. Without an HDD
   mount the cell is skipped automatically.

4. **Disk space**: the per-structure LeanStore image files dominate. The
   `tpch-headline` cell builds both families (vanilla + Invoice-extended) for
   both backends at S1–S4, all co-resident on the SSD:

   | Backend | SF    | Images (2 families × S1–S4) |
   | ------- | ----- | --------------------------- |
   | B-tree  | 4000  | ~132 GiB                    |
   | LSM     | 10000 | ~99 GiB                     |

   Budget **≥ 300 GiB free on `/mnt/ssd`** (≈230 GiB peak + a transient
   per-structure refresh copy + result CSVs + headroom). The `tpch-headline-hdd`
   cell writes LSM images to `/mnt/hdd` and needs **~100 GiB free there**.
   For a quick check without the full footprint, use `make smoke` (a few GiB at SF=15).

5. **RAM**: ~10 GiB free is plenty. The sweep deliberately caps the engine
   DRAM budget per run (`--dram_gib` 0.1–1.0; beyond-memory operation is the
   point being measured), so a large-memory machine is **not** required.

6. **ISA floor**: AVX2. The image is compiled with `-march=haswell`; no
   AVX-512 is baked in. Any post-2013 Intel/AMD CPU works.

### Run

```bash
docker pull ghcr.io/alicia-lyu/leanstore:vldb26
make paper-ready
```

`make paper-ready` invokes `docker_run.sh` (sweep matrix → result tree
under `results/`) and then `main.py` (copies tex-referenced PDFs and
macros into `paper-ready/`). Expected end-to-end wall-clock is **≈ 24 h**
at paper SF (≈ 18 h without the optional HDD cell); image loading
dominates (see the per-cell table below).

Pinned image digest (verify with `docker inspect --format
'{{index .RepoDigests 0}}' ghcr.io/alicia-lyu/leanstore:vldb26`):

```text
ghcr.io/alicia-lyu/leanstore:vldb26@sha256:ec7cd8333c4d46048accba5c80d275a60ac2f58cc32185203703a8f299f8cbdc
```

### Sweep matrix

`docker_run.sh` invokes the same image once per cell, with `CELL=<name>`
as the only env that changes per call. Each cell writes a paper-data-
shaped subtree (`manifest.yaml` + `raw/` + `summary/`) under
`results/<cell>/`.

| Cell                | Figures                                         | Walltime (c220g2, 5 reps) | Notes                                   |
|---------------------|-------------------------------------------------|---------------------------|-----------------------------------------|
| `tpch-headline`     | Fig. 4a, 4b, 5 (q10), `diag_ssd_lsm_sst_path` (suppl.) | ~14 h         | SSD; btree + LSM; q3/q5/q10 families; sstables.csv captured |
| `tpch-headline-hdd` | `paper_tpch_lsm_headline_hdd` (suppl.)          | ~6 h                      | HDD; LSM only; skipped if no HDD mount   |
| `refresh`           | Fig. 7 (`refresh_lsm_vs_btree`)                 | ~3 h                      | RF1/RF2 update sweep; both backends     |
| `dbtoaster`         | refresh overlay column                          | ~45 min                   | In-memory DBToaster baseline            |
| `plots`             | all PDFs + macros                               | ~5 min                    | Pure post-processing                    |

End-to-end is **≈ 24 h at paper SF** (SF 4000 btree / 10000 lsm, 5 reps) on a
c220g2. **Image loading dominates**: building the 16 SSD headline images alone
measured ~13 h (per-image 9–112 min; LSM at SF 10000 is the slowest), and the
timed query runs are minor on top. The `tpch-headline-hdd` cell (~6 h) is
**optional**: it needs a rotational HDD and is auto-skipped without one, so a
reviewer with only an SSD reproduces the main results in **≈ 18 h**. Use
`make smoke` (SF=15, 1 rep) for a few-minute validation of the whole pipeline first.

Fig. 5 (`paper_q10.pdf`) is produced by the `tpch-headline` cell — the
q3/q5/q10 vanilla and q3i/q5i/q10i invoice families sweep together.

Env knobs accepted by all cells:

| Variable     | Default     | Description                                        |
|--------------|-------------|----------------------------------------------------|
| `REPS`       | `5`         | Repetitions per (binary, cell, structure, bg)      |
| `SSD_MOUNT`  | `/mnt/ssd`  | SSD bind-mount point                               |
| `HDD_MOUNT`  | `/mnt/hdd`  | HDD bind-mount point                               |

For a fast end-to-end sanity check before the multi-hour run, use **`make smoke`**
(each cell runs its smallest configuration at SF=15, 1 rep; the whole
pull → cells → plots → copy path completes in a few minutes, and the figures
land in `paper-ready/` exactly like the full run — just sparser):

```bash
make smoke
```

### Expected outputs

After `make paper-ready` finishes, `paper-ready/` contains:

- `paper_tpch_btree_headline.pdf` — Fig. 4a (B-tree headline throughput)
- `paper_tpch_lsm_headline.pdf` — Fig. 4b (LSM headline throughput)
- `paper_q10.pdf` — Fig. 5 (Q10 / Q10i breakdown)
- `refresh_lsm_vs_btree.pdf` — Fig. 7 (LSM vs. B-tree refresh)
- `paper_tpch_lsm_headline_hdd.pdf` — supplementary HDD figure (absent if no HDD mount; see §Supplementary figures)
- `paper_lsm_sst_path.pdf` — supplementary LSM SST-path diagnostics (see §Supplementary figures)
- `experiment_numbers.json` — all `\auto*` macro values
- `experiment_numbers.tex` — LaTeX macro definitions
- `space_table.txt` — Table 3 (storage sizes)

### Verifying numbers

The regenerated `experiment_numbers.json` reflects your hardware. Absolute
numbers may differ from the paper's values (Table 3, Fig. 4–5, Fig. 7) by
10–20% depending on CPU model, SSD, and background noise on shared hardware.
What should hold regardless of hardware:

- The **ordering of throughput across structures** (e.g., `Merged-Idx` beats
  `Base-Hash` on scan-heavy queries) is the key qualitative claim.
- **Relative storage sizes** (Table 3) are data-dependent and should be close
  to the paper values at the same scale factor.

### Makefile targets

```bash
make smoke         # fast SF=15 validation: docker_run.sh --smoke + main.py
make paper-ready   # docker_run.sh + main.py (full end-to-end, ~18-24 h)
make plots         # main.py only (skip sweep if stamp exists)
make clean         # remove paper-ready/ and results/
```

### Troubleshooting

- **`SIGILL` inside the container** — the image requires AVX2. Check
  `grep avx2 /proc/cpuinfo`. On CloudLab `c220g2` this is always
  present; on older hardware it may not be.
- **Sweep stalls / fails at load phase** — the load phase writes the
  per-structure LeanStore image files to the SSD mount (~230 GiB total
  for `tpch-headline`; see §Host prerequisites). Confirm the SSD is
  mounted and has enough free space (`df -h /mnt/ssd`).
- **Re-running a single cell after failure**:

  ```bash
  rm results/.stamp_tpch_headline
  ./docker_run.sh
  ```

## Supplementary figures

Two PDFs are committed under [`paper-ready/`](paper-ready/) ahead of any
reproduction run. They are **supplementary material**, not part of the paper
main text: both figures overflowed the page budget and were moved to the
supplement, so they are checked in here rather than left to the reproduction
pipeline alone. The reproduction path (`make paper-ready`) regenerates them
in place.

- [`paper_tpch_lsm_headline_hdd.pdf`](paper-ready/paper_tpch_lsm_headline_hdd.pdf)
  — the **HDD counterpart to Fig. 4b** (LSM headline). Median query latency
  (seconds/query) for Q3, Q5, Q3i, Q5i across the four structures
  (`Base-Hash`, `Base-Merge`, `Mat-View`, `Merged-Idx`) with the LSM images
  on the rotational HDD instead of the SSD. `Base-Hash` overflows the axis
  (annotated 194 / 739 / 314 / 1716 s), making the order-based structures'
  advantage starker on rotational media than on SSD. Produced by the optional
  `tpch-headline-hdd` cell; absent if no HDD is mounted.

- [`diag_ssd_lsm_sst_path_ssd.pdf`](paper-ready/diag_ssd_lsm_sst_path_ssd.pdf)
  — **LSM SST-path diagnostics on SSD** for Q3 and Q3i. Per-transaction SST
  read time (solid bars, left axis) and total SST compaction time (hatched
  bars, right axis) across the four structures. This decomposes the LSM read
  vs. compaction trade-off behind the SSD LSM headline (Fig. 4b): the
  order-based structures cut per-transaction SST reads sharply while paying
  more in background compaction. Captured from `sstables.csv` by the
  `tpch-headline` cell.

## Workload Specification

### Dataset

Experiments run on **TPC-H**. The reported sweeps use two scale factors:
**SF 4000** on the B-tree backend (LeanStore) and **SF 10000** on the
LSM-tree backend (RocksDB). The two scale factors land at comparable
absolute footprints rather than comparable row counts (≈2.7× on-disk density
difference between backends); see §"What 1 SF means" for per-table row counts.

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

`i_totaldue` is back-filled at load time over the lineitems bundled into the
invoice. Each lineitem's `l_invoicekey` is assigned by the same loader.

The Invoice-extended query texts (Q3i, Q5i, Q10i) are reproduced verbatim
in Appendix A of the paper's response document.

### Queries

| Query | Pipeline | Merged index used |
| --- | --- | --- |
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

### Background workload

Every measured query runs against a live background workload of two TPC-H
worker threads plus a uniform-random point-lookup stream over all base tables.
Each query is repeated **five** times and the median per-query latency is
reported.

### Per-query parameters

Substitution parameters and the validation seed used for parity checking
are recorded per query. The sweep rotates `--param_seed ∈ {0..4}` across
the five repetitions, picking a different parameter combination per rep
but holding the combination identical across structures within a rep.

| Query | Parameter ranges | Validation seed |
| --- | --- | --- |
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
| --- | --- | --- | --- | --- |
| 1 | `Base-Merge` | all queries | Order-based execution (e.g., merge joins) over per-table secondary indexes | Secondary indexes equivalent to `MI_B` / `MI_C`, used as pre-sorted inputs |
| 2 | `Mat-View` | all queries | Pre-computed indexed join view per query, scanned at query time | One indexed join view per query plus the secondary indexes required to maintain it |
| 3 | `Merged-Idx` | all queries | Order-based execution over a multi-table merged index | `MI_B` (vanilla) or `MI_C` (Invoice-extended) |
| 4 | `Base-Hash` | all queries | Hash-based execution | None |
| 5 | `Merged-Idx`, per-order partial aggregate (aCOL / aCOLI) | Q10 / Q10i only | Order-based execution over the merged index with the per-order aggregate baked into the record | `MI_B` (Q10) or `MI_C` (Q10i) extended with the baked aggregate column(s) |
| 7 | `Mat-View`, per-order preagg | Q10 / Q10i only | Per-order partial-aggregation view; query-time code filters by `orderdate` and sums per customer | Per-order view (shares the S2 disk image) |

`Mat-View` (S2 and S7) is a parameter-reusable indexed join view, not a
per-query result cache.

Within an image, `Base-Merge` (S1) and `Merged-Idx` (S3) share their
custkey-sorted secondaries / merged index across the queries in the
family; `Mat-View` (S2) stores one view per query.

### Joint database sizes (B-tree, SF 4000)

LeanStore on-disk footprint in GiB for the joint image at SF 4000:

| Approach | Joint DB size (GiB) |
| --- | --- |
| `Base-Hash` (S4 = base tables only) | 8.57 |
| `Base-Merge` (S1 shared secondary indexes) | 10.81 |
| `Merged-Idx` (S3 shared `MI_B`) | 11.06 |
| `Mat-View` (S2 joint, naive Q10 view) | 31.48 |
| `Mat-View, partial` (S2 for Q3/Q5 + S7 for Q10) | 15.84 |

`Base-Hash` is base tables only; the four other rows are the joint size
when the structure jointly supports Q3, Q5, and Q10 in the vanilla
family (or Q3i, Q5i, Q10i in the Invoice-extended family — the joint
image always groups three queries together).

### Per-query storage deltas

Per-query disk delta over the SF 4000 base tables. S5.q10 is reported as
an incremental size on top of the shared `MI_B` baseline.

| Per-query delta | B-tree (GiB) | LSM (GiB) |
| --- | --- | --- |
| S1 secondary indexes — Q3 | 2.23 | — |
| S2 view — Q3 | 2.83 | 1.74 |
| S2 view — Q5 | 3.20 | 1.68 |
| S2 view — Q10 (naive, per-customer grain) | 16.88 | — |
| S7 view — Q10 (per-order partial aggregate) | 1.24 | — |
| S3 merged index `MI_B` — Q3 | 2.49 | 1.81 |
| S5 merged index — Q10 (incremental over `MI_B`) | 0.29 | — |
| S2 views joint (Q3 + Q5 + Q10 partial) | 7.26 | — |
| Base tables only (S4) | 8.57 | 7.88 |

### What 1 SF means

The TPC-H scale factor (`--tpch_scale_factor = N`) scales the row
counts of the seven non-fixed base tables linearly in `N`:

| Table | Row count at `--tpch_scale_factor = N` |
| --- | --- |
| `part` | `200 × N` |
| `supplier` | `10 × N` |
| `partsupp` | `800 × N` (exactly 4 per `part`) |
| `customer` | `150 × N` |
| `orders` | `1500 × N` |
| `lineitem` | `≈ 6000 × N` (avg 4 per order, range 1–7) |
| `invoice` (Invoice-extended only) | `3000 × N` |
| `nation` | 25 (fixed) |
| `region` | 5 (fixed) |

The loader's `--tpch_scale_factor` is **not** the standard TPC-H SF.
The per-SF multipliers above are 1000× smaller than the TPC-H §4.2.3
constants. The mapping is `standard_SF = --tpch_scale_factor / 1000`:

- `--tpch_scale_factor = 1000` ⇒ standard TPC-H SF 1 cardinalities
  (200 K part, 1.5 M orders, ≈ 6 M lineitem, etc.)
- `--tpch_scale_factor = 4000` (B-tree sweep) ⇒ standard TPC-H SF 4
  cardinalities; raw-data footprint comparable to a 4 GB TPC-H dataset.
- `--tpch_scale_factor = 10000` (LSM sweep) ⇒ standard TPC-H SF 10
  cardinalities; raw-data footprint comparable to a 10 GB TPC-H dataset.

Two loader details affect downstream sizes:

- `o_orderkey` is sparse per TPC-H §4.2.3: only the first 8 of every 32
  consecutive integers are populated initially. At `--tpch_scale_factor = N`
  the `1500 × N` populated orderkeys span a key domain of `6000 × N`.
- Every third customer (`custkey % 3 == 0`) receives no orders, so roughly
  two-thirds of customers are reachable through the order-sharing pipeline.

Reported settings:

- **`--tpch_scale_factor = 4000`** on the B-tree backend:
  800 K parts, 40 K suppliers, 3.2 M partsupp, 600 K customers, 6 M
  orders, ≈ 24 M lineitems, 12 M invoices (Invoice-extended). Base
  tables 8.57 GiB on disk.
- **`--tpch_scale_factor = 10000`** on the LSM backend: same
  per-SF multipliers at 2.5× the row counts. Base tables 7.88 GiB on
  disk.

The B-tree and LSM backends are reported at different scale factors so
that their absolute disk footprints are comparable (≈2.7× on-disk density
difference between backends).

DBToaster is reported at its largest viable scale factor (SF ≈ 0.36,
peak RSS ≈ 8.2 GiB). The `Mat-View` and `Merged-Idx` update-side
comparisons against it run at SF 4000 with `dram_gib = 20` and `bg = 0`.

### Query plans (Q5)

Q5 is the most demanding (six-table) query in the workload. The four
LeanStore plans — one per structure label (`Base-Hash`, `Base-Merge`,
`Mat-View`, `Merged-Idx`) — are provided as LeanStore plan-dump files
alongside the paper source (`q5-plans/` in the response material).

### Calcite optimizer prototype (external)

The optimizer-side prototype described in the paper response lives in a
separate repository at <https://github.com/alicia-lyu/calcite>. It is
not part of this artifact's reproduction path; this is a pointer only.

## Environment

| Component | Specification |
| --- | --- |
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

Flag / option values used in the camera-ready sweep. Per-cell overrides
(e.g. `dram_gib` per sweep subtree) are recorded in each
`results/<subtree>/manifest.yaml`.

### LeanStore flags

| Flag                       | Value                              | Description                                                              |
|----------------------------|------------------------------------|--------------------------------------------------------------------------|
| `--tpch_scale_factor`      | 4000 (btree) / 10000 (lsm)         | TPC-H scale (≈8.6 GiB / 7.9 GiB base tables)                            |
| `--storage_structure`      | 1 / 2 / 3 / 4 (+ 5, 7 for Q10)     | 1=`Base-Merge`, 2=`Mat-View`, 3=`Merged-Idx`, 4=`Base-Hash`. Swept.      |
| `--tx_seconds`             | 15                                 | Seconds per measured transaction type.                                   |
| `--warmup_seconds`         | 5                                  | Warm-up before measurement.                                              |
| `--param_seed`             | 0..4 (per rep)                     | Substitution-parameter rotation; identical across structures within a rep. |
| `--dram_gib`               | 0.1 / 1.0 / 9.0                    | Engine DRAM budget. 1.0 is the headline; 9.0 is the DBToaster point.     |
| `--ssd_path`               | `/mnt/ssd`                         | Bind-mounted into the container.                                         |
| `--isolation_level`        | `si`                               | Snapshot isolation.                                                      |
| `--worker_threads`         | 4                                  | Foreground workers.                                                      |
| `--pp_threads`             | 1                                  | Page-provider threads.                                                   |
| `--bg_query_thread`        | true                               | Background TPC-H query thread.                                           |
| `--bg_point_lookups`       | true                               | Point-lookup noisy-neighbor stream.                                      |

### RocksDB options

| Option                                       | Value                            | Description                                                |
|----------------------------------------------|----------------------------------|------------------------------------------------------------|
| `use_direct_reads`                           | true                             | Bypass the OS page cache.                                  |
| `use_direct_io_for_flush_and_compaction`     | true                             | Bypass page cache for background IO.                       |
| `max_background_jobs`                        | 1                                | Single background thread for transparency.                 |
| `compression`                                | `kNoCompression`                 | Disabled — we measure the raw scan path.                   |
| `compaction_style`                           | `kCompactionStyleLevel`          | Plus `OptimizeLevelStyleCompaction()`.                     |
| `target_file_size_base`                      | 1 MiB                            | Dataset is smaller than RocksDB's 64 MiB default.          |
| `target_file_size_multiplier`                | 2                                |                                                            |
| `block_cache` (LRU)                          | `dram_gib × block_share`         | `strict_capacity_limit=true`.                              |
| `table.metadata_block_size`                  | 64 KiB                           | 4 KiB default is too small for modern hardware.            |
| `table.filter_policy`                        | Bloom(bits=10, use_block_based=false) | Full filter.                                          |
| `table.index_type`                           | `kTwoLevelIndexSearch`           | Partitioned index.                                         |
| `table.partition_filters`                    | true                             | Partitioned filter blocks.                                 |
| `table.cache_index_and_filter_blocks`        | true                             | Index/filter charged to the block cache.                   |
| `table.pin_l0_filter_and_index_blocks_in_cache` | true                          |                                                            |
| `write_buffer_manager`                       | `memtable_budget × 0.9`          | Shared across CFs.                                         |
| `max_write_buffer_number`                    | 10                               |                                                            |

## Source repositories

The Docker image is built upstream from the LeanStore source repo;
the reproducer only needs to `docker pull` it. The source tree is
listed here for inspection and rebuild.

| Image                                  | Source                                                            | Commit                            |
|----------------------------------------|-------------------------------------------------------------------|-----------------------------------|
| `ghcr.io/alicia-lyu/leanstore:vldb26`  | <https://github.com/alicia-lyu/leanstore>                         | `sha256:24dd53af…54c3f3`         |

DBToaster is no longer a separate image — it lives in
`leanstore/dbtoaster/` and is built as part of the single artifact
image. It contributes the DBToaster baseline numbers from
`results/dbtoaster/summary/refresh_sales_dbtoaster_throughput.csv`;
rerun via `CELL=dbtoaster`.
