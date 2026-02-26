import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

COLORS = OrderedDict([
    ('trad_idx_mj',     'C0'),
    ('trad_idx',        'C0'),
    ('trad_idx_hj',     'C3'),
    ('mat_view',        'C1'),
    ('traditional IVM', 'C1'),
    ('DBToaster IVM',   'C4'),
    ('merged_idx',      'C2'),
])

# Maps raw CSV method names to the display names used in COLORS.
METHOD_RENAME = {'base_idx': 'trad_idx_mj', 'hash': 'trad_idx_hj'}

DBTOASTER_CSV = 'update_times.csv'


def _color_order(columns) -> list:
    """Return items from `columns` filtered and reordered by COLORS key order."""
    return [col for col in COLORS if col in columns]


def _bar_colors(df: pd.DataFrame) -> list:
    """Return a per-column color list aligned to df's current column order."""
    return [COLORS.get(col, 'C5') for col in df.columns]


class DataPerConfig:
    """
    Holds benchmark data for one (scale, bgw_pct, dram_size) configuration and
    produces all associated plots and LaTeX tables.
    """

    _ENGINE_MAP = {'geo_btree': 'B-Tree', 'geo_lsm': 'LSM-Tree'}

    def __init__(self, scale: int, bgw_pct: int, dram_size: float,
                 df: pd.DataFrame, parent_dir: str):
        self.scale = scale
        self.bgw_pct = bgw_pct
        self.dram_size = dram_size
        self.engine = self._ENGINE_MAP.get(parent_dir, parent_dir)
        self.directory = f"{parent_dir}/scale{scale}/bgw{bgw_pct}-dram{dram_size}"
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.raw_df = df
        self.anonymized_df, self.anonymized_df_cpu, self.y_label, self.unit = \
            self.group_and_anonymize()
        self.anonymized_df_cpu.to_latex(
            f"{self.directory}/anonymized_cpu.tex", float_format="%.1f")
        self.size_df = self.get_size_df()
        self.size_df.T.to_latex(f"{self.directory}/size.tex", float_format="%.3f")
        print("----------------------------------------------------------------------")
        print(f"Initialized DataPerConfig: scale={scale}, bgw_pct={bgw_pct}, "
              f"dram_size={dram_size} GiB")

    def __str__(self):
        return (f"DataPerConfig(scale={self.scale}, bgw_pct={self.bgw_pct}, "
                f"dram_size={self.dram_size} GiB)")

    @property
    def _update_rows(self) -> pd.DataFrame:
        """Rows of anonymized_df corresponding to Update transactions."""
        return self.anonymized_df[self.anonymized_df.index.str.contains('Update')]

    # ------------------------------------------------------------------
    # CPU utilization
    # ------------------------------------------------------------------

    def plot_cpu(self):
        if self.anonymized_df_cpu.empty:
            print(f"No CPU data to plot for {self.directory}")
            return
        cpu_df = self.anonymized_df_cpu[~self.anonymized_df_cpu.index.str.contains('Update')]
        fig, ax = plt.subplots(figsize=(len(cpu_df.columns) + 1, 3))
        sns.heatmap(cpu_df, annot=True, fmt=".1f", cmap="Reds", vmin=0, vmax=100,
                    cbar_kws={'format': '%.0f%%'}, linewidths=.5, ax=ax)
        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        plt.savefig(f'{self.directory}/cpu_utilization.png', bbox_inches='tight', dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # DB size
    # ------------------------------------------------------------------

    def get_size_df(self) -> pd.DataFrame:
        size_s = (
            self.raw_df[['method', 'size (MiB)']]
            .groupby('method').mean()['size (MiB)']
            .round(0).astype(int)
        )
        size_s.rename(index=METHOD_RENAME, inplace=True)
        size_s = size_s[_color_order(size_s.index)]
        size_s = (size_s / 1024).round(3)   # MiB -> GiB
        return pd.DataFrame({'size (GiB)': size_s}).T

    def _plot_size(self, ax):
        series_to_draw = self.size_df.squeeze()
        if not isinstance(series_to_draw, pd.Series):
            print(f"Not enough size data to plot for {self.directory}")
            return
        self.size_df.plot(kind='bar', ylabel=self.size_df.index[0], ax=ax,
                          xlabel='', legend=False, color=_bar_colors(self.size_df))
        ax.set_xticks([], [])

    def plot_size(self):
        if self.size_df.empty:
            print(f"No size data to plot for {self.directory}")
            return
        fig, ax = plt.subplots(figsize=(3, 3))
        self._plot_size(ax)
        plt.savefig(f'{self.directory}/size.png', bbox_inches='tight', dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Throughput / elapsed-time helpers
    # ------------------------------------------------------------------

    def _prepare_tput_data(self, df: pd.DataFrame):
        """Convert throughput (TX/s) to elapsed time with an appropriate unit."""
        max_tput = df['TPut (TX/s)'].max()
        if max_tput < 1_000:
            unit, scale = 's', 1
        elif max_tput < 1_000_000:
            unit, scale = 'ms', 1_000
        else:
            unit, scale = 'us', 1_000_000
        col = f'Elapsed ({unit})'
        df_copy = df.copy()
        df_copy[col] = (scale / df_copy['TPut (TX/s)']).round(3)
        return df_copy, col, unit

    def optimal_skip(self, tx_df: pd.DataFrame, tx: str, y_label: str):
        merged_df = tx_df[tx_df['method'] == 'merged_idx'].drop(columns=['method'])
        skip_means = merged_df.groupby('tentative_skip_bytes')[y_label].mean()
        if skip_means.empty:
            return None
        optimal = skip_means.idxmax()
        print(f'optimal skip for {tx}: {optimal}')
        return optimal

    # ------------------------------------------------------------------
    # Data grouping / anonymization
    # ------------------------------------------------------------------

    def group_and_anonymize(self):
        df_copy, y_label, unit = self._prepare_tput_data(self.raw_df)
        cpu_label = (
            "worker_0 CPU Util (%)" if 'worker_0 CPU Util (%)' in df_copy.columns
            else "t0 CPU Util (%)"
        )
        query_suffix = {"join": "join", "distinct": "count distinct", "mixed": "count"}
        bucket_sort = OrderedDict({"join": [], "mixed": [], "distinct": [], "maintain": []})

        for tx, tx_df in df_copy.groupby('tx'):
            if tx.endswith("-n"):   # "-n" variants not yet included
                continue
            if len(tx_df['method'].unique()) < 3:
                print(f"Skipping {tx}: not enough unique methods.", tx_df['method'].unique())
                continue
            tx_df = tx_df.drop(columns='tx')
            optimal = self.optimal_skip(tx_df, tx, y_label)
            tx_df = tx_df[
                (tx_df['tentative_skip_bytes'] == optimal) |
                (tx_df['method'] != 'merged_idx')
            ]
            category = tx.split('-')[0]
            assert category in bucket_sort, f"Unknown tx category: {category}"
            y_mean = tx_df.groupby('method').mean()[y_label].round(3)
            data = [y_mean.rename(index=METHOD_RENAME)]
            if cpu_label in tx_df.columns:
                cpu_mean = tx_df.groupby('method').mean()[cpu_label].round(1)
                data.append(cpu_mean.rename(index=METHOD_RENAME))
            bucket_sort[category].append(tuple(data))

        def _build_df(rows: dict) -> pd.DataFrame:
            frame = pd.DataFrame(rows).T
            frame.rename(columns=METHOD_RENAME, inplace=True)
            frame = frame[_color_order(frame.columns)]
            frame.index.name = 'tx'
            return frame

        tput_rows, cpu_rows = {}, {}
        tx_count = 0
        for category, series_list in bucket_sort.items():
            if category == 'maintain':
                assert len(series_list) <= 1, "Expected at most one maintain tx"
                if series_list:
                    tput_rows['Update'] = series_list[0][0]
                    if len(series_list[0]) > 1:
                        cpu_rows['Update']  = series_list[0][1]
                    else:
                        cpu_rows['Update']  = pd.Series(dtype=float)  # Empty series if no CPU data
                continue
            for series_tuple in series_list:
                if len(series_tuple) == 1:
                    tput_s = series_tuple[0]
                    cpu_s = pd.Series(dtype=float)  # Empty series if no CPU data
                else:
                    tput_s, cpu_s = series_tuple
                tx_count += 1
                label = f"Q{tx_count} {query_suffix[category]}"
                tput_rows[label] = tput_s
                cpu_rows[label]  = cpu_s
                

        return _build_df(tput_rows), _build_df(cpu_rows), y_label, unit

    # ------------------------------------------------------------------
    # Query plots
    # ------------------------------------------------------------------

    def plot_queries_detailed(self):
        if len(self.raw_df) < 4:
            return
        tx_groups = [
            ('join',           [tx for tx in self.anonymized_df.index if 'join' in tx]),
            ('count',          [tx for tx in self.anonymized_df.index
                                if 'count' in tx and 'distinct' not in tx]),
            ('count distinct', [tx for tx in self.anonymized_df.index if 'distinct' in tx]),
        ]
        for tx_group, txs in tx_groups:
            if not txs:
                continue
            first_group = (tx_group == 'join')
            width = len(txs) + (3 if first_group else 1)
            fig, axes = plt.subplots(nrows=1, ncols=len(txs),
                                     figsize=(width, 1.5), layout='constrained')
            axes_arr = np.atleast_1d(axes)
            for i, tx in enumerate(txs):
                row_df = pd.DataFrame(self.anonymized_df.loc[tx]).T
                row_df.plot(kind='bar', ylabel="", legend=False, ax=axes_arr[i],
                            xlabel='', color=_bar_colors(row_df))
                axes_arr[i].set_title(tx.split(' ')[0])
                axes_arr[i].set_xticks([], [])
            axes_arr[0].set_ylabel(f'Elapsed time ({self.unit})')
            if first_group:
                axes_arr[-1].legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
            fig.savefig(f'{self.directory}/queries-{tx_group.replace(" ", "-")}.png',
                        bbox_inches='tight', dpi=300)
            plt.close(fig)

        # Update & DB Size panel
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5, 3), layout='constrained')
        update_df = self._update_rows
        if not update_df.empty:
            row_df = pd.DataFrame(update_df.iloc[0]).T
            row_df.plot(kind='bar', ylabel="", legend=False, ax=axes[0],
                        xlabel='', color=_bar_colors(row_df))
            axes[0].set_title('Update')
            axes[0].set_xticks([], [])
            axes[0].set_ylabel(f'Elapsed time ({self.unit})')
        self._plot_size(axes[1])
        axes[1].set_title('DB Size')
        axes[1].legend(loc='center left', bbox_to_anchor=(1.2, 1))
        fig.savefig(f'{self.directory}/update-size.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_queries(self, detailed: bool):
        if detailed:
            self.plot_queries_detailed()
            return

        # Summary bar chart (log scale, all txs grouped)
        width = len(self.anonymized_df.columns) * 2 + 2
        fig, ax = plt.subplots(figsize=(width, 3), layout='constrained')
        self.anonymized_df.plot(
            kind='bar', ylabel=f'Elapsed time ({self.unit})', legend=True,
            ax=ax, xlabel='', logy=True, color=_bar_colors(self.anonymized_df),
        )
        ax.set_xticklabels(self.anonymized_df.index, rotation=0)
        plt.savefig(f'{self.directory}/queries-less.png', bbox_inches='tight', dpi=300)
        plt.close()

        # One subplot per tx + DB size
        ncols = len(self.anonymized_df) + 1
        fig, axes = plt.subplots(nrows=1, ncols=ncols,
                                 figsize=(ncols * 1.2 + 2, 2.5), layout='constrained')
        axes_arr = np.atleast_1d(axes)
        for i, (tx, row) in enumerate(self.anonymized_df.iterrows()):
            row_df = pd.DataFrame(row).T
            print(f"Plotting {tx} with data:\n{row_df}")
            row_df.plot(kind='bar', ylabel="", legend=False, ax=axes_arr[i],
                        xlabel='', color=_bar_colors(row_df))
            axes_arr[i].set_title(tx.replace(' ', '\n', 1))
            axes_arr[i].set_xticks([], [])
        self._plot_size(axes_arr[-1])
        axes_arr[-1].set_title('DB Size')
        axes_arr[0].set_ylabel(f'Elapsed time ({self.unit})')
        handles, labels = axes_arr[-2].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05),
                   ncol=len(self.anonymized_df.columns), frameon=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(f'{self.directory}/queries-individual-less.png',
                    bbox_inches='tight', dpi=300)
        plt.close()

    def get_update_df(self) -> pd.DataFrame:
        if self.anonymized_df.empty:
            return pd.DataFrame()
        update_df = self._update_rows.copy()
        update_df = update_df[[col for col in update_df.columns if col != 'trad_idx_hj']]
        update_df.rename(columns={'trad_idx_mj': 'trad_idx'}, inplace=True)
        return update_df


class BenchmarkAnalyzer:
    """
    Loads benchmark CSVs for a storage-engine directory, creates per-configuration
    DataPerConfig objects, and orchestrates all plots and LaTeX tables.
    """

    def __init__(self, directory: str):
        print("======================================================================")
        print(f"Initializing BenchmarkAnalyzer for directory: {directory}")
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.dir}")
        self.tput_df = pd.read_csv(self.dir / f'TPut.csv')
        # Normalize 0.09 -> 0.1 (hash table adds ~0.01 GiB overhead)
        self.tput_df['DRAM (GiB)'] = self.tput_df['DRAM (GiB)'].replace(0.09, 0.1)
        self.dfs_per_config: dict = {}
        for (scale, bgw_pct, dram), df in self.tput_df.groupby(
                ['scale', 'bgw_pct', 'DRAM (GiB)']):
            config = DataPerConfig(scale, bgw_pct, dram, df, str(self.dir))
            self.dfs_per_config.setdefault(scale, []).append(config)

    def _all_configs(self):
        """Yield every DataPerConfig across all scale groups."""
        for dpc_list in self.dfs_per_config.values():
            yield from dpc_list

    def plot_data_per_config(self):
        print("======================================================================")
        print("Plotting data for configurations...")
        for dpc in self._all_configs():
            print("----------------------------------------------------------------------")
            print(f"Plotting for {dpc.directory}...")
            dpc.plot_queries(detailed=True)
            dpc.plot_queries(detailed=False)
            dpc.plot_size()

    def plot_cpu_utilization(self):
        print("======================================================================")
        print("Plotting CPU utilization for configurations...")
        for dpc in self._all_configs():
            print("----------------------------------------------------------------------")
            print(f"Plotting CPU utilization for {dpc.directory}...")
            dpc.plot_cpu()

    def update_of_all_configs(self):
        dbtoaster = pd.read_csv(DBTOASTER_CSV).mean()
        dbtoaster_dram_gib = dbtoaster['memory_kb'] / 1024
        for scale, dpc_list in self.dfs_per_config.items():
            update_by_dram = {}
            for dpc in dpc_list:
                update_df = dpc.get_update_df()
                assert dpc.unit == 'ms', f"Expected unit 'ms', got '{dpc.unit}'"
                if update_df.empty:
                    continue
                update_by_dram[dpc.dram_size] = (update_df * 1000).round(2).iloc[0]  # ms -> us
            update_by_dram = pd.DataFrame(update_by_dram).T
            if update_by_dram.empty:
                print(f"No update data to plot for scale={scale}")
                continue

            # Place DBToaster at the closest DRAM level
            closest_dram = update_by_dram.index[
                np.abs(update_by_dram.index.to_numpy(dtype=float) - dbtoaster_dram_gib).argmin()
            ]
            update_by_dram.loc[closest_dram, 'DBToaster IVM'] = round(
                dbtoaster['update_time_us'], 2)
            update_by_dram = update_by_dram[_color_order(update_by_dram.columns)]

            self._plot_update(update_by_dram, scale, 'us', summary=False)

            if len(update_by_dram) < 2:
                print(f"Not enough DRAM configs for update summary at scale={scale}")
                continue
            df_summary = update_by_dram.loc[
                [update_by_dram.index[1], closest_dram]
            ].copy()
            df_summary.index = ['Small DRAM', 'Large DRAM']
            self._plot_update(df_summary, scale, 'us', summary=True)

    def _plot_update(self, df: pd.DataFrame, scale_value: int, unit: str, summary: bool):
        inf_df = df.replace(np.nan, np.inf)
        inf_df.rename(columns={'mat_view': 'traditional IVM'}, inplace=True)
        inf_df.to_latex(f'{self.dir}/scale{scale_value}/update.tex', float_format="%.2f")

        fig, ax = plt.subplots(figsize=(len(df) * 1.5 + 1.5, 2))
        inf_df.plot(kind='bar', ylabel=f'Update time ({unit})', legend=True, ax=ax,
                    logy=False, color=_bar_colors(inf_df))
        if not summary:
            ax.set_xlabel("DRAM (GiB)")

        # Draw hatched bars for methods that could not complete (recorded as inf/NaN)
        ymax = ax.get_ylim()[1]
        for col in inf_df.columns:
            inf_heights = (inf_df[col]
                           .where(inf_df[col] == np.inf, np.nan)
                           .replace(np.inf, ymax * 2))
            col_df = pd.DataFrame(np.nan, index=inf_df.index, columns=inf_df.columns)
            col_df[col] = inf_heights
            col_df.plot(kind='bar', ax=ax, logy=False, clip_on=True, legend=False,
                        hatch='//', color='none', edgecolor=COLORS.get(col, 'C5'))

        ax.set_xticklabels(df.index, rotation=0)
        y_cols = [c for c in (['merged_idx', 'trad_idx'] + (['mat_view'] if summary else []))
                  if c in df.columns]
        if y_cols:
            ax.set_ylim(0, df[y_cols].max().max() * 1.1)
        suffix = "less" if summary else "more"
        plt.savefig(f'{self.dir}/scale{scale_value}/update-{suffix}.png',
                    bbox_inches='tight', dpi=300)
        plt.close()


def main():
    """Initialize analyzers for LSM and B-Tree benchmarks and run all analyses."""
    try:
        analyzers = {
            'B-Tree': BenchmarkAnalyzer('geo_btree'),
            'LSM':    BenchmarkAnalyzer('geo_lsm'),
        }
        for analyzer in analyzers.values():
            analyzer.plot_cpu_utilization()
            analyzer.plot_data_per_config()
            analyzer.update_of_all_configs()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")


if __name__ == '__main__':
    main()
