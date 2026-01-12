from unicodedata import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

COLORS = OrderedDict([
            ('trad_idx_mj', 'C0'),
            ('trad_idx', 'C0'),
            ('trad_idx_hj', 'C3'),
            ('mat_view', 'C1'),
            ('traditional IVM', 'C1'),
            ('DBToaster IVM', 'C4'),
            ('merged_idx', 'C2'),
        ])

COLORS_SORT_KEY = lambda method: list(COLORS.keys()).index(method) if method in COLORS else len(COLORS)

class DataPerConfig:
    """
    Holds data for a specific configuration, including scale, background write percentage, dram size,
    and the corresponding DataFrame.
    """
    def __init__(self, scale: int, bgw_pct: int, dram_size: float, df: pd.DataFrame, parent_dir: str):
        self.scale = scale
        self.bgw_pct = bgw_pct
        self.dram_size = dram_size
        if parent_dir == 'geo_btree':
            self.engine = 'B-Tree'
        elif parent_dir == 'geo_lsm':
            self.engine = 'LSM-Tree'
        self.directory = f"{parent_dir}/scale{scale}/bgw{bgw_pct}-dram{dram_size}"
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.raw_df = df
        self.anonymized_df, self.anonymized_df_cpu, self.y_label, self.unit = self.group_and_anonymize()
        # output anonymized_df_cpu to latex code
        self.anonymized_df_cpu.to_latex(f"{self.directory}/anonymized_cpu.tex", float_format="%.1f")
        self.size_df = self.get_size_df()
        self.size_df.T.to_latex(f"{self.directory}/size.tex", float_format="%.3f")
        print("----------------------------------------------------------------------")
        print(f"Initialized DataPerConfig: scale={scale}, bgw_pct={bgw_pct}, dram_size={dram_size} GiB")
    
    def __str__(self):
        return f"DataPerConfig(scale={self.scale}, bgw_pct={self.bgw_pct}, dram_size={self.dram_size} GiB)"   
        
    def get_size_df(self):
        size_df = self.raw_df[['method', 'size (MiB)']].groupby('method').mean()['size (MiB)'].round(0).astype(int)
        # rename index base_idx to trad_idx_mj and hash to trad_idx_hj
        size_df.rename(index={'base_idx': 'trad_idx_mj', 'hash': 'trad_idx_hj'}, inplace=True)
        size_df = size_df[[col for col in COLORS.keys() if col in size_df.index]]
        # convert MiB to GiB
        size_df = (size_df / 1024).round(3)
        return pd.DataFrame({'size (GiB)': size_df}).T

    def _plot_size(self, ax):
        # pass a series with no column names
        series_to_draw = self.size_df.squeeze()
        if not isinstance(series_to_draw, pd.Series):
            print(f"Not enough size data to plot for {self.directory}")
            return
        series_to_draw.index = [''] * len(series_to_draw)

        self.size_df.plot(kind='bar', ylabel=self.size_df.index[0], ax=ax, xlabel='', legend=False, color=[COLORS.get(col, 'C5') for col in self.size_df.columns])
        ax.set_xticks([], [])
    
    def plot_size(self):
        if self.size_df.empty:
            print(f"No size data to plot for {self.directory}")
            return
        fig, ax = plt.subplots(figsize=(3, 3))

        self._plot_size(ax)

        plt.savefig(f'{self.directory}/size.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    def _prepare_tput_data(self, df: pd.DataFrame):
        """
        Calculate elapsed time with appropriate unit and prepare throughput data. Add a column for elapsed.
        """
        df_copy = df.copy()
        if df_copy['TPut (TX/s)'].max() < 1000:
            df_copy['Elapsed (s)'] = 1 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (s)'] = df_copy['Elapsed (s)'].round(3)
            return df_copy, 'Elapsed (s)', 's'
        elif df_copy['TPut (TX/s)'].max() < 10**6:
            df_copy['Elapsed (ms)'] = 1000 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (ms)'] = df_copy['Elapsed (ms)'].round(3)
            return df_copy, 'Elapsed (ms)', 'ms'
        else:
            df_copy['Elapsed (us)'] = 10**6 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (us)'] = df_copy['Elapsed (us)'].round(3)
            return df_copy, 'Elapsed (us)', 'us'

    def optimal_skip(self, tx_df: pd.DataFrame, tx: str, y_label: str):
        merged_index_df = tx_df[tx_df['method'] == 'merged_idx'].drop(columns=['method'])
        skip_means = merged_index_df.groupby('tentative_skip_bytes')[y_label].mean()
        if skip_means.empty:
            return
        optimal_skip = skip_means.idxmax()
        print(f'optimal skip for {tx}: {optimal_skip}')
        return optimal_skip

    def group_and_anonymize(self):
        df_copy, y_label, unit = self._prepare_tput_data(self.raw_df)
        cpu_label = "worker_0 CPU Util (%)" if 'worker_0 CPU Util (%)' in df_copy.columns \
                else "t0 CPU Util (%)"
        point_per_tx = df_copy.groupby('tx')
        bucket_sort = OrderedDict({
            "join": [],
            "mixed": [],
            "distinct": [],
            "maintain": []
        })
        query_suffix_dict = {
            "join": "join",
            "distinct": "count distinct",
            "mixed": "count",
        }
        
        for (tx), tx_df in point_per_tx:
            if len(tx_df['method'].unique()) < 3:
                print(f"Skipping anonymization for {tx} as there are not enough unique methods.", tx_df['method'].unique())
                continue
            tx_df = tx_df.drop(columns='tx')
            # keep optimal skip bytes for merged_idx
            optimal_skip_value = self.optimal_skip(tx_df, tx, y_label)
            tx_df = tx_df[(tx_df['tentative_skip_bytes'] == optimal_skip_value) | (tx_df['method'] != 'merged_idx')]
            series = tx_df.groupby('method').mean()[y_label].round(3)
            cpu_series = tx_df.groupby('method').mean()[cpu_label].round(1)
            tx_category = tx.split('-')[0]
            if tx.endswith("-n"): # PENDING
                continue
            assert tx_category in bucket_sort, f"Unknown tx category: {tx_category}"
            bucket_sort[tx_category].append((series, cpu_series))
        
        txs = {}
        txs_cpu = {}
        tx_count = 0
        for tx_category, list_of_series in bucket_sort.items():
            if tx_category == 'maintain':
                assert len(list_of_series) <= 1, "There should be at most one maintain tx"
                if list_of_series:
                    anonymized_name = "Update"
                    txs[anonymized_name] = list_of_series[0][0]
                    txs_cpu[anonymized_name] = list_of_series[0][1]
                continue
            for series in list_of_series:
                tx_count += 1
                anonymized_name = f"Q{tx_count} {query_suffix_dict[tx_category]}"
                txs[anonymized_name] = series[0]
                txs_cpu[anonymized_name] = series[1]
                
        anonymized_df = pd.DataFrame(txs)
        anonymized_df_cpu = pd.DataFrame(txs_cpu)
        anonymized_df = anonymized_df.T
        anonymized_df_cpu = anonymized_df_cpu.T
        # sort columns by COLORS order
        # rename base_idx to trad_idx_mj and hash to trad_idx_hj
        anonymized_df.rename(columns={'base_idx': 'trad_idx_mj', 'hash': 'trad_idx_hj'}, inplace=True)
        anonymized_df_cpu.rename(columns={'base_idx': 'trad_idx_mj', 'hash': 'trad_idx_hj'}, inplace=True)
        anonymized_df = anonymized_df[[col for col in COLORS.keys() if col in anonymized_df.columns]]
        anonymized_df_cpu = anonymized_df_cpu[[col for col in COLORS.keys() if col in anonymized_df_cpu.columns]]
        anonymized_df.index.name = 'tx'
        anonymized_df_cpu.index.name = 'tx'
        # | ---- | base_idx | hash | mat_view | merged_idx |
        # | Q1-join |...
        # | Q2-join |...
        return anonymized_df, anonymized_df_cpu, y_label, unit
    
    def plot_queries_detailed(self):
        if len(self.raw_df) < 4:
            return
        join_txs = [tx for tx in self.anonymized_df.index if 'join' in tx]
        count_txs = [tx for tx in self.anonymized_df.index if 'count' in tx and 'distinct' not in tx]
        distinct_txs = [tx for tx in self.anonymized_df.index if 'distinct' in tx]
        for tx_group, txs in [('join', join_txs), ('count', count_txs), ('count distinct', distinct_txs)]:
            if not txs:
                continue
            last_group = (tx_group == 'count distinct')
            width = len(txs) + 1 if not last_group else len(txs) + 2
            fig, axes = plt.subplots(nrows=1, ncols=len(txs), figsize=(width + 2, 3), layout='constrained')
            for i, tx in enumerate(txs):
                ax = axes[i] if len(txs) > 1 else axes
                row = self.anonymized_df.loc[tx]
                row_df = pd.DataFrame(row).T
                row_df.plot(kind='bar', ylabel="", legend=False,
                    ax=ax, xlabel='', color=[COLORS.get(col, 'C5') for col in row_df.columns])
                ax.set_title(tx.split(' ')[0])
                ax.set_xticks([], [])
            # fig.suptitle(f'{self.engine} - {tx_group.capitalize()} Queries', fontsize=16)
            axes[0].set_ylabel(f'Elapsed time ({self.unit})')
            if last_group:
                ax.legend(loc='center left', bbox_to_anchor=(1.2, 1)) # last query ax
            fig.savefig(f'{self.directory}/queries-{tx_group.replace(" ", "-")}.png', bbox_inches='tight', dpi=300)
        # update & space
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2 + 3, 3), layout='constrained')
        # update
        ax = axes[0]
        update_df = self.anonymized_df[self.anonymized_df.index.str.contains('Update')]
        if not update_df.empty:
            row = update_df.iloc[0]
            row_df = pd.DataFrame(row).T
            row_df.plot(kind='bar', ylabel="", legend=False, ax=ax, xlabel='', color=[COLORS.get(col, 'C5') for col in row_df.columns])
            ax.set_title('Update')
            ax.set_xticks([], [])
            ax.set_ylabel(f'Elapsed time ({self.unit})')
        # space for size
        self._plot_size(axes[1])
        axes[1].set_title('DB Size')
        axes[1].legend(loc='center left', bbox_to_anchor=(1.2, 1))
        
        # fig.suptitle(f'{self.engine} - Update & Size', fontsize=16)
        fig.savefig(f'{self.directory}/update-size.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_queries(self, detailed: bool):

        # all in one plot
        width = len(self.anonymized_df.columns) * 2 + 2
        if detailed:
            self.plot_queries_detailed()
            return
        else:
            fig, ax = plt.subplots(figsize=(width, 3), layout='constrained')
        self.anonymized_df.plot(kind='bar', ylabel=f'Elapsed time ({self.unit})', legend=True, ax=ax, xlabel='', table=(detailed), logy=True, color=[COLORS.get(col, 'C5') for col in self.anonymized_df.columns])
        ax.set_xticklabels(self.anonymized_df.index, rotation=0)
        filename_suffix = 'more' if detailed else 'less'
        plt.savefig(f'{self.directory}/queries-{filename_suffix}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # each tx in its own plot
        width_multiplier = 1.5
        ncols = len(self.anonymized_df) + 1
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * width_multiplier + 2, 3), layout='constrained')
        for i, (tx, row) in enumerate(self.anonymized_df.iterrows()):
            ax = axes[i] if len(self.anonymized_df) > 1 else axes
            row_df = pd.DataFrame(row).T
            row_df.plot(kind='bar', ylabel="", legend=False, ax=ax, xlabel='', color=[COLORS.get(col, 'C5') for col in row_df.columns])
            ax.set_title(tx)
            ax.set_xticks([], [])
        self._plot_size(axes[-1])
        axes[-1].set_title('DB Size')
        leftmost_ax = axes[0] if len(self.anonymized_df) > 1 else axes
        leftmost_ax.set_ylabel(f'Elapsed time ({self.unit})')
        # rightmost_ax = axes[-1] if len(self.anonymized_df) > 1 else axes
        leftmost_ax.legend(loc='center right', bbox_to_anchor=(-0.2, 1))
        fig.suptitle(self.engine, fontsize=16)
        plt.savefig(f'{self.directory}/queries-individual-{filename_suffix}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def get_update_df(self):
        if self.anonymized_df.empty:
            return pd.DataFrame()
        update_df = self.anonymized_df[self.anonymized_df.index.str.contains('Update')]
        columns = list(update_df.columns)
        update_df = update_df[[col for col in columns if col != 'trad_idx_hj']]  # remove hash column if exists
        # rename trad_idx_mj to trad_idx
        update_df.rename(columns={'trad_idx_mj': 'trad_idx'}, inplace=True)
        return update_df

class BenchmarkAnalyzer:
    """
    Analyzes benchmark data from a specified directory, generating plots and CSVs.

    This class encapsulates the data (throughput and elapsed time DataFrames) and
    the methods to process and visualize different benchmark scenarios like joins,
    maintenance, and mixed workloads.
    """

    def __init__(self, directory: str):
        """
        Initializes the analyzer by loading data from the specified directory.

        Args:
            directory (str): The path to the directory containing 'Elapsed.csv' and 'TPut.csv'.
        """
        print("======================================================================")
        print(f"Initializing BenchmarkAnalyzer for directory: {directory}")
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.dir}")
            
        self.tput_df = pd.read_csv(self.dir / 'TPut.csv')
        self.tput_df['DRAM (GiB)'] = self.tput_df['DRAM (GiB)'].replace(0.09, 0.1)  # Normalize 0.09 to 0.1 (0.01 for hash table)
        self.dfs_per_config = {}
        for (scale, bgw_pct, dram), df in self.tput_df.groupby(['scale', 'bgw_pct', 'DRAM (GiB)']):
            if scale not in list(self.dfs_per_config.keys()):
                self.dfs_per_config[scale] = [DataPerConfig(scale, bgw_pct, dram, df, str(self.dir))]
            else:
                self.dfs_per_config[scale].append(DataPerConfig(scale, bgw_pct, dram, df, str(self.dir)))
    
    def plot_data_per_config(self):
        print("======================================================================")
        print(f"Plotting data for configurations...")
        for _, dpc_list in self.dfs_per_config.items():
            for dpc in dpc_list:
                print("----------------------------------------------------------------------")
                print(f"Plotting for {dpc.directory}...")
                dpc.plot_queries(detailed=True)
                dpc.plot_queries(detailed=False)
                dpc.plot_size()

    def update_of_all_configs(self):
        dbtoaster_update_df = pd.read_csv('update_times.csv')
        dbtoaster_update_df = dbtoaster_update_df.mean()
        for scale, dpc_list in self.dfs_per_config.items():
            update_by_dram = {}
            for dpc in dpc_list:
                update_df = dpc.get_update_df()
                assert(dpc.unit == 'ms')
                if update_df.empty:
                    continue
                update_df = (update_df * 1000).round(0).astype('int')  # convert to us
                update_by_dram[dpc.dram_size] = update_df.iloc[0]
            update_by_dram = pd.DataFrame(update_by_dram).T
            if update_by_dram.empty:
                print(f"No update data to plot for scale={scale}")
                continue
            dbtoaster_dram = dbtoaster_update_df['memory_kb'] / 1024
            # closest dram in df to dbtoaster_dram
            closest_dram = update_by_dram.iloc[np.abs(np.array(update_by_dram.index) - dbtoaster_dram).argsort()[:1]].index[0]
            dbtoaster_col = {'DBToaster IVM': dbtoaster_update_df['update_time_us']}
            dbtoaster_col_df = pd.DataFrame(dbtoaster_col, index=[closest_dram])
            dbtoaster_col_df['DBToaster IVM'] = dbtoaster_col_df['DBToaster IVM'].round(0).astype('int')
            update_by_dram = pd.concat([update_by_dram, dbtoaster_col_df], axis=1)
            update_by_dram = update_by_dram[[col for col in COLORS.keys() if col in update_by_dram.columns]]
            self._plot_update(update_by_dram, scale, 'us', False)
            # update summary
            if len(update_by_dram) < 2:
                print(f"Not enough DRAM configurations for update summary plot at scale={scale}")
                continue
            second_smallest_dram = update_by_dram.index[1]
            df_summary = update_by_dram.loc[[second_smallest_dram, closest_dram]]
            # update dram values to small dram and large dram
            df_summary.index = ['Small DRAM', 'Large DRAM']
            self._plot_update(df_summary, scale, 'us', True)

    def _plot_update(self, df: pd.DataFrame, scale_value: int, unit: str, summary: bool):
        fig, ax = plt.subplots(figsize=(len(df) * 1.5 + 1.5, 3))
        inf_df = df.replace(np.nan, np.inf, inplace=False)
        # replace mat_view with traditional IVM
        inf_df.rename(columns={'mat_view': 'traditional IVM'}, inplace=True)
        inf_df.plot(kind='bar', ylabel=f'Update time ({unit})', legend=True, ax=ax, logy=False, color=[COLORS.get(col, 'C5') for col in inf_df.columns])
        if not summary:
            ax.set_xlabel("DRAM (GiB)")
        ymax = ax.get_ylim()[1]
        infs_only = inf_df[inf_df == np.inf]
        replaced_df = infs_only.replace(np.inf, ymax * 2, inplace=False)
        for i, col in enumerate(replaced_df.columns):
            # set all other columns to 0
            col_df = pd.DataFrame({c: [0] * len(replaced_df) for c in replaced_df.columns}, index=replaced_df.index)
            col_df[col] = replaced_df[col]
            col_df.plot(kind='bar', ylabel=f'Update time ({unit})', ax=ax, logy=False, clip_on=True, legend=False, hatch='//', color='none', edgecolor=COLORS.get(col, 'C5'))
        ax.set_xticklabels(df.index, rotation=0)
        if not summary:
            ax.set_ylim(0, max(df[['merged_idx', 'trad_idx']].max()) * 1.1)
        else:
            ax.set_ylim(0, max(df[['merged_idx', 'trad_idx', 'mat_view']].max()) * 1.1)
        suffix = "less" if summary else "more"
        plt.savefig(f'{self.dir}/scale{scale_value}/update-{suffix}.png', bbox_inches='tight', dpi=300)
        plt.close()
        

def main():
    """
    Main execution function.
    Initializes analyzers for LSM and B-tree benchmarks, runs all analyses,
    and performs a comparative skip study.
    """
    try:
        lsm_analyzer = BenchmarkAnalyzer('geo_lsm')
        btree_analyzer = BenchmarkAnalyzer('geo_btree')

        analyzers = {'B-Tree': btree_analyzer, 
                     'LSM': lsm_analyzer
                    }

        for name, analyzer in analyzers.items():
            analyzer.plot_data_per_config()
            analyzer.update_of_all_configs()
        
        

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()