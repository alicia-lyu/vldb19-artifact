from unicodedata import name
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

COLORS = OrderedDict([
            ('base_idx', 'C0'),
            ('hash', 'C3'),
            ('mat_view', 'C1'),
            ('DBToaster', 'C4'),
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
        self.directory = f"{parent_dir}/scale{scale}-bgw{bgw_pct}/dram{dram_size}"
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.raw_df = df
        self.anonymized_df, self.y_label, self.unit = self.group_and_anonymize()
        self.size_df = self.get_size_df()
        print("----------------------------------------------------------------------")
        print(f"Initialized DataPerConfig: scale={scale}, bgw_pct={bgw_pct}, dram_size={dram_size} GiB")
        
        
    def get_size_df(self):
        size_df = self.raw_df[['method', 'size (MiB)']].groupby('method').mean()['size (MiB)'].round(0).astype(int)
        size_df = size_df[[col for col in COLORS.keys() if col in size_df.index]]
        return pd.DataFrame({'size (MiB)': size_df}).T
    
    def _plot_size(self, ax):
        # pass a series with no column names
        series_to_draw = self.size_df.squeeze()
        if not isinstance(series_to_draw, pd.Series):
            print(f"Not enough size data to plot for {self.directory}")
            return
        series_to_draw.index = [''] * len(series_to_draw)

        self.size_df.plot(kind='bar', ylabel='Size (MiB)', ax=ax, xlabel='', table=series_to_draw, legend=False, color=[COLORS.get(col, 'C5') for col in self.size_df.columns])
        ax.set_xticks([], [])
    
    def plot_size(self):
        if self.size_df.empty:
            print(f"No size data to plot for {self.directory}")
            return
        fig, ax = plt.subplots(figsize=(3, 4))

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
        point_per_tx = df_copy.groupby('tx')
        txs = {}
        tx_count = 0
        for (tx), tx_df in point_per_tx:
            if len(tx_df['method'].unique()) < 3:
                print(f"Skipping anonymization for {tx} as there are not enough unique methods.", tx_df['method'].unique())
                continue
            tx_count += 1
            tx_df = tx_df.drop(columns='tx')
            # keep optimal skip bytes for merged_idx
            optimal_skip_value = self.optimal_skip(tx_df, tx, y_label)
            tx_df = tx_df[(tx_df['tentative_skip_bytes'] == optimal_skip_value) | (tx_df['method'] != 'merged_idx')]
            suffix = ''
            if tx.startswith('join'):
                suffix = 'join'
            elif tx.startswith('mixed'):
                suffix = 'mixed'
            elif tx.startswith('distinct'):
                suffix = 'distinct'
            anonymized_name = f"Q{tx_count}-{suffix}"
            if tx.startswith('maintain'):
                suffix = 'update'
                anonymized_name = "Update"
            txs[anonymized_name] = tx_df.groupby('method').mean()[y_label].round(3)
            print(f"{tx} anonymized to {anonymized_name}")
        anonymized_df = pd.DataFrame(txs)
        anonymized_df = anonymized_df.T
        # sort columns by COLORS order
        anonymized_df = anonymized_df[[col for col in COLORS.keys() if col in anonymized_df.columns]]
        anonymized_df.index.name = 'tx'
        # | ---- | base_idx | hash | mat_view | merged_idx |
        # | Q1-join |...
        # | Q2-join |...
        return anonymized_df, y_label, unit

    def plot_queries(self, detailed: bool):
        if detailed and len(self.raw_df) < 4:
            print(f"Skipping anonymization for {name} as there are not enough unique transactions.")
            return
        df_copy = self.anonymized_df.copy()
        if df_copy.empty:
            print(f"No data to plot for {self.directory}")
            return
        if detailed: # exclude update (in index)
            df_copy = df_copy[~df_copy.index.str.contains('Update')]
        else: # exclude mixed, distinct
            df_copy = df_copy[~df_copy.index.str.contains('mixed')]
            df_copy = df_copy[~df_copy.index.str.contains('distinct')]

        width = len(df_copy.columns) * 2 + 1.5
        if detailed:
            fig, (ax, ax_size) = plt.subplots(ncols=2, figsize=(width + 3, 6), gridspec_kw={'width_ratios': [len(df_copy.columns), 1]})
        else:
            fig, ax = plt.subplots(figsize=(width, 4))
        df_copy.plot(kind='bar', ylabel=f'Query time ({self.unit})', legend=True, ax=ax, xlabel='', table=(detailed), logy=True, color=[COLORS.get(col, 'C5') for col in df_copy.columns])
        if detailed:
            ax.set_xticks([], [])
            self._plot_size(ax_size)
        else:
            ax.set_xticklabels(df_copy.index, rotation=0)
        filename_suffix = 'more' if detailed else 'less'
        plt.savefig(f'{self.directory}/queries-{filename_suffix}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def get_update_df(self):
        if self.anonymized_df.empty:
            return pd.DataFrame()
        update_df = self.anonymized_df[self.anonymized_df.index.str.contains('Update')]
        columns = list(update_df.columns)
        update_df = update_df[[col for col in columns if col != 'hash']]  # remove hash column if exists
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
            if (scale, bgw_pct) not in self.dfs_per_config:
                self.dfs_per_config[(scale, bgw_pct)] = [DataPerConfig(scale, bgw_pct, dram, df, str(self.dir))]
            else:
                self.dfs_per_config[(scale, bgw_pct)].append(DataPerConfig(scale, bgw_pct, dram, df, str(self.dir)))
    
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
        for config, dpc_list in self.dfs_per_config.items():
            scale, bgw_pct = config
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
                print(f"No update data to plot for scale={scale}, bgw_pct={bgw_pct}")
                continue
            dbtoaster_dram = dbtoaster_update_df['memory_kb'] / 1024
            # closest dram in df to dbtoaster_dram
            closest_dram = update_by_dram.iloc[np.abs(np.array(update_by_dram.index) - dbtoaster_dram).argsort()[:1]].index[0]
            dbtoaster_col = {'DBToaster': dbtoaster_update_df['update_time_us']}
            dbtoaster_col_df = pd.DataFrame(dbtoaster_col, index=[closest_dram])
            dbtoaster_col_df['DBToaster'] = dbtoaster_col_df['DBToaster'].round(0).astype('int')
            update_by_dram = pd.concat([update_by_dram, dbtoaster_col_df], axis=1)
            update_by_dram = update_by_dram[[col for col in COLORS.keys() if col in update_by_dram.columns]]
            self._plot_update(update_by_dram, scale, bgw_pct, 'us', False)
            # update summary
            if len(update_by_dram) < 2:
                print(f"Not enough DRAM configurations for update summary plot at scale={scale}, bgw_pct={bgw_pct}")
                continue
            second_smallest_dram = update_by_dram.index[1]
            df_summary = update_by_dram.loc[[second_smallest_dram, closest_dram]]
            # update dram values to small dram and large dram
            df_summary.index = ['Small DRAM', 'Large DRAM']
            self._plot_update(df_summary, scale, bgw_pct, 'us', True)

    def _plot_update(self, df: pd.DataFrame, scale_value: int, bgw_pct: int, unit: str, summary: bool):
        fig, ax = plt.subplots(figsize=(len(df) * 1.5 + 1.5, 4))
        inf_df = df.replace(np.nan, np.inf, inplace=False)
        inf_df.plot(kind='bar', ylabel=f'Update time ({unit})', legend=True, ax=ax, xlabel="DRAM (GiB)", logy=False, color=[COLORS.get(col, 'C5') for col in inf_df.columns])
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
            ax.set_ylim(0, max(df[['merged_idx', 'base_idx']].max()) * 1.1)
        else:
            ax.set_ylim(0, max(df[['merged_idx', 'base_idx', 'mat_view']].max()) * 1.1)
        suffix = "less" if summary else "more"
        plt.savefig(f'{self.dir}/scale{scale_value}-bgw{bgw_pct}/update-{suffix}.png', bbox_inches='tight', dpi=300)
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
                    #  'LSM': lsm_analyzer
                    }

        for name, analyzer in analyzers.items():
            analyzer.plot_data_per_config()
            analyzer.update_of_all_configs()

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()