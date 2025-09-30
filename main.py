import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple

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
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.dir}")
            
        self.elapsed_df = pd.read_csv(self.dir / 'Elapsed.csv')
        self.tput_df = pd.read_csv(self.dir / 'TPut.csv')
        
    def _prepare_tput_data(self, df: pd.DataFrame):
        """
        Calculate elapsed time with appropriate unit and prepare throughput data. Add a column for elapsed.
        """
        df_copy = df.copy()
        if df_copy['TPut (TX/s)'].max() < 1000:
            df_copy['Elapsed (ms)'] = 1000 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (ms)'] = df_copy['Elapsed (ms)'].round(0).astype(int)
            return df_copy, 'Elapsed (ms)', 'ms'
        elif df_copy['TPut (TX/s)'].max() < 10**6:
            df_copy['Elapsed (us)'] = 10**6 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (us)'] = df_copy['Elapsed (us)'].round(0).astype(int)
            return df_copy, 'Elapsed (us)', 'us'
        else:
            df_copy['Elapsed (ns)'] = 10**9 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (ns)'] = df_copy['Elapsed (ns)'].round(0).astype(int)
            return df_copy, 'Elapsed (ns)', 'ns'
        
    def update_details(self):
        update_df = self.tput_df[self.tput_df['tx'] == 'maintain'].drop(columns=['tx', 'tentative_skip_bytes'])
        update_df, y_label, unit = self._prepare_tput_data(update_df)
        dfs_by_config = update_df.groupby(['scale', 'bgw_pct'])
        dbtoaster_update_df = pd.read_csv('update_times.csv')
        dbtoaster_update_df = dbtoaster_update_df.mean()
        for config, df in dfs_by_config:
            scale_value, bgw_pct = config
            df = df.drop(columns=['scale', 'bgw_pct'])
            df = df.groupby(['DRAM (GiB)', 'method']).mean().reset_index()
            # only keep dram, method, y_label
            df = df[['DRAM (GiB)', 'method', y_label]]
            dbtoaster_dram = dbtoaster_update_df['memory_kb'] / 1024
            # closest dram in df to dbtoaster_dram
            closest_dram = df.iloc[(df['DRAM (GiB)'] - dbtoaster_dram).abs().argsort()[:1]]['DRAM (GiB)'].values[0]
            df = pd.concat([df, pd.DataFrame({'DRAM (GiB)': [closest_dram], 'method': ['DBToaster'], y_label: [dbtoaster_update_df['update_time_us']]})], ignore_index=True)
            df = df.pivot(index='DRAM (GiB)', columns='method', values=y_label).reset_index()
            # put dbtoaster as the last column
            df = df[[col for col in df.columns if col != 'DBToaster'] + ['DBToaster']]
            # make DRAM (GiB) the index (not a column)
            df.set_index('DRAM (GiB)', inplace=True)
            fig, ax = plt.subplots(figsize=(len(df) + 2, 6))
            inf_df = df.replace(np.nan, np.inf, inplace=False)
            inf_df.plot(kind='bar', ylabel=f'Update time ({unit})', legend=True, ax=ax, xlabel='', table=True, logy=False)
            ymax = ax.get_ylim()[1]
            replaced_df = df.replace(np.nan, ymax * 2, inplace=False)
            replaced_df.plot(kind='bar', ylabel=f'Update time ({unit})', ax=ax, xlabel='', logy=False, clip_on=True, legend=False)
            ax.set_xticks([], [])
            ax.set_ylim(0, ymax)
            plt.savefig(f'{self.dir}/update-details-{scale_value}-{bgw_pct}.png', bbox_inches='tight', dpi=300)
            plt.close()
        
            
    def anonymized_points(self, include_suffix=True):
        point_df_groups = self.tput_df.groupby(by=['DRAM (GiB)', 'scale', 'bgw_pct'])
        
        for (name, group_df) in point_df_groups:
            """
            PLOT 1: Size of different methods
            """
            self.plot_size(name, group_df)
            dram_value, scale_value, bgw_pct = name
            group_df, y_label, unit = self._prepare_tput_data(group_df)
            point_per_tx = group_df.groupby('tx')
            txs = {}
            tx_count = 0
            for (tx), tx_df in point_per_tx:
                if tx == 'maintain':
                    """
                    PLOT 2: Update time
                    """
                    self.plot_update(name, unit, y_label, tx_df)
                    continue
                if len(tx_df['method'].unique()) < 3:
                    print(f"Skipping anonymization for {tx} as there are not enough unique methods.")
                    continue
                tx_count += 1
                # keep optimal skip bytes for merged_idx
                skip_means = tx_df[tx_df['method'] == 'merged_idx'].drop(columns=['method', 'tx']).groupby('tentative_skip_bytes').mean()[y_label]
                if skip_means.empty:
                    continue
                optimal_skip = skip_means.idxmax()
                print(f'optimal skip for {tx} in {self.dir} with dram={dram_value}, scale={scale_value}, bgw_pct={bgw_pct}: {optimal_skip}')
                tx_df = tx_df[(tx_df['tentative_skip_bytes'] == optimal_skip) | (tx_df['method'] != 'merged_idx')]
                
                suffix = ''
                if tx.startswith('join'):
                    suffix = 'join'
                    anonymized_name = f"Q{tx_count}" if not include_suffix else f"Q{tx_count}-{suffix}"
                elif tx.startswith('mixed'):
                    suffix = 'mixed'
                    if not include_suffix:
                        anonymized_name = f"Q{tx_count}"
                        continue # only include mixed in more detailed drawings
                    else:
                        anonymized_name = f"Q{tx_count}-{suffix}"
                else:
                    suffix = 'update'
                    anonymized_name = "Update"
                txs[anonymized_name] = tx_df.drop(columns='tx').groupby('method').mean()[y_label].round(0).astype(int)
                print(f"{tx} anonymized to Q{tx_count} ({suffix})")
            if tx_count < 4:
                print(f"Skipping anonymization for {name} as there are not enough unique transactions.")
                continue
            # sort txs by key
            txs = dict(sorted(txs.items(), key=lambda item: item[0]))
            anonymized_df = pd.DataFrame(txs)
            anonymized_df = anonymized_df.T
            self.plot_queries(anonymized_df, name, unit, include_suffix)
            
    def plot_queries(self, anonymized_df, config_name, unit, include_suffix):
        width = len(anonymized_df.columns) + 2
        fig, ax = plt.subplots(figsize=(width, 6))
        anonymized_df.plot(kind='bar', ylabel=f'Query time ({unit})', legend=True, ax=ax, xlabel='', table=include_suffix, logy=True)
        if include_suffix:
            ax.set_xticks([], [])
        else:
            ax.set_xticklabels(anonymized_df.index, rotation=0)
        dram_value, scale_value, bgw_pct = config_name
        filename_suffix = 'more' if include_suffix else 'less'
        plt.savefig(f'{self.dir}/queries-{dram_value}-{scale_value}-{bgw_pct}-{filename_suffix}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
    
    def plot_update(self, config_name, unit, y_label, tx_df_by_config):
        fig, ax = plt.subplots(figsize=(4, 6))
        # pass a series with no column names
        
        update_df = pd.DataFrame([tx_df_by_config.drop(columns='tx').groupby('method').mean()[y_label].round(0).astype(int)])
        
        series_to_draw = update_df.iloc[0]
        series_to_draw.index = [''] * len(series_to_draw)

        update_df.plot(kind='bar', ylabel='Update Time (unit)', ax=ax, xlabel='', table=series_to_draw)
        ax.set_xticks([], [])
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        # ax.legend(loc='upper right', bbox_to_anchor=(0, 1))
        
        dram_value, scale_value, bgw_pct = config_name
        plt.savefig(f'{self.dir}/update-{dram_value}-{scale_value}-{bgw_pct}.png', bbox_inches='tight', dpi=300)
        plt.close()
            
    def plot_size(self, config_name, group_df_by_config):
        fig, ax = plt.subplots(figsize=(4, 6))
        size_df = pd.DataFrame([group_df_by_config.drop(columns=['tx']).groupby('method').mean()['size (MiB)'].round(0).astype(int)])
        
        # pass a series with no column names
        series_to_draw = size_df.iloc[0]
        series_to_draw.index = [''] * len(series_to_draw)

        size_df.plot(kind='bar', ylabel='Size (MiB)', legend=True, ax=ax, xlabel='', table=series_to_draw)
        ax.set_xticks([], [])
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        
        dram_value, scale_value, bgw_pct = config_name
        plt.savefig(f'{self.dir}/size-{dram_value}-{scale_value}-{bgw_pct}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def process_join_scan(self):
        """Processes and plots join scan elapsed time."""
        join_df_groups = self.elapsed_df[self.elapsed_df['tx'] == 'join'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale', 'bgw_pct'])

        for (dram_value, scale_value, bgw_pct), group_df in join_df_groups:
            processed_df, y_label, _, round_digits, dtype = self._prepare_elapsed_data(group_df)
            
            mean_df = processed_df.groupby('method').mean()[y_label]
            mean_df = mean_df.round(round_digits).astype(dtype)
            
            file_prefix = self.dir / f'join-{dram_value}-{scale_value}-{bgw_pct}'
                
            ax = mean_df.plot(title='Join scan elapsed time', ylabel=y_label, kind='bar', legend=False, table=True)
            ax.set_xlabel('')
            ax.set_xticks([], [])
            plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
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

        analyzers = {'B-Tree': btree_analyzer, 'LSM': lsm_analyzer}

        for name, analyzer in analyzers.items():
            print(f"--- Processing {name} Benchmark Data ---")
            analyzer.update_details()
            analyzer.anonymized_points()
            analyzer.anonymized_points(False)
            
            print(f"--- Finished processing {name} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()