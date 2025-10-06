import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import OrderedDict

class DataPerConfig:
    """
    Holds data for a specific configuration, including scale, background write percentage, dram size,
    and the corresponding DataFrame.
    """
    def __init__(self, scale: int, bgw_pct: int, dram_size: int, df: pd.DataFrame):
        self.scale = scale
        self.bgw_pct = bgw_pct
        self.dram_size = dram_size
        self.directory = f"{scale}-{bgw_pct}"
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        self.df = df

    def group_and_anonymize(self):
        pass

    def plot_queries(self, detailed: bool):
        pass
        


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
        self.colors = OrderedDict([
            ('base_idx', 'C0'),
            ('hash', 'C3'),
            ('mat_view', 'C1'),
            ('DBToaster', 'C4'),
            ('merged_idx', 'C2'),
        ])
        
        
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
        
    def sort_lambda(self, method: str) -> int:
        order = {
            'base_idx': 0,
            'mat_view': 1,
            'merged_idx': 2,
            'hash': 3,
            'DBToaster': 4
        }
        return order.get(method, 5)
        
    def plot_update(self):
        update_df = self.tput_df[(self.tput_df['tx'] == 'maintain') & (self.tput_df['method'] != 'hash')].drop(columns=['tx', 'tentative_skip_bytes'])
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
            df[y_label] = df[y_label].round(0).astype(int)
            df = df.pivot(index='DRAM (GiB)', columns='method', values=y_label).reset_index()
            # make DRAM (GiB) the index (not a column)
            df.set_index('DRAM (GiB)', inplace=True)
            df = df[[col for col in self.colors.keys() if col in df.columns]]
            # update details
            self._plot_update(df, scale_value, bgw_pct, unit, False)
            # update summary
            second_smallest_dram = df.index[1]
            df_summary = df.loc[[second_smallest_dram, closest_dram]]
            # update dram values to small dram and large dram
            df_summary.index = ['Small DRAM', 'Large DRAM']
            self._plot_update(df_summary, scale_value, bgw_pct, unit, True)

    def _plot_update(self, df: pd.DataFrame, scale_value: int, bgw_pct: int, unit: str, summary: bool):
        fig, ax = plt.subplots(figsize=(len(df) * 1.5 + 1.5, 4))
        inf_df = df.replace(np.nan, np.inf, inplace=False)
        inf_df.plot(kind='bar', ylabel=f'Update time ({unit})', legend=True, ax=ax, xlabel='', table=(not summary), logy=False, color=[self.colors.get(col, 'C5') for col in inf_df.columns])
        ymax = ax.get_ylim()[1]
        infs_only = inf_df[inf_df == np.inf]
        replaced_df = infs_only.replace(np.inf, ymax * 2, inplace=False)
        for i, col in enumerate(replaced_df.columns):
            # set all other columns to 0
            col_df = pd.DataFrame({c: [0] * len(replaced_df) for c in replaced_df.columns}, index=replaced_df.index)
            col_df[col] = replaced_df[col]
            col_df.plot(kind='bar', ylabel=f'Update time ({unit})', ax=ax, xlabel='', logy=False, clip_on=True, legend=False, hatch='//', color='none', edgecolor=self.colors.get(col, 'C5'))
        if not summary:
            ax.set_xticks([], [])
            ax.text(-0.7, 0, 'DRAM (GiB)', horizontalalignment='right', verticalalignment='top')
            ax.set_ylim(0, max(df[['merged_idx', 'base_idx']].max()) * 1.1)
        else:
            ax.set_xticklabels(df.index, rotation=0)
            ax.set_ylim(0, max(df[['merged_idx', 'base_idx', 'mat_view']].max()) * 1.1)
        suffix = "less" if summary else "more"
        plt.savefig(f'{self.dir}/{scale_value}-{bgw_pct}-update-{suffix}.png', bbox_inches='tight', dpi=300)
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
            anonymized_df = pd.DataFrame(txs)
            anonymized_df = anonymized_df.T
            # sort columns by self.colors order
            anonymized_df = anonymized_df[[col for col in self.colors.keys() if col in anonymized_df.columns]]
            self.plot_queries(anonymized_df, name, unit, include_suffix)
            
    def plot_queries(self, anonymized_df, config_name, unit, include_suffix):
        width = len(anonymized_df.columns) * 1.5 + 1.5
        fig, ax = plt.subplots(figsize=(width, 4))
        anonymized_df.plot(kind='bar', ylabel=f'Query time ({unit})', legend=True, ax=ax, xlabel='', table=include_suffix, logy=True, color=[self.colors.get(col, 'C5') for col in anonymized_df.columns])
        if include_suffix:
            ax.set_xticks([], [])
        else:
            ax.set_xticklabels(anonymized_df.index, rotation=0)
        dram_value, scale_value, bgw_pct = config_name
        filename_suffix = 'more' if include_suffix else 'less'
        plt.savefig(f'{self.dir}/{dram_value}-{scale_value}-{bgw_pct}-queries-{filename_suffix}.png', bbox_inches='tight', dpi=300)
        plt.close()
            
    def plot_size(self, config_name, group_df_by_config):
        fig, ax = plt.subplots(figsize=(3, 4))
        size_df = pd.DataFrame([group_df_by_config.drop(columns=['tx']).groupby('method').mean()['size (MiB)'].round(0).astype(int)])
        
        # pass a series with no column names
        series_to_draw = size_df.iloc[0]
        series_to_draw.index = [''] * len(series_to_draw)
        size_df = size_df[[col for col in self.colors.keys() if col in size_df.columns]]

        size_df.plot(kind='bar', ylabel='Size (MiB)', ax=ax, xlabel='', table=series_to_draw, legend=False, color=[self.colors.get(col, 'C5') for col in size_df.columns])
        ax.set_xticks([], [])
        # ax.yaxis.set_label_position("right")
        # ax.yaxis.tick_right()
        
        dram_value, scale_value, bgw_pct = config_name
        plt.savefig(f'{self.dir}/{dram_value}-{scale_value}-{bgw_pct}-size.png', bbox_inches='tight', dpi=300)
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
            analyzer.anonymized_points()
            analyzer.anonymized_points(False)
            analyzer.plot_update()
            
            print(f"--- Finished processing {name} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()