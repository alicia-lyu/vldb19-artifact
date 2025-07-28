import pandas as pd
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

    def _prepare_elapsed_data(self, df: pd.DataFrame, col_name: str = 'Elapsed (ms)') -> Tuple[pd.DataFrame, str, str, int, type]:
        """
        Checks if elapsed time should be converted to seconds and prepares data accordingly.
        If the mean elapsed time is 1000ms or more, it converts the unit to seconds.

        Returns:
            A tuple containing the processed DataFrame, the Y-axis label, the unit,
            rounding precision, and the data type for casting.
        """
        df_copy = df.copy()
        if df_copy[col_name].mean() >= 1000:
            new_col_name = 'Elapsed (s)'
            df_copy[new_col_name] = (df_copy[col_name] / 1000)
            df_copy = df_copy.drop(columns=[col_name])
            return df_copy, new_col_name, 's', 2, float
        else:
            return df_copy, col_name, 'ms', 0, int

    def process_join_scan(self):
        """Processes and plots join scan elapsed time."""
        join_df_groups = self.elapsed_df[self.elapsed_df['tx'] == 'join'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
        
        for (dram_value, scale_value), group_df in join_df_groups:
            processed_df, y_label, _, round_digits, dtype = self._prepare_elapsed_data(group_df)
            
            mean_df = processed_df.groupby('method').mean()[y_label]
            mean_df = mean_df.round(round_digits).astype(dtype)
            
            file_prefix = self.dir / f'join-{dram_value}-{scale_value}'
            with open(f'{file_prefix}.csv', 'w') as f:
                f.write(mean_df.to_string())
                
            ax = mean_df.plot(title='Join scan elapsed time', ylabel=y_label, kind='bar', legend=False, table=True)
            ax.set_xlabel('')
            ax.set_xticks([], [])
            plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
            plt.close()

    def process_maintain(self):
        """Processes and plots maintenance throughput."""
        tput_df_groups = self.tput_df[self.tput_df['tx'] == 'maintain'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
        
        for (dram_value, scale_value), group_df in tput_df_groups:
            mean_df = group_df.groupby('method').mean()['TPut (TX/s)']
            mean_df.name = 'TPut (TX/s)'
            mean_df = mean_df.round(2)
            
            file_prefix = self.dir / f'maintain-{dram_value}-{scale_value}'
            with open(f'{file_prefix}.csv', 'w') as f:
                f.write(mean_df.to_string())
                
            ax = mean_df.plot(title='Maintain throughput', ylabel='TPut (TX/s)', kind='bar', legend=False, table=True)
            ax.set_xlabel('')
            ax.set_xticks([], [])
            plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
            plt.close()

    def process_mixed(self):
        """Processes and plots mixed workload performance (scan elapsed and point throughput)."""
        mixed_scan_df = self.elapsed_df[self.elapsed_df['tx'] == 'mixed'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
        mixed_point_df = self.tput_df[self.tput_df['tx'] == 'mixed-point'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
        
        for (name, scan_group), (_, point_group) in zip(mixed_scan_df, mixed_point_df):
            dram_value, scale_value = name
            
            processed_scan_df, y_label, _, round_digits, dtype = self._prepare_elapsed_data(scan_group)
            scan_mean = processed_scan_df.groupby('method').mean()[y_label]
            
            point_mean = point_group.groupby('method').mean()['TPut (TX/s)']
            
            combined_df = pd.DataFrame({y_label: scan_mean, 'Point TPut (TX/s)': point_mean})
            combined_df = combined_df.round({y_label: round_digits, 'Point TPut (TX/s)': 2})
            combined_df = combined_df.astype({y_label: dtype, 'Point TPut (TX/s)': float})
            
            file_prefix = self.dir / f'mixed-{dram_value}-{scale_value}'
            with open(f'{file_prefix}.csv', 'w') as f:
                f.write(combined_df.to_string())
                
            axes = combined_df.plot(subplots=True, layout=(2, 1), legend=True, marker='o', sharex=True, xlabel='')
            axes[0, 0].set_ylabel(y_label)
            axes[1, 0].set_ylabel('TPut (TX/s)')
            plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
            plt.close()

    def process_join_point(self):
        """Processes and plots point join query throughput."""
        setup_groups = self.tput_df[self.tput_df['tx'].isin(['join-ns', 'join-nsc', 'join-nscci'])].groupby(by=['DRAM (GiB)', 'scale'])
        
        for name, setup_df in setup_groups:
            dram_value, scale_value = name
            series_dict = {}
            for method_name, method_df in setup_df.groupby('method'):
                if method_name != 'merged_idx':
                    series_to_plot = method_df.drop(columns=['method','tentative_skip_bytes']).groupby('tx').mean()['TPut (TX/s)']
                else:
                    series_to_plot = []
                    for tx, tx_df in method_df.groupby('tx'):
                        skip_means = tx_df.drop(columns=['method', 'tx']).groupby('tentative_skip_bytes').mean()['TPut (TX/s)']
                        print(f"Finding optimal skip for {tx} in {self.dir}")
                        print(skip_means.to_string())
                        series_to_plot.append(skip_means.max())
                    series_to_plot = pd.Series(series_to_plot, index=method_df['tx'].unique())
                series_dict[method_name] = series_to_plot
                
            plot_df = pd.DataFrame(series_dict).round(2)
            plot_df.columns.name = ''
            self._plot_join_point_series(plot_df, dram_value, scale_value)
            
    def _plot_join_point_series(self, plot_df: pd.DataFrame, dram_value: int, scale_value: int):
        """Helper to plot point join series."""
        ax = plot_df.plot(title='Point join queries', legend=True, marker='o', ylabel='TPut (TX/s)', logy=True, table=True)
        ax.set_xlabel('')
        ax.set_xticks([], [])
        file_path = self.dir / f'join_point-{dram_value}-{scale_value}.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()

    @staticmethod
    def process_skip_studies(lsm_analyzer: 'BenchmarkAnalyzer', btree_analyzer: 'BenchmarkAnalyzer', output_filename: str = 'skip_study'):
        """
        Compares LSM and B-tree performance for a skip study and generates a plot.
        This is a static method as it compares data from two different analyzer objects.
        """
        lsm_df = lsm_analyzer.tput_df
        btree_df = btree_analyzer.tput_df
        
        lsm_filtered = lsm_df[(lsm_df['tx'] == 'join-nscci') & (lsm_df['method'] == 'merged_idx')]
        btree_filtered = btree_df[(btree_df['tx'] == 'join-nscci') & (btree_df['method'] == 'merged_idx')]

        lsm_groups = lsm_filtered.groupby(['DRAM (GiB)', 'scale'])
        btree_groups = btree_filtered.groupby(['DRAM (GiB)', 'scale'])

        for (_, lsm_group), (_, btree_group) in zip(lsm_groups, btree_groups):
            lsm_mean = lsm_group.drop(columns=['method','tx']).groupby('tentative_skip_bytes').mean()['TPut (TX/s)']
            lsm_mean.rename('LSM TPut', inplace=True)
            btree_mean = btree_group.drop(columns=['method', 'tx']).groupby('tentative_skip_bytes').mean()[['TPut (TX/s)', 'pp_0 CPU Util (%)']]
            btree_mean.rename(columns={'TPut (TX/s)': 'B-Tree TPut'}, inplace=True)

            print("Skip Study Comparison:")
            print(btree_mean.to_string())

            ax = btree_mean.plot(title='Skip study with various tentative skip bytes', ylabel='TPut (TX/s)', xlabel='', marker='o', legend=True, secondary_y='pp_0 CPU Util (%)')
            # plot lsm on ax too
            lsm_mean.plot(ax=ax, marker='o', legend=True)
            xlabels = list(set(lsm_mean.index).union(set(btree_mean.index)))
            ax.set_xticks(xlabels, xlabels)
            ax.set_ylabel('TPut (TX/s)')
            ax.right_ax.set_ylabel('pp_0 CPU Util (%)')
            plt.savefig(f'{output_filename}.png', bbox_inches='tight', dpi=300)
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

        analyzers = {'LSM': lsm_analyzer, 'B-Tree': btree_analyzer}

        for name, analyzer in analyzers.items():
            print(f"--- Processing {name} Benchmark Data ---")
            analyzer.process_join_scan()
            analyzer.process_maintain()
            analyzer.process_mixed()
            analyzer.process_join_point()
            print(f"--- Finished processing {name} ---")

        print("\n--- Processing Comparative Skip Study ---")
        BenchmarkAnalyzer.process_skip_studies(lsm_analyzer, btree_analyzer)
        print("--- Analysis Complete ---")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()