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
        
    def _prepare_tput_data(self, df: pd.DataFrame):
        """
        Calculate elapsed time with appropriate unit and prepare throughput data. Add a column for elapsed.
        """
        df_copy = df.copy()
        if df_copy['TPut (TX/s)'].mean() < 1000:
            df_copy['Elapsed (ms)'] = df_copy['Elapsed (ms)'].round(0).astype(int)
            return df_copy, 'Elapsed (ms)', 'ms'
        elif df_copy['TPut (TX/s)'].mean() < 10**6:
            df_copy['Elapsed (us)'] = 10**6 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (us)'] = df_copy['Elapsed (us)'].round(0).astype(int)
            return df_copy, 'Elapsed (us)', 'us'
        else:
            df_copy['Elapsed (ns)'] = 10**9 / df_copy['TPut (TX/s)']
            df_copy['Elapsed (ns)'] = df_copy['Elapsed (ns)'].round(0).astype(int)
            return df_copy, 'Elapsed (ns)', 'ns'

    def process_join_scan(self):
        """Processes and plots join scan elapsed time."""
        join_df_groups = self.elapsed_df[self.elapsed_df['tx'] == 'join'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
        
        for (dram_value, scale_value), group_df in join_df_groups:
            processed_df, y_label, _, round_digits, dtype = self._prepare_elapsed_data(group_df)
            
            mean_df = processed_df.groupby('method').mean()[y_label]
            mean_df = mean_df.round(round_digits).astype(dtype)
            
            file_prefix = self.dir / f'join-{dram_value}-{scale_value}'
                
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
                
            axes = combined_df.plot(subplots=True, layout=(2, 1), legend=True, marker='o', sharex=True, xlabel='')
            axes[0, 0].set_ylabel(y_label)
            axes[1, 0].set_ylabel('TPut (TX/s)')
            plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
            plt.close()

    def process_join_point(self):
        """Processes and plots point join query throughput."""
        txs_df = self.tput_df[self.tput_df['tx'].isin(['join-ns', 'join-nsc', 'join-nscci'])]
        txs_df, tput_newcol, unit = self._prepare_tput_data(txs_df)
        join_df = self.elapsed_df[self.elapsed_df['tx'] == 'join']
        if unit == 'ms':
            join_df['Elapsed (ms)'] = join_df['Elapsed (ms)'] / 25 # 25 nations
        elif unit == 'us':
            # join_df['Elapsed (us)'] = join_df['Elapsed (us)'] / 25 * 1000
            join_df.rename(columns={'Elapsed (ms)': tput_newcol}, inplace=True)
            join_df[tput_newcol] = join_df[tput_newcol] / 25 * 1000
        elif unit == 'ns':
            join_df.rename(columns={'Elapsed (ms)': tput_newcol}, inplace=True)
            join_df[tput_newcol] = join_df[tput_newcol] / 25 * 1000 * 1000
        join_df[tput_newcol] = join_df[tput_newcol].round(0).astype(int)
        
        setup_tput_groups = txs_df.groupby(by=['DRAM (GiB)', 'scale'])
        setup_join_groups = join_df.groupby(by=['DRAM (GiB)', 'scale'])

        for name, setup_tput_df in setup_tput_groups:
            dram_value, scale_value = name
            setup_join_df = setup_join_groups.get_group(name)
            setup_join_method_groups = setup_join_df.groupby("method")
            series_dict = {}
            for method_name, method_df in setup_tput_df.groupby('method'):
                method_join_df = setup_join_method_groups.get_group(method_name)
                join_n_mean = method_join_df.drop(columns=['method', 'tx', 'tentative_skip_bytes']).mean()[tput_newcol]
                if method_name != 'merged_idx':
                    series_to_plot = method_df.drop(columns=['method','tentative_skip_bytes']).groupby('tx').mean()[tput_newcol]
                    # append join value
                    series_to_plot = pd.concat([pd.Series({'join-n': join_n_mean}), series_to_plot])
                else:
                    series_to_plot = []
                    for tx, tx_df in method_df.groupby('tx'):
                        if tx != "join-nscci":
                            # mean over all skip bytes
                            all_means = tx_df.drop(columns=['method', 'tx']).mean()
                            series_to_plot.append(all_means[tput_newcol])
                        else:
                            skip_means = tx_df.drop(columns=['method', 'tx']).groupby('tentative_skip_bytes').mean()[tput_newcol]
                            print(f"Finding optimal skip for {tx} in {self.dir}")
                            print(skip_means.to_string())
                            series_to_plot.append(skip_means.max())
                    # dumb skipping
                    dumb_skipping_series = method_df[method_df['tentative_skip_bytes'] == 0].drop(columns=['method','tentative_skip_bytes']).groupby('tx').mean()[tput_newcol]
                    dumb_skipping_series = pd.concat([pd.Series({'join-n': join_n_mean}), dumb_skipping_series])
                    
                    series_to_plot = pd.Series(series_to_plot, index=method_df['tx'].unique())
                    series_to_plot = pd.concat([pd.Series({'join-n': join_n_mean}), series_to_plot])
                series_dict[method_name] = series_to_plot
                
            plot_df = pd.DataFrame(series_dict).round(2)
            plot_df.columns.name = ''
            self._plot_join_point_series(plot_df, tput_newcol, dram_value, scale_value, dumb_skipping_series)

    def _plot_join_point_series(self, plot_df: pd.DataFrame, ylabel, dram_value: int, scale_value: int, dumb_skipping_series = None):
        """Helper to plot point join series."""
        plot_df.rename(index={'join-n': '0 for an\nentire nation', 'join-ns': "1 for a state\nout of a nation", 'join-nsc': "2 for a county\nout of a nation", 'join-nscci': "3 for a city\nout of a nation"}, inplace=True)
        ax = plot_df.plot(title='Point join queries', legend=True, marker='o', ylabel=ylabel, logy=True)
        scatter = ax.scatter(plot_df.index, dumb_skipping_series, label='merged_idx w/o\nsmart skipping', color='red', marker='x', zorder=10)
        handles, labels = ax.get_legend_handles_labels()
        handles.append(scatter)
        ax.legend(handles, labels)
        ax.set_xlabel('Number of inner-instance searches')
        ax.set_xticks(range(len(plot_df.index)), plot_df.index)
        ax.set_title(f'Factor of locality quantified by number of inner-instance searches')
        file_path = self.dir / f'join_point-{dram_value}-{scale_value}.png'
        plt.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close()

    def process_skip_studies(self):
        """
        Compares LSM and B-tree performance for a skip study and generates a plot.
        This is a static method as it compares data from two different analyzer objects.
        """

        filtered = self.tput_df[self.tput_df['tx'] == 'join-nscci']
        groups = filtered.groupby(['DRAM (GiB)', 'scale'])

        for (name, group) in groups:
            dram, scale = name
            group, col, _ = self._prepare_tput_data(group)
            base_elapsed = group[group['method'] == 'base_idx'][col].mean()
            view_elapsed = group[group['method'] == 'mat_view'][col].mean()
            merged_group = group[group['method'] == 'merged_idx']
            filter_columns = [col] + (['pp_0 CPU Util (%)'] if 'pp_0 CPU Util (%)' in merged_group.columns else [])
            merged_means = merged_group.drop(columns=['method', 'tx']).groupby('tentative_skip_bytes').mean()[filter_columns]
            merged_means.rename(columns={col: f'{col} of merged_idx'}, inplace=True)

            print("Skip Study Comparison:")
            print(merged_means.to_string())

            ax = merged_means.plot(title='Skip study with various tentative skip bytes', ylabel=col, xlabel='', marker='o', legend=True, secondary_y='pp_0 CPU Util (%)' if 'pp_0 CPU Util (%)' in merged_means.columns else False)
            base_line = ax.axhline(base_elapsed, linestyle='--', label=f'{col} of base_idx')
            view_line = ax.axhline(view_elapsed, linestyle='-.', label=f'{col} of mat_view')
            # # plot lsm on ax too
            # lsm_mean.plot(ax=ax, marker='o', legend=True)
            # xlabels = list(set(lsm_mean.index).union(set(btree_mean.index)))
            # aggregate all legends
            handles, labels = ax.get_legend_handles_labels()
            handles.extend([base_line, view_line])
            ax.legend(handles, labels)
            xlabels = merged_means.index
            ax.set_xticks(xlabels, xlabels)
            ax.set_ylabel(col)
            if 'pp_0 CPU Util (%)' in merged_means.columns:
                ax.right_ax.set_ylabel('Page Provider Thread CPU Util (%)')
            plt.savefig(f'{self.dir}/skip-study-{dram}-{scale}.png', bbox_inches='tight', dpi=300)
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
            analyzer.process_skip_studies()
            print(f"--- Finished processing {name} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()