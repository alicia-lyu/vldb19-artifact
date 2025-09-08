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
            
    def anonymized_points(self, include_suffix=True):
        point_df_groups = self.tput_df.groupby(by=['DRAM (GiB)', 'scale', 'bgw_pct'])
        
        for (name, group_df) in point_df_groups:
            dram_value, scale_value, bgw_pct = name
            group_df, y_label, unit = self._prepare_tput_data(group_df)
            point_per_tx = group_df.groupby('tx')
            txs = {}
            tx_count = 0
            for (tx), tx_df in point_per_tx:
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

            size_df = pd.DataFrame([group_df.drop(columns=['tx']).groupby('method').mean()['size (MiB)'].round(0).astype(int)])

            params = {'kind': 'bar', 'ylabel': f"Query time ({unit})", 'logy': True}
            if include_suffix:
                params['table'] = True
                params['xlabel'] = ''
                params['legend'] = False # include legend in size plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [tx_count + 1, 1.5]})
                # little space between subplots
                plt.subplots_adjust(wspace=0.1)
                params['ax'] = axes[0]
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                params['ax'] = ax
                params['legend'] = True
            ax = anonymized_df.plot(**params)

            if include_suffix:
                ax.set_xticks([], [])
                # ax legend outside
                
                ax2 = axes[1]
                # pass a series with no column names
                series_to_draw = size_df.iloc[0]
                series_to_draw.index = [''] * len(series_to_draw)

                size_df.plot(kind='bar', ylabel='Size (MiB)', legend=True, ax=ax2, xlabel='', table=series_to_draw)
                ax2.set_xticks([], [])
                ax2.yaxis.set_label_position("right")
                ax2.yaxis.tick_right()
                ax2.legend(loc='upper right', bbox_to_anchor=(0, 1))
            else:
                ax.set_xticklabels(anonymized_df.index, rotation=0)
            filename_suffix = 'more' if include_suffix else 'less'
            plt.savefig(f'{self.dir}/anonymized_points-{dram_value}-{scale_value}-{bgw_pct}-{filename_suffix}.png', bbox_inches='tight', dpi=300)
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

    def process_skip_studies(self, filter_tx: str = 'join-nscci'):
        """
        Compares LSM and B-tree performance for a skip study and generates a plot.
        This is a static method as it compares data from two different analyzer objects.
        """

        filtered = self.tput_df[self.tput_df['tx'] == filter_tx]
        groups = filtered.groupby(['DRAM (GiB)', 'scale', 'bgw_pct'])

        for (name, group) in groups:
            dram, scale, bgw_pct = name
            group, col, _ = self._prepare_tput_data(group)
            base_elapsed = group[group['method'] == 'base_idx'][col].mean()
            view_elapsed = group[group['method'] == 'mat_view'][col].mean()
            merged_group = group[group['method'] == 'merged_idx']
            filter_columns = [col]
            # + (['pp_0 CPU Util (%)'] if 'pp_0 CPU Util (%)' in merged_group.columns else [])
            merged_means = merged_group.drop(columns=['method', 'tx']).groupby('tentative_skip_bytes')[filter_columns].mean()
            merged_means.rename(columns={col: 'merged_idx'}, inplace=True)
            print(merged_means)

            ax = merged_means.plot(title='Smart Skipping', ylabel=col, xlabel='tentatively skipped bytes', marker='o', legend=True, secondary_y='pp_0 CPU Util (%)' if 'pp_0 CPU Util (%)' in merged_means.columns else False)
            base_line = ax.axhline(base_elapsed, linestyle=':', label=f'base_idx')
            view_line = ax.axhline(view_elapsed, linestyle='--', label=f'mat_view')
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
            plt.savefig(f'{self.dir}/skip-study-{filter_tx}-{dram}-{scale}-{bgw_pct}.png', bbox_inches='tight', dpi=300)
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
            # analyzer.anonymized_scans()
            analyzer.anonymized_points()
            analyzer.anonymized_points(False)
            analyzer.process_skip_studies()
            analyzer.process_skip_studies('mixed-nscci')
            print(f"--- Finished processing {name} ---")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the data directories and CSV files exist.")

if __name__ == '__main__':
    main()