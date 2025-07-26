import matplotlib.pyplot as plt
import pandas as pd

def join_scan(dir, elapsed_df: pd.DataFrame):
    join_df = elapsed_df[elapsed_df['tx'] == 'join'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
    for name, group_df in join_df:
        dram_value, scale_value = name
        mean_df = group_df.groupby('method').mean()['Elapsed (ms)']
        mean_df = mean_df.round(0).astype(int)
        file_prefix = f'{dir}/join-{dram_value}-{scale_value}'
        with open(f'{file_prefix}.csv', 'w') as f:
            f.write(mean_df.to_string())
        ax = mean_df.plot(title=f'Join scan elapsed time', ylabel='Elapsed (ms)', xlabel='Method', kind='bar', legend=False, table=True)
        ax.set_xlabel('')
        ax.set_xticks([], [])  # Hide x-ticks because table is drawn below
        plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
        plt.close()
        

def maintain(dir, tput_df: pd.DataFrame):
    tput_df = tput_df[tput_df['tx'] == 'maintain'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
    for name, group_df in tput_df:
        dram_value, scale_value = name
        mean_df = group_df.groupby('method').mean()['TPut (TX/s)']
        mean_df.name = 'TPut (TX/s)'
        mean_df = mean_df.round(2)
        file_prefix = f'{dir}/maintain-{dram_value}-{scale_value}'
        with open(f'{file_prefix}.csv', 'w') as f:
            f.write(mean_df.to_string())
        ax = mean_df.plot(title=f'Maintain throughput', ylabel='TPut (TX/s)', xlabel='Method', kind='bar', legend=False, table=True)
        ax.set_xlabel('')
        ax.set_xticks([], [])  # Hide x-ticks because table is drawn below
        plt.savefig(f'{file_prefix}.png', bbox_inches='tight', dpi=300)
        plt.close()
        
def mixed(dir, elapsed_df: pd.DataFrame, tput_df: pd.DataFrame):
    mixed_scan_df = elapsed_df[elapsed_df['tx'] == 'mixed'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
    mixed_point_df = tput_df[tput_df['tx'] == 'mixed-point'].drop(columns=['tx', 'tentative_skip_bytes']).groupby(by=['DRAM (GiB)', 'scale'])
    for (name, scan_group), (_, point_group) in zip(mixed_scan_df, mixed_point_df):
        dram_value, scale_value = name
        scan_mean = scan_group.groupby('method').mean()['Elapsed (ms)']
        point_mean = point_group.groupby('method').mean()['TPut (TX/s)']
        combined_df = pd.DataFrame({'Scan Elapsed (ms)': scan_mean, 'Point TPut (TX/s)': point_mean})
        combined_df = combined_df.round({'Scan Elapsed (ms)': 0, 'Point TPut (TX/s)': 2})
        combined_df = combined_df.astype({'Scan Elapsed (ms)': int, 'Point TPut (TX/s)': float})
        with open(f'{dir}/mixed-{dram_value}-{scale_value}.csv', 'w') as f:
            f.write(combined_df.to_string())
        axes = combined_df.plot(subplots=True, layout=(2, 1), legend=True, marker='o', sharex=True, xlabel='')
        axes[0, 0].set_ylabel('Elapsed (ms)')
        axes[1, 0].set_ylabel('TPut (TX/s)')
        plt.savefig(f'{dir}/mixed-{dram_value}-{scale_value}.png', bbox_inches='tight', dpi=300)
        plt.close()
    
def skip_studies(lsm_df: pd.DataFrame, btree_df: pd.DataFrame):
    # revise (0, 8096) to
    lsm_df = lsm_df[(lsm_df['tx'] == 'join-nscci') & (lsm_df['method'] == 'merged_idx')].drop(columns=['tx', 'method'])
    btree_df = btree_df[(btree_df['tx'] == 'join-nscci') & (btree_df['method'] == 'merged_idx')].drop(columns=['tx', 'method'])
    lsm_df = lsm_df.groupby(by=['DRAM (GiB)', 'scale'])
    btree_df = btree_df.groupby(by=['DRAM (GiB)', 'scale'])
    for (_, lsm_group), (_, btree_group) in zip(lsm_df, btree_df): # one group each
        lsm_mean = lsm_group.groupby('tentative_skip_bytes').mean()['TPut (TX/s)']
        btree_mean = btree_group.groupby('tentative_skip_bytes').mean()['TPut (TX/s)']
        combined_df = pd.DataFrame({'LSM': lsm_mean, 'BTree': btree_mean})
        print(combined_df.to_string())
        plot_skip_study(combined_df)
    

def join_point(dir, tput_df: pd.DataFrame):
    setup_dfs = tput_df[tput_df['tx'].isin(['join-ns', 'join-nsc', 'join-nscci'])].groupby(by=['DRAM (GiB)', 'scale'])
    for name, setup_df in setup_dfs:
        dram_value, scale_value = name
        method_dfs = setup_df.groupby(by=['method'])
        series_dict = {}
        for method_name_tuple, method_df in method_dfs:
            method_df = method_df.drop(columns=['method','DRAM (GiB)','scale'])
            method_name = method_name_tuple[0]
            if method_name != 'merged_idx':
                series_to_plot = method_df.drop(columns=['tentative_skip_bytes']).groupby(by=['tx']).mean()['TPut (TX/s)']
            else:
                tx_dfs = method_df.groupby('tx')
                series_to_plot = []
                for tx, tx_df in tx_dfs:
                    skip_means = tx_df.drop(columns='tx').groupby(by=['tentative_skip_bytes']).mean()
                    print(f"Finding optimal skip for {tx} in {dir}")
                    print(skip_means['TPut (TX/s)'].to_string())
                    optimal_skip = skip_means['TPut (TX/s)'].max()
                    series_to_plot.append(optimal_skip)
                series_to_plot = pd.Series(series_to_plot, index=tx_dfs.groups.keys())
            series_dict[method_name] = series_to_plot
        plot_df = pd.DataFrame(series_dict)
        plot_df.columns.name = ''
        plot_df = plot_df.round(2)
        plot_join_point_series(plot_df, dram_value, scale_value, dir)

def plot_join_point_series(plot_df: pd.DataFrame, dram_value, scale_value, dir):
    ax = plot_df.plot(title=f'Point join queries', legend=True, marker='o', ylabel='TPut (TX/s)', logy=True, table=True)
    ax.set_xlabel('')
    ax.set_xticks([],[])  # Hide x-ticks because table is drawn below
    plt.savefig(f'{dir}/join_point-{dram_value}-{scale_value}.png', bbox_inches='tight', dpi=300)
    plt.close()
    
def plot_skip_study(merged_idx_nscci_df: pd.DataFrame):
    ax = merged_idx_nscci_df.plot(title=f'Skip study', ylabel='TPut (TX/s)', xlabel='Tentative skip bytes', marker='o', legend=True)
    ax.set_xticks(merged_idx_nscci_df.index)
    plt.savefig(f'skip_study', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    for dir in ['geo_lsm', 'geo_btree']:
        elapsed_df = pd.read_csv(f'{dir}/Elapsed.csv')
        tput_df = pd.read_csv(f'{dir}/TPut.csv')
        if dir == 'geo_lsm':
            lsm_tput_df = tput_df
        else:
            btree_tput_df = tput_df
        
        # Process elapsed data for join scan
        join_scan(dir, elapsed_df)
        maintain(dir, tput_df)
        mixed(dir, elapsed_df, tput_df)
        
        # Process throughput data for join point
        join_point(dir, tput_df)
    skip_studies(lsm_tput_df, btree_tput_df)

if __name__ == '__main__':
    main()