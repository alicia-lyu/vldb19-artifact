import matplotlib.pyplot as plt
import pandas as pd

# 'method', 'skipping', 'tentative_skip_bytes'
def join_scan(dir, elapsed_df: pd.DataFrame):
    elapsed_df = elapsed_df.drop(columns=['skipping', 'tentative_skip_bytes']).groupby('tx') # no meaningful data in these columns for elapsed_df
    join_df = elapsed_df.get_group('join').drop(columns='tx').groupby(by=['DRAM (GiB)', 'scale'])
    for name, group_df in join_df:
        dram_value, scale_value = name
        mean_df = group_df.groupby('method').mean()
        mean_df.to_csv(f'{dir}/join-{dram_value}-{scale_value}.csv')
        print(mean_df.to_string())
    
def join_point(dir, tput_df: pd.DataFrame):
    setup_dfs = tput_df[tput_df['tx'].isin(['join-ns', 'join-nsc', 'join-nscci'])].groupby(by=['DRAM (GiB)', 'scale'])
    for name, setup_df in setup_dfs:
        dram_value, scale_value = name
        method_dfs = setup_df.groupby(by=['method'])
        series_dict = {}
        for method_name_tuple, method_df in method_dfs:
            method_df = method_df.drop(columns=['method','DRAM (GiB)','scale'])
            method_name = method_name_tuple[0]
            if method_name != 'merged':
                series_to_plot = method_df.drop(columns=['skipping','tentative_skip_bytes']).groupby(by=['tx']).mean()['TPut (TX/s)']
            else:
                tx_dfs = method_df.groupby(by=['tx'])
                series_to_plot = []
                for _, tx_df in tx_dfs:
                    skip_means = tx_df.drop(columns='tx').groupby(by=['skipping', 'tentative_skip_bytes']).mean()
                    print(skip_means.to_string())
                    optimal_skip = skip_means['TPut (TX/s)'].max()
                    series_to_plot.append(optimal_skip)
                series_to_plot = pd.Series(series_to_plot, index=tx_dfs.groups.keys())
            series_dict[method_name] = series_to_plot
        plot_df = pd.DataFrame(series_dict)
        plot_df.columns.name = ''
        plot_df = plot_df.round().astype(int)
        plot_join_point_series(plot_df, dram_value, scale_value, dir)

def plot_join_point_series(plot_df: pd.DataFrame, dram_value, scale_value, dir):
    print(plot_df.to_string())
    ax = plot_df.plot(title=f'Point join queries - DRAM: {dram_value} GiB, Scale: {scale_value}', legend=True, marker='o', ylabel='TPut (TX/s)', logy=True, table=True)
    ax.set_xlabel('')
    ax.set_xticks([],[])  # Hide x-ticks because table is drawn below
    plt.savefig(f'{dir}/join_point-{dram_value}-{scale_value}.png', bbox_inches='tight', dpi=300)
    
    
def main():
    for dir in ['geo_lsm', 'geo_btree']:
        elapsed_df = pd.read_csv(f'{dir}/Elapsed.csv')
        tput_df = pd.read_csv(f'{dir}/TPut.csv')
        
        # Process elapsed data for join scan
        join_scan(dir, elapsed_df)
        
        # Process throughput data for join point
        join_point(dir, tput_df)


if __name__ == '__main__':
    main()