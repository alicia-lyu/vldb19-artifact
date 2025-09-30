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