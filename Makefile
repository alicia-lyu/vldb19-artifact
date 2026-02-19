clean:
	rm -rf *.png

paper-ready: geo_btree/scale15/bgw0-dram0.1/queries-individual-less.png \
			  geo_lsm/scale40/bgw0-dram0.1/queries-individual-less.png \
			  geo_btree/scale15/update-less.png
	mkdir -p paper-ready
# 	cp geo_btree/scale15/bgw0-dram0.1/queries-individual-less.png paper-ready/btree_summary.png
	cp geo_lsm/scale40/bgw0-dram0.1/queries-individual-less.png paper-ready/lsm_summary.png
	cp geo_lsm/scale40/bgw0-dram0.1/cpu_utilization.png paper-ready/lsm_cpu_utilization.png
	cp geo_btree/scale15/update-less.png paper-ready/btree_update_summary.png
	cp geo_btree/scale15/bgw0-dram0.1/queries-count.png paper-ready/btree_count.png
	cp geo_btree/scale15/bgw0-dram0.1/queries-count-distinct.png paper-ready/btree_count_distinct.png
	cp geo_btree/scale15/bgw0-dram0.1/update-size.png paper-ready/btree_update_size.png
	cp geo_btree/scale15/bgw0-dram0.1/queries-join.png paper-ready/btree_join.png
	cp geo_btree/scale15/bgw0-dram0.1/cpu_utilization.png paper-ready/btree_cpu_utilization.png