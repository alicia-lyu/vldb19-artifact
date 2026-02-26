.PHONY: clean paper-ready plots

setup:
	mkdir -p geo_btree geo_lsm

clean:
	rm -rf *.png

paper-ready: plots
	mkdir -p paper-ready
	cp geo_lsm/scale40/bgw0-dram0.1/queries-individual-less.png paper-ready/lsm_summary.png
	cp geo_lsm/scale40/bgw0-dram0.1/cpu_utilization.png paper-ready/lsm_cpu_utilization.png
	cp geo_btree/scale15/update-less.png paper-ready/btree_update_summary.png
	cp geo_btree/scale15/bgw0-dram0.1/queries-count.png paper-ready/btree_count.png
	cp geo_btree/scale15/bgw0-dram0.1/queries-count-distinct.png paper-ready/btree_count_distinct.png
	cp geo_btree/scale15/bgw0-dram0.1/queries-join.png paper-ready/btree_join.png
	cp geo_btree/scale15/bgw0-dram0.1/cpu_utilization.png paper-ready/btree_cpu_utilization.png

geo_btree/TPut.csv:
	./docker_run.sh

geo_lsm/TPut.csv:
	./docker_run.sh

plots: geo_btree/TPut.csv geo_lsm/TPut.csv setup
	python main.py