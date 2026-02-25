docker run --name geodb-run geodb
docker run --name geodb-dbtoaster-run geodb-dbtoaster

docker cp geodb-run:/results/geo_btree_TPut.csv ./geo_btree/TPut.csv
docker cp geodb-run:/results/geo_lsm_TPut.csv ./geo_lsm/TPut.csv
docker cp geodb-dbtoaster-run:/results/update_times.csv ./update_times.csv