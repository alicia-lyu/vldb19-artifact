docker pull ghcr.io/alicia-lyu/geodb-dbtoaster:latest
DBTOASTER_ID=$(docker run -d ghcr.io/alicia-lyu/geodb-dbtoaster:latest)
echo "Waiting for DBToaster container" $DBTOASTER_ID "to finish..."
docker wait $DBTOASTER_ID
docker cp $DBTOASTER_ID:/results/update_times.csv ./update_times.csv

docker pull ghcr.io/alicia-lyu/geodb:latest
MAIN_ID=$(docker run -d ghcr.io/alicia-lyu/geodb:latest)
echo "Waiting for main container" $MAIN_ID "to finish..."
docker wait $MAIN_ID
docker cp $MAIN_ID:/results/geo_btree_TPut.csv ./geo_btree/TPut.csv
docker cp $MAIN_ID:/results/geo_lsm_TPut.csv ./geo_lsm/TPut.csv

docker rm $MAIN_ID
docker rm $DBTOASTER_ID