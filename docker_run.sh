docker run --name dbtoaster ghcr.io/alicia-lyu/geodb-dbtoaster:latest
docker cp dbtoaster:/results/update_times.csv ./update_times.csv

docker run --name main ghcr.io/alicia-lyu/geodb:latest
docker cp main:/results/geo_btree_TPut.csv ./geo_btree/TPut.csv
docker cp main:/results/geo_lsm_TPut.csv ./geo_lsm/TPut.csv

docker rm main
docker rm dbtoaster