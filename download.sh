ROOT_URL=https://api.radiant.earth/mlhub/v1/archive
KEY=17238db9dae8c654b106ec7ab1c10ccb8613176aaeca8fd30ad2806b5e93b87c

curl -L "$ROOT_URL/$1?key=$KEY" --output data/$1.tar.gz