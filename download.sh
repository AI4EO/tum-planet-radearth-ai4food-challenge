ROOT_URL=https://api.radiant.earth/mlhub/v1/archive
KEY=16a2d3dd90787e39dbc13979a7d7187f71528bf66f03ef7eff9f6af5c5d18065

curl -L "$ROOT_URL/$1?key=$KEY" --output data/$1.tar.gz
