from pathlib import Path
from tqdm import tqdm

import geopandas as gpd
import pdb

from helper import ivan_data_root, kevin_data_root


def crawl(path: Path, record: str, prefix=""):
    """Crawls path and records files present"""
    if path.stem == "_common":
        return record

    if path.stem == "time_series":
        to_print = f"{prefix}{path.parent.stem}/{path.stem}"
    else:
        to_print = f"{prefix}{path.stem}"

    npz_files = 0
    csv_files = 0
    children = list(path.glob("*"))
    for p in tqdm(children):
        if p.is_dir():
            tifs = len(list(p.glob("*/*.tif")))
            if tifs == 0:
                record = crawl(p, record, prefix=prefix + "  ")
            else:
                record += f"{prefix}    {p.stem}: {tifs} tifs\n"

        elif p.name == "labels.geojson":
            labels = gpd.read_file(p)
            record += f"{to_print}: {len(labels)} labels\n"

        elif p.suffix == ".npz":
            npz_files += 1

        elif p.suffix == ".csv":
            csv_files += 1

    if npz_files > 0:
        record += f"{to_print}: {npz_files} .npz files\n"
    if csv_files > 0:
        record += f"{to_print}: {csv_files} .csv files\n"

    return record


record = ""
for root in [ivan_data_root, kevin_data_root]:
    record += root + "\n"
    record = crawl(Path(root), record)

print(record)
Path("data_overview.txt").write_text(record)
