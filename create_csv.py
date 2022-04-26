import geopandas as gpd
import json
import numpy as np
import os
import pandas as pd
import pdb
import zipfile

from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from helper import ivan_data_root, kevin_data_root


def get_paths(competition: str, train_or_test: str, pos: str):
    """
    Get paths for labels and remote sensing directories
    """
    if competition == "south_africa":
        country = "ref_fusion_competition_south_africa"
        root = ivan_data_root
        year = "2017"
    elif competition == "germany":
        country = "dlr_fusion_competition_germany"
        root = kevin_data_root
        if train_or_test == "train":
            year = "2018"
        else:
            year = "2019"
    else:
        raise NameError("Please respecify competition correctly.")

    fill = ""
    if train_or_test == "train" and competition == "south_africa":
        fill = f"_{pos}"

    sentinel_1_tif_folder = f"{country}_{train_or_test}_source_sentinel_1"
    sentinel_2_tif_folder = f"{country}_{train_or_test}_source_sentinel_2"
    planet_5day_tif_folder = f"{country}_{train_or_test}_source_planet_5day"
    planet_daily_tif_folder = f"{country}_{train_or_test}_source_planet"
    input_dirs = {
        "s1": f"{root}/{sentinel_1_tif_folder}/{sentinel_1_tif_folder}{fill}_asc_{pos}_{year}",
        "s2": f"{root}/{sentinel_2_tif_folder}/{sentinel_2_tif_folder}{fill}_{pos}_{year}",
        "planet_5day": f"{root}/{planet_5day_tif_folder}",
        "planet_daily": f"{root}/{planet_daily_tif_folder}",
    }

    label_file = f"{root}/{country}_{train_or_test}_labels/{country}_{train_or_test}_labels_{pos}/labels.geojson"
    return label_file, input_dirs


def load_labels_and_ids(label_file: str, npyfolder: str, competition: str):
    """
    Load labels and ids from geojson file
    """
    labels = gpd.read_file(label_file)
    # if competition == "germany":
    #     with Path("src/s1g_redflag.txt").open("r") as f:
    #         sentinel_1_redflags = f.readlines()
    #         sentinel_1_redflags = [int(f) for f in "".join(sentinel_1_redflags).split("\n")]
    #         labels = labels[~labels.fid.isin(sentinel_1_redflags)]

    label_ids = labels["crop_id"].unique()
    label_names = labels["crop_name"].unique()

    zipped_lists = zip(label_ids, label_names)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    label_ids, label_names = [list(tuple) for tuple in tuples]

    labels["path"] = labels["fid"].apply(lambda fid: os.path.join(npyfolder, f"fid_{fid}.npz"))
    labels["exists"] = labels.path.apply(os.path.exists)
    return labels, label_ids


def get_timesteps(input_dir: str):
    """
    Get timesteps for one remote sensing instrument (s1, s2, planet_5day, planet_daily)
    """
    with (Path(input_dir) / "collection.json").open("rb") as f:
        planet_collection = json.load(f)

    start_str, end_str = planet_collection["extent"]["temporal"]["interval"][0]
    start = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ")
    end = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ")
    timesteps = np.array(
        [
            datetime.fromordinal(o).replace(tzinfo=None)
            for o in range(start.toordinal(), end.toordinal() + 1)
        ]
    )
    return timesteps


def read_in_image_stack(npyfile):
    try:
        object = np.load(npyfile)
        image_stack = object["image_stack"]
        mask = object["mask"]
    except zipfile.BadZipFile:
        print("ERROR: {} is a bad zipfile...".format(npyfile))
        raise

    # Normalize by scaling
    image_stack = image_stack * 1e-4

    # Ignore all values not in mask
    image_stack = image_stack[:, :, mask > 0]
    return image_stack


def compute_stats(image_stack_channel, prefix):
    """
    Compute statistics for a single channel from the image_stack
    """
    assert (
        len(image_stack_channel.shape) == 2
    ), "Expected image stack with only temporal and flat spatial dimension"
    return {
        f"{prefix}_mean": np.mean(image_stack_channel, axis=1),
        f"{prefix}_std": np.std(image_stack_channel, axis=1),
        f"{prefix}_min": np.min(image_stack_channel, axis=1),
        f"{prefix}_max": np.max(image_stack_channel, axis=1),
        f"{prefix}_median": np.median(image_stack_channel, axis=1),
    }


########################################################################################################################
# Set up the config
########################################################################################################################
satellite = "planet_daily"
competition = "germany"
train_or_test = "test"
pos = "33N_17E_243N"

########################################################################################################################
# Read in features
########################################################################################################################

label_file, input_dirs = get_paths(competition=competition, train_or_test=train_or_test, pos=pos)
input_dir = Path(input_dirs[satellite])
npyfolder = Path(input_dir / "time_series")
labels, label_ids = load_labels_and_ids(label_file, npyfolder, competition)

if not satellite.startswith("planet"):
    raise NotImplementedError("Only planet currently supported")

bands = ["blue", "green", "red", "nir"]

timesteps = get_timesteps(input_dir)

folder = Path(f"{ivan_data_root}/{competition}/{train_or_test}/{pos}")
folder.mkdir(parents=True, exist_ok=True)

feature_dfs = []
missing_fids = []
for i, row in tqdm(labels.iterrows(), total=len(labels)):
    p = Path(f"{folder}/{i}.csv")
    if p.exists():
        continue

    npyfile = Path(f"{npyfolder}/fid_{row.fid}.npz")
    if not npyfile.exists():
        missing_fids.append(npyfile.stem.split("_")[1])
        continue

    image_stack = read_in_image_stack(npyfile)

    stats_per_channel_list = []
    for j, channel in enumerate(bands):
        stats = compute_stats(image_stack[:, j], channel)
        stats_per_channel_list.append(stats)

    red = image_stack[:, bands.index("red")]
    nir = image_stack[:, bands.index("nir")]
    ndvi = (nir - red) / (nir + red)
    stats = compute_stats(ndvi, "ndvi")
    stats_per_channel_list.append(stats)

    stats_per_channel = {k: v for s in stats_per_channel_list for k, v in s.items()}

    # Create feature dictionary
    feature_dict = {
        "daily_fid": row.fid,
        "label": label_ids.index(row.crop_id),
        "timesteps": timesteps,
        **stats_per_channel,
    }
    feature_df = pd.DataFrame(feature_dict)
    feature_df.to_csv(p, index=False)

print(f"Found missing: {len(missing_fids)}")
with Path(folder / "missing.txt").open("wb") as f:
    for missing in missing_fids:
        f.write(f"{missing}\n".encode("utf-8"))
print(f"Wrote missing fids to {folder}/missing.txt")
