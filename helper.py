import geopandas as gpd
import pdb
import warnings

from pathlib import Path
from shapely.errors import ShapelyDeprecationWarning
from src.utils.data_transform import (
    Sentinel1Transform,
    Sentinel2Transform,
    PlanetTransform,
)
from src.utils.planet_reader import PlanetReader
from src.utils.s1_s2_planet_reader import S1S2PlanetReader
from src.utils.sentinel_1_reader import S1Reader
from src.utils.sentinel_2_reader import S2Reader
from src.utils.s1_s2_reader import S1S2Reader


warnings.filterwarnings(action="ignore", category=ShapelyDeprecationWarning)

ivan_data_root = "/cmlscratch/izvonkov/tum-planet-radearth-ai4food-challenge/data"
kevin_data_root = "/cmlscratch/hkjoo/repo/ai4eo/data"

def load_reader(
    competition: str,
    satellite: str,
    pos: str,
    include_bands: bool,
    include_cloud: bool,
    include_ndvi: bool,
    image_size: int,
    spatial_backbone: str,
    include_rvi: bool = False,
    pse_sample_size: int = 64,
    min_area_to_ignore: int = 1000,
    train_or_test: str = "train",
    alignment: str = "1to2",
    s1_temporal_dropout: float = 0.0,
    s2_temporal_dropout: float = 0.0,
    planet_temporal_dropout: float = 0.0,
):
    if competition == "south_africa":
        country = "ref_fusion_competition_south_africa"
        root = ivan_data_root
        year = '2017'
    elif competition == "germany":
        country = "dlr_fusion_competition_germany"
        root = kevin_data_root
        if train_or_test == 'train':
            year = '2018'
        else:
            year = '2019'
    else:
        raise NameError("Please respecify competition correctly.")

    label_file = f"{root}/{country}_{train_or_test}_labels/{country}_{train_or_test}_labels_{pos}/labels.geojson"
    labels = gpd.read_file(label_file)

    if competition == 'germany':
        with Path("src/s1g_redflag.txt").open('r') as f:
            sentinel_1_redflags = f.readlines()
            sentinel_1_redflags = [int(f) for f in "".join(sentinel_1_redflags).split('\n')]
            labels = labels[~labels.fid.isin(sentinel_1_redflags)]

    label_ids = labels["crop_id"].unique()
    label_names = labels["crop_name"].unique()

    zipped_lists = zip(label_ids, label_names)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    label_ids, label_names = [list(tuple) for tuple in tuples]

    kwargs = dict(
        image_size=image_size,
        pse_sample_size=pse_sample_size,
        spatial_backbone=spatial_backbone,
        normalize=True,
        is_train=train_or_test == "train",
    )

    fill = ""

    if train_or_test == "train" and competition == 'south_africa':
        fill = f"_{pos}"

    sentinel_1_tif_folder = f"{country}_{train_or_test}_source_sentinel_1"
    sentinel_2_tif_folder = f"{country}_{train_or_test}_source_sentinel_2"
    planet_5day_tif_folder = f"{country}_{train_or_test}_source_planet_5day"
    planet_daily_tif_folder = f"{country}_{train_or_test}_source_planet"
    s1_input_dir = f"{root}/{sentinel_1_tif_folder}/{sentinel_1_tif_folder}{fill}_asc_{pos}_{year}"
    s2_input_dir = f"{root}/{sentinel_2_tif_folder}/{sentinel_2_tif_folder}{fill}_{pos}_{year}"
    planet_5day_input_dir = f"{root}/{planet_5day_tif_folder}"
    planet_daily_input_dir = f"{root}/{planet_daily_tif_folder}"
    
    if pos == "34S_19E_259N":
        planet_5day_input_dir = planet_5day_input_dir + "_259"
        planet_daily_input_dir = planet_daily_input_dir + "_259"

    s1_transform = Sentinel1Transform(include_rvi=include_rvi, **kwargs).transform
    s2_transform = Sentinel2Transform(
        include_cloud=include_cloud,
        include_ndvi=include_ndvi,
        include_bands=include_bands,
        **kwargs,
    ).transform
    planet_transform = PlanetTransform(
        include_bands=include_bands, include_ndvi=include_ndvi, **kwargs
    ).transform

    if satellite == "sentinel_1":
        reader = S1Reader(
            input_dir=s1_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            filter=sentinel_1_redflags if competition == 'germany' else None,
            transform=s1_transform,
            temporal_dropout=s1_temporal_dropout,
        )
    elif satellite == "sentinel_2":
        reader = S2Reader(
            input_dir=s2_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            include_cloud=include_cloud,
            filter=sentinel_1_redflags if competition == 'germany' else None,
            transform=s2_transform,
            temporal_dropout=s2_temporal_dropout,
        )
    elif satellite == "s1_s2":
        reader = S1S2Reader(
            s1_input_dir=s1_input_dir,
            s2_input_dir=s2_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            include_cloud=include_cloud,
            s1_transform=s1_transform,
            s2_transform=s2_transform,
            alignment=alignment,
            filter=sentinel_1_redflags if competition == 'germany' else None,
            s1_temporal_dropout=s1_temporal_dropout,
            s2_temporal_dropout=s2_temporal_dropout,
        )
    elif satellite == "planet_5day":
        reader = PlanetReader(
            input_dir=planet_5day_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=planet_transform,
            temporal_dropout=planet_temporal_dropout,
        )
    elif satellite == "planet_daily":
        reader = PlanetReader(
            input_dir=planet_daily_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=planet_transform,
            temporal_dropout=planet_temporal_dropout,
        )
    elif satellite == "s1_s2_planet_daily":
        reader = S1S2PlanetReader(
            s1_input_dir=s1_input_dir,
            s2_input_dir=s2_input_dir,
            planet_input_dir=planet_daily_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            filter=sentinel_1_redflags if competition == 'germany' else None,
            s1_transform=s1_transform,
            s2_transform=s2_transform,
            planet_transform=planet_transform,
            s1_temporal_dropout=s1_temporal_dropout,
            s2_temporal_dropout=s2_temporal_dropout,
            planet_temporal_dropout=planet_temporal_dropout,
        )
        
    return label_names, reader
