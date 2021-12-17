import geopandas as gpd
import pdb
import warnings

from pathlib import Path
from shapely.errors import ShapelyDeprecationWarning
from src.utils.data_transform import EOTransformer, Sentinel2Transform
from src.utils.planet_reader import PlanetReader
from src.utils.sentinel_1_reader import S1Reader
from src.utils.sentinel_2_reader import S2Reader
from src.utils.s1_s2_reader import S1S2Reader


warnings.filterwarnings(action="ignore", category=ShapelyDeprecationWarning)

competition = "ref_fusion_competition_south_africa"
root = Path(__file__).parent / "data"


def load_reader(
    satellite: str,
    pos: str,
    include_bands: bool,
    include_cloud: bool,
    include_ndvi: bool,
    image_size: int,
    spatial_backbone: str,
    pse_sample_size: int = 64,
    min_area_to_ignore: int = 1000,
    train_or_test: str = "train",
):
    label_file = (
        root
        / f"{competition}_{train_or_test}_labels/{competition}_{train_or_test}_labels_{pos}/labels.geojson"
    )
    labels = gpd.read_file(label_file)
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
    )
    default_transform = EOTransformer(**kwargs).transform

    fill = ""
    if train_or_test == "train":
        fill = f"_{pos}"

    sentinel_1_tif_folder = f"{competition}_{train_or_test}_source_sentinel_1"
    sentinel_2_tif_folder = f"{competition}_{train_or_test}_source_sentinel_2"
    planet_5day_tif_folder = f"{competition}_{train_or_test}_source_planet_5day"
    planet_daily_tif_folder = f"{competition}_{train_or_test}_source_planet"
    s1_input_dir = str(
        root / f"{sentinel_1_tif_folder}/{sentinel_1_tif_folder}{fill}_asc_{pos}_2017"
    )
    s2_input_dir = str(root / f"{sentinel_2_tif_folder}/{sentinel_2_tif_folder}{fill}_{pos}_2017")
    planet_5day_input_dir = str(root / f"{planet_5day_tif_folder}")
    planet_daily_input_dir = str(root / f"{planet_daily_tif_folder}")
    if pos == "34S_19E_259N":
        planet_5day_input_dir = planet_5day_input_dir + "_259"
        planet_daily_input_dir = planet_daily_input_dir + "_259"

    if satellite == "sentinel_1":
        reader = S1Reader(
            input_dir=s1_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=default_transform,
        )
    elif satellite == "sentinel_2":
        reader = S2Reader(
            input_dir=s2_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            include_cloud=include_cloud,
            transform=Sentinel2Transform(
                include_cloud=include_cloud,
                include_ndvi=include_ndvi,
                include_bands=include_bands,
                **kwargs,
            ).transform,
        )
    elif satellite == "s1_s2":
        reader = S1S2Reader(
            s1_input_dir=s1_input_dir,
            s2_input_dir=s2_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            include_cloud=include_cloud,
            s1_transform=default_transform,
            s2_transform=Sentinel2Transform(
                include_cloud=include_cloud,
                include_ndvi=include_ndvi,
                include_bands=include_bands,
                **kwargs,
            ).transform,
        )
    elif satellite == "planet_5day":
        reader = PlanetReader(
            input_dir=planet_5day_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=default_transform,
        )
    elif satellite == "planet_daily":
        reader = PlanetReader(
            input_dir=planet_daily_input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=default_transform,
        )

    return label_names, reader
