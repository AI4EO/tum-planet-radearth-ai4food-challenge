import geopandas as gpd
import pdb
import warnings
from shapely.errors import ShapelyDeprecationWarning
from notebook.utils.data_transform import EOTransformer, Sentinel2Transform
from notebook.utils.planet_reader import PlanetReader
from notebook.utils.sentinel_1_reader import S1Reader
from notebook.utils.sentinel_2_reader import S2Reader

warnings.filterwarnings(action="ignore", category=ShapelyDeprecationWarning)

competition = "ref_fusion_competition_south_africa"


def load_reader(
    satellite: str,
    pos: str,
    include_bands: bool,
    include_cloud: bool,
    include_ndvi: bool,
    image_size: int,
    spatial_backbone: str,
    min_area_to_ignore: int = 1000,
    train_or_test: str = "train",
):
    label_file = f"data/{competition}_{train_or_test}_labels/{competition}_{train_or_test}_labels_{pos}/labels.geojson"
    labels = gpd.read_file(label_file)
    label_ids = labels["crop_id"].unique()
    label_names = labels["crop_name"].unique()

    zipped_lists = zip(label_ids, label_names)
    sorted_pairs = sorted(zipped_lists)

    tuples = zip(*sorted_pairs)
    label_ids, label_names = [list(tuple) for tuple in tuples]

    tif_folder = f"{competition}_{train_or_test}_source_{satellite}"

    kwargs = dict(
        image_size=image_size,
        spatial_encoder=spatial_backbone != "none",
        normalize=True,
    )
    default_transform = EOTransformer(**kwargs).transform

    input_dir = f"data/{tif_folder}"

    fill = ""
    if train_or_test == "train":
        fill = f"_{pos}"

    if satellite == "sentinel_1":
        reader = S1Reader(
            input_dir=f"{input_dir}/{tif_folder}{fill}_asc_{pos}_2017",
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=default_transform,
        )
    elif satellite == "sentinel_2":
        reader = S2Reader(
            input_dir=f"{input_dir}/{tif_folder}{fill}_{pos}_2017",
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
    elif satellite == "planet_5day":
        if pos == "34S_19E_259N":
            input_dir = input_dir + "_259"
        reader = PlanetReader(
            input_dir=input_dir,
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=min_area_to_ignore,
            transform=default_transform,
        )

    return label_names, reader
