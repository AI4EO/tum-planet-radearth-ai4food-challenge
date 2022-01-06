"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines a data reader for Planet Fusion eath observation data
"""
import torch
import geopandas as gpd
import rasterio as rio
from rasterio import features
import numpy as np
import os
import json
import zipfile
import glob
import pdb

from datetime import datetime
from pathlib import Path
from tqdm import tqdm


class PlanetReader(torch.utils.data.Dataset):
    """
    THIS CLASS INITIALIZES THE DATA READER FOR PLANET DATA
    """

    def __init__(
        self,
        input_dir,
        label_dir,
        label_ids=None,
        transform=None,
        min_area_to_ignore=1000,
        selected_time_points=None,
        tzinfo=None,
        temporal_dropout=0.0,
        return_timesteps=False,
    ):
        """
        THIS FUNCTION INITIALIZES DATA READER.
        :param input_dir: directory of input images in TIF format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param label_ids: an array of crop IDs in order. if the crop labels in GeoJSON data is not started from index 0 it can be used. Otherwise it is not required.
        :param transform: data transformer function for the augmentation or data processing
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :param selected_time_points: If a sub set of the time series will be exploited, it can determine the index of those times in a given time series dataset

        :return: None
        """

        self.data_transform = transform
        self.selected_time_points = selected_time_points
        self.crop_ids = label_ids
        if label_ids is not None and not isinstance(label_ids, list):
            self.crop_ids = label_ids.tolist()

        self.npyfolder = os.path.abspath(input_dir + "/time_series")
        self.labels = PlanetReader._setup(input_dir, label_dir, self.npyfolder, min_area_to_ignore)

        with (Path(input_dir) / "collection.json").open("rb") as f:
            planet_collection = json.load(f)

        start_str, end_str = planet_collection["extent"]["temporal"]["interval"][0]
        start = datetime.strptime(start_str, "%Y-%m-%dT%H:%M:%SZ")
        end = datetime.strptime(end_str, "%Y-%m-%dT%H:%M:%SZ")
        self.timesteps = np.array(
            [
                datetime.fromordinal(o).replace(tzinfo=tzinfo)
                for o in range(start.toordinal(), end.toordinal() + 1)
            ]
        )

        self.temporal_dropout = temporal_dropout
        self.return_timesteps = return_timesteps

    def __len__(self):
        """
        THIS FUNCTION RETURNS THE LENGTH OF DATASET
        """
        return len(self.labels)

    def __getitem__(self, item):
        """
        THIS FUNCTION ITERATE OVER THE DATASET BY GIVEN ITEM NO AND RETURNS FOLLOWINGS:
        :return: image_stack in size of [Time Stamp, Image Dimension (Channel), Height, Width] , crop_label, field_mask in size of [Height, Width], field_id, timesteps
        """

        feature = self.labels.iloc[item]

        npyfile = os.path.join(self.npyfolder, "fid_{}.npz".format(feature.fid))
        if os.path.exists(npyfile):  # use saved numpy array if already created
            try:
                object = np.load(npyfile)
                image_stack = object["image_stack"]
                mask = object["mask"]
            except zipfile.BadZipFile:
                print("ERROR: {} is a bad zipfile...".format(npyfile))
                raise
        else:
            print("ERROR: {} is a missing...".format(npyfile))
            raise

        if self.data_transform is not None:
            image_stack, mask = self.data_transform(image_stack, mask)

        if self.selected_time_points is not None:
            image_stack = image_stack[self.selected_time_points]

        if self.crop_ids is not None:
            label = self.crop_ids.index(feature.crop_id)
        else:
            label = feature.crop_id

        if self.temporal_dropout > 0:
            dropout_timesteps = np.random.rand(image_stack.shape[0]) > self.temporal_dropout
            image_stack = image_stack[dropout_timesteps]
            timesteps = self.timesteps[dropout_timesteps]
        else:
            timesteps = self.timesteps

        if self.return_timesteps:
            return image_stack, label, mask, feature.fid, timesteps
        else:
            return image_stack, label, mask, feature.fid

    @staticmethod
    def _setup(input_dir, label_dir, npyfolder, min_area_to_ignore=1000):
        """
        THIS FUNCTION PREPARES THE PLANET READER BY SPLITTING AND RASTERIZING EACH CROP FIELD AND SAVING INTO SEPERATE FILES FOR SPEED UP THE FURTHER USE OF DATA.
        :param input_dir: directory of input images in TIF format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param npyfolder: folder to save the field data for each field polygon
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :return: labels of the saved fields
        """
        labels = gpd.read_file(label_dir)
        labels["path"] = labels["fid"].apply(lambda fid: os.path.join(npyfolder, f"fid_{fid}.npz"))
        labels["exists"] = labels.path.apply(os.path.exists)
        if labels["exists"].all():
            return labels

        inputs = glob.glob(input_dir + "/*/*sr.tif", recursive=True)
        tifs = sorted(inputs)

        # read coordinate system of tifs and project labels to the same coordinate reference system (crs)
        with rio.open(tifs[0]) as image:
            crs = image.crs
            print("INFO: Coordinate system of the data is: {}".format(crs))
            transform = image.transform

        mask = labels.geometry.area > min_area_to_ignore
        print(
            f"INFO: Ignoring {(~mask).sum()}/{len(mask)} fields with area < {min_area_to_ignore}m2"
        )

        labels = labels.loc[mask]
        labels = labels.to_crs(crs)  # TODO: CHECK IF REQUIRED

        for index, feature in tqdm(
            labels.iterrows(), total=len(labels), position=0, leave=True
        ):  # , desc="INFO: Extracting time series into the folder: {}".format(npyfolder)):

            if os.path.exists(feature.path):
                continue

            left, bottom, right, top = feature.geometry.bounds
            window = rio.windows.from_bounds(
                left=left, bottom=bottom, right=right, top=top, transform=transform
            )

            # reads each tif in tifs on the bounds of the feature. shape T x D x H x W
            image_stack = np.stack([rio.open(tif).read(window=window) for tif in tifs])

            with rio.open(tifs[0]) as src:
                win_transform = src.window_transform(window)

            out_shape = image_stack[0, 0].shape
            assert (
                out_shape[0] > 0 and out_shape[1] > 0
            ), "WARNING: fid:{} image stack shape {} is zero in one dimension".format(
                feature.fid, image_stack.shape
            )

            # rasterize polygon to get positions of field within crop
            mask = features.rasterize(
                feature.geometry,
                all_touched=True,
                transform=win_transform,
                out_shape=image_stack[0, 0].shape,
            )

            # mask[mask != feature.fid] = 0
            # mask[mask == feature.fid] = 1
            os.makedirs(npyfolder, exist_ok=True)
            np.savez(
                feature.path,
                image_stack=image_stack,
                mask=mask,
                feature=feature.drop("geometry").to_dict(),
            )

        return labels


if __name__ == "__main__":
    """
    EXAMPLE USAGE OF DATA READER
    """

    zippath = "../data/dlr_fusion_competition_germany_train_source_planet_5day"

    labelgeojson = "../data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson"
    ds = PlanetReader(zippath, labelgeojson, selected_time_points=[2, 3, 4])
    X, y, m, fid = ds[0]
