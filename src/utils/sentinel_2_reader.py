"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines a data reader for Sentinel-2 eath observation data
"""

import os
import pdb
from torch.utils.data import Dataset
import zipfile
import pickle
import geopandas as gpd
import numpy as np
import math
import rasterio as rio
from pathlib import Path
from rasterio import features
from tqdm import tqdm


class S2Reader(Dataset):
    """
    THIS CLASS INITIALIZES THE DATA READER FOR SENTINEL-2 DATA
    """

    def __init__(
        self,
        input_dir,
        label_dir,
        label_ids=None,
        transform=None,
        min_area_to_ignore=1000,
        selected_time_points=None,
        include_cloud=False,
        filter=None,
        temporal_dropout=0.0,
        return_timesteps=False,
    ):
        """
        THIS FUNCTION INITIALIZES DATA READER.
        :param input_dir: directory of input images in zip format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param label_ids: an array of crop IDs in order. if the crop labels in GeoJSON data is not
            started from index 0 it can be used. Otherwise it is not required.
        :param transform: data transformer function for the augmentation or data processing
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a
            certain threshold. By default, threshold is 1000 m2
        :param selected_time_points: If a sub set of the time series will be exploited,
            it can determine the index of those times in a given time series dataset

        :return: None
        """
        self.data_transform = transform.transform
        self.selected_time_points = selected_time_points
        self.crop_ids = label_ids
        if label_ids is not None and not isinstance(label_ids, list):
            self.crop_ids = label_ids.tolist()

        self.npyfolder = input_dir.replace(".zip", "/time_series")
        self.labels = S2Reader._setup(
            input_dir,
            label_dir,
            self.npyfolder,
            min_area_to_ignore,
            include_cloud=include_cloud,
            filter=filter,
        )

        with (Path(input_dir) / "timestamp.pkl").open("rb") as f:
            self.timesteps = np.array(pickle.load(f))

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
            assert (
                np.argwhere(np.isnan(image_stack)).size == 0
                and np.argwhere(np.isinf(image_stack)).size == 0
            )
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
    def _setup(
        rootpath, labelgeojson, npyfolder, min_area_to_ignore=1000, include_cloud=False, filter=None
    ):
        """
        THIS FUNCTION PREPARES THE PLANET READER BY SPLITTING AND RASTERIZING EACH CROP FIELD AND SAVING INTO SEPERATE FILES FOR SPEED UP THE FURTHER USE OF DATA.

        This utility function unzipps a dataset and performs a field-wise aggregation.
        results are written to a .npz cache with same name as zippath

        :param rootpath: directory of input images in ZIP format
        :param labelgeojson: directory of ground-truth polygons in GeoJSON format
        :param npyfolder: folder to save the field data for each field polygon
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :param include_cloud: It includes cloud probabilities inti image_stack if TRUE, othervise it saves the cloud info as sepeate array
        :return: labels of the saved fields
        """

        with open(os.path.join(rootpath, "bbox.pkl"), "rb") as f:
            bbox = pickle.load(f)
            crs = str(bbox.crs)
            minx, miny, maxx, maxy = bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y

        labels = gpd.read_file(labelgeojson)
        if filter is not None:
            labels = labels[~labels.fid.isin(filter)]
        # project to same coordinate reference system (crs) as the imagery
        ignore = labels.geometry.area > min_area_to_ignore
        print(
            f"INFO: Ignoring {(~ignore).sum()}/{len(ignore)} fields with area < {min_area_to_ignore}m2"
        )
        labels = labels.loc[ignore]
        labels = labels.to_crs(crs)  # TODO: CHECK IF NECESSARY
        labels["path"] = labels["fid"].apply(lambda fid: os.path.join(npyfolder, f"fid_{fid}.npz"))
        labels["exists"] = labels.path.apply(os.path.exists)
        if labels["exists"].all():
            return labels

        bands = np.load(os.path.join(rootpath, "bands.npy"))
        clp = np.load(os.path.join(rootpath, "clp.npy"))  # CLOUD PROBABILITY

        if include_cloud:
            bands = np.concatenate([bands, clp], axis=-1)  # concat cloud probability
        _, width, height, _ = bands.shape

        bands = bands.transpose(0, 3, 1, 2)
        clp = clp.transpose(0, 3, 1, 2)

        transform = rio.transform.from_bounds(
            west=minx, south=miny, east=maxx, north=maxy, width=width, height=height
        )

        fid_mask = features.rasterize(
            zip(labels.geometry, labels.fid),
            all_touched=True,
            transform=transform,
            out_shape=(width, height),
        )
        ids_in_mask = np.unique(fid_mask)
        assert len(ids_in_mask) > 0, (
            f"WARNING: Vectorized fid mask contains no fields. "
            f"Does the label geojson {labelgeojson} cover the region defined by {rootpath}?"
        )
        ids_in_labels = np.unique(labels.fid)
        ids_missing_in_mask = np.setdiff1d(ids_in_labels, ids_in_mask)
        if len(ids_missing_in_mask) > 0:
            print(
                f"WARNING: {len(ids_missing_in_mask)}/{len(ids_in_labels)} fields are missing from the fid mask"
            )
            labels = labels.loc[labels.fid.isin(ids_in_mask)]

        crop_mask = features.rasterize(
            zip(labels.geometry, labels.crop_id),
            all_touched=True,
            transform=transform,
            out_shape=(width, height),
        )
        assert len(np.unique(crop_mask)) > 0, (
            f"WARNING: Vectorized fid mask contains no fields. "
            f"Does the label geojson {labelgeojson} cover the region defined by {rootpath}?"
        )

        for index, feature in tqdm(
            labels.iterrows(), total=len(labels), position=0, leave=True
        ):  # , desc="INFO: Extracting time series into the folder: {}".format(npyfolder)):
            npyfile = os.path.join(npyfolder, "fid_{}.npz".format(feature.fid))
            if os.path.exists(npyfile):
                continue

            left, bottom, right, top = feature.geometry.bounds
            window = rio.windows.from_bounds(left, bottom, right, top, transform)

            row_start = round(window.row_off) if window.row_off > 0 else 0
            row_end = math.ceil(window.row_off + window.height)
            col_start = round(window.col_off) if window.col_off > 0 else 0
            col_end = math.ceil(window.col_off + window.width)

            image_stack = bands[:, :, row_start:row_end, col_start:col_end]
            cloud_stack = clp[:, :, row_start:row_end, col_start:col_end]
            mask = fid_mask[row_start:row_end, col_start:col_end].copy()
            mask[mask != feature.fid] = 0
            mask[mask == feature.fid] = 1

            if (mask == 0).all():
                pdb.set_trace()

            os.makedirs(npyfolder, exist_ok=True)
            np.savez(
                npyfile,
                image_stack=image_stack.astype(np.float32),
                cloud_stack=cloud_stack.astype(np.float32),
                mask=mask.astype(np.float32),
                feature=feature.drop("geometry").to_dict(),
            )

        return labels


if __name__ == "__main__":
    """
    EXAMPLE USAGE OF DATA READER
    """

    zippath = "../data/dlr_fusion_competition_germany_train_source_sentinel_2.tar.gz"

    labelgeojson = "../data/dlr_fusion_competition_germany_train_labels/dlr_fusion_competition_germany_train_labels_33N_18E_242N/labels.geojson"
    ds = S2Reader(zippath, labelgeojson)
    X, y, m, fid = ds[0]
