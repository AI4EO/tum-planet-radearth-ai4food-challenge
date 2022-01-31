import json
import pickle
import numpy as np
import pdb
import torch

from datetime import datetime
from pathlib import Path

from torch.utils.data import Dataset
from src.utils.sentinel_1_reader import S1Reader
from src.utils.sentinel_2_reader import S2Reader
from src.utils.planet_reader import PlanetReader


class S1S2PlanetReader(Dataset):
    def __init__(
        self,
        s1_input_dir,
        s2_input_dir,
        planet_input_dir,
        label_dir,
        label_ids=None,
        s1_transform=None,
        s2_transform=None,
        planet_transform=None,
        min_area_to_ignore=1000,
        selected_time_points=None,
        include_cloud=False,
        s1_temporal_dropout=0.0,
        s2_temporal_dropout=0.0,
        planet_temporal_dropout=0.0,
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
        :param selected_time_points: If a sub set of the time series will be exploited, it can
            determine the index of those times in a given time series dataset

        :return: None
        """

        print(
            "WARNING: If using the pixelsetencoder or random pixel spatial encoding the pixels from"
            " each image (S1, S2, planet daily) are not aligned. "
            "This is deliberate to avoid interpolation and code changes."
        )

        self.s1_reader = S1Reader(
            input_dir=s1_input_dir,
            label_dir=label_dir,
            label_ids=label_ids,
            transform=s1_transform,
            min_area_to_ignore=min_area_to_ignore,
            selected_time_points=selected_time_points,
            temporal_dropout=s1_temporal_dropout,
            return_timesteps=True,
        )

        self.s2_reader = S2Reader(
            input_dir=s2_input_dir,
            label_dir=label_dir,
            label_ids=label_ids,
            transform=s2_transform,
            min_area_to_ignore=min_area_to_ignore,
            selected_time_points=selected_time_points,
            include_cloud=include_cloud,
            temporal_dropout=s2_temporal_dropout,
            return_timesteps=True,
        )

        self.planet_reader = PlanetReader(
            input_dir=planet_input_dir,
            label_dir=label_dir,
            label_ids=label_ids,
            transform=planet_transform,
            min_area_to_ignore=min_area_to_ignore,
            selected_time_points=selected_time_points,
            tzinfo=self.s1_reader.timesteps[0].tzinfo,
            temporal_dropout=planet_temporal_dropout,
            return_timesteps=True,
        )

        self.timesteps = self.planet_reader.timesteps

        assert self.s1_reader.labels.drop("path", axis=1).equals(
            self.s2_reader.labels.drop("path", axis=1)
        )
        assert self.s1_reader.labels.drop("path", axis=1).equals(
            self.planet_reader.labels.drop("path", axis=1)
        )
        self.labels = self.s1_reader.labels.drop("path", axis=1)

    @staticmethod
    def nearest_ind(items, pivot):
        time_diff = np.abs([date - pivot for date in items])
        return time_diff.argmin(0)

    def __len__(self):
        return len(self.s1_reader.labels)

    def __getitem__(self, idx):

        s1_image_stack, s1_label, _, s1_fid, s1_timesteps = self.s1_reader[idx]
        s2_image_stack, s2_label, _, s2_fid, s2_timesteps = self.s2_reader[idx]
        (
            planet_image_stack,
            planet_label,
            planet_mask,
            planet_fid,
            planet_timesteps,
        ) = self.planet_reader[idx]

        assert s1_fid == s2_fid == planet_fid
        assert s1_label == s2_label == planet_label

        # If returned timesteps do not match the timesteps of the S1, S2 and Planet data, that means
        # temporal dropout is being used and the data needs to be realigned
        s1_aligned_index = [self.nearest_ind(s1_timesteps, d) for d in planet_timesteps]
        s2_aligned_index = [self.nearest_ind(s2_timesteps, d) for d in planet_timesteps]

        s1_aligned = s1_image_stack[s1_aligned_index]
        s2_aligned = s2_image_stack[s2_aligned_index]
        s1_s2_planet_image_stack = torch.cat((planet_image_stack, s2_aligned, s1_aligned), dim=1)

        # pad with zeros to make the image stack of the same size
        target = torch.zeros(len(self.timesteps), *s1_s2_planet_image_stack.shape[1:])
        target[: len(planet_timesteps)] = s1_s2_planet_image_stack

        return target, planet_label, planet_mask, planet_fid
