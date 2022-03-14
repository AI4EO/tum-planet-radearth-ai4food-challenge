import pickle
import numpy as np
import pdb
import torch

from pathlib import Path

from torch.utils.data import Dataset
from src.utils.sentinel_1_reader import S1Reader
from src.utils.sentinel_2_reader import S2Reader


class S1S2Reader(Dataset):
    def __init__(
        self,
        s1_input_dir,
        s2_input_dir,
        label_dir,
        label_ids=None,
        s1_transform=None,
        s2_transform=None,
        min_area_to_ignore=1000,
        selected_time_points=None,
        include_cloud=False,
        alignment="1to2",
        filter=None,
        s1_temporal_dropout=0.0,
        s2_temporal_dropout=0.0,
    ):
        """
        THIS FUNCTION INITIALIZES DATA READER.
        :param input_dir: directory of input images in zip format
        :param label_dir: directory of ground-truth polygons in GeoJSON format
        :param label_ids: an array of crop IDs in order. if the crop labels in GeoJSON data is not started from index 0 it can be used. Otherwise it is not required.
        :param transform: data transformer function for the augmentation or data processing
        :param min_area_to_ignore: threshold m2 to eliminate small agricultural fields less than a certain threshold. By default, threshold is 1000 m2
        :param selected_time_points: If a sub set of the time series will be exploited, it can determine the index of those times in a given time series dataset
        :param alignment: [AtoB] Align sentinel_A's timeseries to sentinel_B's timeseries.

        :return: None
        """

        self.s1_reader = S1Reader(
            input_dir=s1_input_dir,
            label_dir=label_dir,
            label_ids=label_ids,
            transform=s1_transform,
            min_area_to_ignore=min_area_to_ignore,
            selected_time_points=selected_time_points,
            filter=filter,
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
            filter=filter,
            temporal_dropout=s2_temporal_dropout,
            return_timesteps=True,
        )

        if alignment == "1to2":
            self.timesteps = self.s2_reader.timesteps
            self.aligned_index = [
                self.nearest_ind(self.s1_reader.timesteps, d) for d in self.timesteps
            ]
        elif alignment == "2to1":
            self.timesteps = self.s1_reader.timesteps
            self.aligned_index = [
                self.nearest_ind(self.s2_reader.timesteps, d) for d in self.timesteps
            ]

        else:
            raise ValueError("Please specify the alignment correctly.")

        assert self.s1_reader.labels.drop("path", axis=1).equals(
            self.s2_reader.labels.drop("path", axis=1)
        )
        self.labels = self.s1_reader.labels
        self.alignment = alignment

        assert len(self.s1_reader.labels) == len(self.s2_reader.labels)
        
    @staticmethod
    def nearest_ind(items, pivot):
        time_diff = np.abs([date - pivot for date in items])
        return time_diff.argmin(0)

    def __len__(self):
        return len(self.s2_reader.labels)

    def __getitem__(self, idx):
        s1_image_stack, s1_label, s1_mask, s1_fid, s1_timesteps = self.s1_reader[idx]
        s2_image_stack, s2_label, s2_mask, s2_fid, s2_timesteps = self.s2_reader[idx]

        assert s1_fid == s2_fid
        assert s1_label == s2_label

        realign = (s1_timesteps != self.s1_reader.timesteps).any() or (s2_timesteps != self.s2_reader.timesteps).any()

        timesteps = self.timesteps
        if self.alignment == "1to2":
            if realign:
                timesteps = s2_timesteps
                aligned_index = [self.nearest_ind(s1_timesteps, d) for d in timesteps]
            else:
                aligned_index = self.aligned_index

            s1_s2_image_stack = torch.cat((s2_image_stack, s1_image_stack[aligned_index]), dim=1)
            s1_s2_mask = s2_mask
        elif self.alignment == "2to1":
            if realign:
                timesteps = s1_timesteps
                aligned_index = [self.nearest_ind(s2_timesteps, d) for d in timesteps]
            else:
                aligned_index = self.aligned_index
            s1_s2_image_stack = torch.cat((s2_image_stack[aligned_index], s1_image_stack), dim=1)
            s1_s2_mask = s1_mask

        return s1_s2_image_stack, s1_label, s1_s2_mask, s1_fid, timesteps
