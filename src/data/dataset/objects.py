import copy
import os.path as osp
import pickle
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate

from src.utils.misc import to_one_hot


class ObjectsSplitHeadsDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_transform_str="",
        processed_dir_str="processed",
        num_train=100,
        num_test=100,
    ):
        """
        :param str split: choose between 'train', 'test'
        """

        splits = ["train", "test"]
        assert split in splits

        self.split = split

        self.pre_transform_str = pre_transform_str
        self.processed_dir_str = processed_dir_str

        self.num_train = num_train
        self.num_test = num_test

        super().__init__(root, transform, pre_transform, pre_filter=None)
        path = self.processed_paths[splits.index(split)]
        # self.slices is None if using one object mesh
        self._data, self.slices = torch.load(path)

    @property
    def raw_file_names(self) -> str:
        num_episodes = {
            "train": self.num_train,
            "test": self.num_test,
        }

        return [
            f"{split}/episode{i+1}.h5"
            for split, n_ep in num_episodes.items()
            for i in range(n_ep)
        ]

    @property
    def processed_file_names(self):
        base_paths = ["train", "test"]
        return [
            f"{self.pre_transform_str + '_' if self.pre_transform_str else ''}{s}.pt"
            for s in base_paths
        ]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_str)

    def process(self):
        train_list = []
        test_list = []

        for i, f in enumerate(self.raw_file_names):
            cur_split = f.split("/")[0]

            data = self._read_pkl(osp.join(self.raw_dir, f))

            # Label for which episode/mesh
            data.mesh_idx = torch.tensor([i], dtype=torch.long)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if cur_split == "train":
                train_list.append(data)
            elif cur_split == "test":
                test_list.append(data)

        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(test_list), self.processed_paths[1])

    def _read_pkl(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Normalize orientations
        data["orientation"] = data["orientation"] / (
            np.linalg.norm(data["orientation"], axis=-1, keepdims=True) + 1e-16
        )

        return Data(
            pos=torch.from_numpy(data["pos"].squeeze()).float(),
            face=torch.from_numpy(data["faces"].squeeze()).long(),
            occupancy=torch.from_numpy(data["occupancy"])
            .permute(1, 0)
            .long(),  # [Num_nodes, T]
            orientation=torch.from_numpy(data["orientation"])
            .permute(1, 0, 2)
            .float(),  # [Num_nodes, T, 3]
            action=torch.from_numpy(data["action"]).long(),
        )

    def len(self) -> int:
        return self._data.action.shape[0]

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            data = copy.copy(self._data)
            return data

        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        t_per_sample = self._data.occupancy.shape[1] - 1
        sample_idx = idx // t_per_sample
        current_t_idx = idx % t_per_sample

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=sample_idx,
            slice_dict=self.slices,
            decrement=False,
        )

        # Get slice of node features
        # Change occupancy and action to one hot
        occupancy = copy.copy(data.occupancy)
        orientation = copy.copy(data.orientation)
        num_objects = occupancy.max()
        data.occupancy = to_one_hot(occupancy[:, current_t_idx], num_objects + 1)
        data.orientation = orientation[:, current_t_idx]
        data.action = to_one_hot(data.action[current_t_idx], num_objects * 3)

        # Concatenate pos, occupancy, action, and orientation to use as input
        rho0_idx = data.pos.shape[1] + data.occupancy.shape[1] + data.action.shape[0]
        x = torch.zeros(
            data.pos.shape[0],
            rho0_idx + 1,
            3,
        ).to(data.pos.device)
        # rho0
        x[:, :-1, 0] = torch.cat(
            [data.pos, data.occupancy, data.action.repeat(data.pos.shape[0], 1)], dim=1
        )
        # rho1
        x[:, -1, 1:] = data.orientation

        data.x = x

        # Concatenate occupancy and orientation as label
        data.next_occupancy = occupancy[:, current_t_idx + 1]
        data.next_orientation = orientation[:, current_t_idx + 1, :]

        y = torch.zeros(data.pos.shape[0], 2, 3).to(data.pos.device)
        # rho0
        y[:, 0, 0] = occupancy[:, current_t_idx + 1]
        # rho1
        y[:, 1, 1:] = orientation[:, current_t_idx + 1, :]

        data.y = y

        self._data_list[idx] = copy.copy(data)

        return data

    def num_trajectories(self):
        return self._data.mesh_idx.shape[0]

    def get_trajectory(self, idx):
        n_trajs = self.num_trajectories()

        if not hasattr(self, "_traj_list") or self._traj_list is None:
            self._traj_list = n_trajs * [None]
        elif self._traj_list[idx] is not None:
            return copy.copy(self._traj_list[idx])

        assert idx < n_trajs, "incorrect idx for trajectory"

        traj = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False,
        )

        num_objects = traj.occupancy.max()
        traj.occupancy = to_one_hot(traj.occupancy, num_objects + 1)
        traj.action = to_one_hot(traj.action, num_objects * 3)

        traj.state = torch.cat(
            [
                traj.pos.unsqueeze(1).repeat(1, traj.occupancy.shape[1], 1),
                traj.occupancy,
                traj.orientation,
            ],
            dim=-1,
        )

        self._traj_list[idx] = copy.copy(traj)

        return traj
