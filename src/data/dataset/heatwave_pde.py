import copy
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from plyfile import PlyData
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.separate import separate


class HeatWavePDEonMesh(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        maxT=200,
        num_samples=10,
        input_length=5,
        output_length=3,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_transform_str="",
        processed_dir_str="processed",
    ):
        """
        :param str split: choose between 'train', 'test_time', 'test_init', 'test_mesh'
        """

        splits = ["train", "test_time", "test_init", "test_mesh"]
        assert split in splits

        self.split = split
        self.num_samples = num_samples
        self.maxT = maxT
        self.input_length = input_length
        self.output_length = output_length
        self.pre_transform_str = pre_transform_str
        self.processed_dir_str = processed_dir_str

        super().__init__(root, transform, pre_transform, pre_filter=None)
        path = self.processed_paths[splits.index(split)]
        # self.slices is None if using one object mesh
        self._data, self.slices = torch.load(path)
        self.num_meshes = self.slices["u"].size(0) - 1

    @property
    def mesh_names(self):
        return [
            "armadillo",  # test
            "bunny_coarse",
            "lucy",
            "sphere",
            "spider",
            "urn",  # test
            "woman",
        ]

    @property
    def raw_file_names(self) -> str:
        return [
            f"{name}/{name}_{i}"
            for name in self.mesh_names
            for i in range(self.num_samples)
        ]

    @property
    def processed_file_names(self):
        base_paths = ["train", "test_time", "test_init", "test_mesh"]
        return [
            f"{self.pre_transform_str + '_' if self.pre_transform_str else ''}{s}.pt"
            for s in base_paths
        ]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_str)

    def process(self):
        train_list = []
        test_time_list = []
        test_sample_list = []
        test_mesh_list = []

        test_time_idx = int(self.maxT * 0.75) + 1
        test_sample_idx = int(self.num_samples * 0.7)
        test_meshes = ["armadillo", "urn"]

        for f in self.raw_file_names:
            data = self._read_data(osp.join(self.raw_dir, f))

            # Label for which mesh and sample
            mesh_name = f.split("/")[0]
            mesh_idx = self.mesh_names.index(mesh_name)
            data.mesh_idx = torch.tensor([mesh_idx], dtype=torch.long)
            sample_idx = int(f.rsplit("_", 1)[1])
            data.sample_idx = torch.tensor([sample_idx], dtype=torch.long)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            if mesh_name in test_meshes:
                # Test mesh
                test_mesh_list.append(data)
            else:
                if sample_idx < test_sample_idx:
                    # Train data
                    train_data = copy.copy(data)
                    u = data.u[:, :test_time_idx]
                    train_data.u = u
                    train_list.append(train_data)

                    # Test time data
                    test_time_data = copy.copy(data)
                    u = data.u[:, test_time_idx:]
                    test_time_data.u = u
                    test_time_list.append(test_time_data)
                else:
                    # Test sample data
                    test_sample_list.append(data)

        torch.save(self.collate(train_list), self.processed_paths[0])
        torch.save(self.collate(test_time_list), self.processed_paths[1])
        torch.save(self.collate(test_sample_list), self.processed_paths[2])
        torch.save(self.collate(test_mesh_list), self.processed_paths[3])

    def _read_data(self, path):
        # Load mesh
        prefix_path = path.rsplit("_", 1)[0]
        plydata = PlyData.read(f"{prefix_path}.ply")
        face = np.vstack(plydata["face"].data["vertex_indices"]).T.astype(np.float32)
        face = torch.from_numpy(face)
        pos = np.vstack([[v[0], v[1], v[2]] for v in plydata["vertex"]]).astype(
            np.float32
        )
        pos = torch.from_numpy(pos)

        # Load values
        values = np.load(f"{path}.npy").astype(np.float32)
        u = torch.from_numpy(values).permute(1, 0).contiguous()

        return Data(pos=pos, face=face, u=u)

    def len(self) -> int:
        # Num meshes * num_samples * num_time_windows
        n = self._data.mesh_idx.shape[0] * (
            self._data.u.shape[-1] - self.input_length - self.output_length + 1
        )
        return n

    def get(self, idx: int) -> Data:
        if self.len() == 1:
            data = copy.copy(self._data)
            return data

        if not hasattr(self, "_data_list") or self._data_list is None:
            self._data_list = self.len() * [None]
        elif self._data_list[idx] is not None:
            return copy.copy(self._data_list[idx])

        t_per_sample = (
            self._data.u.shape[-1] - self.output_length - self.input_length + 1
        )

        mesh_sample_idx = idx // t_per_sample
        rem = idx % t_per_sample
        current_t_idx = rem + self.input_length

        data = separate(
            cls=self._data.__class__,
            batch=self._data,
            idx=mesh_sample_idx,
            slice_dict=self.slices,
            decrement=False,
        )

        assert (data.edge_index < data.pos.shape[0]).all().item()

        # Input feature
        x = data.u[..., current_t_idx - self.input_length : current_t_idx][:, :, None]
        # Target
        data.y = data.u[..., current_t_idx : current_t_idx + self.output_length]

        data.x = x

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
        traj.x = traj.u

        self._traj_list[idx] = copy.copy(traj)

        return traj
