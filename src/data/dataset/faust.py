import copy
import os.path as osp
import shutil
from typing import Callable, Optional

import torch
from torch_geometric import datasets
from torch_geometric.data import extract_zip
from torch_geometric.io import read_ply


class FAUST(datasets.FAUST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_transform_str="",
        processed_dir_str="processed",
        pre_filter: Optional[Callable] = None,
    ):

        self.pre_transform_str = pre_transform_str
        self.processed_dir_str = processed_dir_str

        super().__init__(
            root, train, transform, pre_transform=pre_transform, pre_filter=pre_filter
        )

    @property
    def processed_file_names(self):
        base_paths = ["test", "train"]
        return [
            f"{self.pre_transform_str + '_' if self.pre_transform_str else ''}{s}.pt"
            for s in base_paths
        ]

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.processed_dir_str)

    def process(self):
        """
        Same as torch_geometric.data.datasets.FAUST except output labels are added for node segmentation
        """
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)

        path = osp.join(self.raw_dir, "MPI-FAUST", "training", "registrations")
        path = osp.join(path, "tr_reg_{0:03d}.ply")
        data_list = []
        for i in range(100):
            data = read_ply(path.format(i))
            # Label for which person
            data.person_idx = torch.tensor([i % 10], dtype=torch.long)

            # Input features (as trivial rep)
            data.x = copy.copy(data.pos)[:, :, None]
            # Labels for node segmentation
            data.y = torch.arange(data.num_nodes, dtype=torch.long)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list[:80]), self.processed_paths[0])
        torch.save(self.collate(data_list[80:]), self.processed_paths[1])

        shutil.rmtree(osp.join(self.raw_dir, "MPI-FAUST"))
