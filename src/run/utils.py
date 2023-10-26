import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from hydra.utils import instantiate
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from torch_geometric.loader import DataLoader

from src.transform.edge_features import empty_edge_attr
from src.transform.simple_geometry import SimpleGeometry
from src.transform.vector_normals import compute_vertex_normals


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def prepare_batch_fn(key="y"):
    def prepare_batch(batch, device, non_blocking=False):
        data = batch.to(device)
        return data, data[key].to(device)

    return prepare_batch


def create_dataset_loaders(cfg, return_datasets=False):
    print("Creating datasets")

    if cfg.dataset.name == "FAUST":
        pre_tf = T.Compose([compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test", "test_gauge"]
    elif any(cfg.dataset.name.startswith(s) for s in ["Heat", "Wave", "Cahn-Hilliard"]):
        pre_tf = T.Compose([compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test_time", "test_init", "test_mesh"]
    elif cfg.dataset.name.startswith("Other"):
        pre_tf = T.Compose([compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test_time", "test_init"]
    elif cfg.dataset.name.startswith("Objects"):
        pre_tf = T.Compose([compute_vertex_normals, empty_edge_attr, SimpleGeometry()])
        splits = ["train", "test"]
    else:
        raise NotImplementedError(f"Incorrect cfg.dataset.name {cfg.dataset.name}")

    out_dict = {}
    for split in splits:
        train = split == "train"

        if any(cfg.dataset.name.startswith(prefix) for prefix in ["FAUST"]):
            dataset = instantiate(cfg.dataset.cls, train=train, pre_transform=pre_tf)
        else:
            dataset = instantiate(cfg.dataset.cls, split=split, pre_transform=pre_tf)

        if any(
            cfg.dataset.name.startswith(s) for s in ["Heat", "Wave", "Cahn-Hilliard"]
        ):
            print(
                f"[{split}] Len: {len(dataset)}, Num nodes: {dataset._data.num_nodes}"
            )
        else:
            print(f"[{split}] Len: {len(dataset)}, Num nodes: {dataset[0].num_nodes}")

        if return_datasets:
            out_dict[split] = dataset

        else:
            out_dict[split] = DataLoader(
                dataset,
                batch_size=cfg.train.batch_size,
                shuffle=train,
                pin_memory=True,
            )

    return out_dict


class GaugeInvarianceNLLLoss(Metric):
    """
    Custom metric to compute difference in NLLLoss between original dataset and random gauge-transformed dataset
    """

    @reinit__is_reduced
    def reset(self):
        self._gauge_error = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_orig, y_t, y = output[0].detach(), output[1].detach(), output[2].detach()

        loss_orig = F.nll_loss(y_orig, y, reduction="none")
        loss_t = F.nll_loss(y_t, y, reduction="none")

        self._gauge_error += torch.abs(loss_orig - loss_t).sum()

        self._num_examples += loss_orig.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct:SUM")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "GaugeInvarianceNLLLoss must have at least one example before it can be computed."
            )
        return self._gauge_error.item() / self._num_examples
