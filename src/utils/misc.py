import torch


def to_one_hot(indices, max_index):
    one_hot = torch.zeros(
        *indices.size() + (max_index,), dtype=torch.float32, device=indices.device
    )

    return one_hot.scatter_(-1, indices.unsqueeze(-1), 1)
