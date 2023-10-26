# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
import torch
import trimesh

# Modified by Hermes authors

# Gets different result than torch_geometric.transforms.GenerateMeshNormals
def compute_vertex_normals(data):
    mesh = trimesh.Trimesh(
        vertices=data.pos.cpu().numpy(),
        faces=data.face.cpu().numpy().T,
        process=False,
        validate=False,
    )
    data.normal = torch.tensor(
        mesh.vertex_normals.copy(), dtype=data.pos.dtype, device=data.pos.device
    )

    assert (mesh.edges.T < data.pos.shape[0]).all().item()

    data.edge_index = torch.tensor(
        mesh.edges.T.copy(), dtype=torch.long, device=data.pos.device
    )

    return data
