import torch
from torch_geometric.utils import remove_isolated_nodes

from src.nn.eman_res_net_block import EmanAttResNetBlock
from src.transform.gem_precomp import GemPrecomp


class EMAN(torch.nn.Module):
    def __init__(
        self,
        block_dims,
        block_orders,
        reltan_features,
        null_isolated,
        n_rings,
        band_limit,
        num_samples,
        checkpoint,
        node_batch_size,
        equiv_bias,
        regular_non_lin,
        batch_norm,
        dropout,
        n_heads,
        final_activation,
        **kwargs
    ):
        super().__init__()

        if not reltan_features:
            assert kwargs == {}, "kwargs not empty but reltan_features=False"

        assert len(block_dims) >= 3, "minimum length of block_dims must be >= 3"
        assert len(block_orders) >= 3, "minimum length of block_orders must be >= 3"
        assert len(block_dims) == len(
            block_orders
        ), "length of block_dims and block_orders must be equal"
        self.block_dims = block_dims
        self.block_orders = block_orders
        self.out_dim = self.block_dims[-1]

        self.reltan_features = reltan_features
        self.null_isolated = null_isolated

        block_kwargs = dict(
            n_rings=n_rings,
            band_limit=band_limit,
            num_samples=num_samples,
            checkpoint=checkpoint,
            node_batch_size=node_batch_size,
            equiv_bias=equiv_bias,
            regular_non_lin=regular_non_lin,
            batch_norm=batch_norm,
            dropout=dropout,
            n_heads=n_heads,
        )

        self.transforms = [GemPrecomp(n_rings, band_limit)]

        self.layers = torch.nn.ModuleList()
        for i in range(len(self.block_dims) - 3):
            self.layers.append(
                EmanAttResNetBlock(
                    self.block_dims[i],
                    self.block_dims[i + 1],
                    self.block_dims[i + 2],
                    self.block_orders[i],
                    self.block_orders[i + 1],
                    self.block_orders[i + 2],
                    final_activation=True,
                    **block_kwargs,
                )
            )
        # Add final block
        self.layers.append(
            EmanAttResNetBlock(
                self.block_dims[-3],
                self.block_dims[-2],
                self.block_dims[-1],
                self.block_orders[-3],
                self.block_orders[-2],
                self.block_orders[-1],
                final_activation=final_activation,
                **block_kwargs,
            )
        )

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        # rel_transform adds rel_tang_feat (check Sec. 4 in the draft) feature to data
        for transform in self.transforms:
            data = transform(data)

        edge_index, precomp_neigh_edge, precomp_self_edge, connection = (
            data.edge_index,
            data.precomp_neigh_edge,
            data.precomp_self_edge,
            data.connection,
        )

        # Input node features
        assert data.x.dim() == 3
        x = data.x

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(edge_index)[-1]
            x[~non_isol_mask] = 0.0

        for layer in self.layers:
            x = layer(x, edge_index, precomp_neigh_edge, precomp_self_edge, connection)

        return x
