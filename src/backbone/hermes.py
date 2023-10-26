import torch
from torch_geometric.utils import remove_isolated_nodes

from src.nn.hermes_conv import HermesLayer
from src.transform.gem_precomp import GemPrecomp


class Hermes(torch.nn.Module):
    def __init__(
        self,
        message_dims,
        message_orders,
        update_dims,
        update_orders,
        edge_dims,
        reltan_features,
        null_isolated,
        n_rings,
        band_limit,
        num_samples,
        checkpoint,
        node_batch_size,
        equiv_bias,
        message_norm,
        update_norm,
        message_dropout,
        update_dropout,
        residual,
        final_activation,
        **kwargs,
    ):
        super().__init__()

        if not reltan_features:
            assert kwargs == {}, "kwargs not empty but reltan_features=False"

        assert len(message_dims) == len(
            message_orders
        ), "message_dims and message_orders should have same number of blocks"
        for i in range(len(message_dims)):
            assert len(message_orders[i]) == len(
                message_orders[i]
            ), "message_dims and message_orders should have same number of layers"

        assert len(update_dims) == len(
            update_orders
        ), "update_dims and update_orders should have same number of blocks"
        for i in range(len(update_dims)):
            assert len(update_orders[i]) == len(
                update_orders[i]
            ), "update_dims and update_orders should have same number of layers"

        assert len(message_dims) == len(
            update_dims
        ), "message_dims and update_dims should have same number of blocks"

        self.message_dims = message_dims
        self.message_orders = message_orders
        self.update_dims = update_dims
        self.update_orders = update_orders
        self.edge_dims = edge_dims

        self.out_dim = self.update_dims[-1][-1]

        self.reltan_features = reltan_features
        self.null_isolated = null_isolated

        # TODO checkpoint not implemented yet
        block_kwargs = dict(
            edge_dims=edge_dims,
            n_rings=n_rings,
            band_limit=band_limit,
            num_samples=num_samples,
            checkpoint=checkpoint,
            node_batch_size=node_batch_size,
            equiv_bias=equiv_bias,
            message_norm=message_norm,
            update_norm=update_norm,
            message_dropout=message_dropout,
            update_dropout=update_dropout,
            residual=residual,
        )

        self.transforms = [GemPrecomp(n_rings, band_limit)]

        # Construct blocks
        self.blocks = torch.nn.ModuleList()
        for i in range(len(message_dims) - 1):
            self.blocks.append(
                HermesLayer(
                    self.message_dims[i],
                    self.message_orders[i],
                    self.update_dims[i],
                    self.update_orders[i],
                    final_activation=True,
                    **block_kwargs,
                )
            )
        # Add final block
        self.blocks.append(
            HermesLayer(
                self.message_dims[-1],
                self.message_orders[-1],
                self.update_dims[-1],
                self.update_orders[-1],
                final_activation=final_activation,
                **block_kwargs,
            )
        )

    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        # rel_transform adds rel_tang_feat (check Sec. 4 in the draft) feature to data
        for transform in self.transforms:
            data = transform(data)

        edge_index, connection, precomp_neigh_edge, precomp_self_node = (
            data.edge_index,
            data.connection,
            data.precomp_neigh_edge,
            data.precomp_self_node,
        )

        # Input node features
        assert data.x.dim() == 3
        x = data.x

        # Edge features
        if data.edge_attr is not None:
            edge_attr = data.edge_attr[..., None]
        else:
            edge_attr = None

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(data.edge_index)[-1]
            x[~non_isol_mask] = 0.0

        for block in self.blocks:
            x = block(
                x,
                edge_index,
                connection,
                precomp_neigh_edge,
                precomp_self_node,
                edge_attr,
            )

        return x
