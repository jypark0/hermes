import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, InstanceNorm1d, Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot

from src.nn.regular_nonlin import RegularNonlinearity
from src.utils.einsum import einsum
from src.utils.kernel import build_kernel, build_self_kernel
from src.utils.rep_act import rep_act


class HermesLayer(MessagePassing):
    """
    Gauge equivariant message passing layer

    Args:
        in_channels (int): number of input features
        hid_channels (int):
        out_channels (int):
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        band_limit (int, optional): maximum theta frequency used
        node_batch_size (int, optional): compute edges in batches, checkpointed to save memory
    """

    def __init__(
        self,
        message_dims,
        message_orders,
        update_dims,
        update_orders,
        edge_dims,
        n_rings,
        band_limit,
        num_samples,
        checkpoint=False,
        node_batch_size=None,
        equiv_bias=False,
        message_norm=None,
        update_norm=None,
        message_dropout=False,
        update_dropout=False,
        final_activation=True,
        residual=True,
    ):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)

        # assert len(message_dims) >= 2, "minimum length of message_dims must be >= 2"
        # assert len(message_orders) >= 2, "minimum length of message_orders must be >= 2"
        # assert len(update_dims) >= 2, "minimum length of update_dims must be >= 2"
        # assert len(update_orders) >= 2, "minimum length of update_orders must be >= 2"

        assert len(message_dims) == len(message_orders) and len(update_dims) == len(
            update_orders
        ), "length of message_dims!=message_orders or length of update_dims!=update_orders"

        self.edge_dims = edge_dims

        self.num_samples = num_samples
        self.message_norm = message_norm
        self.update_norm = update_norm
        self.message_dropout = message_dropout
        self.update_dropout = update_dropout
        self.final_activation = final_activation
        self.residual = residual

        layer_kwargs = dict(
            n_rings=n_rings,
            band_limit=band_limit,
            node_batch_size=node_batch_size,
            equiv_bias=equiv_bias,
        )

        # Construct message layers
        self.message_layers = torch.nn.ModuleList()
        for i in range(len(message_dims) - 1):
            if i == 0:
                # Source + destination + edge features
                in_dim = message_dims[0] + message_dims[0] + self.edge_dims
                # in_dim = message_dims[0] + message_dims[0]
            else:
                in_dim = message_dims[i]

            self.message_layers.append(
                HermesMessageLayer(
                    in_dim,
                    message_dims[i + 1],
                    message_orders[i],
                    message_orders[i + 1],
                    kernel_constraint="neigh",
                    **layer_kwargs,
                )
            )

            message_act = []

            if (
                i == len(message_dims) - 2
                and len(update_dims) == 1
                and not final_activation
            ):
                continue
            if self.message_norm == "batch":
                message_act.append(BatchNorm1d(message_dims[i + 1]))
            elif self.message_norm == "instance":
                message_act.append(InstanceNorm1d(message_dims[i + 1]))
            if self.message_dropout:
                message_act.append(nn.Dropout())

            message_act.append(nn.ReLU())

            self.message_layers.append(
                RegularNonlinearity(
                    message_orders[i + 1], num_samples, nn.Sequential(*message_act)
                )
            )

        # Construct update layers
        self.update_layers = torch.nn.ModuleList()
        for i in range(len(update_dims) - 1):
            if i == 0 and self.residual:
                # message dims + residual (source node features)
                in_dim = update_dims[0] + message_dims[0]
                # in_dim = update_dims[0]
            else:
                in_dim = update_dims[i]

            self.update_layers.append(
                HermesMessageLayer(
                    in_dim,
                    update_dims[i + 1],
                    update_orders[i],
                    update_orders[i + 1],
                    kernel_constraint="self",
                    **layer_kwargs,
                )
            )

            update_act = []
            if i < len(update_dims) - 2 or (
                i == len(update_dims) - 2 and final_activation
            ):
                if self.update_norm == "batch":
                    update_act.append(BatchNorm1d(update_dims[i + 1]))
                elif self.update_norm == "instance":
                    update_act.append(InstanceNorm1d(update_dims[i + 1]))

                if self.update_dropout:
                    update_act.append(nn.Dropout())

                self.update_layers.append(
                    RegularNonlinearity(
                        update_orders[i + 1], num_samples, nn.Sequential(*update_act)
                    )
                )

        if message_dims[0] != update_dims[-1]:
            self.lin = nn.Conv1d(
                message_dims[0], update_dims[-1], kernel_size=1, bias=False
            )  # Equivariant linear map mixing channels
        else:
            self.lin = nn.Identity()

    def add_residual(self, y, x):
        ### self.lin only used to make in_channels (residual) == out_channels (y)
        residual = self.lin(x)
        o = min(y.shape[2], residual.shape[2])
        y[:, :, :o] = y[:, :, :o] + residual[:, :, :o]  # Handle varying orders
        return y

    def forward(
        self, x, edge_index, connection, precomp_neigh, precomp_self, edge_attr=None
    ):
        # Propagate messages along edges
        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            connection=connection,
            precomp_neigh=precomp_neigh,
            precomp_self=precomp_self,
        )

        return x

    def message(self, x_i, x_j, precomp_neigh, connection, edge_attr=None):
        x_j_transported = rep_act(x_j, connection)

        if edge_attr is not None:
            if edge_attr.shape[-1] != x_i.shape[-1]:
                # edge_attr is trivial rep
                # pad edge_attr to same order as x_i with zeros
                right_pad = x_i.shape[-1] - edge_attr.shape[-1]
                edge_attr = F.pad(
                    edge_attr, pad=(0, right_pad), mode="constant", value=0
                )

            message = torch.cat([x_i, x_j_transported, edge_attr], dim=1)
        else:
            message = torch.cat([x_i, x_j_transported], dim=1)

        for layer in self.message_layers:
            if isinstance(layer, HermesMessageLayer):
                message = layer(message, precomp_neigh)
            elif isinstance(layer, RegularNonlinearity):
                message = layer(message)
            else:
                raise NotImplementedError

        return message

    def update(self, message, x, precomp_self):
        if x.shape[-1] != message.shape[-1]:
            # pad x to same order as message with zeros
            right_pad = message.shape[-1] - x.shape[-1]
            x = F.pad(x, pad=(0, right_pad), mode="constant", value=0)

        if len(self.update_layers) >= 1 and self.residual:
            update = torch.cat([x, message], dim=1)
        else:
            update = message

        for layer in self.update_layers:
            if isinstance(layer, HermesMessageLayer):
                update = layer(update, precomp_self)
            elif isinstance(layer, RegularNonlinearity):
                update = layer(update)
            else:
                raise NotImplementedError

        # Residual connection
        update = self.add_residual(update, x)

        return update

    def __repr__(self):
        return f"\n(message_layers): {repr(self.message_layers)}\n(update_layers):  {repr(self.update_layers)}"


class HermesMessageLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_order,
        out_order,
        kernel_constraint,
        n_rings,
        band_limit=None,
        node_batch_size=None,
        equiv_bias=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_order = in_order
        self.out_order = out_order
        self.kernel_constraint = kernel_constraint

        self.n_rings = n_rings
        self.band_limit = band_limit
        self.node_batch_size = node_batch_size
        self.equiv_bias = equiv_bias

        if self.kernel_constraint == "neigh":
            self.register_buffer(
                "kernel",
                torch.tensor(
                    build_kernel(in_order, out_order, band_limit), dtype=torch.float32
                ),
            )
        elif self.kernel_constraint == "self":
            self.register_buffer(
                "kernel",
                torch.tensor(
                    build_self_kernel(in_order, out_order, band_limit),
                    dtype=torch.float32,
                ),
            )
        self.weight = Parameter(
            torch.Tensor(self.kernel.shape[0], n_rings, out_channels, in_channels)
        )

        self.bias = Parameter(torch.Tensor(out_channels))

        if self.equiv_bias:
            self.ang_bias1 = Parameter(torch.Tensor(out_channels))
            self.ang_bias2 = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        torch.nn.init.zeros_(self.bias)
        if self.equiv_bias:
            torch.nn.init.normal_(self.ang_bias1, 0, 0.1)
            torch.nn.init.normal_(self.ang_bias2, 0, 0.1)

    @staticmethod
    def get_rot_matrix(bias):
        cos_, sin_ = torch.cos(bias), torch.sin(bias)
        return torch.stack([cos_, -sin_, sin_, cos_]).permute(1, 0).reshape(-1, 2, 2)

    def forward(self, inp, precomp):
        if self.node_batch_size is None:
            out = einsum(
                "ejm,efr,bfnm,brij->ein",
                inp,
                precomp,
                self.kernel,
                self.weight,
            )
        else:
            outs = []
            for i in range(0, inp.shape[0], self.node_batch_size):
                out = einsum(
                    "ejm,efr,bfnm,brij->ein",
                    inp[i : i + self.node_batch_size],
                    precomp[i : i + self.node_batch_size],
                    self.kernel,
                    self.weight,
                )
                outs.append(out)

            out = torch.cat(outs)

        if self.equiv_bias:
            out[:, :, 0] += self.bias[None, :]
            if out.shape[2] > 1:
                # Applying rotational/angular bias for rho_1, rho_2 etc features
                # to preserve equivariance
                rot_matrix1 = self.get_rot_matrix(self.ang_bias1)

                # Note that tensor.clone is a autodiff compatible operation
                out[:, :, 1:3] = torch.einsum(
                    "cij, ncj -> nci", rot_matrix1, out[:, :, 1:3].clone()
                )
            if out.shape[2] > 3:
                rot_matrix2 = self.get_rot_matrix(self.ang_bias2)
                out[:, :, 3:5] = torch.einsum(
                    "cij, ncj -> nci", rot_matrix2, out[:, :, 3:5].clone()
                )

        else:
            out += self.bias[None, :, None]

        return out

    def __repr__(self):
        return f"{self.__class__.__name__} ('{self.kernel_constraint}', channels=({self.in_channels},{self.out_channels}), orders=({self.in_order},{self.out_order}))"
