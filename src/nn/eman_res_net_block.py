from functools import partial

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from src.nn.eman_conv import EmanAttConv
from src.nn.regular_nonlin import RegularNonlinearity


class EmanAttResNetBlock(torch.nn.Module):
    """
    ResNet block with convolutions, linearities, and non-linearities

    Args:
        in_channels (int): number of input features
        out_channels (int): number of output features
        in_order (int): order of input
        out_order (int): order of output
        n_rings (int): number of radial rings
        num_samples (int): number of samples to use for non-linearity. Should be odd
        band_limit (int, optional): maximum theta frequency used
        final_activation (bool): whether to apply final non-linearity
        checkpoint (bool): whether to call GemConv within a torch checkpoint, saving lots of memory
        batch_norm (bool): whether use batch norm before non-lienarities
        batch (int, optional): if not None, comptue conv in batches of this size, checkpointed
    """

    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        in_order,
        hid_order,
        out_order,
        n_rings,
        band_limit,
        num_samples,
        checkpoint=False,
        node_batch_size=None,
        equiv_bias=False,
        regular_non_lin=False,
        batch_norm=False,
        dropout=False,
        n_heads=2,
        final_activation=True,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.conv1 = EmanAttConv(
            in_channels,
            hid_channels,
            in_order,
            hid_order,
            n_rings,
            band_limit,
            node_batch_size,
            n_heads,
            equiv_bias,
        )
        self.conv2 = EmanAttConv(
            hid_channels,
            out_channels,
            hid_order,
            out_order,
            n_rings,
            band_limit,
            node_batch_size,
            n_heads,
            equiv_bias,
        )

        # Apply batch norm and dropout inside RegularNonLinearity
        act1 = []
        act2 = []
        if batch_norm:
            act1.append(nn.BatchNorm1d(hid_channels))
            act2.append(nn.BatchNorm1d(out_channels))
        if dropout:
            act1.append(nn.Dropout())
            act2.append(nn.Dropout())
        act1.append(nn.ReLU())
        act2.append(nn.ReLU())

        self.nonlin1 = RegularNonlinearity(hid_order, num_samples, nn.Sequential(*act1))
        self.regular_non_lin = regular_non_lin

        if final_activation:
            self.nonlin2 = RegularNonlinearity(
                out_order, num_samples, nn.Sequential(*act2)
            )
        else:
            self.nonlin2 = nn.Identity()

        if in_channels != out_channels:
            self.lin = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, bias=False
            )  # Equivariant linear map mixing channels
        else:
            self.lin = nn.Identity()

    @staticmethod
    def call_conv_dummy(
        conv, x, edge_index, precomp_neigh_edge, precomp_self_edge, connection, _dummy
    ):
        return conv(x, edge_index, precomp_neigh_edge, precomp_self_edge, connection)

    def call_conv(
        self, conv, x, edge_index, precomp_neigh_edge, precomp_self_edge, connection
    ):
        if self.checkpoint:
            # Create dummy requires_grad argument to suppress pytorch checkpoint warning
            dummy = torch.zeros(1, device=x.device).requires_grad_()
            return checkpoint(
                partial(self.call_conv_dummy, conv),
                x,
                edge_index,
                precomp_neigh_edge,
                precomp_self_edge,
                connection,
                dummy,
            )
        else:
            return conv(
                x, edge_index, precomp_neigh_edge, precomp_self_edge, connection
            )

    def add_residual(self, y, x):
        residual = self.lin(x)
        o = min(y.shape[2], residual.shape[2])
        y[:, :, :o] = y[:, :, :o] + residual[:, :, :o]  # Handle varying orders
        return y

    def forward(self, x, edge_index, precomp_neigh_edge, precomp_self_edge, connection):
        """
        Forward pass.

        :param x: [num_v, in_channels, 2*in_order+1]
        :param edge_index: [n_edges, 2]
        :param precomp_neigh_edge: [n_edges, 2*band_limit+1, n_rings] computed by GemPrecomp
        :param connection: [num_edges]
        :return: [num_v, out_channels, 2*out_order+1]
        """
        y = self.call_conv(
            self.conv1, x, edge_index, precomp_neigh_edge, precomp_self_edge, connection
        )
        if self.regular_non_lin:
            y = self.nonlin1(y)
        y = self.call_conv(
            self.conv2, y, edge_index, precomp_neigh_edge, precomp_self_edge, connection
        )
        y = self.add_residual(y, x)

        return self.nonlin2(y)
