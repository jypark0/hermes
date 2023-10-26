# This file was modified by The EMAN Authors to include self-contributions

# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
"""
Adapted from: hsn/transforms/harmonic_precomp.py by Ruben Wiersma at github.com/rubenwiersma/hsn

MIT License

Copyright (c) 2020 rubenwiersma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch


def linear_interpolation_weights(x, n_points, zero_falloff=False):
    """
    Compute linear interpolation weights
    to points at x from regularly interspersed points.
    :param x: coordinates of points to interpolate to, in range [0, 1].
    :param n_points: number of regularly interspersed points.
    :param zero_falloff: if set to True, the interpolated function falls to 0 at x = 1.
    """
    assert x.dim() == 1
    if zero_falloff:
        n_points += 1
    x = x * (n_points - 1)
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    x0 = torch.clamp(x0, 0, n_points - 2)
    x1 = torch.clamp(x1, 1, n_points - 1)
    w0 = x1.type(x.dtype) - x
    w1 = x - x0.type(x.dtype)

    weights = torch.zeros((x.size(0), n_points), device=x.device, dtype=x.dtype)
    weights[torch.arange(x.size(0)), x0] = w0
    weights[torch.arange(x.size(0)), x1] = w1

    if zero_falloff:
        weights = weights[:, :-1]

    return weights


class GemPrecomp(object):
    r"""Precomputation for
    Asserts that a logmap and vertex weights have been computed
    and stored in data.edge_coords, data.weight.

    .. math::
        w_j \mu_{\q}(r_{ij}) e^{\i m\theta_{ij}}\right

    Args:
        n_rings (int, optional): number of rings used to parametrize
            the radial profile, defaults to 2.
        max_order (int, optional): the maximum rotation order of the network,
            defaults to 1.
        max_r (float, optional): the radius of the kernel,
            if not supplied, maximum radius is used.
    """

    def __init__(self, n_rings=2, max_order=1, max_r=None, dtype=torch.float32):
        self.n_rings = n_rings
        self.max_order = max_order
        self.max_r = max_r
        self.dtype = dtype

    def __call__(self, data):
        assert hasattr(data, "edge_coords")
        # assert hasattr(data, 'weight')

        # Conveniently name variables
        # (row, col), (r, theta), weight = data.edge_index, data.edge_coords.T, data.weight

        r, theta = data.edge_coords.T
        N_edges, M, R = r.size(0), 2 * self.max_order + 1, self.n_rings
        N_nodes = data.pos.shape[0]

        # Normalize radius to range [0, 1]
        r = r / self.max_r if self.max_r is not None else r / r.max()

        # Compute interpolation weights for the radial profile function
        radial_profile_weights = linear_interpolation_weights(r, R, zero_falloff=False)

        # Compute exponential component for each point
        freq = torch.arange(
            1, self.max_order + 1, device=theta.device, dtype=self.dtype
        )
        angles = theta.view(-1, 1) * freq

        exponential = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).view(
            N_edges, M - 1
        )

        exponential = torch.cat(
            (
                torch.ones(N_edges, 1, device=theta.device, dtype=self.dtype),
                exponential,
            ),
            1,
        )

        # Set kernel for center points to 0
        exponential[torch.nonzero(r == 0), 1:] = 0

        precomp_neigh_edge = radial_profile_weights.view(
            N_edges, 1, R
        ) * exponential.view(N_edges, M, 1)
        data.precomp_neigh_edge = precomp_neigh_edge

        # Self kernel
        # Per edge
        angles_self = torch.zeros_like(angles)
        exponential_self = torch.stack(
            (torch.cos(angles_self), torch.sin(angles_self)), dim=-1
        ).view(N_edges, M - 1)
        exponential_self = torch.cat(
            (
                torch.ones(N_edges, 1, device=theta.device, dtype=self.dtype),
                exponential_self,
            ),
            1,
        )
        exponential_self[torch.nonzero(r == 0), 1:] = 0

        precomp_self_edge = radial_profile_weights.view(
            N_edges, 1, R
        ) * exponential_self.view(N_edges, M, 1)
        data.precomp_self_edge = precomp_self_edge

        # Per node
        angles_self = torch.zeros(
            N_nodes, freq.shape[0], device=theta.device, dtype=self.dtype
        )
        radii_self = torch.zeros(N_nodes, device=theta.device, dtype=self.dtype)
        radial_profile_weights_self = linear_interpolation_weights(
            radii_self, R, zero_falloff=False
        )

        exponential_self = torch.stack(
            (torch.cos(angles_self), torch.sin(angles_self)), dim=-1
        ).view(N_nodes, 2 * self.max_order)
        exponential_self = torch.cat(
            (
                torch.ones(N_nodes, 1, device=theta.device, dtype=self.dtype),
                exponential_self,
            ),
            1,
        )

        precomp_self_node = radial_profile_weights_self.view(
            N_nodes, 1, R
        ) * exponential_self.view(N_nodes, M, 1)
        data.precomp_self_node = precomp_self_node

        return data

    def __repr__(self):
        return "{}(n_rings={}, max_order={}, max_r={})".format(
            self.__class__.__name__, self.n_rings, self.max_order, self.max_r
        )
