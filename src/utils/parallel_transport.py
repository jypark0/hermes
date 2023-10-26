# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.

# Modified by GEMP authors

import numpy as np


def so3_matrix_generator(axis, theta):
    """
    SO(3) exponential map, vectorized.
    By Rodriguez formula.
    :param axis: in S2 rotation axis (B, 3)
    :param theta: Rotation angle (B,)
    :return: Rotation matrix in SO(3), (B, 3, 3)
    """
    # To handle single theta and single axis
    theta = np.atleast_1d(theta)
    axis = np.atleast_2d(axis)

    theta = theta[:, None, None]
    x, y, z = axis.T
    zero = np.zeros_like(x)
    k = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], 1).reshape((-1, 3, 3))
    rot = np.eye(3)[None] + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k

    return rot


def spherical_parallel_transport(p_from, p_to, v):
    """
    Parallel transport according to Levi-Civita connection on sphere along geodesic.
    Vectorized.
    :param p_from: in S2, (B, 3)
    :param p_to: in S2, (B, 3)
    :param v: in T_{p_from} S2 (B, 3)
    :return: v_transformed in T_{p_to} S2 (B, 3)
    """

    assert p_from.shape == p_to.shape == v.shape
    axis = np.cross(p_from, p_to)
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-20)
    theta = np.arccos(np.sum(p_to * p_from, axis=-1).clip(-1, 1))
    rot = so3_matrix_generator(axis, theta)
    # To handle single v
    v = np.atleast_2d(v)
    v_transformed = np.einsum("nij,nj->ni", rot, v)
    return v_transformed.squeeze()
