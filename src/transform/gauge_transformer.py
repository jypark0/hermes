# Copyright (c) 2021 Qualcomm Technologies, Inc.
# All rights reserved.
from copy import deepcopy
from math import pi

import torch


class GaugeTransformer:
    """
    Transform geometry by a gauge transformation.

    log map v: v -> g_p^{-1} v, so angle th: th - g_p
    transformer g_{q -> p}  ->  g_p^{-1} g_{q->p} g_q
    """

    def __init__(self, transform_angle=None):
        """
        Set transform_angle = None to randomize for every __call__
        Use after SimpleGeometry()
        """
        self.transform_angle = transform_angle

    def __call__(self, data):
        if self.transform_angle is None:
            transform_angle = 2 * pi * torch.rand(data.pos.shape[0]).to(data.device)
        else:
            transform_angle = self.transform_angle

        assert len(data.pos) == len(transform_angle)
        new_data = deepcopy(data)
        idx_to, idx_from = data.edge_index  # (q, p)
        new_data.connection = (
            -transform_angle[idx_to] + transform_angle[idx_from] + data.connection
        )
        r, th = data.edge_coords.T
        new_data.edge_coords = torch.stack([r, th - transform_angle[idx_to]], 1)

        return new_data
