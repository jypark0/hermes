from math import pi

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def rotate_frame(frame, rot_direction, rot_angle):
    rot_direction = np.array(rot_direction) / np.linalg.norm(rot_direction)
    # get rotation matrices
    rot_vec = R.from_rotvec(rot_angle * rot_direction)
    rot_matrix = rot_vec.as_matrix()
    # compute rotated positions
    frame_rot = torch.einsum(
        "ij, jk -> ik", frame, torch.tensor(rot_matrix, dtype=frame.dtype)
    )
    return frame_rot


class RotoTranslationTransformer:
    """
    Transform data by SO(3) rotation matrix and translations
    """

    def __init__(self, rot_angle=None, rot_direction=None, translation=None):
        """
        Set all arguments to None to randomize for every __call__
        """
        self.rot_angle = rot_angle  # magnitude of rotation about some chosen axis
        self.rot_direction = rot_direction  # axis of rotation
        self.translation = translation

    def __call__(self, data):
        # Generate random rotation angle if needed
        if self.rot_angle is None:
            rot_angle = np.random.uniform(0, 2 * pi)
        else:
            rot_angle = self.rot_angle

        # Generate random axis of rotation if needed
        if self.rot_direction is None:
            rot_direction = np.random.rand(3)

        else:
            rot_direction = self.rot_direction
        # normalize the direction of axis so that rot_angle remains same
        rot_direction = rot_direction / np.linalg.norm(rot_direction)

        # get rotation matrices
        rot_vec = R.from_rotvec(rot_angle * rot_direction)
        rot_matrix = rot_vec.as_matrix()
        # compute rotated positions
        data.pos = torch.einsum(
            "ij, jk -> ik",
            data.pos,
            torch.tensor(rot_matrix, dtype=data.pos.dtype, device=data.pos.device),
        )

        # compute rotated normal vectors
        if data.get("normal") is not None:
            data.normal = torch.einsum(
                "ij,jk -> ik",
                data.normal,
                torch.tensor(
                    rot_matrix, dtype=data.normal.dtype, device=data.normal.device
                ),
            )

        # Translation

        # Generate random translation if needed
        if self.translation is None:
            translation_mag = 100 * np.random.uniform(0, 1)
            translation_direction = np.random.rand(3)
            translation_direction = np.array(translation_direction) / np.linalg.norm(
                translation_direction
            )
            translation = torch.tensor(
                translation_mag * translation_direction,
                dtype=data.pos.dtype,
                device=data.pos.device,
            )
        else:
            translation = torch.tensor(
                self.translation, dtype=data.pos.dtype, device=data.pos.device
            )

        # Translate positions
        data.pos = data.pos + translation[None, :]
        return data
