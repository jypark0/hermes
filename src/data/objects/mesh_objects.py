from collections import OrderedDict

import gymnasium as gym
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import pyvistaqt as pvqt
import trimesh
from gymnasium import spaces

from src.utils.parallel_transport import spherical_parallel_transport


class MeshObjects(gym.Env):
    """Gym environment for pushing objects with orientation on mesh

    actions are turn left, move forward, turn right
    """

    # COLORS = plt.get_cmap("tab10").colors
    COLORS = plt.get_cmap("Greys")

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 2}

    def __init__(
        self,
        num_nodes,
        num_objects,
        rules="123",
        rule2_threshold=2,
        rule3_threshold=np.pi / 4,
        render_mode="rgb_array",
    ):
        self.orig_num_nodes = num_nodes
        self.num_objects = num_objects

        assert rules in ["1", "12", "13", "123"]
        self.rules = rules
        self.rule2_threshold = rule2_threshold
        self.rule3_threshold = rule3_threshold

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Actions: turn left, go forward, turn right
        self.num_actions = 3
        self.action_space = spaces.Discrete(self.num_actions * self.num_objects)
        # Will be set in reset()
        self.observation_space = None

        # Trimesh object for underlying mesh
        self.mesh = None
        # List of object node indices and orientations
        # Orientation: index of which neighbor it is facing, ranges from 0 to node degree - 1
        self.object_node_idxs = None
        self.object_ori_idxs = None

        # Node occupancy
        # 0 means no object
        # i > 0 means object with idx (i - 1) is here
        self.node_occupancy = None

        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        info = self._reset_mesh(options)

        self._reset_objects()

        return self._get_obs(), info

    def _reset_mesh(self, options):
        self.num_nodes = self.orig_num_nodes
        if options is not None and options.get("random_hill_parameters"):
            hillxvariance, hillyvariance, hillamplitude = self.np_random.uniform(
                [1, 1, 1], [1.5, 1.5, 2]
            )
        else:
            hillxvariance = hillyvariance = 1.25
            hillamplitude = 1.5

        numberofhills = 200
        u_res = v_res = w_res = 100

        # Use pyvista random hills generation and decimate to get random meshes
        # Need to generate new random seed first
        seed = self.np_random.integers(0, 2**31 - 1)
        mesh = pv.ParametricRandomHills(
            numberofhills=numberofhills,
            hillxvariance=hillxvariance,
            hillyvariance=hillyvariance,
            hillamplitude=hillamplitude,
            randomseed=seed,
            u_res=u_res,
            v_res=v_res,
            w_res=w_res,
        )

        cur_N = 0
        high = 1
        low = 0
        # Binary search to get exact self.num_nodes
        while cur_N != self.num_nodes:
            cur_reduction = (low + high) / 2
            dec_mesh = mesh.decimate(cur_reduction)
            cur_N = dec_mesh.n_points

            if cur_N > self.num_nodes:
                low = cur_reduction
            else:
                high = cur_reduction

        faces_as_array = dec_mesh.faces.reshape((dec_mesh.n_faces, 4))[:, 1:]
        self.mesh = trimesh.Trimesh(dec_mesh.points, faces_as_array, process=False)

        # Register faces as observation_space
        self.observation_space = spaces.Dict(
            {
                "pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_nodes, 3),
                    dtype=np.float32,
                ),
                "occupancy": spaces.Discrete(self.num_nodes),
                "faces": spaces.Box(
                    low=0,
                    high=self.num_nodes - 1,
                    shape=faces_as_array.shape,
                    dtype=int,
                ),
                "orientation": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_nodes, 2),
                    dtype=np.float32,
                ),
                "object_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_objects, 3),
                    dtype=np.float32,
                ),
                "object_orientation": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_objects, 2),
                    dtype=np.float32,
                ),
            }
        )

        return {
            "num_nodes": self.num_nodes,
            "numberofhills": numberofhills,
            "hillxvariance": hillxvariance,
            "hillyvariance": hillyvariance,
            "hillamplitude": hillamplitude,
            "randomseed": seed,
            "u_res": u_res,
            "v_res": v_res,
            "w_res": w_res,
        }

    def _reset_objects(self):
        # Pick random node index for each object
        node_idxs = self.np_random.choice(
            np.arange(self.num_nodes), size=self.num_objects, replace=False
        )
        self.object_node_idxs = node_idxs
        self.node_occupancy = np.zeros((self.num_nodes), dtype=int)
        self.node_occupancy[node_idxs] = np.arange(1, self.num_objects + 1)

        # Pick random orientations for each object
        all_neighbors = self.mesh.vertex_neighbors
        node_degrees = np.asarray(
            [len(neighbors) for neighbors in all_neighbors], dtype=int
        )
        node_ori_idxs = self.np_random.integers(node_degrees)
        self.object_ori_idxs = node_ori_idxs[self.object_node_idxs]

        # Precompute tangent planes of all nodes
        normals = self._normalize(self.mesh.vertex_normals)
        # I - n * n^T
        self.projector = np.eye(3) - np.einsum("oi,oj->oij", normals, normals)

        node_ori_node_idxs = np.empty((self.num_nodes), dtype=int)
        for i in range(self.num_nodes):
            node_ori_node_idxs[i] = all_neighbors[i][node_ori_idxs[i]]

        # Compute x,y gauges at every node
        gauge_vectors = self.mesh.vertices[node_ori_node_idxs] - self.mesh.vertices

        self.gauge_x = self._normalize(
            np.einsum("nij,nj->ni", self.projector, gauge_vectors)
        )
        self.gauge_y = np.cross(normals, self.gauge_x, axis=1)

    def render(self):
        def get_cmap():
            colors = np.array(self.COLORS)
            # Use white gray as first color for empty occupancy
            colors = np.concatenate([[[0.9, 0.9, 0.9]], colors], axis=0)

            return mpl.colors.ListedColormap(colors)

        if self.render_mode == "rgb_array":
            pv_mesh = pv.wrap(self.mesh)

            if self.viewer is None:
                self.viewer = pv.Plotter(off_screen=True, window_size=(1000, 700))

            self.viewer.clear()

            self.viewer.add_axes()
            self.viewer.add_mesh(pv_mesh, show_edges=True, opacity=0.7)
            self.viewer.add_points(
                pv_mesh.points,
                preference="points",
                color="gray",
                scalars=self.node_occupancy,
                render_points_as_spheres=True,
                point_size=15,
                cmap="viridis",
                show_scalar_bar=False,
                scalar_bar_args={
                    "n_labels": (self.num_objects + 1) // 5,
                    "fmt": "%.0f",
                },
            )

            self.viewer.camera.elevation = -10

            img = self.viewer.screenshot()
            return img

        elif self.render_mode == "human":
            pv_mesh = pv.wrap(self.mesh)

            if self.viewer is None:
                self.viewer = pvqt.BackgroundPlotter(
                    window_size=(1000, 700), auto_update=True
                )

            self.viewer.clear()

            self.viewer.add_axes()
            self.viewer.add_mesh(pv_mesh, show_edges=True, opacity=0.7)
            self.viewer.add_points(
                pv_mesh.points,
                preference="points",
                color="gray",
                scalars=self.node_occupancy,
                render_points_as_spheres=True,
                point_size=15,
                cmap="viridis",
                show_scalar_bar=False,
                scalar_bar_args={
                    "n_labels": (self.num_objects + 1) // 5,
                    "fmt": "%.0f",
                },
            )
            self.viewer.camera.elevation = -10

            self.viewer.show()

            # self.viewer.show(auto_close=False, interactive_update=True)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _normalize(self, x):
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-16)

    def _get_current_orientation(self):
        ori_node_idxs = np.empty((self.num_objects), dtype=int)
        all_neighbors = self.mesh.vertex_neighbors
        for i in range(self.num_objects):
            ori_node_idxs[i] = all_neighbors[self.object_node_idxs[i]][
                self.object_ori_idxs[i]
            ]

        unprojected_ori = (
            self.mesh.vertices[ori_node_idxs]
            - self.mesh.vertices[self.object_node_idxs]
        )

        # Project orientation onto current node's tangent plane
        projected_ori = self._normalize(
            np.einsum(
                "oij,oj->oi", self.projector[self.object_node_idxs], unprojected_ori
            )
        )

        # Compute orientation in current node reference frame (tangent plane)
        ori_x = np.sum(projected_ori * self.gauge_x[ori_node_idxs], axis=1)
        ori_y = np.sum(projected_ori * self.gauge_y[ori_node_idxs], axis=1)

        return self._normalize(np.stack([ori_x, ori_y], axis=1))

    def _get_next_orientation(self, cur_node_idx, next_node_idx, cur_ori):
        # Parallel transport current orientation
        transported_ori = spherical_parallel_transport(
            self.mesh.vertex_normals[cur_node_idx],
            self.mesh.vertex_normals[next_node_idx],
            cur_ori,
        )

        next_node_neighbor_idxs = self.mesh.vertex_neighbors[next_node_idx]
        next_node_orientations = (
            self.mesh.vertices[next_node_neighbor_idxs]
            - self.mesh.vertices[next_node_idx][None]
        )
        next_node_orientations = self._normalize(next_node_orientations)

        # Pick closest neighbor by minimum distance
        distances = np.linalg.norm(next_node_orientations - transported_ori, axis=-1)
        min_dist_neighbor_idx = np.argsort(distances)[0]

        return min_dist_neighbor_idx

    def _get_obs(self):
        obs_space = self.observation_space

        object_ori = self._get_current_orientation()

        # Masked orientation over nodes
        nodes_ori = np.zeros((self.num_nodes, 2))
        nodes_ori[self.object_node_idxs] = object_ori

        obs = OrderedDict(
            {
                "pos": self.mesh.vertices.view(np.ndarray).astype(
                    obs_space["pos"].dtype
                ),
                "faces": self.mesh.faces.view(np.ndarray).T.astype(
                    obs_space["faces"].dtype
                ),
                "occupancy": self.node_occupancy.astype(obs_space["occupancy"].dtype),
                "orientation": nodes_ori.astype(obs_space["orientation"].dtype),
                "object_pos": self.mesh.vertices[self.object_node_idxs]
                .view(np.ndarray)
                .astype(obs_space["object_pos"].dtype),
                "object_orientation": object_ori.astype(
                    obs_space["object_orientation"].dtype
                ),
            }
        )

        return obs

    def move(self, obj_id, move):
        """Check if move is valid and return updated features

        Rules:
        1. Can't move if node is occupied by another object
        2. Can't move if | z_old - z_new | > hillamplitude
        3. Can't move if angle between vertex normals > np.pi/4
        """

        cur_node_idx = self.object_node_idxs[obj_id]
        cur_node_degree = len(self.mesh.vertex_neighbors[cur_node_idx])

        # Check that node_occupancy and object_node_idxs match for obj_id
        assert self.node_occupancy[cur_node_idx] == obj_id + 1

        # Check that current object_ori_idxs < cur_node_degree
        assert self.object_ori_idxs[obj_id] < cur_node_degree

        info = {"rule": 0}

        # Turn left
        if move == 0:
            self.object_ori_idxs[obj_id] = (
                self.object_ori_idxs[obj_id] - 1
            ) % cur_node_degree
            return info
        # Turn right
        elif move == 2:
            self.object_ori_idxs[obj_id] = (
                self.object_ori_idxs[obj_id] + 1
            ) % cur_node_degree
            return info
        # Go forward
        else:
            # Get next node index of current object
            next_node_idx = self.mesh.vertex_neighbors[cur_node_idx][
                self.object_ori_idxs[obj_id]
            ]

            # Rule 1
            if self.node_occupancy[next_node_idx] != 0:
                # No changes
                info["rule"] = 1
                return info

            # Rule 2
            if self.rules in ["12", "123"]:
                if (
                    np.abs(
                        self.mesh.vertices[cur_node_idx, 2]
                        - self.mesh.vertices[next_node_idx, 2]
                    )
                    > self.rule2_threshold
                ):
                    # edge = [cur_node_idx, next_node_idx]
                    # face_adj_idx = np.where(
                    #     np.all(
                    #         (self.mesh.face_adjacency_edges == edge)
                    #         | (self.mesh.face_adjacency_edges == edge[::-1]),
                    #         axis=1,
                    #     )
                    # )
                    # # Doesn't work
                    # assert face_adj_idx[0].size == 1
                    # if self.mesh.face_adjacency_angles[face_adj_idx] > np.pi / 3:
                    info["rule"] = 2
                    return info

            # Rule 3
            if self.rules in ["13", "123"]:
                old_vn = self.mesh.vertex_normals[cur_node_idx]
                new_vn = self.mesh.vertex_normals[next_node_idx]
                if np.arccos(np.sum(old_vn * new_vn, axis=-1).clip(-1, 1)) > (
                    self.rule3_threshold
                ):
                    info["rule"] = 3
                    return info

            # Update node_occupancy
            self.node_occupancy[next_node_idx] = self.node_occupancy[cur_node_idx]
            self.node_occupancy[cur_node_idx] = 0

            # Update object_ori_idxs and object_node_idxs
            # Compute current orientation
            ori_node_idx = self.mesh.vertex_neighbors[cur_node_idx][
                self.object_ori_idxs[obj_id]
            ]
            cur_ori = (
                self.mesh.vertices[ori_node_idx] - self.mesh.vertices[cur_node_idx]
            )
            cur_ori = self._normalize(self.projector[cur_node_idx] @ cur_ori).view(
                np.ndarray
            )

            # Get next orientation using parallel transport and pick nearest neighbor
            self.object_ori_idxs[obj_id] = self._get_next_orientation(
                cur_node_idx, next_node_idx, cur_ori
            )
            self.object_node_idxs[obj_id] = next_node_idx

            return info

    def step(self, action):
        terminated = False
        reward = 0
        truncated = False

        move = action % self.num_actions
        obj_id = action // self.num_actions

        info = self.move(obj_id, move)

        return self._get_obs(), reward, terminated, truncated, info
