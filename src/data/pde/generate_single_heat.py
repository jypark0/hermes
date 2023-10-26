from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from pyvista import examples

from src.data.pde.utils import get_mesh_laplacian, plot_mesh, screenshot_mesh

orig_mesh = examples.download_armadillo()
n_samples = 15
Tmax = 100
sigma = 1e-4
plot = False

pv.set_plot_theme("paraview")

# Fineness
decimate_ratio = [0.998, 0.995, 0.99, 0.98, 0.95]
dts = [0.2, 0.1, 0.07, 0.03, 0.015]

save_dir = Path("data/fineness/heat")
for idx in range(len(decimate_ratio)):
    # for name in objects.keys():
    r = decimate_ratio[idx]
    mesh = orig_mesh.decimate(r)
    _ = mesh.clean(inplace=True)

    # Current mesh
    pos = mesh.points
    face = mesh.faces.reshape(-1, 4)[:, 1:]

    pos = torch.from_numpy(pos)
    face = torch.from_numpy(face).T.long()

    edge_index, edge_weight = get_mesh_laplacian(pos, face, normalization="sym")

    tg = torch.zeros(pos.shape[0], pos.shape[0])
    tg[tuple(edge_index)] = edge_weight

    # Parameters
    dt = dts[idx]
    N = pos.shape[0]
    num_start_points = N // 5

    # Create save_dir and save some files
    (save_dir / f"reduce_{r}" / "raw").mkdir(parents=True, exist_ok=True)

    s = 0
    while s < n_samples:
        # Store values
        U = torch.zeros(Tmax + 1, N, dtype=torch.float32)

        # Initial conditions
        idx = torch.randperm(N)[:num_start_points]
        dist = torch.cdist(pos, pos)
        U[0] += (1 * torch.exp(-dist[idx].pow(2) / (2 * sigma**2))).sum(0)

        u = U[0].clone()
        print(f"[Fineness {r}] Sample: {s}, num nodes: {N}")

        if plot and s == 0:
            p = plot_mesh(mesh, torch.zeros(N), "Heat", "armadillo")
            p.add_text(f"time: 0.0", font_size=12, name="timelabel")
            p.add_text(
                f"Num Nodes: {N}", font_size=12, name="num_nodes", position="lower_left"
            )

            p.show()

        t = 0.0
        for i in range(1, Tmax + 1):
            u = u + (dt * tg @ u)
            t += dt

            U[i] = u.clone()

            if plot and s == 0:
                p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
                mesh.point_data["c"] = u
                p.app.processEvents()

                if i == 1 or i == Tmax // 4 or i == Tmax:
                    screenshot = (
                        save_dir / f"reduce_{r}" / "raw" / f"armadillo_{s}_T{i}.png"
                    )
                    screenshot_mesh(mesh, u, "Heat", "armadillo", screenshot)

        if plot and s == 0:
            p.close()
        # Save results
        if u.max() < 1:
            mesh.save(save_dir / f"reduce_{r}" / "raw" / f"armadillo.ply")
            np.save(save_dir / f"reduce_{r}" / "raw" / f"armadillo_{s}.npy", U)
            s += 1
        else:
            print("diverging, continue")


# Roughness
start_mesh = orig_mesh.decimate(0.99)
start_mesh.compute_normals(cell_normals=False, point_normals=True, inplace=True)

scales = [0.1, 0.5, 1.0, 1.5, 3.0]
dt = 0.07
save_dir = Path("data/roughness/heat")
for idx in range(len(scales)):
    # for name in objects.keys():
    scale = scales[idx]

    # Warp by normals
    scaled_normals = (
        scale * np.random.randn(start_mesh.n_points, 1) * start_mesh["Normals"]
    )
    mesh = start_mesh.copy(deep=True)
    mesh["scaled_normals"] = scaled_normals
    mesh.warp_by_vector("scaled_normals", inplace=True)
    _ = mesh.clean(inplace=True)

    # Current mesh
    pos = mesh.points
    face = mesh.faces.reshape(-1, 4)[:, 1:]

    pos = torch.from_numpy(pos)
    face = torch.from_numpy(face).T.long()

    edge_index, edge_weight = get_mesh_laplacian(pos, face, normalization="sym")

    tg = torch.zeros(pos.shape[0], pos.shape[0])
    tg[tuple(edge_index)] = edge_weight

    # Parameters
    N = pos.shape[0]
    num_start_points = N // 5

    # Create save_dir and save some files
    (save_dir / f"roughness_{scale}" / "raw").mkdir(parents=True, exist_ok=True)

    s = 0
    while s < n_samples:
        # Store values
        U = torch.zeros(Tmax + 1, N, dtype=torch.float32)

        # Initial conditions
        idx = torch.randperm(N)[:num_start_points]
        dist = torch.cdist(pos, pos)
        U[0] += (1 * torch.exp(-dist[idx].pow(2) / (2 * sigma**2))).sum(0)

        u = U[0].clone()
        print(f"[Roughness {scale}] Sample: {s}, num nodes: {N}")

        if plot and s == 0:
            p = plot_mesh(mesh, torch.zeros(N), "Heat", "armadillo")
            p.add_text(f"time: 0.0", font_size=12, name="timelabel")
            p.add_text(
                f"Num Nodes: {N}", font_size=12, name="num_nodes", position="lower_left"
            )

            p.show()

        t = 0.0
        for i in range(1, Tmax + 1):
            u = u + (dt * tg @ u)
            t += dt

            U[i] = u.clone()

            if plot and s == 0:
                p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
                mesh.point_data["c"] = u
                p.app.processEvents()

                if i == 1 or i == Tmax // 4 or i == Tmax:
                    screenshot = (
                        save_dir
                        / f"roughness_{scale}"
                        / "raw"
                        / f"armadillo_{s}_T{i}.png"
                    )
                    screenshot_mesh(mesh, u, "Heat", "armadillo", screenshot)

        if plot and s == 0:
            p.close()
        # Save results
        if u.max() < 1:
            mesh.save(save_dir / f"roughness_{scale}" / "raw" / f"armadillo.ply")
            np.save(save_dir / f"roughness_{scale}" / "raw" / f"armadillo_{s}.npy", U)
            s += 1
        else:
            print("diverging, continue")
