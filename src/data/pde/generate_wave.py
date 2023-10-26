from pathlib import Path

import numpy as np
import pyvista as pv
import torch
from pyvista import examples

from src.data.pde.utils import get_mesh_laplacian, plot_mesh, screenshot_mesh

objects = {
    "sphere": pv.Sphere(),
    "armadillo": examples.download_armadillo(),
    "bunny_coarse": examples.download_bunny_coarse(),
    "lucy": examples.download_lucy(),
    "woman": examples.download_woman(),
    "spider": examples.download_spider(),
    "urn": examples.download_urn(),
}
decimate_ratio = {
    "sphere": 0.0,
    "armadillo": 0.99,
    "bunny_coarse": 0.0,
    "lucy": 0.95,
    "woman": 0.98,
    "spider": 0.0,
    "urn": 0.98,
}
dts = {
    "sphere": 3e-3,
    "armadillo": 0.3,
    "bunny_coarse": 1e-3,
    "lucy": 0.8,
    "woman": 0.8,
    "spider": 5e-3,
    "urn": 0.6,
}

save_dir = Path("data/wave/raw")
n_samples = 5
Tmax = 200
sigma = 5e-3
names = objects.keys()
plot = False

pv.set_plot_theme("paraview")

# for name in objects.keys():
for name in names:
    mesh = objects[name]
    mesh = mesh.decimate(decimate_ratio[name])
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
    dt = dts[name]
    num_start_points = N // 40

    # Create save_dir
    (save_dir / name).mkdir(parents=True, exist_ok=True)
    mesh.save(save_dir / name / f"{name}.ply")
    np.save(save_dir / name / f"dt.npy", dt)

    for s in range(n_samples):
        # Store values
        U = torch.zeros(Tmax + 1, N, dtype=torch.float32)

        # Initial conditions
        idx = torch.randperm(N)[:num_start_points]
        dist = torch.cdist(pos, pos)
        coeff = (-1) ** torch.remainder(torch.arange(num_start_points), 2)[:, None]
        U[0] += (1 * coeff * torch.exp(-dist[idx].pow(2) / (2 * sigma**2))).sum(0)

        # previous timestep
        u1 = U[0].clone()
        # current timestep
        u = U[0].clone()
        print(f"[{name}] Sample: {s}, num nodes: {N}")

        if plot and s == 0:
            p = plot_mesh(mesh, torch.zeros(N), "Wave", name)

            p.add_text(f"time: 0.0", font_size=12, name="timelabel")
            p.add_text(
                f"Num Nodes: {N}", font_size=12, name="num_nodes", position="lower_left"
            )

            p.show()

        t = 0.0
        for i in range(1, Tmax + 1):
            prev_u1 = u1.clone()
            u1 = u.clone()
            u = 2 * u - prev_u1 + dt**2 * (tg @ u)
            t += dt

            U[i] = u.clone()

            if plot and s == 0:
                p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
                mesh.point_data["c"] = u
                p.app.processEvents()

                if i == 1 or i == Tmax // 4 or i == Tmax:
                    screenshot = save_dir / name / f"{name}_{s}_T{i}.png"
                    screenshot_mesh(mesh, u, "Wave", name, screenshot)

        # Save results
        np.save(save_dir / name / f"{name}_{s}.npy", U)
        if plot and s == 0:
            p.close()
