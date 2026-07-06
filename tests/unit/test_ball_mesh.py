from __future__ import annotations

import numpy as np

from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh import oriented_tetra_volume6
from magcore.mesh.mesh_generators import build_ball_tetra_mesh


def _cell_volumes(mesh) -> np.ndarray:
    return np.array([oriented_tetra_volume6(mesh.vertices[c]) / 6.0 for c in mesh.cells])


def test_ball_mesh_cells_positively_oriented() -> None:
    mesh = build_ball_tetra_mesh(4, radius=1.0)
    vols = _cell_volumes(mesh)
    assert np.all(vols > 0.0)


def test_ball_mesh_boundary_lies_on_sphere() -> None:
    mesh = build_ball_tetra_mesh(4, radius=2.0, center=(0.5, -0.3, 0.1))
    ci = CouplingInterface.from_tetra_mesh(mesh)
    c = np.array([0.5, -0.3, 0.1])
    r = np.linalg.norm(ci.surface_mesh.vertices - c, axis=1)
    assert np.allclose(r, 2.0, atol=1e-10)
    # Замкнутая поверхность с согласованными внешними нормалями: ∮ n dS = 0.
    n_int = np.sum(ci.outward_normals * ci.face_areas[:, None], axis=0)
    assert np.linalg.norm(n_int) < 1e-10


def test_ball_mesh_volume_and_area_converge() -> None:
    exact_vol = 4.0 / 3.0 * np.pi
    exact_area = 4.0 * np.pi
    vols, areas = [], []
    for n in (3, 6):
        mesh = build_ball_tetra_mesh(n, radius=1.0)
        vols.append(float(_cell_volumes(mesh).sum()))
        ci = CouplingInterface.from_tetra_mesh(mesh)
        areas.append(float(ci.face_areas.sum()))
    # Полиэдральная аппроксимация шара снизу, монотонно растёт к точному.
    assert vols[0] < vols[1] <= exact_vol + 1e-9
    assert areas[0] < areas[1] <= exact_area + 1e-9
    # Уже при n=6 — в пределах ~3% по объёму.
    assert abs(vols[1] - exact_vol) / exact_vol < 0.05
