from __future__ import annotations

import numpy as np

from magcore.femcore.periodic import match_periodic_vertices
from magcore.mesh.mesh_generators import (
    build_annular_sector_tetra_mesh,
    tag_sector_regions_by_radius,
)

R_IN, R_OUT, THETA, ZL = 1.0, 2.0, np.pi / 7.0, 0.5  # ~один полюс из 14
TOL = 1e-9


def _mesh(nr=3, nt=2, nz=2):
    return build_annular_sector_tetra_mesh(
        nr, nt, nz, r_in=R_IN, r_out=R_OUT, theta_seg=THETA, z_len=ZL
    )


def test_sector_geometry_in_range_and_oriented() -> None:
    mesh = _mesh()
    V = mesh.vertices
    r = np.hypot(V[:, 0], V[:, 1])
    ang = np.arctan2(V[:, 1], V[:, 0])
    assert r.min() >= R_IN - TOL and r.max() <= R_OUT + TOL
    assert ang.min() >= -TOL and ang.max() <= THETA + TOL
    assert V[:, 2].min() >= -TOL and V[:, 2].max() <= ZL + TOL
    # все ячейки положительно ориентированы (объём > 0)
    vols = np.array([mesh.cell_volume(c) for c in range(mesh.n_cells)])
    assert np.all(vols > 0.0)
    assert mesh.n_cells == 3 * 2 * 2 * 6


def test_sector_cut_planes_are_periodic_conforming() -> None:
    mesh = _mesh()
    c, s = np.cos(THETA), np.sin(THETA)

    def rot(p):  # поворот на +θ_seg вокруг z (θ=0 → θ=θ_seg)
        return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])

    def on0(p):  # разрез θ=0 (y≈0, x>0)
        return abs(p[1]) < 1e-7 and p[0] > 0.0

    def onT(p):  # разрез θ=θ_seg
        return abs(np.arctan2(p[1], p[0]) - THETA) < 1e-7

    vmap = match_periodic_vertices(mesh, on0, onT, rot, tol=1e-7)
    # узлов на разрезе = (nr+1)*(nz+1) = 4*3 = 12, биекция
    assert len(vmap) == 12
    assert len(set(vmap.values())) == 12


def test_sector_region_tagging_by_radius() -> None:
    mesh = _mesh()
    dr = (R_OUT - R_IN) / 3
    # три радиальных пояса = три радиальных слоя ячеек (ярмо / магнит / зазор)
    labels = tag_sector_regions_by_radius(
        mesh, band_edges=(R_IN, R_IN + dr, R_IN + 2 * dr, R_OUT), labels=(0, 1, 2)
    )
    assert labels.shape == (mesh.n_cells,)
    assert np.all(labels >= 0)  # все ячейки помечены
    assert set(np.unique(labels)) == {0, 1, 2}
    assert int((labels == 0).sum() + (labels == 1).sum() + (labels == 2).sum()) == mesh.n_cells
    # пояса радиально разделены: средний радиус центроидов растёт с меткой
    rad = np.array([np.hypot(*mesh.cell_centroid(c)[:2]) for c in range(mesh.n_cells)])
    assert rad[labels == 0].mean() < rad[labels == 1].mean() < rad[labels == 2].mean()
