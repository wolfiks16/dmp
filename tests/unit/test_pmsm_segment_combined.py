"""
§8.7 — COMBINED Picard (сталь+магнит) на сегменте PMSM (ограниченный периодический FEM).

Кульминация фазы C: в ОДНОЙ задаче — нелинейная сталь ν(|B|) (ярмо) + магнит-источник
(радиальный N42SH) + воздушный зазор, на аннулярном секторе одного полюса с АНТИ-
периодическими разрезами (чередование N–S) и Dirichlet на r_in/r_out/z. Решается
ограниченным периодическим нелинейным Picard. Верификация: сходимость; нелинейность
стали активна; сталь — высокопроницаема (ν_steel ≪ ν_air); магнит гонит поток;
combined-решение отличается от линейно-стального (saturation реально влияет).

Единицы: нормировка μ₀=1 (ν относительная). Сталь ν_rel=μ₀·ν_SI(|B|); магнит ν=1/μ_rec;
воздух ν=1. Источник магнита = ν·B_r(20 °C)·ê_r (статич., безопасная точка — колено не
срабатывает, что верно для NdFeB при 20 °C; срабатывание колена проверено отдельно).
"""
from __future__ import annotations

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.femcore.periodic import build_periodic_with_dirichlet, match_periodic_vertices
from magcore.femcore.periodic_nonlinear import solve_periodic_nonlinear_mixed_picard
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.mesh.mesh_generators import (
    build_annular_sector_tetra_mesh,
    tag_sector_regions_by_radius,
)

TOL = 1e-7
R_IN, R_OUT, THETA, ZL = 1.0, 1.6, np.pi / 7.0, 0.3
STEEL, MAGNET, AIR = 0, 1, 2


def _setup():
    mesh = build_annular_sector_tetra_mesh(
        6, 3, 2, r_in=R_IN, r_out=R_OUT, theta_seg=THETA, z_len=ZL
    )
    vs, ss = NedelecP1Space.from_mesh(mesh), LagrangeP1Space(mesh)
    labels = tag_sector_regions_by_radius(
        mesh, band_edges=(R_IN, 1.2, 1.4, R_OUT), labels=(STEEL, MAGNET, AIR)
    )
    # радиальная ось по ячейке (в плоскости xy)
    axis = np.zeros((mesh.n_cells, 3))
    for c in range(mesh.n_cells):
        cen = mesh.cell_centroid(c)
        r = np.hypot(cen[0], cen[1])
        axis[c] = [cen[0] / r, cen[1] / r, 0.0]
    return mesh, vs, ss, labels, axis


def _periodic_dirichlet_maps(mesh, vs, ss):
    V = mesh.vertices
    n_a = vs.ndofs
    e2d = vs.edge_to_dof_map()
    rr = np.hypot(V[:, 0], V[:, 1])

    def on_dir_vertex(v):
        return (
            abs(rr[v] - R_IN) < 1e-6 or abs(rr[v] - R_OUT) < 1e-6
            or abs(V[v, 2]) < 1e-9 or abs(V[v, 2] - ZL) < 1e-9
        )

    def edge_in_dir(a, b):
        for val in (R_IN, R_OUT):
            if abs(rr[a] - val) < 1e-6 and abs(rr[b] - val) < 1e-6:
                return True
        for val in (0.0, ZL):
            if abs(V[a, 2] - val) < 1e-9 and abs(V[b, 2] - val) < 1e-9:
                return True
        return False

    dirichlet_global: set[int] = set()
    for v in range(V.shape[0]):
        if on_dir_vertex(v):
            dirichlet_global.add(n_a + v)
    for (a, b), d in e2d.items():
        if edge_in_dir(a, b):
            dirichlet_global.add(int(d))

    c, s = np.cos(THETA), np.sin(THETA)
    vmap = match_periodic_vertices(
        mesh,
        lambda p: abs(p[1]) < 1e-7 and p[0] > 0.0,
        lambda p: abs(np.arctan2(p[1], p[0]) - THETA) < 1e-7,
        lambda p: np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]),
        tol=1e-7,
    )
    # АНТИ-период (один полюс, чередование N–S)
    return build_periodic_with_dirichlet(vs, ss, vmap, dirichlet_global, antiperiodic=True)


def _nu_of_B_factory(labels, curve):
    nu_air, nu_mag = 1.0, 1.0 / n42sh_magnet([0, 0, 1]).mu_rec
    steel = labels == STEEL
    magnet = labels == MAGNET

    def nu_of_B(B):
        nu = np.ones(labels.size)
        nu[magnet] = nu_mag
        mag = np.linalg.norm(np.asarray(B), axis=1)
        nu_steel = MU0 * np.array([curve.nu_chord(float(b)) for b in mag])  # ν_rel=μ₀·ν_SI
        nu[steel] = nu_steel[steel]
        nu[~(steel | magnet)] = nu_air
        return nu

    return nu_of_B


def _static_magnet(labels, axis):
    m = n42sh_magnet([0, 0, 1])
    nu_rec = 1.0 / m.mu_rec
    br = m.Br(20.0)  # безопасная точка
    mag = np.zeros((labels.size, 3))
    mag[labels == MAGNET] = (nu_rec * br) * axis[labels == MAGNET]
    return mag


def _solve(nu_of_B, magnet_src, mesh, vs, ss, red, dir_red):
    curve = m270_35a_bh_curve()
    nu0 = np.where(  # стартовая ν: сталь — начальная (low-B), иначе из nu_of_B@B=0
        True, nu_of_B(np.zeros((mesh.n_cells, 3))), 0.0
    )
    return solve_periodic_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=nu_of_B, nu_init=nu0,
        reduction=red, dirichlet_reduced_dofs=dir_red,
        magnetization=magnet_src, relaxation=0.5, tol=1e-6, max_iter=120,
    )


def test_pmsm_segment_combined_steel_magnet_converges() -> None:
    mesh, vs, ss, labels, axis = _setup()
    red, dir_red = _periodic_dirichlet_maps(mesh, vs, ss)
    curve = m270_35a_bh_curve()
    nu_of_B = _nu_of_B_factory(labels, curve)
    magnet_src = _static_magnet(labels, axis)

    res = _solve(nu_of_B, magnet_src, mesh, vs, ss, red, dir_red)

    steel = labels == STEEL
    magnet = labels == MAGNET
    assert res.converged, res.rel_change_history[-3:] if res.rel_change_history else None

    # магнит гонит поток: |B| в магните и ярме заметно ненулевой
    bmag = np.linalg.norm(res.B_cells, axis=1)
    assert bmag[magnet].mean() > 1e-2
    assert bmag[steel].mean() > 1e-2

    # сталь высокопроницаема: ν_steel ≪ ν_air(=1)
    assert res.nu_cells[steel].max() < 0.5
    # нелинейность стали активна: ν варьируется по ячейкам ярма
    assert res.nu_cells[steel].max() > res.nu_cells[steel].min() * (1.0 + 1e-3)


def test_pmsm_segment_nonlinear_steel_differs_from_linear() -> None:
    """combined нелинейная сталь даёт иное поле, чем линейная (saturation влияет)."""
    mesh, vs, ss, labels, axis = _setup()
    red, dir_red = _periodic_dirichlet_maps(mesh, vs, ss)
    curve = m270_35a_bh_curve()
    magnet_src = _static_magnet(labels, axis)
    steel = labels == STEEL

    nl = _solve(_nu_of_B_factory(labels, curve), magnet_src, mesh, vs, ss, red, dir_red)

    # линейная сталь: ν заморожена на начальной (low-B) — const
    nu_lin = nl.nu_cells.copy()
    nu_lin_steel0 = MU0 * curve.nu_chord(1e-6)
    nu_lin[steel] = nu_lin_steel0

    def nu_const(B):
        return nu_lin

    lin = solve_periodic_nonlinear_mixed_picard(
        mesh, vs, ss, nu_of_B=nu_const, nu_init=nu_lin,
        reduction=red, dirichlet_reduced_dofs=dir_red,
        magnetization=magnet_src, relaxation=1.0, tol=1e-9, max_iter=5,
    )
    assert lin.converged
    # нелинейное и линейное решения различаются (saturation реально меняет поле)
    assert not np.allclose(nl.a, lin.a, atol=1e-6)
