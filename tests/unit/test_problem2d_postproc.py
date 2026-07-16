import numpy as np
import pytest

from magcore.constants import MU0
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.model import (
    Air,
    Problem2D,
    Region2D,
    flux_between_points,
    force_maxwell_band,
    interpolate_Az,
    magnetic_energy,
    operating_point,
    solve_problem2d,
    torque_arkkio,
)

# Общий пост-процессинг на Problem2D. Оракулы: энергия ∝ |B|² и ∝ L; поток антисимметричен
# и ∝ L; интерполяция точна в узле; и — эквивалентность машинному пост-процу (момент/раб.точка).


def _air_current(n, jval):
    mesh = build_structured_rectangle_tri_mesh(n, n)
    nc = mesh.n_cells
    cen = np.array([mesh.cell_centroid(c) for c in range(nc)])
    j = np.zeros(nc)
    j[(np.abs(cen[:, 0] - 0.5) < 0.18) & (np.abs(cen[:, 1] - 0.5) < 0.18)] = jval
    p = Problem2D(mesh=mesh, cell_region=np.zeros(nc, int),
                  regions={0: Region2D(0, "air", Air())}, j_cells=j)
    return solve_problem2d(p, max_iter=40)


def test_energy_scales_with_current_squared_and_length():
    s1 = _air_current(10, 1.0e5)
    s2 = _air_current(10, 2.0e5)
    e1 = magnetic_energy(s1, axial_length=1.0)
    assert e1 > 0.0
    assert np.isclose(magnetic_energy(s2, axial_length=1.0), 4.0 * e1, rtol=1e-6)  # линейно: W∝j²
    assert np.isclose(magnetic_energy(s1, axial_length=2.0), 2.0 * e1, rtol=1e-9)


def test_interpolate_at_node_exact():
    s = _air_current(6, 1.0e5)
    v = s.problem.mesh.vertices[5]
    assert np.isclose(interpolate_Az(s, (v[0], v[1])), s.field.a[5], atol=1e-9)


def test_flux_antisymmetric_and_scales():
    s = _air_current(10, 1.0e5)
    p1, p2 = (0.3, 0.4), (0.7, 0.6)
    f = flux_between_points(s, p1, p2, axial_length=1.0)
    assert np.isclose(flux_between_points(s, p2, p1, axial_length=1.0), -f)
    assert np.isclose(flux_between_points(s, p1, p2, axial_length=3.0), 3.0 * f)


@pytest.fixture(scope="module")
def pmsm_geom():
    pytest.importorskip("gmsh")
    from magcore.domain.magnet_model import n42sh_magnet
    from magcore.domain.steel_curves import m270_35a_bh_curve
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        star_of_slots_layout,
    )
    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.004))
    return (g, n42sh_magnet((1, 0, 0)), m270_35a_bh_curve(),
            star_of_slots_layout(g.params.n_slots, g.params.n_poles))


@pytest.fixture(scope="module")
def pmsm_solution(pmsm_geom):
    from magcore.fem2d.machines import pmsm_to_problem
    from magcore.fem2d.machines.excitation import winding_current_density
    g, magnet, steel, lay = pmsm_geom
    jz = winding_current_density(g, lay, i_peak=20.0, gamma_elec=np.deg2rad(45.0), turns_per_slot=40.0)
    return g, solve_problem2d(pmsm_to_problem(g, magnet, steel, T=20.0, j_cells=jz),
                              relaxation=0.1, max_iter=300)


@pytest.fixture(scope="module")
def pmsm_noload(pmsm_geom):
    from magcore.fem2d.machines import pmsm_to_problem
    g, magnet, steel, _ = pmsm_geom
    return g, solve_problem2d(pmsm_to_problem(g, magnet, steel, T=20.0), relaxation=0.1, max_iter=300)


def test_torque_matches_machine_postproc(pmsm_solution):
    from magcore.fem2d.machines.postproc import airgap_torque_arkkio
    g, sol = pmsm_solution
    L = g.params.axial_length
    t_gen = torque_arkkio(sol, g.params.R_s_out, g.params.R_mag_in, axial_length=L)
    t_mach = airgap_torque_arkkio(g, sol.B_cells, axial_length=L)
    assert np.isclose(t_gen, t_mach, rtol=1e-9)
    assert abs(t_gen) > 0.1                                   # нагруженный момент существен


def test_operating_point_matches_risk(pmsm_solution):
    g, sol = pmsm_solution
    op = operating_point(sol, axial_length=g.params.axial_length)
    assert np.allclose(op.H_op, sol.risk.H_par)               # та же рабочая точка, что risk-map
    assert op.total_volume > 0.0
    assert op.worst_H_op() < 0.0                              # 2-й квадрант


def test_force_symmetric_near_zero_and_scales(pmsm_noload):
    g, sol = pmsm_noload
    L, ri, ro = g.params.axial_length, g.params.R_s_out, g.params.R_mag_in
    Fx, Fy = force_maxwell_band(sol, ri, ro, axial_length=L)
    # Характерный масштаб радиальной тяги в кольце (несокращённая величина).
    cen = np.array([g.mesh.cell_centroid(c) for c in range(g.mesh.n_cells)])
    r = np.hypot(cen[:, 0], cen[:, 1])
    band = np.where((r >= ri) & (r <= ro))[0]
    B2 = sol.B_cells[band, 0] ** 2 + sol.B_cells[band, 1] ** 2
    areas = np.array([g.mesh.cell_area(int(c)) for c in band])
    scale = L / (MU0 * (ro - ri)) * float((0.5 * B2 * areas).sum())
    assert np.hypot(Fx, Fy) < 0.08 * scale                    # симметрия: вектор гасится
    Fx2, Fy2 = force_maxwell_band(sol, ri, ro, axial_length=2.0 * L)
    assert np.isclose(Fx2, 2.0 * Fx) and np.isclose(Fy2, 2.0 * Fy)  # ∝ осевой длине
