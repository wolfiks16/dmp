import numpy as np

from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import (
    assemble_capacity,
    solve_thermal,
    solve_thermal_transient,
)

# S3, инкремент 1: нестационарный тепловой решатель C·∂T/∂t − div(k∇T)=q + конвекция.
# ОРАКУЛ (аналитика, не «работает без ошибки»): почти изотермичное тело (Bi=hL/k≪1) ведёт
# себя как СОСРЕДОТОЧЕННАЯ ёмкость — C_tot·dT/dt = q·V − h·P·(T−T_amb), решение
# T(t)=T_amb+(T_ss−T_amb)(1−e^{−t/τ}), T_ss−T_amb=qV/(hP), τ=cV/(hP). Здесь V,P берём из
# самой сетки (∫area, ∫граница) ⇒ независимо от домена. Плюс кросс-проверка: t→∞ сходится к
# независимому СТАЦИОНАРНОМУ решателю solve_thermal.


def _geom(mesh):
    V = float(sum(mesh.cell_area(c) for c in range(mesh.n_cells)))
    P = float(sum(np.linalg.norm(mesh.vertices[i] - mesh.vertices[j])
                  for i, j in mesh.boundary_edges()))
    return V, P


def test_capacity_matrix_row_sums_to_c_times_area():
    # ∫ c φ_i dx суммарно = c·V (строчные суммы матрицы ёмкости = ∫ c φ_i, их сумма = c·|Ω|).
    mesh = build_structured_rectangle_tri_mesh(6, 6)
    space = LagrangeP1Space2D(mesh)
    C = assemble_capacity(space, 3.0)
    V, _ = _geom(mesh)
    assert abs(float(C.sum()) - 3.0 * V) < 1e-9


def test_transient_matches_lumped_capacitance():
    mesh = build_structured_rectangle_tri_mesh(10, 10)
    space = LagrangeP1Space2D(mesh)
    V, P = _geom(mesh)
    k, c, q, h, Tamb = 1.0e5, 800.0, 4.0e4, 50.0, 20.0   # k велико ⇒ Bi≪1 (изотермично)
    dt, nsteps = 0.1, 250

    times, Th = solve_thermal_transient(
        space, k, c, source=np.full(mesh.n_cells, q), dt=dt, n_steps=nsteps, h=h, T_amb=Tamb)
    Tmean = Th.mean(axis=1)

    T_ss = Tamb + q * V / (h * P)
    tau = c * V / (h * P)
    T_lump = Tamb + (T_ss - Tamb) * (1.0 - np.exp(-times / tau))

    dT = T_ss - Tamb
    assert np.max(np.abs(Tmean - T_lump)) < 0.03 * dT     # вся кривая ≈ аналитике
    assert abs(Tmean[-1] - T_ss) < 0.01 * dT              # выход на установившееся
    assert np.all(np.diff(Tmean) > 0)                     # монотонный нагрев
    assert Th[0].std() < 1e-9                             # старт с однородного T_amb

    # кросс-проверка: t→∞ = независимый стационарный решатель.
    T_steady = solve_thermal(space, k, source=np.full(mesh.n_cells, q), h=h, T_amb=Tamb)
    assert abs(Tmean[-1] - float(T_steady.mean())) < 0.005 * dT


def test_no_source_stays_at_ambient():
    # Без источника и с T0=T_amb поле остаётся T_amb (равновесие).
    mesh = build_structured_rectangle_tri_mesh(6, 6)
    space = LagrangeP1Space2D(mesh)
    times, Th = solve_thermal_transient(
        space, 50.0, 800.0, source=np.zeros(mesh.n_cells), dt=1.0, n_steps=5, h=20.0, T_amb=30.0)
    assert np.max(np.abs(Th - 30.0)) < 1e-9
