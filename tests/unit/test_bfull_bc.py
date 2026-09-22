import numpy as np

from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.assembly import assemble_stiffness
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams
from magcore.fem2d.machines.scenario import machine_scenario
from magcore.fem2d.machines.thermal_scenario import MachineThermalProperties
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.solver import solve_scalar
from magcore.fem2d.spaces import LagrangeP1Space2D
from magcore.fem2d.thermal import (
    ImplicitEulerThermalStepper,
    assemble_robin_multi,
    assemble_source_rhs,
    classify_boundary_edges,
)

# P-B1: дифференцированные ГУ конвекции (h_in≠h_out, разные T_ref). Оракулы:
#  (1) РЕГРЕССИЯ — дифф. ГУ с одинаковыми (h, T_ref) на всей границе ≡ скалярное однородное;
#  (2) ЭНЕРГОБАЛАНС точен и при дифф. ГУ (ΔЗапас/dt = Источник − Отвод);
#  (3) классификация границ машины на внутр.(расточка)/наружн.(ротор);
#  (4) ФИЗИКА (стационар) — сильный h_in (статор→рама) даёт МЕНЬШУЮ установившуюся T меди,
#      чем слабое однородное h. Эффект ГУ — стационарный (за короткий транзиент медь греется
#      своим I²R быстрее, чем охлаждение рамы доходит до горячей точки в пазах).


def _rect_space(n=8):
    mesh = build_structured_rectangle_tri_mesh(n, n, x0=-1.0, x1=1.0, y0=-1.0, y1=1.0)
    return mesh, LagrangeP1Space2D(mesh)


def test_multi_robin_reduces_to_uniform():
    # Дифф. ГУ, где ВСЯ граница = один регион с (h, T_amb), должно совпасть со скалярным путём.
    mesh, space = _rect_space()
    nc = mesh.n_cells
    k = np.full(nc, 1.5)
    c = np.full(nc, 1.0e5)
    q = np.full(nc, 200.0)
    h, T_amb = 12.0, 20.0

    s_uniform = ImplicitEulerThermalStepper(space, k, c, dt=1.0, h=h, T_amb=T_amb)
    R, rhs = assemble_robin_multi(space, [(list(mesh.boundary_edges()), h, T_amb)])
    s_multi = ImplicitEulerThermalStepper(space, k, c, dt=1.0, robin=(R, rhs))

    Tu = np.full(space.ndofs, float(T_amb))
    Tm = Tu.copy()
    for _ in range(6):
        Tu = s_uniform.step(Tu, q)
        Tm = s_multi.step(Tm, q)
    assert np.allclose(Tu, Tm, atol=1e-12)
    assert abs(s_uniform.convective_outflow(Tu) - s_multi.convective_outflow(Tm)) < 1e-9
    assert abs(s_uniform.stored_energy(Tu) - s_multi.stored_energy(Tm)) < 1e-9


def test_differentiated_bc_energy_balance_exact():
    # Дифф. ГУ: левая/правая половины границы с РАЗНЫМИ (h, T_ref). Дискретный энергобаланс
    # неявного Эйлера точен: (ΔЗапас)/dt = ∫q − Отвод (1ᵀK T=0, 1ᵀf=источник).
    mesh, space = _rect_space(10)
    nc = mesh.n_cells
    k = np.full(nc, 2.0)
    c = np.full(nc, 2.0e5)
    q = np.full(nc, 400.0)

    left, right = [], []
    for i, j in mesh.boundary_edges():
        xm = 0.5 * (mesh.vertices[i][0] + mesh.vertices[j][0])
        (left if xm < 0.0 else right).append((i, j))
    R, rhs = assemble_robin_multi(space, [(left, 5.0, 15.0), (right, 20.0, 45.0)])

    dt = 0.5
    s = ImplicitEulerThermalStepper(space, k, c, dt=dt, robin=(R, rhs))
    areas = np.array([mesh.cell_area(ci) for ci in range(nc)])
    src = float((q * areas).sum())

    T = np.full(space.ndofs, 20.0)
    prev = s.stored_energy(T)
    for _ in range(8):
        T = s.step(T, q)
        store = s.stored_energy(T)
        out = s.convective_outflow(T)
        d_store = (store - prev) / dt
        assert abs(d_store - (src - out)) / max(abs(src), 1.0) < 1e-9
        prev = store


def test_classify_boundary_edges_on_machine():
    sc = machine_scenario(
        OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.004),
        n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve(),
    )
    space = LagrangeP1Space2D(sc.geometry.mesh)
    p = sc.geometry.params
    inner, outer = classify_boundary_edges(space, p.R_bore, p.R_out)
    assert len(inner) > 0 and len(outer) > 0

    def rad(e):
        m = 0.5 * (space.mesh.vertices[e[0]] + space.mesh.vertices[e[1]])
        return float(np.hypot(m[0], m[1]))

    # Внутренние рёбра строго ближе к оси, чем наружные; средние радиусы ≈ R_bore / R_out.
    assert max(rad(e) for e in inner) < min(rad(e) for e in outer)
    assert abs(np.mean([rad(e) for e in inner]) - p.R_bore) < 0.003
    assert abs(np.mean([rad(e) for e in outer]) - p.R_out) < 0.003


def test_frame_cooling_reduces_steady_copper_temperature():
    # ФИЗИКА P-B1 (стационар): источник тепла в меди (пазах); сильный h_in (статор→рама) —
    # реальный сток тепла статора ⇒ МЕНЬШАЯ установившаяся T меди, чем при слабом однородном h.
    sc = machine_scenario(
        OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.004),
        n42sh_magnet((1.0, 0.0, 0.0)), m270_35a_bh_curve(),
    )
    g = sc.geometry
    space = LagrangeP1Space2D(g.mesh)
    k, _ = MachineThermalProperties.representative().cell_fields(g)
    K = assemble_stiffness(space, k)

    q = np.where(g.slot_id >= 0, 5.0e6, 0.0)      # тепло только в меди (пазах) [Вт/м³]
    f = assemble_source_rhs(space, q)
    inner, outer = classify_boundary_edges(space, g.params.R_bore, g.params.R_out)

    # (a) слабое однородное h на всей границе
    Ru, rhsu = assemble_robin_multi(space, [(list(g.mesh.boundary_edges()), 20.0, 40.0)])
    Tu = solve_scalar(K + Ru, f + rhsu)
    # (b) сильный сток статора в раму (h_in) + слабый обдув ротора (h_out)
    Rf, rhsf = assemble_robin_multi(space, [(inner, 800.0, 40.0), (outer, 20.0, 40.0)])
    Tf = solve_scalar(K + Rf, f + rhsf)

    assert Tf.max() < Tu.max()                     # сток в раму снижает пик T меди
    assert Tf.max() > 40.0                          # но не ниже среды (санити)
