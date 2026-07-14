import math

import numpy as np
import pytest

from magcore.fem2d.assembly import (
    assemble_current_rhs,
    assemble_current_rhs_piecewise,
)
from magcore.fem2d.machines.excitation import (
    phase_currents,
    worst_case_d_axis_currents,
)
from magcore.fem2d.mesh_generators import build_structured_rectangle_tri_mesh
from magcore.fem2d.spaces import LagrangeP1Space2D


# --- фазные токи (не требуют геометрии/gmsh) ---

def test_phase_currents_balanced_zero_sum_and_amplitude():
    for g in np.linspace(0.0, 2.0 * math.pi, 13):
        i = phase_currents(5.0, g)
        assert abs(float(i.sum())) < 1e-12                     # нет нулевой последовательности
        # Амплитуда пространственного вектора (Кларк, amplitude-invariant) = 1.5·i_peak.
        alpha = i[0] - 0.5 * i[1] - 0.5 * i[2]
        beta = (math.sqrt(3.0) / 2.0) * (i[1] - i[2])
        assert math.isclose(math.hypot(alpha, beta), 1.5 * 5.0, rel_tol=1e-9)


def test_phase_currents_gamma_zero():
    assert np.allclose(phase_currents(2.0, 0.0), [2.0, -1.0, -1.0])


def test_worst_case_is_negative_d_axis():
    # I_d = -i_peak, I_q = 0  ⇒  [-I, I/2, I/2] (МДС статора против магнита).
    assert np.allclose(worst_case_d_axis_currents(3.0), [-3.0, 1.5, 1.5])


# --- кусочно-постоянный токовый ассемблер (оракулы без gmsh) ---

def test_current_rhs_piecewise_partition_of_unity():
    # Σ_i f_i = ∫ J_z dx (т.к. Σ_i φ_i ≡ 1): равномерный J_z → J_z·площадь.
    mesh = build_structured_rectangle_tri_mesh(4, 3, x1=2.0, y1=1.5)
    space = LagrangeP1Space2D(mesh)
    f = assemble_current_rhs_piecewise(space, np.full(mesh.n_cells, 3.0))
    assert math.isclose(float(f.sum()), 3.0 * (2.0 * 1.5), rel_tol=1e-12)


def test_current_rhs_piecewise_matches_continuous_constant():
    # Для постоянного J_z оба ассемблера точны ⇒ должны совпасть поэлементно.
    mesh = build_structured_rectangle_tri_mesh(5, 5)
    space = LagrangeP1Space2D(mesh)
    f_pw = assemble_current_rhs_piecewise(space, np.full(mesh.n_cells, 2.5))
    f_c = assemble_current_rhs(space, lambda x: 2.5, quadrature_order=2)
    assert np.allclose(f_pw, f_c, atol=1e-12)


def test_current_rhs_piecewise_shape_check():
    mesh = build_structured_rectangle_tri_mesh(2, 2)
    space = LagrangeP1Space2D(mesh)
    with pytest.raises(ValueError):
        assemble_current_rhs_piecewise(space, np.zeros(mesh.n_cells + 1))


# --- обмотка → J_z в реальной геометрии (главный физический оракул) ---

def test_slot_current_density_ampere_turns():
    pytest.importorskip("gmsh")
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        star_of_slots_layout,
    )
    from magcore.fem2d.machines.excitation import slot_current_density

    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.0018))
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    N = 20.0
    pc = np.array([7.0, -3.0, -4.0])          # произвольные фазные токи
    jz = slot_current_density(g, lay, pc, N)

    # Вне пазов тока нет.
    assert np.all(jz[g.slot_id < 0] == 0.0)
    # ∫_s J_z dA = sign_s · N · I_{phase_s} — заданные А·витки со знаком фазы.
    areas = np.array([g.mesh.cell_area(c) for c in range(g.mesh.n_cells)])
    for s in range(g.params.n_slots):
        cells = g.slot_id == s
        at = float((jz[cells] * areas[cells]).sum())
        expect = lay.sign_of_slot[s] * N * pc[lay.phase_of_slot[s]]
        assert math.isclose(at, expect, rel_tol=1e-9, abs_tol=1e-9)


def test_slot_current_density_balanced_net_zero():
    # Сбалансированные токи (сумма фаз = 0) ⇒ суммарные А·витки по всем пазам = 0.
    pytest.importorskip("gmsh")
    from magcore.fem2d.machines import (
        OutrunnerPMSMParams,
        build_outrunner_spm_pmsm,
        star_of_slots_layout,
        winding_current_density,
    )

    g = build_outrunner_spm_pmsm(OutrunnerPMSMParams(mesh_size=0.0018))
    lay = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    jz = winding_current_density(g, lay, i_peak=10.0, gamma_elec=0.7, turns_per_slot=15.0)
    areas = np.array([g.mesh.cell_area(c) for c in range(g.mesh.n_cells)])
    total_at = float((jz * areas).sum())
    assert abs(total_at) < 1e-7
