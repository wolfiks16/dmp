from __future__ import annotations

import numpy as np

from magcore.femcore.assembly import assemble_magnetization_rhs
from magcore.femcore.post import evaluate_curl_on_cell
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.solver import solve_coupled_block_system
from magcore.mesh.mesh_generators import build_ball_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def test_uniformly_magnetized_sphere_demagnetizing_factor() -> None:
    """
    Канонический бенчмарк НОВИЗНЫ (1): равномерно намагниченная сфера в открытой
    области (FEM-магнит ↔ BEM-убывающий экстерьер). Аналитика (μ₀=1, ν=1, B_r=M):
        H_in = -M/3  (размагничивающий фактор сферы 1/3),  B_in = 2M/3, изотропно.
    Первая ФИЗИЧЕСКАЯ проверка всей связанной системы (структурные тесты не ловят
    относительный знак внешнего блока — его фиксирует именно этот бенчмарк).
    """
    m_vec = np.array([0.0, 0.0, 1.0])
    mesh = build_ball_tetra_mesh(3, radius=1.0)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)

    nu_br = np.tile(m_vec, (mesh.n_cells, 1))
    f_br = assemble_magnetization_rhs(mesh, vs, nu_br)
    coupled = assemble_coupled_block_system(
        ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=f_br, mu0=1.0
    )
    sol = solve_coupled_block_system(coupled, ss)

    assert sol.residual_norm < 1e-8
    assert abs(sol.gauge_multiplier) < 1e-8

    b_cells = np.array([evaluate_curl_on_cell(vs, sol.a, c) for c in range(mesh.n_cells)])
    vols = np.array([mesh.cell_volume(c) for c in range(mesh.n_cells)])
    b_mean = (b_cells * vols[:, None]).sum(axis=0) / vols.sum()
    h_mean = b_mean - m_vec

    # Размагничивающий фактор 1/3: H_in,z = -1/3, B_in,z = 2/3.
    assert abs(h_mean[2] + 1.0 / 3.0) < 5e-3
    assert abs(b_mean[2] - 2.0 / 3.0) < 5e-3
    # Поперечные компоненты ≈ 0 (изотропия отклика сферы).
    assert abs(h_mean[0]) < 1e-2
    assert abs(h_mean[1]) < 1e-2

    # Однородность внутреннего поля (классический результат для сферы).
    hz = b_cells[:, 2] - 1.0
    hz_mean = float((hz * vols).sum() / vols.sum())
    hz_std = float(np.sqrt((vols * (hz - hz_mean) ** 2).sum() / vols.sum()))
    assert hz_std < 0.03


def test_permeable_sphere_in_uniform_field() -> None:
    """
    Бенчмарк (2) НОВИЗНЫ (1): проницаемая сфера (контраст μ_r) в однородном поле H₀=ẑ.
    Тестирует приложенное поле (расщепление ψ=ψ_app+ψ_scat, coupling_block_system §11)
    + контраст проницаемости. Аналитика (μ_out=1): H_in = 3/(μ_r+2)·H₀ равномерно;
    μ_r=1 ⇒ H_in=H₀ (поле не возмущено — sanity, что приложенное поле подано верно).
    """
    h0 = np.array([0.0, 0.0, 1.0])
    for mu_r, expect_hz in ((1.0, 1.0), (2.0, 0.75)):
        mesh = build_ball_tetra_mesh(3, radius=1.0)
        ci = CouplingInterface.from_tetra_mesh(mesh)
        vs = NedelecP1Space.from_mesh(mesh)
        ss = LagrangeP1Space(mesh)
        coupled = assemble_coupled_block_system(
            ci, vs, ss, nu=1.0 / mu_r, j_fn=_zero_j, applied_field_h0=h0, mu0=1.0
        )
        sol = solve_coupled_block_system(coupled, ss)
        assert sol.residual_norm < 1e-8

        b_cells = np.array([evaluate_curl_on_cell(vs, sol.a, c) for c in range(mesh.n_cells)])
        vols = np.array([mesh.cell_volume(c) for c in range(mesh.n_cells)])
        b_mean = (b_cells * vols[:, None]).sum(axis=0) / vols.sum()
        h_in = b_mean / mu_r  # H = ν B, ν = 1/μ_r

        assert abs(h_in[2] - expect_hz) < 2e-2
        assert abs(h_in[0]) < 1e-2
        assert abs(h_in[1]) < 1e-2
