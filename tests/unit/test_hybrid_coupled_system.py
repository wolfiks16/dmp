from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import (
    assemble_coupled_block_system,
    assemble_coupling_block,
    assemble_exterior_steklov_poincare,
)
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def _build(n: int):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    return mesh, ci, vs, ss


def test_coupled_block_system_shape_symmetry_and_zero_rhs() -> None:
    _, ci, vs, ss = _build(1)
    sysm = assemble_coupled_block_system(ci, vs, ss, nu=1.0, j_fn=_zero_j, mu0=1.0)

    n = sysm.n_a + sysm.n_p + sysm.n_psi + sysm.n_lam
    assert sysm.matrix.shape == (n, n)
    assert sysm.n_psi == ci.n_phi_dofs
    assert sysm.n_lam == ci.n_flux_dofs

    # Симметрия по построению (свойство Costabel — самосопряжённость для adjoint).
    assert np.allclose(sysm.matrix, sysm.matrix.T, atol=1e-10)
    # Нет источника при J=0 и без намагниченности.
    assert np.allclose(sysm.rhs, 0.0)


def test_coupled_block_placement_matches_component_blocks() -> None:
    _, ci, vs, ss = _build(1)
    cfg = AdaptiveIntegrationConfig()
    sysm = assemble_coupled_block_system(ci, vs, ss, nu=1.0, j_fn=_zero_j, mu0=1.0, config=cfg)
    m = sysm.matrix
    opsi, olam = sysm.off_psi, sysm.off_lam

    # Блок связи A↔ψ совпадает с независимо собранным B_FΓ.
    b_fg = assemble_coupling_block(ci, vs)
    assert np.allclose(m[: sysm.n_a, opsi:olam], b_fg, atol=1e-12)
    # Внешний 2-блок входит со знаком МИНУС: ψ-диагональ = −μ₀W, λ-диагональ = +V
    # (даёт K_eff=K+B_FΓ S_ext⁻¹ B_FΓᵀ ⇒ размагничивание; подтверждено сферой H_in=−M/3).
    ext = assemble_exterior_steklov_poincare(ci, cfg, mu0=1.0)
    assert np.allclose(m[opsi:olam, opsi:olam], -ext.W, atol=1e-12)
    assert np.allclose(m[olam:, olam:], ext.V, atol=1e-12)


def test_coupled_exterior_schur_equals_minus_steklov_poincare() -> None:
    # Ключевая проверка: дополнение Шура блок-системы на ψ (исключая λ) равно
    # −S_ext (внешний 2-блок входит со знаком минус). Это и даёт физически верное
    # K_eff = K + B_FΓ S_ext⁻¹ B_FΓᵀ (реакция добавляет жёсткость ⇒ размагничивание;
    # подтверждено бенчмарком намагниченной сферы H_in=−M/3). Подтверждает расстановку
    # внешних блоков и согласованность знака C_ΓB.
    _, ci, vs, ss = _build(2)
    cfg = AdaptiveIntegrationConfig()
    sysm = assemble_coupled_block_system(ci, vs, ss, nu=1.0, j_fn=_zero_j, mu0=1.0, config=cfg)
    m = sysm.matrix
    opsi, olam = sysm.off_psi, sysm.off_lam

    m_pp = m[opsi:olam, opsi:olam]
    m_pl = m[opsi:olam, olam:]
    m_ll = m[olam:, olam:]
    m_lp = m[olam:, opsi:olam]
    schur = m_pp - m_pl @ np.linalg.solve(m_ll, m_lp)

    ext = assemble_exterior_steklov_poincare(ci, cfg, mu0=1.0)
    assert np.allclose(schur, -ext.S_ext, atol=1e-9)
