from __future__ import annotations

import numpy as np

from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.assembly import assemble_coupled_block_system
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.solver import (
    coupled_nullspace_dim,
    interior_p_integral_vector,
    solve_coupled_block_system,
    solve_coupled_min_norm,
)
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _zero_j(_x):
    return np.zeros(3)


def _coupled_with_source(n: int):
    mesh = build_structured_unit_cube_tetra_mesh(n)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    # Совместимый источник: только в a-блоке (⊥ нуль-моде «константа p»).
    src = np.random.default_rng(0).standard_normal(vs.ndofs)
    coupled = assemble_coupled_block_system(
        ci, vs, ss, nu=1.0, j_fn=_zero_j, extra_vector_rhs=src, mu0=1.0
    )
    return coupled, ss


def test_coupled_nullspace_is_one_dimensional() -> None:
    # Диагностика C (SVD): на стягиваемом кубе ядро ровно 1-мерно (константа p).
    coupled, _ = _coupled_with_source(2)
    assert coupled_nullspace_dim(coupled) == 1


def test_coupled_solver_A_zero_mean_gauge() -> None:
    coupled, ss = _coupled_with_source(2)
    sol = solve_coupled_block_system(coupled, ss)

    # Бордюр снял сингулярность: малый невязка, множитель μ≈0 (совместность RHS).
    assert sol.residual_norm < 1e-8
    assert abs(sol.gauge_multiplier) < 1e-8
    # Калибровка ∫_Ω p = 0 выполнена.
    m_p = interior_p_integral_vector(ss)
    denom = float(np.linalg.norm(m_p) * np.linalg.norm(sol.p)) + 1.0
    assert abs(float(m_p @ sol.p)) / denom < 1e-9


def test_coupled_solver_A_matches_min_norm_C_on_physics() -> None:
    # A (бордюр ∫p=0) и C (min-norm) дают ИДЕНТИЧНУЮ физику (a,ψ,λ); различается лишь
    # калибровка множителя p (∫p=0 у A vs Σp=0 у C) — подтверждает инвариантность
    # физики к выбору калибровочной константы.
    coupled, ss = _coupled_with_source(2)
    sol = solve_coupled_block_system(coupled, ss)
    a_c, p_c, psi_c, lam_c = solve_coupled_min_norm(coupled)

    assert np.allclose(sol.a, a_c, atol=1e-7)
    assert np.allclose(sol.psi, psi_c, atol=1e-7)
    assert np.allclose(sol.lam, lam_c, atol=1e-7)
    # p различается нормировкой калибровки (это ожидаемо, не баг).
    assert not np.allclose(sol.p, p_c, atol=1e-7) or np.linalg.norm(sol.p) < 1e-12
