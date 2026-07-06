from __future__ import annotations

import numpy as np

from magcore.bem.adaptive_single_layer import AdaptiveIntegrationConfig
from magcore.hybrid.assembly import (
    assemble_exterior_steklov_poincare,
    assemble_interface_mass,
)
from magcore.hybrid.interface import CouplingInterface
from magcore.mesh.mesh_generators import build_structured_unit_cube_tetra_mesh


def _interface(n: int) -> CouplingInterface:
    return CouplingInterface.from_tetra_mesh(build_structured_unit_cube_tetra_mesh(n))


def test_interface_mass_identities() -> None:
    ci = _interface(2)
    m = assemble_interface_mass(ci)

    assert m.shape == (ci.n_phi_dofs, ci.n_flux_dofs)
    # Σ_m M_PG[m,f] = ∫_{T_f} Σλ_m = A_f (для каждой грани f).
    assert np.allclose(m.sum(axis=0), ci.face_areas, atol=1e-12)
    # Полная сумма = площадь Γ (единичный куб: 6).
    assert np.isclose(m.sum(), float(ci.face_areas.sum()), atol=1e-12)
    # Ровно 3 ненуля на столбец (3 вершины грани).
    assert np.all((m != 0.0).sum(axis=0) == 3)


def test_exterior_steklov_poincare_sign_is_pinned_by_spd() -> None:
    # Закрепляет знак Кальдерона C_ΓB = ½M_PG − Kᵀ (k_sign=-1).
    # Внешняя DtN положительно ОПРЕДЕЛЕНА (нет ядра констант): постоянный след даёт
    # ненулевой убывающий внешний поток. Правильный знак ⇒ S_ext SPD и S_ext·1 ≠ 0.
    ci = _interface(2)
    cfg = AdaptiveIntegrationConfig()
    ext = assemble_exterior_steklov_poincare(ci, cfg, mu0=1.0, k_sign=-1.0)
    s = ext.S_ext
    nv = s.shape[0]

    assert s.shape == (ci.n_phi_dofs, ci.n_phi_dofs)
    assert np.allclose(s, s.T, atol=1e-10)             # симметрия (Costabel)
    eig = np.linalg.eigvalsh(s)
    assert eig[0] > 1e-3                                 # SPD — нет ядра констант
    assert float(np.linalg.norm(s @ np.ones(nv))) > 1e-1  # S_ext·1 ≠ 0


def test_exterior_steklov_wrong_sign_has_spurious_constant_kernel() -> None:
    # Контроль: неверный знак (½I+K) даёт паразитное ядро констант (поведение
    # ВНУТРЕННЕЙ DtN) ⇒ S_ext·1 ≈ 0 и наименьшее с.з. ≈ 0. Это и есть дискриминатор.
    ci = _interface(2)
    cfg = AdaptiveIntegrationConfig()
    ext = assemble_exterior_steklov_poincare(ci, cfg, mu0=1.0, k_sign=+1.0)
    s = ext.S_ext
    nv = s.shape[0]

    eig = np.linalg.eigvalsh(s)
    assert eig[0] < 1e-6                                  # паразитное нулевое с.з.
    assert float(np.linalg.norm(s @ np.ones(nv))) < 1e-3  # константы в ядре
