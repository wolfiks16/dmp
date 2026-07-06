from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.hybrid.assembly import CoupledBlockSystem


@dataclass(frozen=True)
class CoupledSolution:
    """Решение связанной задачи: внутренние (a,p) и следы (ψ,λ) + диагностика."""

    a: np.ndarray            # Неделек-дофы A
    p: np.ndarray            # множитель Кулоновой калибровки
    psi: np.ndarray          # внешний след ψ (P1 на Γ)
    lam: np.ndarray          # внешний поток λ=∂ψ/∂n (P0 на Γ)
    gauge_multiplier: float  # μ бордюра ∫p=0 (должен быть ≈0 — признак совместности)
    residual_norm: float     # ‖M x − b‖


def interior_p_integral_vector(scalar_space: LagrangeP1Space) -> np.ndarray:
    """
    Вектор калибровки множителя `∫_Ω p dV = 0`:
        m_p[m] = ∫_Ω λ_m dV = Σ_{ячеек c ∋ m} vol(c)/4   (P1, ∫_tet λ_вершины = vol/4).
    """
    mesh = scalar_space.mesh
    m_p = np.zeros(scalar_space.ndofs, dtype=float)
    for c in range(mesh.n_cells):
        cv = np.asarray(mesh.cell_vertices(c), dtype=float)
        vol = abs(float(np.linalg.det(np.stack([cv[1] - cv[0], cv[2] - cv[0], cv[3] - cv[0]])))) / 6.0
        for vdof in scalar_space.cell_dof_indices(c):
            m_p[int(vdof)] += vol / 4.0
    return m_p


def solve_coupled_block_system(
    coupled: CoupledBlockSystem,
    scalar_space: LagrangeP1Space,
) -> CoupledSolution:
    """
    Решатель связанной симметричной системы — **вариант A**: снятие единственной
    нуль-моды (константа множителя `p`) симметричным бордюром калибровки `∫_Ω p = 0`:

        ⎡ M  c ⎤ ⎡x⎤   ⎡b⎤
        ⎣ cᵀ 0 ⎦ ⎣μ⎦ = ⎣0⎦,   c = (0, m_p, 0, 0)/‖·‖.

    Бордюр сохраняет симметрию (нарратив Costabel + самосопряжённый adjoint фазы E),
    масштабируем и **падает явно** при неожиданном расширении ядра (когомологии
    нестягиваемого мотора → нужны cuts). μ выходит ≈0 (RHS совместна: нуль-мода —
    чистая константа в p-блоке, ортогональна источнику в a-блоке), физика (a,ψ,λ)
    инвариантна к выбору калибровочной константы.
    """
    m = coupled.matrix
    b = coupled.rhs
    n = m.shape[0]

    c = np.zeros(n, dtype=float)
    c[coupled.off_p : coupled.off_psi] = interior_p_integral_vector(scalar_space)
    nrm = float(np.linalg.norm(c))
    if nrm == 0.0:
        raise ValueError("Gauge vector is zero; scalar space has no interior dofs.")
    c /= nrm  # единичная нормировка → бордюр не ухудшает обусловленность

    m_aug = np.zeros((n + 1, n + 1), dtype=float)
    m_aug[:n, :n] = m
    m_aug[:n, n] = c
    m_aug[n, :n] = c
    b_aug = np.zeros(n + 1, dtype=float)
    b_aug[:n] = b

    x_aug = np.linalg.solve(m_aug, b_aug)
    x = x_aug[:n]
    mu = float(x_aug[n])
    a, p, psi, lam = coupled.split(x)
    residual = float(np.linalg.norm(m @ x - b))
    return CoupledSolution(a=a, p=p, psi=psi, lam=lam, gauge_multiplier=mu, residual_norm=residual)


def coupled_nullspace_dim(coupled: CoupledBlockSystem, rtol: float = 1e-9) -> int:
    """
    Диагностика (**вариант C**): размерность ядра системы через SVD (число σ ≈ 0).
    На стягиваемой области (шар/куб) ожидается ровно 1 (константа `p`). Рост (мотор:
    когомологии) сигнализирует о необходимости физических разрезов (cuts) ДО того,
    как решатель A «упадёт».
    """
    sv = np.linalg.svd(coupled.matrix, compute_uv=False)
    return int(np.sum(sv < rtol * sv[0]))


def solve_coupled_min_norm(
    coupled: CoupledBlockSystem,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Кросс-проверка (**вариант C**): решение минимальной нормы (lstsq, ортогонально
    ядру). Совпадение физических полей (a,ψ,λ) с вариантом A подтверждает, что (i) ядро
    — это именно константа `p`, (ii) бордюр A не искажает физику. НЕ для продакшна
    (плотный SVD не масштабируется; на моторе тихо замаскировал бы когомологии).
    """
    x, _res, _rank, _sv = np.linalg.lstsq(coupled.matrix, coupled.rhs, rcond=None)
    return coupled.split(x)
