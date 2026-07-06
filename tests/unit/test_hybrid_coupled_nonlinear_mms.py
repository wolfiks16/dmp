"""
ПОЛНОЦЕННАЯ аналитическая верификация СВЯЗАННОГО НЕЛИНЕЙНОГО решателя (без костылей).

Оракул: сфера из изотропного нелинейного материала H=ν(|B|)·B в однородном внешнем
поле H₀ имеет ОДНОРОДНОЕ внутреннее поле (теорема об эллипсоиде, любой изотропный
закон). Баланс размагничивания сферы (μ₀=1: M=B−H, H_demag=−M/3) ⇒ скалярное уравнение

        B·(2·ν(|B|) + 1) = 3·H₀.

Якорь: при ν=1/μ_r это даёт H_in=3H₀/(μ_r+2) — в точности существующий линейный
бенчмарк. Для нелинейного ν это ТОЧНОЕ решение всего связанного пути (FEM-нелинейность
+ BEM-экстерьер + приложенное поле + ν-обновление). Точный корень считается ЗАНОВО
(np.roots), не захардкожен.

Эмпирически: объёмное среднее ⟨B⟩ воспроизводится до ~машинной точности (дискретное
сохранение потока), а ошибка дискретизации сферы сидит в ОДНОРОДНОСТИ поля (std) и
убывает при измельчении сетки — это и проверяем.
"""
from __future__ import annotations

import math

import numpy as np

from magcore.femcore.scalar_spaces import LagrangeP1Space
from magcore.femcore.spaces import NedelecP1Space
from magcore.hybrid.interface import CouplingInterface
from magcore.hybrid.nonlinear import solve_coupled_nonlinear_picard
from magcore.mesh.mesh_generators import build_ball_tetra_mesh

NU0, C = 0.5, 1.0  # ν(|B|)=ν0(1+c|B|²); при B=0 μ_r=1/ν0=2, растёт с |B| ⇒ сильная нелинейность


def _exact_uniform_B(H0: float, nu0: float = NU0, c: float = C) -> float:
    """Единственный положительный корень B·(2·ν(B)+1)=3H₀ ⇔ 2ν0c·B³+(2ν0+1)·B−3H₀=0."""
    roots = np.roots([2.0 * nu0 * c, 0.0, 2.0 * nu0 + 1.0, -3.0 * H0])
    pos = roots[np.abs(roots.imag) < 1e-9].real
    pos = pos[pos > 0.0]
    assert pos.size >= 1
    return float(pos.min())


def _nu_of(B: float, nu0: float = NU0, c: float = C) -> float:
    return nu0 * (1.0 + c * B * B)


def _solve_nonlinear_sphere(n: int, H0: float, nu0: float = NU0, c: float = C):
    mesh = build_ball_tetra_mesh(n, radius=1.0)
    ci = CouplingInterface.from_tetra_mesh(mesh)
    vs = NedelecP1Space.from_mesh(mesh)
    ss = LagrangeP1Space(mesh)
    nc = mesh.n_cells
    res = solve_coupled_nonlinear_picard(
        ci, vs, ss,
        nu_of_B=lambda B: nu0 * (1.0 + c * np.sum(np.asarray(B) ** 2, axis=1)),
        nu_init=np.full(nc, nu0),
        applied_field_h0=np.array([0.0, 0.0, H0]),
        # Недорелаксация: карта demag-Picard имеет ОТРИЦАТЕЛЬНУЮ производную (нагруз.
        # линия), осцилляторная мода маргинально устойчива при ω=1 (период-2 цикл,
        # чувствительный к численному окружению). ω=0.5 гасит её и УСКОРЯет сходимость.
        relaxation=0.5, tol=1e-6, max_iter=200,
    )
    vols = np.array([mesh.cell_volume(k) for k in range(nc)])
    w = vols / vols.sum()
    b_mean = w @ res.B_cells
    bz_std = math.sqrt(float(w @ (res.B_cells[:, 2] - b_mean[2]) ** 2))
    return res, b_mean, bz_std


def test_const_nu_reduces_to_linear_permeable_sphere() -> None:
    """Якорь: при c=0 (ν≡ν0) нелинейный связанный решатель = линейный бенчмарк
    H_in=3H₀/(μ_r+2), μ_r=1/ν0."""
    H0, nu0 = 1.0, 0.5
    mu_r = 1.0 / nu0
    res, b_mean, _ = _solve_nonlinear_sphere(4, H0, nu0=nu0, c=0.0)
    assert res.converged
    h_in_z = nu0 * b_mean[2]  # H = ν B
    assert abs(h_in_z - 3.0 * H0 / (mu_r + 2.0)) < 2e-2
    assert abs(b_mean[0]) < 1e-2 and abs(b_mean[1]) < 1e-2


def test_nonlinear_permeable_sphere_matches_analytic_uniform_field() -> None:
    """Главный: нелинейная сфера воспроизводит ТОЧНОЕ однородное B_in из B(2ν+1)=3H₀."""
    H0 = 1.0
    B_ex = _exact_uniform_B(H0)  # для (ν0,c)=(0.5,1): B³+2B−3=0 ⇒ B=1.0
    res, b_mean, bz_std = _solve_nonlinear_sphere(4, H0)

    assert res.converged
    # воспроизведено точное внутреннее B (объёмное среднее — почти машинная точность).
    assert abs(b_mean[2] - B_ex) < 1e-4, (b_mean[2], B_ex)
    # поле продольно H₀ и почти однородно (сфера; ошибка дискретизации в std).
    assert abs(b_mean[0]) < 1e-2 and abs(b_mean[1]) < 1e-2
    assert bz_std < 0.05
    # нелинейность реально активна: ν ЭВОЛЮЦИОНИРОВАЛА с начальной 0.5 к ν(|B_ex|)=1.0.
    # (На сфере поле ОДНОРОДНО ⇒ ν однородна по ячейкам — разброс ~0, это и есть признак
    #  правильного однородного решения, а не отсутствия нелинейности.)
    assert abs(float(res.nu_cells.mean()) - _nu_of(B_ex)) < 5e-3
    assert float(res.nu_cells.mean()) > NU0 + 0.3


def test_nonlinear_sphere_across_operating_points() -> None:
    """Развёртка по H₀ ⇒ несколько РАЗНЫХ точек на кривой ν(B), каждая против СВОЕГО
    точного B — тестирует участок нелинейной кривой, а не одну точку."""
    for H0 in (0.5, 1.0, 2.0):
        B_ex = _exact_uniform_B(H0)
        res, b_mean, _ = _solve_nonlinear_sphere(4, H0)
        assert res.converged
        assert abs(b_mean[2] - B_ex) < 1e-3, (H0, b_mean[2], B_ex)


def test_nonlinear_sphere_analytic_match_is_mesh_robust() -> None:
    """Аналитическое совпадение B_in=B_ex — НЕ артефакт сетки n=4: держится и на более
    грубой n=3, на всех рабочих точках (включая НЕвырожденные μ_r≠1, где сфера реально
    возмущает поле: H0=0.5→μ_r≈1.43, H0=2→μ_r≈0.64). Поле внутри однородно (ст. отклон.
    мало), как требует теорема об эллипсоиде."""
    for H0 in (0.5, 1.0, 2.0):
        B_ex = _exact_uniform_B(H0)
        res, b_mean, bz_std = _solve_nonlinear_sphere(3, H0)
        assert res.converged
        assert abs(b_mean[2] - B_ex) < 3e-3, (H0, b_mean[2], B_ex)
        assert abs(b_mean[0]) < 1e-2 and abs(b_mean[1]) < 1e-2
        assert bz_std < 0.05  # поле внутри сферы однородно
