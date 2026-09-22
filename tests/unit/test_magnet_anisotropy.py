# -*- coding: utf-8 -*-
"""
АНИЗОТРОПИЯ МАГНИТА В ПЛАНАРНОЙ СБОРКЕ (этапы 0–1 плана 2026-09).

ЧТО БЫЛО НЕ ТАК. Матрица жёсткости собиралась с ОДНИМ скаляром ν на ячейку, и для ячейки
магнита туда шёл наклон рабочей ветви B(H). Замер (см. `test_transverse_...`): решатель
применял поперёк лёгкой оси ровно ту ν, что подана в сборку, — независимо от `mu_perp`
материала.

ЧТО ИМЕННО ДЕФЕКТ, А ЧТО НЕТ (уточнено 2026-09-07, первая формулировка была неверной).
ВДОЛЬ оси крутой наклон ветви ЛЕГИТИМЕН: источник строится как b_src = b_branch − μ_frozen·h_par,
поэтому подразумеваемый осевой закон B_par = μ_frozen·h_par + b_src = b_branch(h_par) НЕ
зависит от замороженной ν — она влияет только на скорость сходимости (это ньютоновская
линеаризация, а не хордовый Пикар). Проверяется здесь `test_source_correction_...` и
независимо — `test_coupled_transient.py::test_converged_answer_is_independent_of_demag_relaxation`.
ПОПЕРЁК оси такой защиты нет: там источника нет, и ν работает как настоящее свойство
материала. Настоящий дефект — ровно в этом, и лечится он тензором, а не заменой ветви.

Физика поперечного отклика: это когерентный поворот моментов против поля анизотропии,
χ_⊥ = J_s²/(2μ₀K₁). В χ_⊥ входит КВАДРАТ J_s, поэтому перемагниченное зерно (та же ось,
тот же K₁, тот же |M_s|) даёт тот же поперечный отклик ⇒ μ_⊥ от продольной необратимости
НЕ ЗАВИСИТ. Для Nd₂Fe₁₄B μ_⊥ ≈ 1,17…1,25, для Sm₂Co₁₇ ≈ 1,16 при μ_rec ≈ 1,05…1,11.
"""
from __future__ import annotations

import numpy as np
import pytest

from magcore.constants import MU0
from magcore.domain.magnet_model import magnet_from_datasheet, n42sh_magnet
from magcore.fem2d.assembly import (
    assemble_stiffness,
    assemble_stiffness_sparse,
    rotate_nu_to_gradient,
    uniaxial_nu_tensor,
)
from magcore.fem2d.coupled_transient import IrreversibleMagnetState
from magcore.fem2d.mesh_generators import build_disk_tri_mesh, build_structured_rectangle_tri_mesh
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D

EZ = [1.0, 0.0, 0.0]


# ======================================================================================
# 1. ПОВОРОТ ν̃ = Rᵀ ν R — планарная конвенция B = R∇A_z
# ======================================================================================

def test_rotation_matches_hand_derivation() -> None:
    """Для ν=[[a,b],[c,d]] ручной вывод даёт ν̃=[[d,−c],[−b,a]]."""
    nu = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    assert np.allclose(rotate_nu_to_gradient(nu)[0], [[4.0, -3.0], [-2.0, 1.0]])


@pytest.mark.parametrize("alpha", [0.0, 0.7, np.pi / 2, 2.3, -1.1])
def test_rotation_swaps_axial_and_transverse_roles(alpha: float) -> None:
    """
    ν̃ = ν_⊥·eeᵀ + ν_∥·e^⊥(e^⊥)ᵀ — роли МЕНЯЮТСЯ МЕСТАМИ, и это правильно:
    ∇A_z вдоль e отвечает B вдоль e^⊥.
    """
    e = np.array([np.cos(alpha), np.sin(alpha)])
    ep = np.array([-np.sin(alpha), np.cos(alpha)])
    nu_par, nu_perp = 2.0, 5.0
    got = rotate_nu_to_gradient(uniaxial_nu_tensor(nu_par, nu_perp, e))[0]
    assert np.allclose(got, nu_perp * np.outer(e, e) + nu_par * np.outer(ep, ep))


def _rect_space(n: int = 6):
    mesh = build_structured_rectangle_tri_mesh(n, n)
    return LagrangeP1Space2D(mesh), mesh


def test_isotropic_tensor_reproduces_scalar_assembly() -> None:
    """Тензор ν·I обязан давать ТУ ЖЕ матрицу, что скалярная ν. Страж обратной совместимости."""
    space, mesh = _rect_space()
    rng = np.random.default_rng(7)
    nu = 0.5 + rng.random(mesh.n_cells)
    tensor = nu[:, None, None] * np.broadcast_to(np.eye(2), (mesh.n_cells, 2, 2))
    K_scalar = assemble_stiffness_sparse(space, nu).toarray()
    K_tensor = assemble_stiffness_sparse(space, tensor).toarray()
    assert np.allclose(K_scalar, K_tensor, rtol=0.0, atol=1e-13)


def test_dense_and_sparse_agree_for_tensor() -> None:
    """Плотная и разрежённая сборки обязаны совпадать и на тензорном пути."""
    space, mesh = _rect_space(4)
    rng = np.random.default_rng(11)
    axis = rng.normal(size=(mesh.n_cells, 2))
    tensor = uniaxial_nu_tensor(0.9 + rng.random(mesh.n_cells),
                                0.4 + rng.random(mesh.n_cells), axis)
    assert np.allclose(assemble_stiffness(space, tensor),
                       assemble_stiffness_sparse(space, tensor).toarray(), atol=1e-13)


# ======================================================================================
# 2. ПОПЕРЕЧНАЯ ПРОНИЦАЕМОСТЬ — АНАЛИТИЧЕСКИЙ ОРАКУЛ
# ======================================================================================
# Круг радиуса a с относительной проницаемостью μ внутри концентрической границы r=b, на
# которой задано A_z однородного поля B0 поперёк лёгкой оси. Ряд обрывается на первой
# гармонике (A_z = f(r)·cosθ), решение точное:
#       A_in = P·r·cosθ,   A_out = (α·r + β/r)·cosθ,   k = (μ−1)/(μ+1),
#       β = a²·α·k,   α = −B0/(1 + (a/b)²·k),   P = α(1+k),
#       H_in = 2·B0 / [ (μ+1)·(1 + (a/b)²·k) ].
# ⚠ Множитель 1/(1 + (a/b)²k) — поправка на КОНЕЧНУЮ область. Без неё оракул смещён
# (при b/a=3 и μ=3 — на 7,5 %), и ошибка НЕ УБЫВАЕТ при измельчении сетки: это усечение
# области, а не дискретизация. Проверено: с поправкой порядок сходимости 1,97 ≈ 2 = O(h²).

A_MAG, B_OUT, B_APPLIED = 1.0, 3.0, 1.0


def _analytic_H_in(mu_r: float) -> float:
    k = (mu_r - 1.0) / (mu_r + 1.0)
    return 2.0 * B_APPLIED / ((mu_r + 1.0) * (1.0 + (A_MAG / B_OUT) ** 2 * k))


def _measure_H_in(mu_perp: float, *, tensor: bool, n_rings: int = 12, n_theta: int = 48) -> float:
    """Поперечное поле в магните: диск в однородном поле ПОПЕРЁК лёгкой оси x."""
    mag = magnet_from_datasheet("probe", "probe", EZ, Br=1.29, Hcb=9.24e5,
                                Hk=1.353e6, Hcj=1.592e6, mu_perp=mu_perp)
    disk = build_disk_tri_mesh(B_OUT, n_rings, n_theta)
    mesh = disk.mesh
    space = LagrangeP1Space2D(mesh)
    inside = np.array([np.linalg.norm(mesh.cell_centroid(c)) < A_MAG
                       for c in range(mesh.n_cells)], dtype=bool)
    nu_par = np.where(inside, 1.0 / mag.mu_rec, 1.0)
    if tensor:
        nu = uniaxial_nu_tensor(nu_par, np.where(inside, 1.0 / mag.mu_perp, 1.0),
                                np.broadcast_to([1.0, 0.0], (mesh.n_cells, 2)))
    else:
        nu = nu_par                                  # прежний скалярный путь
    dofs = np.asarray(space.boundary_dofs(), dtype=int)
    xy = mesh.vertices[dofs]
    res = solve_nonlinear_2d_picard(
        space, nu_of_B=lambda B: nu.copy(), nu_init=nu,
        dirichlet_dofs=dofs, dirichlet_values=-B_APPLIED * xy[:, 0],
        tol=1e-10, max_iter=5,
    )
    assert res.converged
    core = np.array([np.linalg.norm(mesh.cell_centroid(c)) < 0.5
                     for c in range(mesh.n_cells)], dtype=bool)
    return float(np.mean(res.H_cells[core, 1]))


@pytest.mark.parametrize("mu_perp", [1.17, 3.0])
def test_transverse_permeability_follows_mu_perp(mu_perp: float) -> None:
    """
    ГЛАВНЫЙ ТЕСТ ЭТАПА 1: поперёк лёгкой оси решатель обязан давать μ_perp.

    Контраст 3,0 взят преувеличенным НАМЕРЕННО — это верификация сборки, а не физическое
    значение (физическое μ_⊥ ≈ 1,17…1,25 для NdFeB).
    """
    assert _measure_H_in(mu_perp, tensor=True) == pytest.approx(_analytic_H_in(mu_perp), rel=0.01)


def test_scalar_path_ignores_mu_perp() -> None:
    """
    ДОКУМЕНТИРУЕТ ПРЕЖНЕЕ ПОВЕДЕНИЕ: скалярная сборка применяет поперёк ту ν, что ей подали,
    какой бы ни был μ_perp материала. Тест не «про баг» — он фиксирует, что скалярный путь
    остался прежним и что тензор действительно меняет именно поперечный отклик.
    """
    base = _measure_H_in(1.17, tensor=False)
    assert _measure_H_in(3.0, tensor=False) == pytest.approx(base, rel=1e-12)
    assert base != pytest.approx(_measure_H_in(3.0, tensor=True), rel=0.05)


def test_transverse_oracle_converges_second_order() -> None:
    """Порядок сходимости по сетке ≈ 2 (P1). Если < 1,5 — сборка тензора неверна."""
    exact = _analytic_H_in(3.0)
    err = [abs(_measure_H_in(3.0, tensor=True, n_rings=nr, n_theta=nt) / exact - 1.0)
           for nr, nt in ((12, 48), (24, 96))]
    assert err[0] < 0.01
    assert np.log2(err[0] / err[1]) > 1.5, "порядок сходимости %.2f" % np.log2(err[0] / err[1])


# ======================================================================================
# 3. ОСЕВОЙ ЗАКОН — почему крутой наклон ветви вдоль оси НЕ является дефектом
# ======================================================================================

def test_source_correction_makes_axial_law_independent_of_frozen_nu() -> None:
    """
    Источник строится как b_src = b_branch − μ_frozen·h_par, поэтому подразумеваемый осевой
    закон B_par = μ_frozen·h_par + b_src равен b_branch(h_par) при ЛЮБОЙ замороженной ν.
    Это и есть причина, по которой вдоль оси касательный наклон законен: он не смещает
    неподвижную точку, а лишь ускоряет итерацию.
    """
    mag = n42sh_magnet(EZ)
    T = 150.0
    h_par = -0.5 * (mag.Hk(T) + mag.Hcj(T))        # между коленом и H_cJ
    laws = []
    for nu_frozen in (1.0 / mag.mu_rec, 0.25 / mag.mu_rec):
        st = IrreversibleMagnetState(mag, np.array([True]), 1, axis=(1.0, 0.0))
        st.set_temperature(np.array([T]))
        src = st(np.zeros((1, 2)), np.array([[MU0 * h_par, 0.0]]), np.array([nu_frozen]))
        b_src = float(src[0, 0]) / nu_frozen        # out = nu_frozen·b_src·e
        laws.append((MU0 / nu_frozen) * h_par + b_src)
    assert laws[0] == pytest.approx(laws[1], rel=1e-12), (
        "осевой закон обязан не зависеть от замороженной ν: %r" % (laws,)
    )


def test_reluctivity_above_knee_is_recoil() -> None:
    """Над коленом ветвь совпадает с линией возврата — контроль ветвевой машины."""
    mag = n42sh_magnet(EZ)
    for T in (20.0, 100.0, 150.0):
        st = IrreversibleMagnetState(mag, np.array([True]), 1, axis=(1.0, 0.0))
        st.set_temperature(np.array([T]))
        st(np.zeros((1, 2)), np.array([[MU0 * (-0.5 * mag.Hk(T)), 0.0]]),
           np.array([1.0 / mag.mu_rec]))
        assert 1.0 / st.nu_rel[0] == pytest.approx(mag.mu_rec, rel=0.05)


# ======================================================================================
# 4. ЭКВИВАРИАНТНОСТЬ ПО ПОВОРОТУ — страж от ошибки знака в RᵀνR
# ======================================================================================

@pytest.mark.parametrize("alpha", [0.0, 0.37, np.pi / 3, 2.1, -1.4])
def test_magnet_source_is_rotation_equivariant(alpha: float) -> None:
    mag = n42sh_magnet(EZ)
    T = 120.0
    h_par = -1.02 * mag.Hk(T)
    c, s = np.cos(alpha), np.sin(alpha)

    def probe(axis2, H2):
        st = IrreversibleMagnetState(mag, np.array([True]), 1, axis=axis2)
        st.set_temperature(np.array([T]))
        src = st(np.zeros((1, 2)), np.asarray([H2], dtype=float),
                 np.array([1.0 / mag.mu_rec]))
        return float(st.nu_rel[0]), src[0].copy()

    nu0, src0 = probe((1.0, 0.0), [MU0 * h_par, 0.0])
    nu1, src1 = probe((c, s), [MU0 * h_par * c, MU0 * h_par * s])

    assert nu1 == pytest.approx(nu0, rel=1e-12)
    rot = np.array([c * src0[0] - s * src0[1], s * src0[0] + c * src0[1]])
    assert np.allclose(src1, rot, rtol=1e-12, atol=1e-14)
