import math

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import brentq

from magcore.constants import MU0
from magcore.domain.magnet_model import n42sh_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.assembly import p1_cell_geometry
from magcore.fem2d.magnet_law import MagnetLaw2D
from magcore.fem2d.mesh import TriangleMesh
from magcore.fem2d.model.problem import steel_nu_and_dnu
from magcore.fem2d.newton import solve_nonlinear_2d_newton
from magcore.fem2d.quadrature import triangle_quadrature
from magcore.fem2d.spaces import LagrangeP1Space2D

# ОРАКУЛ «МАГНИТ ЗА КОЛЕНОМ + НАСЫЩЕННАЯ СТАЛЬ В ОДНОЙ ЗАДАЧЕ» (п. 15 плана 2026-11-18). По отдельности оба закона
# под оракулами (сталь — MMS test_fem2d_newton_mms / test_fem2d_nonlinear_mms, колено — круглый магнит
# test_problem2d_magnet_knee), вместе — не было: а в машине они встречаются в одной касательной Ньютона.
#
# ПОСТАНОВКА (точное решение, изготовленное выбором тока). Кольцо R0 ≤ r ≤ R4 из концентрических колец:
#   сталь M270 (с током) | воздух (катушка) | магнит N42SH при 120 °C (с током) | воздух,
# A_z = 0 на ОБЕИХ окружностях. Ток кусочно-постоянен по кольцам ⇒ решение одномерно: A = A(r), B = B_θ(r)·e_θ.
# Закон Ампера даёт r·H_θ(r) = C + ∫_{R0}^{r} J(s)·s ds, а H_θ непрерывна на границах колец (поверхностного тока
# нет, B_r ≡ 0). В каждом кольце B_θ = закон(H_θ): сталь — обращение H(|B|) таблицы; магнит — ось лёгкого
# намагничивания НАКЛОНЕНА к e_θ на β = 30°, поэтому работают и ветвь вдоль оси (главная кривая с коленом), и μ⊥
# поперёк: H_θ = H∥(B_θ·cos β)·cos β + B_θ·sin²β/(μ0·μ⊥). Постоянная C — из условия A(R0) = A(R4) = 0, то есть
# ∫ B_θ dr = 0 по всему кольцу: в неё входят ОБА нелинейных закона сразу (ошибка в стали сдвигает C и тем самым
# рабочую точку магнита — это и есть связь, которую проверяем). Корень один: B_θ монотонно растёт по H_θ, H_θ — по C.
#
# Законы оракул берёт у тех же функций модели, что и решатель (`branch_parallel_inverse`, таблица стали): проверяется
# дискретизация и метод, а не модель материала. Оба закона кусочно-линейны ⇒ обращаются ТОЧНО по узлам таблицы.
#
# РЕЖИМ (проверяется на точном решении, иначе тест пуст): сталь идёт от 1,74 Тл через колено своей кривой до 2,06 Тл
# (глубокое насыщение, дифференциальная μ_r ≈ 2) и через ноль НЕ проходит — иначе у нуля, где μ_r ≈ 4000, B стали
# меняет знак на десятках микрометров, и такой внутренний слой сетка 0,25–1 мм не разрешает (первый вариант оракула
# с R4 = 30 мм на этом и споткнулся: ошибка B стали не убывала); магнит пересекает колено ВНУТРИ кольца (≈ 1/6 площади
# до колена, остальное — на крутой главной кривой), не доходя до −H_cJ (там модель не определена — Л-119).

MAG = n42sh_magnet((0.0, 1.0, 0.0))
STEEL = m270_35a_bh_curve()
T = 120.0
BETA = math.radians(30.0)
RADII = (0.010, 0.014, 0.018, 0.021, 0.034)      # м
MATS = ("steel", "air", "magnet", "air")
J_RING = (1.7e7, -1.37e8, -7.5e7, 0.0)           # А/м²
NU_PERP = 1.0 / (MU0 * MAG.mu_perp)              # поперёк оси магнита [А/(м·Тл)]


# ------------------------------------------------------------------ точное решение

def _steel_B(H):
    """Обращение H(|B|) стали: кусочно-линейное по таблице, за B_max — наклон μ0 (как `SteelBHCurve.H_of_B`)."""
    h = np.abs(H)
    Hv, Bv = STEEL.H_values, STEEL.B_values
    b = np.where(h <= Hv[-1], np.interp(h, Hv, Bv), Bv[-1] + MU0 * (h - Hv[-1]))
    return np.sign(H) * b


def _magnet_H_theta(B):
    h_par, _ = MAG.branch_parallel_inverse(np.asarray(B) * math.cos(BETA), T, np.ones(np.size(B)))
    return h_par * math.cos(BETA) + NU_PERP * np.asarray(B) * math.sin(BETA) ** 2


def _magnet_nodes():
    """Узлы излома H_θ(B) магнита: узлы таблицы главной кривой, точка переключения ветвей, край −H_cJ."""
    curve = MAG.curve_at(T)
    h_star = float(MAG.switch_field(1.0, T))
    b_par = np.concatenate([curve.B_values, [MAG.Br(T) + MU0 * MAG.mu_rec * h_star,
                                             MU0 * MAG.mu_rec * curve.H_values[0], -10.0, 10.0]])
    B = np.unique(b_par / math.cos(BETA))
    return _magnet_H_theta(B), B


_MAG_H, _MAG_B = _magnet_nodes()


def _magnet_B(H_theta):
    return np.interp(H_theta, _MAG_H, _MAG_B)


def _rH(r, C):
    """r·H_θ(r) [А] по закону Ампера: C + Σ по кольцам J·(r² − a²)/2."""
    r = np.asarray(r, dtype=float)
    s = np.full(r.shape, float(C))
    for (a, b), j in zip(zip(RADII[:-1], RADII[1:]), J_RING):
        hi = np.clip(r, a, b)
        s += j * (hi * hi - a * a) / 2.0
    return s


def _B_ring(k, r, C):
    H = _rH(r, C) / r
    if MATS[k] == "steel":
        return _steel_B(H)
    if MATS[k] == "magnet":
        return _magnet_B(H)
    return MU0 * H


_GX, _GW = np.polynomial.legendre.leggauss(6)
N_PANELS = 4000


def _ring_panels(k):
    return np.linspace(RADII[k], RADII[k + 1], N_PANELS + 1)


def _panel_integrals(k, C):
    e = _ring_panels(k)
    lo, hi = e[:-1, None], e[1:, None]
    x = 0.5 * (hi - lo) * _GX[None, :] + 0.5 * (hi + lo)
    return (0.5 * (hi - lo) * _B_ring(k, x, C) * _GW[None, :]).sum(axis=1)


class Exact:
    """Точное решение: постоянная C, A(r) (сплайн Эрмита с A′ = −B_θ по каждому кольцу), B_θ(r), H∥(r) магнита."""

    def __init__(self):
        total = lambda C: sum(float(_panel_integrals(k, C).sum()) for k in range(len(MATS)))  # noqa: E731
        self.C = brentq(total, -1.0e4, 1.0e4, xtol=1e-12, rtol=1e-15, maxiter=200)
        self.splines = []
        a_start = 0.0
        for k in range(len(MATS)):
            e = _ring_panels(k)
            A = a_start - np.concatenate([[0.0], np.cumsum(_panel_integrals(k, self.C))])
            self.splines.append(CubicHermiteSpline(e, A, -_B_ring(k, e, self.C)))
            a_start = A[-1]
        self.A_end = a_start                      # A(R4): должно быть 0 по построению C

    def ring(self, r):
        return np.clip(np.searchsorted(RADII, r, side="right") - 1, 0, len(MATS) - 1)

    def A(self, r):
        r = np.asarray(r, dtype=float)
        k = self.ring(r)
        out = np.empty_like(r)
        for i, sp in enumerate(self.splines):
            sel = k == i
            out[sel] = sp(r[sel])
        return out

    def B_theta(self, r, ring):
        r = np.asarray(r, dtype=float)
        out = np.empty_like(r)
        for i in range(len(MATS)):
            sel = ring == i
            out[sel] = _B_ring(i, r[sel], self.C)
        return out

    def H_par(self, r):
        B = _magnet_B(_rH(r, self.C) / np.asarray(r))
        return MAG.branch_parallel_inverse(B * math.cos(BETA), T, np.ones(np.size(B)))[0]


@pytest.fixture(scope="module")
def exact():
    return Exact()


# ------------------------------------------------------------------ дискретная задача

def _annulus(n_r, n_theta):
    """Структурная сетка кольца: n_r интервалов по радиусу в каждом кольце материала (узлы — на границах колец)."""
    rs = np.concatenate([np.linspace(a, b, n_r + 1)[:-1] for a, b in zip(RADII[:-1], RADII[1:])] + [[RADII[-1]]])
    th = 2.0 * np.pi * np.arange(n_theta) / n_theta
    verts = np.column_stack([np.repeat(rs, n_theta) * np.tile(np.cos(th), rs.size),
                             np.repeat(rs, n_theta) * np.tile(np.sin(th), rs.size)])

    def v(i, k):
        return i * n_theta + (k % n_theta)

    cells, ring = [], []
    for i in range(rs.size - 1):
        for k in range(n_theta):
            cells += [(v(i, k), v(i + 1, k), v(i + 1, k + 1)), (v(i, k), v(i + 1, k + 1), v(i, k + 1))]
            ring += [i // n_r] * 2
    mesh = TriangleMesh(vertices=verts, cells=np.asarray(cells, dtype=int))
    ddofs = np.concatenate([np.arange(n_theta), (rs.size - 1) * n_theta + np.arange(n_theta)])
    return mesh, np.asarray(ring, dtype=int), ddofs


def _axes(mesh):
    """Ось магнита в ячейке: cos β·e_θ + sin β·e_r в центроиде."""
    c = mesh.vertices[mesh.cells].mean(axis=1)
    phi = np.arctan2(c[:, 1], c[:, 0])
    e_t = np.column_stack([-np.sin(phi), np.cos(phi)])
    e_r = np.column_stack([np.cos(phi), np.sin(phi)])
    return math.cos(BETA) * e_t + math.sin(BETA) * e_r


def _solve(n_r, n_theta, *, steel="curve", magnet="knee", max_iter=60):
    """
    Одна задача Ньютона на кольце. `steel='linear'` — сталь без насыщения (начальная ν кривой); `magnet='linear'` —
    магнит без колена (линия возврата B = B_r(T)·e + μ0·μ_rec·H во всех направлениях; у N42SH μ⊥ = μ_rec). Обе
    подмены — отрицательные проверки: оракул обязан их различать.
    """
    mesh, ring, ddofs = _annulus(n_r, n_theta)
    space = LagrangeP1Space2D(mesh)
    nc = mesh.n_cells
    steel_mask, mag_mask = ring == MATS.index("steel"), ring == MATS.index("magnet")
    axes = _axes(mesh)
    nu_base = np.ones(nc)
    if magnet == "linear":
        nu_base[mag_mask] = 1.0 / MAG.mu_rec
    if steel == "linear":
        nu_base[steel_mask] = MU0 * STEEL.nu_initial

    def nu_and_dnu(B):
        nu, dnu = nu_base.copy(), np.zeros(nc)
        if steel == "curve":
            nu[steel_mask], dnu[steel_mask] = steel_nu_and_dnu(STEEL, np.hypot(B[steel_mask, 0], B[steel_mask, 1]))
        return nu, dnu

    kw = {}
    law = None
    if magnet == "knee":
        law = MagnetLaw2D(MAG, mag_mask, nc, T=T, axis=axes)
        kw["magnet_law"] = law
    else:
        nu_br = np.zeros((nc, 2))
        nu_br[mag_mask] = (MAG.Br(T) / MAG.mu_rec) * axes[mag_mask]
        kw["magnetization"] = nu_br
    j = MU0 * np.asarray(J_RING)[ring]
    res = solve_nonlinear_2d_newton(space, nu_and_dnu, nu_init=nu_and_dnu(np.zeros((nc, 2)))[0], j_cells=j,
                                    dirichlet_dofs=ddofs, max_iter=max_iter, tol=1e-11, **kw)
    return mesh, ring, res, law


def _errors(exact, mesh, ring, res):
    """Относительная L²-ошибка A и L²-ошибки B (Тл, среднеквадратичная по площади) в стали и в магните."""
    cells, _, area = p1_cell_geometry(mesh)
    bary, w = triangle_quadrature(5)
    xq = np.einsum("qk,ckd->cqd", bary, mesh.vertices[cells])
    ah = np.einsum("qk,ck->cq", bary, res.a[cells])
    aq = exact.A(np.hypot(xq[..., 0], xq[..., 1]))
    e_A = math.sqrt(float((area[:, None] * w[None, :] * (ah - aq) ** 2).sum())
                    / float((area[:, None] * w[None, :] * aq ** 2).sum()))
    cen = mesh.vertices[cells].mean(axis=1)
    rc = np.hypot(cen[:, 0], cen[:, 1])
    e_t = np.column_stack([-cen[:, 1], cen[:, 0]]) / rc[:, None]
    B_ex = exact.B_theta(rc, ring)[:, None] * e_t
    err = np.sum((res.B_cells - B_ex) ** 2, axis=1)
    e_B = {}
    for name in ("steel", "magnet"):
        sel = ring == MATS.index(name)
        e_B[name] = math.sqrt(float((area[sel] * err[sel]).sum() / area[sel].sum()))
    return e_A, e_B


MESHES = ((4, 64), (8, 128), (16, 256))


@pytest.fixture(scope="module")
def ladder(exact):
    out = []
    for n_r, n_t in MESHES:
        mesh, ring, res, law = _solve(n_r, n_t)
        out.append((n_r, mesh, ring, res, law, *_errors(exact, mesh, ring, res)))
    return out


# ------------------------------------------------------------------ тесты

def test_oracle_is_self_consistent_and_exercises_both_nonlinearities(exact):
    """Точное решение корректно (A(R4) = 0, обращение законов точное) и действительно в нужном режиме."""
    assert abs(exact.A_end) < 1e-9 * max(abs(exact.A(np.array([0.016])))[0], 1e-9)
    # Обращение закона магнита по узлам излома совпадает с самим законом решателя (кусочная линейность — точная).
    B = np.random.default_rng(7).uniform(-0.9, 1.4, 2000)
    assert np.abs(_magnet_B(_magnet_H_theta(B)) - B).max() < 1e-12
    H = np.random.default_rng(8).uniform(-2e5, 2e5, 2000)
    assert np.abs(np.array([STEEL.H_of_B(b) for b in _steel_B(H)]) * np.sign(H) - H).max() < 1e-6

    r_s = np.linspace(RADII[0], RADII[1], 4001)
    B_s = _B_ring(0, r_s, exact.C)
    assert B_s.min() > 0.0                                 # без смены знака (внутреннего слоя нет)
    assert B_s.min() < 1.78 and B_s.max() > 2.0            # колено кривой стали (1,68–1,98 Тл) пройдено целиком…
    assert STEEL.nu_differential(float(B_s.max())) * MU0 > 1.0 / 4.0      # …до глубокого насыщения: дифф. μ_r < 4

    r_m = np.linspace(RADII[2], RADII[3], 4001)
    h_par = exact.H_par(r_m)
    w = r_m                                                # вес площади кольца ∝ r
    past = float((w * (h_par < MAG.knee_field(T))).sum() / w.sum())
    assert 0.5 < past < 0.95                               # колено пересекается ВНУТРИ магнита
    assert h_par.min() > -MAG.Hcj(T)                       # и модель магнита определена всюду (не «каскад»)


def test_newton_solves_the_coupled_case_and_ends_with_an_exact_step(ladder):
    """
    Ньютон решает связку до допуска; последний шаг — точный (невязка падает до округления): оба закона кусочно-линейны
    по таблицам, и как только у каждой ячейки найден верный участок таблицы, задача на шаге линейна.

    ⚠ Число итераций с холодного старта РАСТЁТ при сгущении (на сетках MESHES: 10 → 13 → 23), и причина — таблица
    стали, а не колено и не связка: с ненасыщающейся сталью (магнит за коленом остаётся) — 5, 4, 5 итераций на тех же
    сетках, с магнитом без колена и настоящей сталью — снова рост (9 → 13 → 22). Кусочно-линейная H(|B|) делает
    Ньютон полугладким: участок таблицы меняется у всё большего числа ячеек, и хвост сходится линейно, а не
    квадратично (тот же закон в MMS test_fem2d_newton_mms гладкий — там число итераций от сетки не зависит). Найдено
    этим оракулом 2026-11-18; в транзиенте старт тёплый (поле прошлого шага), поэтому на расчётах это видно слабо.
    """
    its = []
    for n_r, mesh, ring, res, law, e_A, e_B in ladder:
        hist = np.asarray(res.rel_change_history)
        assert res.converged and hist[-1] / hist[0] <= 1e-11
        assert hist[-1] / hist[-2] < 1e-8                  # последний шаг точный: верные участки таблиц найдены
        its.append(res.n_iterations)
    assert max(its) <= 25, its                             # единицы-десятки итераций, не сотни (как у Пикара)
    knee_only = [_solve(n_r, n_t, steel="linear")[2] for n_r, n_t in MESHES]
    assert all(r.converged for r in knee_only)
    assert max(r.n_iterations for r in knee_only) <= 6, [r.n_iterations for r in knee_only]   # колено — без роста


def test_field_converges_to_the_exact_solution_at_the_expected_rate(ladder):
    """A — второй порядок в L²; B в стали и в магните — не ниже первого (P1: B кусочно-постоянна)."""
    e_A = [row[5] for row in ladder]
    e_B = [row[6] for row in ladder]
    assert e_A[0] > e_A[1] > e_A[2] > 0.0
    rate_A = math.log2(e_A[1] / e_A[2])
    assert 1.7 < rate_A < 2.3, (e_A, rate_A)
    assert e_A[2] < 5e-3
    for name in ("steel", "magnet"):
        eb = [d[name] for d in e_B]
        assert eb[0] > eb[1] > eb[2], (name, eb)
        assert math.log2(eb[1] / eb[2]) > 0.8, (name, eb)


def test_knee_crossing_is_located_within_one_radial_layer(exact, ladder):
    """Доля площади магнита за коленом — как у точного решения с точностью до одного слоя ячеек по радиусу."""
    r_m = np.linspace(RADII[2], RADII[3], 20001)
    h_ex = exact.H_par(r_m)
    past_ex = float((r_m * (h_ex < MAG.knee_field(T))).sum() / r_m.sum())
    for n_r, mesh, ring, res, law, e_A, e_B in ladder:
        law(res.B_cells)                                   # закон на выданном решении (H∥ по ячейкам)
        _, _, area = p1_cell_geometry(mesh)
        a_m = area[law.idx]
        past = float(a_m[law.H_par < MAG.knee_field(T)].sum() / a_m.sum())
        assert abs(past - past_ex) <= 1.0 / n_r, (n_r, past, past_ex)
        assert np.all(law.H_par > -MAG.Hcj(T))


def _magnet_mean_B(mesh, ring, B_cells):
    """Средняя по площади магнита B_θ [Тл] — интегральная мера рабочей точки (одной ячейке не верим, Л-29)."""
    cells, _, area = p1_cell_geometry(mesh)
    cen = mesh.vertices[cells].mean(axis=1)
    e_t = np.column_stack([-cen[:, 1], cen[:, 0]]) / np.hypot(cen[:, 0], cen[:, 1])[:, None]
    m = ring == MATS.index("magnet")
    return float((area[m] * np.einsum("ij,ij->i", B_cells[m], e_t[m])).sum() / area[m].sum())


@pytest.mark.parametrize("variant", [dict(steel="linear"), dict(magnet="linear")])
def test_oracle_rejects_a_model_without_saturation_or_without_the_knee(exact, ladder, variant):
    """
    НЕ ТАВТОЛОГИЯ: тот же решатель с ненасыщающейся сталью или с магнитом без колена уводит рабочую точку магнита от
    оракула много дальше ошибки дискретизации. Мера — средняя по магниту B_θ; граница — 50 ошибок дискретизации той же
    меры на той же сетке (разделение больше полутора порядков). Сталь без насыщения при этом в магнит не входит
    вовсе — она сдвигает его рабочую точку только через постоянную C (условие A(R0) = A(R4) = 0): связь законов в
    одной задаче оракул видит. Замер 2026-11-18 (сетка 8×128): ошибка дискретизации 3·10⁻⁴ Тл, сталь без насыщения
    −0,065 Тл, магнит без колена +0,078 Тл.
    """
    r = np.linspace(RADII[2], RADII[3], 200001)
    mean_exact = float((r * _magnet_B(_rH(r, exact.C) / r)).sum() / r.sum())
    ok = ladder[1]
    disc = abs(_magnet_mean_B(ok[1], ok[2], ok[3].B_cells) - mean_exact)
    n_r, n_t = MESHES[1]
    mesh, ring, res, law = _solve(n_r, n_t, **variant)
    assert res.converged
    shift = abs(_magnet_mean_B(mesh, ring, res.B_cells) - mean_exact)
    assert shift > 50.0 * disc, (variant, shift, disc)
