from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from magcore.constants import MU0
from magcore.fem2d.model.materials import Air, LinearMaterial, MagnetMaterial, SteelMaterial
from magcore.fem3d.problem import Problem3D
from magcore.fem3d.scalar import ScalarField3D, assemble_scalar_system, p1_gradients
from magcore.hybrid.magnet_demag import compute_demag_risk_map

# НЕЛИНЕЙНАЯ 3D-МАГНИТОСТАТИКА НА СКАЛЯРНОМ ПОТЕНЦИАЛЕ (этап 3D-3, план — docs/plan_3d_2026-09-11.md).
#
# Материалы:
#   сталь   B = μ₀ μ_c(|H|) H по кривой SteelBHCurve. У неё H(B) — ломаная по таблице, за B_max
#           наклон 1/μ₀; обратная B(H) — ломаная на ТЕХ ЖЕ узлах, за H_max наклон μ₀, поэтому
#           материал в 3D тот же, что в 2D;
#   магнит  поперёк оси B⊥ = μ₀μ⊥H⊥; вдоль оси — закон С ПАМЯТЬЮ (модель магнита та же, что в 2D,
#           память — как в ядре К6′, постановка (S2) в docs/math/coupled_problem.md):
#             r_now(H∥, T) < r:  B∥ = B_главн(H∥, T)          — главная кривая, новая необратимая потеря;
#             r_now(H∥, T) ≥ r:  B∥ = r·B_r(T) + μ₀μ_rec·H∥    — линия возврата,
#           где r ∈ [0, 1] — сохранённая доля ремнантности ячейки после прежних нагружений (новый
#           магнит: r = 1), r_now(H, T) = clip((B_главн(H, T) − μ₀μ_rec·H)/B_r(T), 0, 1) — доля, которую
#           оставило бы поле H (`magnet_model.retention_now`). Ветви сходятся там, где r_now = r, обе
#           возрастают ⇒ закон монотонный, решение единственное. После решения r ← min(r, r_now(H∥, T)) —
#           история для следующего нагружения (`retention`). Хранится доля, а не наихудшее поле (Л-100):
#           при смене температуры потерянная доля сохраняется, при остывании возвращается только
#           обратимая часть через B_r(T). При одной температуре закон тождествен прежнему «по полю».
#   воздух и линейные — как в `scalar.py`.
# Метод Ньютона. Невязка R(φ) = −∫ (B/μ₀)·∇v dΩ + ∮ v H₀·n dS (на границе «поток не выходит»);
# касательная — тензор dB/dH/μ₀ по ячейке: у стали μ_c I + (μ_d − μ_c) ĥĥᵀ (вдоль поля —
# дифференциальная проницаемость, поперёк — хордовая), у магнита поперёк μ⊥, вдоль оси — наклон
# действующей ветви, у воздуха I; собирается той же функцией, что и линейная матрица.
# Глобализация — дробление шага по норме невязки.
# Магнит — СОГЛАСОВАННО, в касательной (Л-93). Внешний цикл «заморозить источник → решить →
# обновить с релаксацией» за крутым коленом перестаёт сжимать (при наклоне кривой μ_d ≈ 40
# множитель итерации ≈ −2,5) — шар N42SH при 175 °C не сходился за 200 итераций.
#
# Линейная задача шага Ньютона: `solver="direct"` — прямой решатель; `solver="cg"` — сопряжённые
# градиенты с диагональным предобуславливателем (этап 3D-6, для больших сеток, Л-96). Касательная
# симметрична и положительно определена: у стали собственные значения μ_c и μ_d, у магнита μ⊥ и
# наклон ветви — все положительны при монотонных законах. Точность шага — «неточный Ньютон»:
# линейная невязка ≤ η·‖R‖, η = min(0,1; ‖R‖/‖R₀‖) — сходимость сверхлинейная, а конечная точность
# задаётся нелинейной невязкой (tol), как и с прямым решателем.
#
# Функционал метода Ньютона — коэнергия J(φ) = ∫ w'(H) dV + μ₀∮ φ H₀·n dS, w'(H) = ∫₀^H B·dH
# (у всех законов B = ∂w'/∂H, w' выпукла ⇒ решение — минимум J). Плотность w' того закона, по
# которому решено, сохраняется в результате — на ней стоят энергия и виртуальная работа (3D-4).

_H_EPS = 1.0e-9


def steel_B_of_H(curve, H) -> np.ndarray:
    """|B|(|H|) [Тл] — обращение ломаной кривой стали (узлы те же, что у H_of_B), за H_max наклон μ₀."""
    H = np.abs(np.asarray(H, dtype=float))
    Hv, Bv = curve.H_values, curve.B_values
    return np.where(H <= Hv[-1], np.interp(H, Hv, Bv), Bv[-1] + MU0 * (H - Hv[-1]))


def steel_mu_rel(curve, H) -> tuple[np.ndarray, np.ndarray]:
    """Относительные хордовая μ_c = B/(μ₀H) и дифференциальная μ_d = (dB/dH)/μ₀ проницаемости стали."""
    H = np.abs(np.asarray(H, dtype=float))
    Hv, Bv = curve.H_values, curve.B_values
    idx = np.clip(np.searchsorted(Hv, H, side="right") - 1, 0, Hv.size - 2)
    slope = (Bv[idx + 1] - Bv[idx]) / (Hv[idx + 1] - Hv[idx])
    mu_d = np.where(H >= Hv[-1], MU0, slope) / MU0
    mu_c = np.where(H > _H_EPS, steel_B_of_H(curve, H) / np.maximum(H, _H_EPS), Bv[1] / Hv[1]) / MU0
    return mu_c, mu_d


def _cumulative_integral(Hv: np.ndarray, Bv: np.ndarray) -> np.ndarray:
    """∫ B dH от первого узла таблицы до каждого узла — точный для ломаной (трапеции)."""
    return np.concatenate(([0.0], np.cumsum(0.5 * (Bv[1:] + Bv[:-1]) * np.diff(Hv))))


def steel_coenergy(curve, H) -> np.ndarray:
    """
    Плотность коэнергии стали w' = ∫₀^|H| B dh [Дж/м³] — точный интеграл той же ломаной B(H), что
    в решателе (таблица начинается в нуле; за H_max наклон μ₀).
    """
    H = np.abs(np.asarray(H, dtype=float))
    Hv, Bv = np.asarray(curve.H_values, dtype=float), np.asarray(curve.B_values, dtype=float)
    cum = _cumulative_integral(Hv, Bv)
    idx = np.clip(np.searchsorted(Hv, H, side="right") - 1, 0, Hv.size - 2)
    inside = cum[idx] + 0.5 * (H - Hv[idx]) * (Bv[idx] + np.interp(H, Hv, Bv))
    dH = H - Hv[-1]
    beyond = cum[-1] + Bv[-1] * dH + 0.5 * MU0 * dH ** 2
    return np.where(H <= Hv[-1], inside, beyond)


def magnet_axial_coenergy(magnet, T, h, retention) -> np.ndarray:
    """
    Коэнергия закона с памятью вдоль оси W∥(h) = ∫₀^h f(x; r) dx [Дж/м³], f — как в решателе: при
    x ≥ H*(r, T) линия возврата r·B_r(T) + μ₀μ_rec·x, ниже — главная кривая (таблица на [−H_cJ, 0],
    правее нуля — линия возврата, левее −H_cJ — линия μ₀μ_rec·x); H* — поле переключения
    (`magnet_model.switch_field`). Точный интеграл.
    """
    h = np.asarray(h, dtype=float)
    r = np.asarray(retention, dtype=float)
    curve = magnet.curve_at(T)
    Hv, Bv = np.asarray(curve.H_values, dtype=float), np.asarray(curve.B_values, dtype=float)
    if Hv[-1] != 0.0:
        raise ValueError("таблица главной кривой должна кончаться в H = 0 (B = B_r).")
    mr = MU0 * float(magnet.mu_rec)
    cum = _cumulative_integral(Hv, Bv)

    def G(x):                                        # ∫₀^x B_главн
        xc = np.clip(x, Hv[0], Hv[-1])
        idx = np.clip(np.searchsorted(Hv, xc, side="right") - 1, 0, Hv.size - 2)
        inside = cum[idx] + 0.5 * (xc - Hv[idx]) * (Bv[idx] + np.interp(xc, Hv, Bv)) - cum[-1]
        above = Bv[-1] * x + 0.5 * mr * x ** 2
        below = -cum[-1] - 0.5 * mr * (Hv[0] ** 2 - x ** 2)
        return np.where(x > Hv[-1], above, np.where(x < Hv[0], below, inside))

    hs = np.asarray(magnet.switch_field(r, T), dtype=float)
    br = r * float(magnet.Br(T))
    recoil = br * h + 0.5 * mr * h ** 2
    major = br * hs + 0.5 * mr * hs ** 2 + G(h) - G(hs)
    return np.where(h >= hs, recoil, major)


def _discrete_system(problem: Problem3D, *, bc: str, applied_field, demag: bool, retention) -> SimpleNamespace:
    """
    Дискретная нелинейная задача — одна для решения методом Ньютона (`solve_nonlinear3d`) и для поля по
    сохранённому потенциалу (`restore_field3d`): невязка R(φ) вместе с состоянием материалов, начальное
    приближение (значения на закреплённых узлах) и сборка итогового поля по φ (`result`).
    """
    problem.check()
    if bc not in ("neumann", "dirichlet"):
        raise ValueError("bc должен быть 'neumann' или 'dirichlet'.")
    H0 = np.zeros(3) if applied_field is None else np.asarray(applied_field, dtype=float)
    if H0.shape != (3,) or not np.isfinite(H0).all():
        raise ValueError("applied_field — три конечных числа [А/м].")

    mesh = problem.mesh
    n, nc = mesh.n_vertices, mesh.n_cells
    r_all = np.ones(nc)
    if retention is not None:
        if not demag:
            raise ValueError("retention задаёт историю колена — без demag она не используется.")
        r_all = np.array(retention, dtype=float).reshape(-1)
        if r_all.shape != (nc,) or not np.isfinite(r_all).all() or (r_all < 0.0).any() or (r_all > 1.0).any():
            raise ValueError("retention — (n_cells,) чисел из [0, 1].")
    cells = mesh.cells
    grads, vol = p1_gradients(mesh)
    reg = np.asarray(problem.cell_region)
    eye = np.eye(3)
    T = problem.T

    # Постоянная часть тензора (воздух, линейные, магнит) и группы стали.
    mu_fixed = np.tile(eye, (nc, 1, 1))
    steel: list[tuple[np.ndarray, object]] = []
    for rid, region in problem.regions.items():
        sel = np.where(reg == rid)[0]
        if sel.size == 0:
            continue
        mat = region.material
        if isinstance(mat, Air):
            continue
        if isinstance(mat, LinearMaterial):
            mu_fixed[sel] = float(mat.mu_r) * eye
        elif isinstance(mat, MagnetMaterial):
            e = np.asarray(problem.magnet_axis, dtype=float)[sel]
            ee = e[:, :, None] * e[:, None, :]
            mu_fixed[sel] = float(mat.magnet.mu_perp) * (eye - ee) + float(mat.magnet.mu_rec) * ee
        elif isinstance(mat, SteelMaterial):
            steel.append((sel, mat.curve))
        else:
            raise TypeError(f"неизвестный материал региона {rid}: {type(mat)}")

    magnet = problem.magnet()
    mmask = problem.magnet_mask()
    M = np.zeros((nc, 3))
    knee = None             # магнит с коленом: (ячейки, оси, сохранённая доля r, B_r(T))
    if magnet is not None and mmask.any():
        axis = np.asarray(problem.magnet_axis, dtype=float)
        if not demag:
            M[mmask] = (float(magnet.Br(T)) / MU0) * axis[mmask]
        else:
            sel_m = np.where(mmask)[0]
            knee = (sel_m, axis[sel_m], r_all[sel_m], float(magnet.Br(T)))
    mu_rec_abs = MU0 * float(magnet.mu_rec) if magnet is not None else 0.0

    def magnet_axial(hpar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """B∥ [Тл] и dB∥/dH∥ по закону с памятью (общая реализация `AnisotropicBHTMagnet.branch_parallel`:
        новая потеря — главная кривая, иначе линия возврата; та же ветвь даёт и наклон)."""
        _, _, r_m, _ = knee
        return magnet.branch_parallel(hpar, T, r_m)

    def field(phi_vec: np.ndarray) -> np.ndarray:
        return -np.einsum("ci,cik->ck", phi_vec[cells], grads)

    def material_state(H: np.ndarray):
        """B/μ₀ по ячейкам [А/м], касательный и хордовый тензоры."""
        Bn = np.einsum("ckl,cl->ck", mu_fixed, H) + M
        tang = mu_fixed.copy()
        chord = mu_fixed.copy()
        for sel, curve in steel:
            h = H[sel]
            hm = np.linalg.norm(h, axis=1)
            mu_c, mu_d = steel_mu_rel(curve, hm)
            Bn[sel] = mu_c[:, None] * h
            u = np.divide(h, hm[:, None], out=np.zeros_like(h), where=hm[:, None] > 0.0)
            tang[sel] = mu_c[:, None, None] * eye + (mu_d - mu_c)[:, None, None] * (u[:, :, None] * u[:, None, :])
            chord[sel] = mu_c[:, None, None] * eye
        if knee is not None:                         # магнит: вдоль оси — закон с памятью
            sel, e = knee[0], knee[1]
            h = H[sel]
            hpar = np.einsum("ij,ij->i", h, e)
            b_par, slope = magnet_axial(hpar)
            mu_p = float(magnet.mu_perp)
            ee = e[:, :, None] * e[:, None, :]
            Bn[sel] = mu_p * (h - hpar[:, None] * e) + (b_par / MU0)[:, None] * e
            tang[sel] = mu_p * (eye - ee) + (slope / MU0)[:, None, None] * ee
        return Bn, tang, chord

    faces, area_vec = mesh.boundary_faces_oriented()
    bnodes = np.unique(faces)
    flux_vec = np.zeros(n)
    if bc == "dirichlet":
        fixed = bnodes
    else:
        flux = area_vec @ H0
        flux_vec = np.bincount(faces.reshape(-1), weights=np.repeat(flux / 3.0, 3), minlength=n)
        fixed = bnodes[:1]
    free = np.setdiff1d(np.arange(n), fixed)
    phi0 = np.zeros(n)
    phi0[fixed] = -mesh.vertices[fixed] @ H0

    def residual(phi_vec: np.ndarray):
        H = field(phi_vec)
        Bn, tang, chord = material_state(H)
        w = vol[:, None] * np.einsum("ck,cik->ci", Bn, grads)
        R = -np.bincount(cells.reshape(-1), weights=w.reshape(-1), minlength=n) + flux_vec
        return H, Bn, tang, chord, R

    def result(phi: np.ndarray, r0, *, n_iterations: int, converged: bool, history) -> ScalarField3D:
        """Поле по потенциалу φ: состояние материалов, коэнергия, доля r после нагружения, карта риска."""
        H, Bn, tang, chord, R = residual(phi)
        # Плотность коэнергии того закона, по которому решено (M здесь — ещё источник, а не отчётная).
        wco = 0.5 * MU0 * np.einsum("ck,ckl,cl->c", H, mu_fixed, H) + MU0 * np.einsum("ck,ck->c", M, H)
        for sel, curve in steel:
            wco[sel] = steel_coenergy(curve, np.linalg.norm(H[sel], axis=1))
        M_out = M.copy()
        retention_out = None
        if knee is not None:         # эффективная намагниченность для отчёта: B∥/μ₀ = μ_rec H∥ + M_eff
            sel, e, r_m = knee[0], knee[1], knee[2]
            hpar = np.einsum("ij,ij->i", H[sel], e)
            b_par, _ = magnet_axial(hpar)
            h_perp2 = np.einsum("ij,ij->i", H[sel], H[sel]) - hpar ** 2
            wco[sel] = (magnet_axial_coenergy(magnet, T, hpar, r_m)
                        + 0.5 * MU0 * float(magnet.mu_perp) * h_perp2)
            M_out[sel] = ((b_par - mu_rec_abs * hpar) / MU0)[:, None] * e
            retention_out = np.ones(nc)
            retention_out[sel] = np.minimum(r_m, magnet.retention_now(hpar, T))
        risk = None
        if magnet is not None and mmask.any():
            risk = compute_demag_risk_map(magnet, SimpleNamespace(H_cells=MU0 * H), mmask, T,
                                          axis=np.asarray(problem.magnet_axis, dtype=float),
                                          retention=r_all if knee is not None else None)
        return ScalarField3D(
            problem=problem, phi=phi, H_cells=H, B_cells=MU0 * Bn, mu_cells=chord, M_cells=M_out,
            volumes=vol, bc=bc, applied_field=H0,
            residual=float(np.linalg.norm(R[free]) / max(r0 or 1e-300, 1e-300)),
            n_iterations=n_iterations, converged=converged, residual_history=tuple(history), risk=risk,
            retention=retention_out, coenergy_density=wco)

    return SimpleNamespace(mesh=mesh, grads=grads, vol=vol, free=free, phi0=phi0, residual=residual, result=result)


def solve_nonlinear3d(problem: Problem3D, *, bc: str = "neumann", applied_field=None,
                      demag: bool = True, retention=None, solver: str = "direct",
                      max_iter: int = 60, tol: float = 1.0e-9) -> ScalarField3D:
    """
    Решить нелинейную 3D-задачу (сталь с насыщением, магнит с коленом) методом Ньютона.

    `bc`, `applied_field` — как в `solve_linear3d`;
    `demag` — учитывать колено магнита (иначе линия возврата с номинальной B_r при T);
    `retention` — история нагружения: сохранённая доля ремнантности r ∈ [0, 1] по ячейкам, форма
    (n_cells,) — берётся из `retention` прежнего результата на той же сетке, температура при этом
    может быть другой (Л-100); вне магнита значения не используются; None — новый магнит;
    `solver` — линейная задача шага: 'direct' (прямой) или 'cg' (сопряжённые градиенты — большие сетки).
    Возвращает `ScalarField3D` с хордовым тензором в `mu_cells`, историей невязки, картой риска
    (потеря — с учётом истории), обновлённой `retention` и плотностью коэнергии.
    """
    if solver not in ("direct", "cg"):
        raise ValueError("solver должен быть 'direct' или 'cg'.")
    s = _discrete_system(problem, bc=bc, applied_field=applied_field, demag=demag, retention=retention)
    free, zero_M = s.free, np.zeros((s.mesh.n_cells, 3))
    phi = s.phi0.copy()
    history: list[float] = []
    r0 = None
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        H, Bn, tang, chord, R = s.residual(phi)
        rnorm = float(np.linalg.norm(R[free]))
        history.append(rnorm)
        if r0 is None:
            r0 = max(rnorm, 1e-300)
        if rnorm <= tol * r0:
            converged = True
            break
        J_ff = assemble_scalar_system(s.mesh, tang, zero_M, s.grads, s.vol)[0][free, :][:, free]
        if solver == "direct":
            dphi = spla.spsolve(J_ff.tocsc(), -R[free])
        else:                                        # неточный Ньютон: линейная невязка ≤ η‖R‖
            eta = min(0.1, max(rnorm / r0, 1.0e-13))
            dphi, _ = spla.cg(J_ff.tocsr(), -R[free], rtol=eta, maxiter=20 * free.size,
                              M=sp.diags(1.0 / J_ff.diagonal()))
        alpha = 1.0
        for _ in range(12):                         # дробление шага по норме невязки
            trial = phi.copy()
            trial[free] += alpha * dphi
            if float(np.linalg.norm(s.residual(trial)[4][free])) < rnorm:
                break
            alpha *= 0.5
        phi[free] += alpha * dphi
        if alpha * float(np.linalg.norm(dphi)) <= tol * (1.0 + float(np.linalg.norm(phi))):
            converged = True
            break
    return s.result(phi, r0, n_iterations=it, converged=converged, history=history)


def restore_field3d(problem: Problem3D, phi, *, bc: str = "neumann", applied_field=None, demag: bool = True,
                    retention=None, tol: float = 1.0e-9) -> ScalarField3D:
    """
    Поле по сохранённому узловому потенциалу φ (например, из файла расчёта) — без итераций: тот же
    дискретный закон, что в `solve_nonlinear3d`, вычисленный при этом φ. `bc`, `applied_field`, `demag`,
    `retention` (история ДО расчёта) — те, с которыми φ получен. Итерации — 0, история невязки пустая.
    `residual` — невязка уравнений при этом φ, отнесённая к невязке начального приближения, как у
    решателя: у φ, сохранённого из решения той же задачи тем же кодом, она та же, что при решении; если с
    тех пор изменились уравнения (материалы, модель магнита, граница, сетка), она больше — по ней
    вызывающий и судит, годится ли сохранённое поле (Л-80). `converged` — `residual` не больше `tol`.
    """
    s = _discrete_system(problem, bc=bc, applied_field=applied_field, demag=demag, retention=retention)
    phi = np.array(phi, dtype=float).reshape(-1)
    if phi.shape != s.phi0.shape or not np.isfinite(phi).all():
        raise ValueError("phi — по конечному числу на узел сетки.")
    r0 = max(float(np.linalg.norm(s.residual(s.phi0)[4][s.free])), 1e-300)
    f = s.result(phi, r0, n_iterations=0, converged=False, history=())
    return replace(f, converged=f.residual <= tol)
