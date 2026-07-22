from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from magcore.constants import MU0
from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.bridge import pmsm_to_problem
from magcore.fem2d.machines.excitation import winding_current_density
from magcore.fem2d.machines.loss import phase_flux_linkage
from magcore.fem2d.machines.pmsm_outrunner import (
    MachineGeometry,
    OutrunnerPMSMParams,
    Region,
    build_outrunner_spm_pmsm,
)
from magcore.fem2d.machines.scenario import MachineScenario
from magcore.fem2d.machines.static_solver import machine_reluctivity
from magcore.fem2d.machines.winding import star_of_slots_layout
from magcore.fem2d.model.postproc import _bary, torque_arkkio
from magcore.fem2d.model.problem import Solution2D
from magcore.fem2d.nonlinear import solve_nonlinear_2d_picard
from magcore.fem2d.spaces import LagrangeP1Space2D

# ВРАЩЕНИЕ РОТОРА — предпосылка для честных момента, K_t и КПД, а не украшение.
# Без него величины снимаются при ОДНОМ положении ротора, и при НЕСИММЕТРИЧНОМ повреждении
# магнита один вопрос получает три разных ответа (момент под нагрузкой −6.9 %, K_t холостого
# хода +1.7 %, инвариантная 1-я гармоника −1.7 %). Свипом угла тока γ это не лечится:
# потокосцепление ПМ холостого хода от γ не зависит ВООБЩЕ — нужен реальный поворот ротора.
#
# КАК СДЕЛАНО: геометрия ПЕРЕСТРАИВАЕТСЯ на каждом положении (`OutrunnerPMSMParams.rotor_angle`
# смещает дуги магнитов и межполюсного воздуха). Сетка поэтому КОНФОРМНА границам магнитов на
# любом угле, и пульсации момента получаются физическими. Плата — перестроение сетки и решение
# на каждом положении; альтернатива «переклассифицировать ячейки неподвижной сетки» дешевле,
# но даёт ступенчатые границы, и пульсация оказалась бы артефактом дискретизации.
#
# ПОВРЕЖДЕНИЕ ЕДЕТ ВМЕСТЕ С РОТОРОМ: доля сохранённой ремнантности привязана к РОТОРНОЙ
# системе координат (`RotorDamage`), поэтому при повороте она поворачивается с магнитами, а
# не остаётся висеть в лабораторной системе.


@dataclass(frozen=True, slots=True)
class RotorSweepResult:
    """Величины машины как функции механического угла ротора."""

    angles: np.ndarray            # (N,) механический угол ротора [рад]
    torque: np.ndarray            # (N,) момент [Н·м]
    flux_linkage: np.ndarray      # (N,3) потокосцепления фаз [Вб] (nan, если х.х. не считался)
    converged: np.ndarray         # (N,) bool
    n_pole_pairs: int
    probe_B: np.ndarray | None = None   # (N, P, 2) B в пробных точках статора (для потерь в железе)

    @property
    def torque_mean(self) -> float:
        """Средний момент — то, что реально разгоняет вал (мгновенный от угла зависит)."""
        return float(np.mean(self.torque))

    @property
    def torque_ripple(self) -> float:
        """Размах пульсаций момента, отнесённый к среднему (0 — идеально ровный момент)."""
        m = self.torque_mean
        span = float(np.max(self.torque) - np.min(self.torque))
        return float(span / abs(m)) if m != 0.0 else float("inf")

    @property
    def all_converged(self) -> bool:
        return bool(np.all(self.converged))

    def flux_linkage_fundamental(self) -> float:
        """
        Амплитуда ПЕРВОЙ (электрической) гармоники потокосцепления фазы A, λ_m [Вб].

        Именно она определяет K_e и K_t. Берётся проекцией λ_A(θ) на e^{i·p·θ} по всей
        прогонке — в отличие от значения при одном положении ротора, эта величина корректна
        и при несимметричном повреждении (тогда «мгновенная» амплитуда вводит в заблуждение
        вплоть до смены знака изменения).

        Требует РАВНОМЕРНОЙ выборки по целому числу электрических периодов.
        """
        lam_a = self.flux_linkage[:, 0]
        if not np.all(np.isfinite(lam_a)):
            raise ValueError("потокосцепление не считалось: запустите прогонку с no_load=True.")
        n = lam_a.size
        proj = np.sum(lam_a * np.exp(-1j * self.n_pole_pairs * self.angles))
        return float(2.0 * np.abs(proj) / n)

    def emf_constant(self) -> float:
        """K_e = p·λ_m [В·с/рад] — пик фазной ЭДС на рад/с механические."""
        return float(self.n_pole_pairs * self.flux_linkage_fundamental())

    def torque_constant(self) -> float:
        """K_t = (3/2)·p·λ_m [Н·м/А] — момент на амплитуду тока по оси q."""
        return float(1.5 * self.n_pole_pairs * self.flux_linkage_fundamental())


class RotorDamage:
    """
    Повреждение магнита, привязанное к РОТОРУ: доля сохранённой ремнантности как функция
    роторных координат. При повороте едет вместе с магнитами.

    Перенос на сетку другого положения — ближайшим соседом в РОТОРНОЙ системе (точки
    сравниваются как декартовы, поэтому переход через 0/2π обрабатывается сам собой).
    Равномерное повреждение переносится ТОЧНО при любом угле — это проверяется тестом.
    """

    def __init__(self, geometry: MachineGeometry, retention: np.ndarray) -> None:
        idx = np.where(geometry.mask(Region.MAGNET))[0]
        r = np.asarray(retention, dtype=float).reshape(-1)
        if r.shape != idx.shape:
            raise ValueError("retention должен быть по ячейкам магнита исходной геометрии.")
        cen = np.array([geometry.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
        self.points = _to_rotor_frame(cen, geometry.params.rotor_angle)
        self.retention = r

    def sample(self, geometry: MachineGeometry) -> np.ndarray:
        """Доля сохранённой ремнантности для ячеек магнита ЗАДАННОЙ (повёрнутой) геометрии."""
        idx = np.where(geometry.mask(Region.MAGNET))[0]
        cen = np.array([geometry.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
        target = _to_rotor_frame(cen, geometry.params.rotor_angle)
        d = ((target[:, None, :] - self.points[None, :, :]) ** 2).sum(axis=2)
        return self.retention[np.argmin(d, axis=1)]


def _to_rotor_frame(points: np.ndarray, rotor_angle: float) -> np.ndarray:
    """Повернуть точки на −rotor_angle: из лабораторной системы в роторную."""
    a = -float(rotor_angle)
    ca, sa = math.cos(a), math.sin(a)
    return np.column_stack([
        points[:, 0] * ca - points[:, 1] * sa,
        points[:, 0] * sa + points[:, 1] * ca,
    ])


def _solve_at_angle(
    geometry: MachineGeometry, magnet: AnisotropicBHTMagnet, steel: SteelBHCurve,
    *, T: float, retention: np.ndarray | None, j_cells: np.ndarray | None,
    relaxation: float, max_iter: int, tol: float,
):
    """Магнитостатика при замороженном состоянии магнита (источник ν·B_r = ν·r·B_r(T))."""
    space = LagrangeP1Space2D(geometry.mesh)
    nu_of_B, nu_init, magnet_mask, _ = machine_reluctivity(geometry, magnet, steel)
    idx = np.where(magnet_mask)[0]

    br = np.full(idx.size, float(magnet.Br(T)), dtype=float)
    if retention is not None:
        br = br * np.asarray(retention, dtype=float).reshape(-1)
    nu_br = np.zeros((geometry.mesh.n_cells, 2), dtype=float)
    nu_br[idx] = (br / magnet.mu_rec)[:, None] * geometry.magnet_easy_axis[idx]

    return solve_nonlinear_2d_picard(
        space, nu_of_B=nu_of_B, nu_init=nu_init,
        j_cells=(None if j_cells is None else MU0 * j_cells),
        magnetization=nu_br, relaxation=relaxation, max_iter=max_iter, tol=tol,
    )


def cogging_period_angles(params: OutrunnerPMSMParams, n: int) -> np.ndarray:
    """
    Углы для ЗУБЦОВОГО момента: его период = 2π/НОК(n_slots, n_poles) — много мельче
    электрического. Выборка по электрическому периоду его АЛИАСИТ (для 12/14 шаг ровно
    совпадает с периодом зубцового момента, и вместо пульсации видно случайное сечение).
    """
    lcm = math.lcm(int(params.n_slots), int(params.n_poles))
    return np.arange(int(n)) * (2.0 * math.pi / lcm) / int(n)


def electrical_period_angles(params: OutrunnerPMSMParams, n: int, periods: int = 1) -> np.ndarray:
    """
    Углы РАВНОМЕРНОЙ выборки по `periods` электрическим периодам: шаг 2π·periods/(p·n).

    Для СИММЕТРИЧНОЙ машины одного периода достаточно. При несимметричном повреждении
    картина повторяется только за полный оборот ⇒ нужен `periods = p` (полный оборот).
    """
    p = params.n_poles // 2
    span = 2.0 * math.pi * periods / p
    return np.arange(int(n)) * span / int(n)


def sweep_rotor(
    params: OutrunnerPMSMParams,
    magnet: AnisotropicBHTMagnet,
    steel: SteelBHCurve,
    *,
    angles,
    i_peak: float = 0.0,
    gamma_elec: float = 0.0,
    turns_per_slot: float = 0.0,
    T: float = 20.0,
    damage: RotorDamage | None = None,
    no_load: bool = True,
    probe_points=None,
    relaxation: float = 0.1,
    max_iter: int = 300,
    tol: float = 1.0e-6,
) -> RotorSweepResult:
    """
    Прогнать машину по положениям ротора и снять момент и потокосцепления фаз.

    На каждом угле геометрия строится заново (конформная сетка), поэтому прогонка стоит
    примерно (перестроение + 1–2 решения) на положение. `i_peak=0` даёт ЗУБЦОВЫЙ момент
    (cogging) — момент от одних магнитов, без тока. `no_load=True` добавляет второе решение
    без тока для потокосцепления ПМ (нужно для K_e/K_t).

    `gamma_elec` — угол вектора тока ОТНОСИТЕЛЬНО РОТОРА (нагрузочный угол), как его держит
    привод синхронной машины. Абсолютный угол тока на каждом положении берётся как
    γ + p·α: без этого ток остаётся стоять, ротор проезжает под ним весь период, и момент
    качается от +M до −M со средним нулём — это была бы прогонка по нагрузочному углу,
    а не вращение машины.

    `damage` — повреждение в РОТОРНОЙ системе; поворачивается вместе с магнитами.
    `probe_points` — (P,2) НЕПОДВИЖНЫЕ точки (обычно в железе статора): в них на каждом угле
    снимается B из НАГРУЗОЧНОГО решения ⇒ `probe_B` (N,P,2) = волна B(θ) в этих точках.
    Именно это нужно потерям в железе: размах ΔB и dB/dθ за электрический период по элементу.
    """
    angles = np.asarray(angles, dtype=float).reshape(-1)
    p_pairs = params.n_poles // 2
    torque = np.empty(angles.size, dtype=float)
    lam = np.full((angles.size, 3), np.nan, dtype=float)
    conv = np.empty(angles.size, dtype=bool)

    probes = None if probe_points is None else np.asarray(probe_points, dtype=float).reshape(-1, 2)
    probe_B = None if probes is None else np.empty((angles.size, probes.shape[0], 2), dtype=float)

    have_current = i_peak != 0.0 and turns_per_slot != 0.0
    layout = star_of_slots_layout(params.n_slots, params.n_poles)

    for i, a in enumerate(angles):
        geo = build_outrunner_spm_pmsm(replace(params, rotor_angle=float(a)))
        ret = None if damage is None else damage.sample(geo)
        gamma_abs = float(gamma_elec) + p_pairs * float(a)     # ток едет вместе с ротором
        jz = (winding_current_density(geo, layout, i_peak=i_peak, gamma_elec=gamma_abs,
                                      turns_per_slot=turns_per_slot)
              if have_current else None)

        kw = dict(T=T, retention=ret, relaxation=relaxation, max_iter=max_iter, tol=tol)
        em = _solve_at_angle(geo, magnet, steel, j_cells=jz, **kw)
        problem = pmsm_to_problem(geo, magnet, steel, T=T, j_cells=jz)
        torque[i] = torque_arkkio(
            Solution2D(problem=problem, field=em, risk=None),
            params.R_s_out, params.R_mag_in, axial_length=params.axial_length,
        )
        ok = bool(em.converged)

        if probes is not None:
            probe_B[i] = sample_B_at_points(geo.mesh, em.B_cells, probes)

        if no_load and turns_per_slot != 0.0:
            em_nl = em if not have_current else _solve_at_angle(
                geo, magnet, steel, j_cells=None, **kw
            )
            lam[i] = phase_flux_linkage(geo, layout, em_nl.a, turns_per_slot=turns_per_slot)
            ok = ok and bool(em_nl.converged)
        conv[i] = ok

    return RotorSweepResult(
        angles=angles, torque=torque, flux_linkage=lam, converged=conv,
        n_pole_pairs=p_pairs, probe_B=probe_B,
    )


def sample_B_at_points(mesh, B_cells: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Значение (кусочно-постоянного) B в точках: B ячейки, содержащей точку.

    Для внутренних точек железа статора это устойчиво даже при пере-сетке между углами:
    B на P1 постоянна в ячейке, а сама сетка перестраивается только в роторной части, но
    точный треугольник статора всё равно меняется ⇒ локализация по вхождению обязательна,
    с запасным вариантом «ближайший центроид» при попадании точки ровно на ребро.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    cent = np.array([mesh.cell_centroid(c) for c in range(mesh.n_cells)], dtype=float)
    out = np.empty((pts.shape[0], 2), dtype=float)
    for i, p in enumerate(pts):
        order = np.argsort(((cent - p) ** 2).sum(axis=1))   # сначала ближайшие центроиды
        hit = -1
        for c in order[:12]:                                # вхождение среди ближайших
            lam = _bary(p, mesh.cell_vertices(int(c)))
            if lam is not None and min(lam) >= -1e-9:
                hit = int(c)
                break
        out[i] = B_cells[hit if hit >= 0 else int(order[0])]
    return out


def scenario_damage(scenario: MachineScenario, retention: np.ndarray) -> RotorDamage:
    """Повреждение из результата теплового сценария, привязанное к ротору его геометрии."""
    return RotorDamage(scenario.geometry, retention)
