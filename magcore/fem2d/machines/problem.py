from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from magcore.domain.magnet_model import AnisotropicBHTMagnet
from magcore.domain.steel_curves import SteelBHCurve
from magcore.fem2d.machines.loss import (
    DemagImpact,
    MagnetLossAggregate,
    evaluate_demag_impact,
    magnet_loss_aggregate,
    phase_flux_linkage,
)
from magcore.fem2d.machines.operating_point import (
    MagnetOperatingPoint,
    magnet_operating_point,
)
from magcore.fem2d.machines.pmsm_outrunner import MachineGeometry
from magcore.fem2d.machines.postproc import airgap_torque_arkkio, back_emf_constant
from magcore.fem2d.machines.static_solver import (
    MachineStaticResult,
    solve_machine_static,
)
from magcore.fem2d.machines.winding import WindingLayout

# P7: СОГЛАСОВАННАЯ МОДЕЛЬ ЗАДАЧИ — единая проверяемая структура всего входа (геометрия +
# материалы + обмотка + режим) + валидация (полнота, диапазоны, согласованность размеров) +
# ОДИН вызов «модель → полный статический расчёт P4–P6». Это deliverable гейта готовности к
# интерфейсу: с этого момента UI лишь заполняет MachineProblem и вызывает solve.


class Scenario(str, Enum):
    S1 = "S1"   # статика при 20 °C (поле магнита + опц. ток)
    S2 = "S2"   # статика при ЗАДАННОЙ температуре T
    # S3 = динамика/тепло (переходный, ядро К6′) — отдельный модуль, пока не реализован


@dataclass(frozen=True, slots=True)
class MachineProblem:
    """Полная постановка статической задачи машины (вход одного расчёта)."""

    geometry: MachineGeometry
    magnet: AnisotropicBHTMagnet
    steel: SteelBHCurve
    layout: WindingLayout
    scenario: Scenario = Scenario.S1
    T: float = 20.0                 # температура [°C]; S1 ⇒ 20, S2 ⇒ задана
    # режим тока (worst-case по-простому: амплитуда + эл. угол; ток включается при i_peak≠0)
    i_peak: float = 0.0
    gamma_elec: float = 0.0
    turns_per_slot: float = 0.0
    # параметры решателя
    relaxation: float = 0.1
    max_iter: int = 200
    tol: float = 1.0e-6

    @property
    def has_current(self) -> bool:
        return self.i_peak != 0.0 and self.turns_per_slot != 0.0

    def validate(self) -> list[str]:
        """Список проблем постановки (пустой ⇒ корректно). Полнота, диапазоны, согласованность."""
        p: list[str] = []
        if self.layout.n_slots != self.geometry.params.n_slots:
            p.append("layout.n_slots != geometry.params.n_slots (несогласованные размеры).")
        if not self.layout.is_balanced():
            p.append("обмотка несбалансирована (разное число пазов на фазу).")
        if not isinstance(self.scenario, Scenario):
            p.append("scenario должен быть Scenario.S1 | S2.")
        if self.scenario == Scenario.S1 and abs(self.T - 20.0) > 1e-9:
            p.append("S1 подразумевает T=20 °C; для иной температуры используйте S2.")
        limit = self.magnet.temperature_limit()
        if self.T >= limit:
            p.append(f"T={self.T:g} °C >= предел модели магнита {limit:.0f} °C (перегрев).")
        if self.T <= -273.15:
            p.append("T ниже абсолютного нуля.")
        if self.i_peak != 0.0 and self.turns_per_slot <= 0.0:
            p.append("ток задан (i_peak≠0), но turns_per_slot<=0 — источник тока не определён.")
        if self.turns_per_slot < 0.0:
            p.append("turns_per_slot < 0.")
        if not (0.0 < self.relaxation <= 1.0):
            p.append("relaxation вне (0, 1].")
        if self.max_iter < 1:
            p.append("max_iter < 1.")
        if self.tol <= 0.0:
            p.append("tol <= 0.")
        return p

    def check(self) -> None:
        """Бросить ValueError, если постановка некорректна (список всех проблем)."""
        issues = self.validate()
        if issues:
            raise ValueError("Некорректная постановка задачи:\n  - " + "\n  - ".join(issues))


@dataclass(frozen=True, slots=True)
class MachineSolution:
    """Полный результат статического расчёта задачи (P4–P6 в одном месте)."""

    field: MachineStaticResult          # P4: поле + risk-map демага
    operating_point: MagnetOperatingPoint  # рабочая точка по объёму магнита
    loss: MagnetLossAggregate           # P5: агрегаты необратимой потери
    torque: float                       # P6: момент [Н·м] в режиме
    pm_flux_linkage: np.ndarray         # (3,) потокосцепление ПМ (ХХ) [Вб]
    back_emf_constant: float            # P6: K_e = K_t [В·с/рад]
    impact: DemagImpact | None          # P5 полный (падение ЭДС/момента), если запрошен


def solve_machine_problem(
    problem: MachineProblem, *, assess_impact: bool = False
) -> MachineSolution:
    """
    Один вызов «модель → полный расчёт»: валидирует постановку, решает поле в режиме (P4),
    извлекает рабочую точку по объёму, агрегаты потери (P5), момент и ЭДС-постоянную (P6).
    `assess_impact=True` дополнительно считает падение ЭДС/момента из-за необратимого демага
    (P5 evaluate_demag_impact, 3 решения). ЧИСТО СТАТИКА (нагрев/потери меди — модуль S2).
    """
    problem.check()
    g, magnet, steel, layout = problem.geometry, problem.magnet, problem.steel, problem.layout

    field = solve_machine_static(
        g, magnet, steel, T=problem.T,
        layout=layout if problem.has_current else None,
        i_peak=problem.i_peak, gamma_elec=problem.gamma_elec,
        turns_per_slot=problem.turns_per_slot,
        relaxation=problem.relaxation, max_iter=problem.max_iter, tol=problem.tol,
    )
    op = magnet_operating_point(g, field, magnet, T=problem.T)
    loss = magnet_loss_aggregate(g, field.risk)
    torque = airgap_torque_arkkio(g, field.B_cells)

    impact: DemagImpact | None = None
    n_turns = problem.turns_per_slot if problem.turns_per_slot > 0.0 else 1.0
    if problem.has_current and assess_impact:
        impact = evaluate_demag_impact(
            g, magnet, steel, layout, i_peak=problem.i_peak, gamma_elec=problem.gamma_elec,
            turns_per_slot=problem.turns_per_slot, T=problem.T,
            relaxation=problem.relaxation, max_iter=problem.max_iter,
        )
        lam_pm = impact.lam_nominal                    # ХХ ПМ уже посчитан внутри impact
    elif problem.has_current:
        # Нужен отдельный ХХ для ЭДС-постоянной (в режиме λ содержит вклад тока).
        nl = solve_machine_static(g, magnet, steel, T=problem.T,
                                  relaxation=problem.relaxation, max_iter=problem.max_iter)
        lam_pm = phase_flux_linkage(g, layout, nl.a, turns_per_slot=n_turns)
    else:
        lam_pm = phase_flux_linkage(g, layout, field.a, turns_per_slot=n_turns)  # режим = ХХ

    return MachineSolution(
        field=field, operating_point=op, loss=loss, torque=torque,
        pm_flux_linkage=lam_pm, back_emf_constant=back_emf_constant(g, lam_pm), impact=impact,
    )
