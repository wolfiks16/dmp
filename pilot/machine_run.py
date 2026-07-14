"""
CLI-интерфейс машины: `python -m pilot.machine_run <config.toml>`.

Читает конфиг (машина + материалы + обмотка + режим) → строит `MachineProblem` →
ВАЛИДИРУЕТ → один вызов полного статического расчёта (P4–P6) → печатает понятный отчёт
(момент, ЭДС, демаг, рабочая точка по объёму) и сохраняет картинки (|B|, карта риска,
рабочая точка) для РУЧНОЙ проверки. Интерфейс = обёртка; вся физика/валидация — в magcore.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from magcore.fem2d.machines import Region, solve_machine_problem
from pilot import viz
from pilot.machine_config import load_machine_run

try:  # консоль Windows (cp1251) не печатает юникод → переключаем на utf-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def _report(run, sol) -> str:
    p = run.problem
    op = sol.operating_point
    risk = sol.field.risk
    L = []
    L.append("=" * 66)
    L.append(f"  МАШИНА: outrunner SPM PMSM {p.geometry.params.n_slots}N{p.geometry.params.n_poles}P"
             f"  ·  материал: {run.material_name}")
    L.append(f"  СЦЕНАРИЙ: {p.scenario.value}   T = {p.T:.0f} °C"
             + (f"   ток: i_peak={p.i_peak:g} A, γ={np.degrees(p.gamma_elec):.0f}°, "
                f"N={p.turns_per_slot:g}" if p.has_current else "   (холостой ход)"))
    L.append("=" * 66)
    L.append(f"  Сходимость: {'ДА' if sol.field.converged else 'НЕТ'} "
             f"({sol.field.n_iterations} итераций)")
    L.append(f"  Момент (Arkkio):        {sol.torque:+.4f} Н·м")
    L.append(f"  ЭДС-постоянная K_e=K_t: {sol.back_emf_constant:.5f} В·с/рад")
    L.append("-" * 66)
    L.append("  ДЕМАГ (необратимое размагничивание):")
    L.append(f"    за коленом:      {risk.n_demagnetized}/{risk.cell_indices.size} ячеек"
             f"  ({sol.loss.demag_area_fraction*100:.1f}% площади)")
    L.append(f"    средняя потеря Br: {sol.loss.mean_loss_frac*100:.2f}%   "
             f"макс: {sol.loss.max_loss_frac*100:.1f}%")
    L.append(f"    худшая маржа к колену: {risk.worst_margin/1e3:+.0f} кА/м")
    if sol.impact is not None:
        L.append(f"    ПАДЕНИЕ ЭДС/момента из-за демага: {sol.impact.flux_linkage_drop_frac*100:.2f}%")
    L.append("-" * 66)
    L.append("  РАБОЧАЯ ТОЧКА МАГНИТА (по объёму):")
    L.append(f"    H_op: среднее(объёмн.) {op.volume_weighted_mean_H_op()/1e3:.0f} кА/м, "
             f"худшая {op.worst_H_op()/1e3:.0f} кА/м")
    L.append(f"    коэфф. проницаемости P_c (объёмн. среднее): {op.volume_weighted_mean_permeance():.2f}")
    L.append(f"    колено H_knee(T) = {op.knee_field/1e3:.0f} кА/м; "
             f"за коленом {op.volume_fraction_below(op.knee_field)*100:.1f}% объёма")
    L.append("=" * 66)
    return "\n".join(L)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Использование: python -m pilot.machine_run <config.toml>")
        return 2
    run = load_machine_run(argv[0])

    issues = run.problem.validate()
    if issues:
        print("НЕКОРРЕКТНАЯ ПОСТАНОВКА ЗАДАЧИ:")
        for s in issues:
            print("  -", s)
        return 1

    sol = solve_machine_problem(run.problem, assess_impact=run.assess_impact)
    report = _report(run, sol)
    print(report)

    out = run.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "machine_report.txt").write_text(report, encoding="utf-8")

    g = run.problem.geometry
    mesh = g.mesh
    magnet_mask = g.region == int(Region.MAGNET)
    viz.plot_B_field(mesh, sol.field.B_cells, title="Поле B (worst-case режим)",
                     save_path=out / "machine_B.png")
    viz.plot_demag_risk(mesh, magnet_mask, sol.field.risk, save_path=out / "machine_risk.png")
    viz.plot_magnet_cell_field(
        mesh, sol.operating_point.cell_indices, sol.operating_point.H_op / 1e3,
        title="Рабочая точка H_op по объёму магнита", label="H_op, кА/м",
        save_path=out / "machine_operating_point.png",
    )
    print(f"\nКартинки и отчёт сохранены в: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
