# -*- coding: utf-8 -*-
"""
ПОДБОР РЕЖИМА ДЛИТЕЛЬНОЙ РАБОТЫ: при каком токе связка выходит на РАВНОВЕСИЕ,
а не упирается в предел изоляции, и до какой температуры при этом греется магнит.

Зачем. В карте B-full (ток 40…140 А, горизонт 45 с) магнит НИ РАЗУ не превысил ~170 °C:
  · при среде 100–130 °C расчёт останавливался по T_cap = 200 °C на ОБМОТКЕ за 24–43 с;
  · при токах ≥70 А обмотка выгорала за 1–7 с, и магнит оставался ДАЖЕ ХОЛОДНЕЕ (61–140 °C);
  · при среде 40–70 °C останов был «горизонт исчерпан» — то есть температура ещё росла.
Значит ось «температура среды» магнит почти не греет, и сравнение материалов шло в
диапазоне, где преимущество Sm-Co не проявляется. Нужен режим, в котором связка реально
УСТАНАВЛИВАЕТСЯ ниже предела изоляции — тогда температура магнита будет рабочей, а не
«сколько успел за 45 секунд».

Сетка здесь ГРУБАЯ намеренно: подбирается ТЕПЛОВОЙ режим, а тепловая часть по сетке сошлась
(T магнита 163.8 / 164.1 / 162.3 °C на трёх уровнях, см. fig_risk_map.py). Демагу на грубой
сетке верить нельзя — итоговый свип пойдёт на сетке cfg.critical_mesh().

Запуск: PYTHONPATH=<repo> python docs/papers/experiments/probe_steady.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
import scenario_paper1 as cfg                                              # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet       # noqa: E402
from magcore.fem2d.machines.magnet_loss import magnet_segment_width        # noqa: E402
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams      # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario               # noqa: E402
from magcore.fem2d.machines.thermal_scenario import run_machine_thermal_demag  # noqa: E402

MESH = 0.004                 # ГРУБАЯ — см. шапку
DT, N_STEPS = 0.5, 1400      # горизонт 700 с вместо прежних 45 с
GAMMA = math.pi / 2.0
AMBIENTS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else [70, 100, 130])]
CURRENTS = [float(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else [10, 15, 20, 25, 30])]
MATERIAL = sys.argv[3] if len(sys.argv) > 3 else "ndfeb"
FINE = len(sys.argv) > 4 and sys.argv[4] == "fine"      # сходившаяся сетка (магнит+зазор)


def run(i_peak: float, T_amb: float):
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH,
                                 mesh_size_by_region=cfg.critical_mesh() if FINE else None)
    mag = (n42sh_magnet if MATERIAL == "ndfeb" else sm2co17_magnet)((1.0, 0.0, 0.0))
    sc = machine_scenario(params, mag, cfg.STEEL_KINDS["steel10"]["curve"]())
    return run_machine_thermal_demag(
        sc, i_peak=i_peak, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=GAMMA,
        slot_fill=cfg.SLOT_FILL, h=cfg.H_OUT, T_amb=T_amb, h_in=cfg.H_IN, T_frame=T_amb,
        dt=DT, n_steps=N_STEPS,
        thermal=cfg.build_thermal_props(MATERIAL, cfg.N_OPER, "steel10"),
        T0=T_amb, T_cap=cfg.T_CAP, steady_tol=cfg.STEADY_TOL, max_substeps=8,
        core_losses=True, speed_rpm=cfg.N_OPER,
        sigma_pm=cfg.magnet_conductivity(MATERIAL, T_amb),
        magnet_seg_width=magnet_segment_width(params, cfg.N_SEG),
        loss_mech_span=2.0 * math.pi / 12, loss_n_positions=12,
        **cfg.steel_loss_kwargs("steel10"),
    )


def main():
    print("ПОДБОР УСТАНОВИВШЕГОСЯ РЕЖИМА (%s, сталь 10, q-ось, сетка %s)"
          % (MATERIAL, "СХОДИВШАЯСЯ магнит+зазор 0.5 мм" if FINE else "грубая %.3f м" % MESH))
    print("горизонт %.0f с, T_cap = %.0f °C, допуск установления %.3g °C/с"
          % (DT * N_STEPS, cfg.T_CAP, cfg.STEADY_TOL))
    print("%-6s %-5s %9s %9s %8s  %s" % ("среда", "ток", "T_магн", "T_max", "t_кон", "исход"))
    t0 = time.time()
    for Ta in AMBIENTS:
        for ip in CURRENTS:
            t = time.time()
            try:
                r = run(ip, Ta)
                tr = r.transient
                # ⚠ печатаем СЫРУЮ причину останова: своя классификация уже один раз
                #   скрыла настоящий механизм (см. историю правок).
                print("%-6.0f %-5.0f %9.1f %9.1f %8.0f  разгон=%s каскад=%s заколен=%d "
                      "ret_min=%.3f" % (Ta, ip, r.T_magnet_max, max(tr.T_max), tr.times[-1],
                                        tr.runaway, tr.magnet_cascade, tr.n_past_knee[-1],
                                        r.retention.min()), flush=True)
                print("        причина: %s  [%.0f с]" % (tr.stop_reason, time.time() - t),
                      flush=True)
            except Exception as e:      # noqa: BLE001 — перегрев за предел модели = результат
                print("%-6.0f %-5.0f %9s %9s %8s  ошибка: %s" % (Ta, ip, "—", "—", "—",
                                                                 str(e)[:70]), flush=True)
    print("[всего %.0f с]" % (time.time() - t0))


if __name__ == "__main__":
    main()
