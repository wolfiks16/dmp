# -*- coding: utf-8 -*-
"""
АБЛЯЦИЯ А — сколько стоила скалярная ν магнита.

ЗАЧЕМ. Заявленный результат ядра К6′ — необратимое размагничивание и каскад. Аудит
2026-09-03 показал, что прежняя сборка отдавала повреждённой ячейке наклон рабочей ветви
ИЗОТРОПНО, то есть и ПОПЕРЁК лёгкой оси, где у магнита постоянная μ_⊥. Повреждённая
ячейка становилась проводником потока вбок и ухудшала рабочую точку соседей — механизм,
неотличимый от заявленного каскада. Артефакт возникает только там, где есть ячейки за
коленом, то есть у NdFeB и не у SmCo ⇒ в сравнении материалов он сидит на ОДНОЙ стороне.

ТРИ ВЕТВИ (условия побитово одинаковы, различается только закон ν магнита):
  (а) скаляр       — прежнее поведение: наклон ветви во все стороны;
  (б) тензор, изотропный материал (μ_⊥ = μ_rec) — утечка убрана, анизотропии ещё нет;
  (в) тензор, физический μ_⊥ — полный правильный закон.
Разность (а)−(б) есть ЦЕНА УТЕЧКИ; (б)−(в) — вклад собственно анизотропии.

⚠ ПОЧЕМУ СРАВНИВАЕМ НЕ КОНЕЧНЫЕ СОСТОЯНИЯ. Первый прогон (2026-09-07) показал, что ветви
ОСТАНАВЛИВАЮТСЯ ПО РАЗНЫМ ПРИЧИНАМ: (а) срывается в каскад при T_магн=165 °C, (б) и (в)
доходят до предела изоляции T_cap=200 °C. Конечные состояния поэтому относятся к разным
моментам траектории, и их прямое сравнение показывало бы «больше повреждений у (в)» лишь
потому, что (в) успела прогреться дольше. Сравнение ведётся В ОБЩЕЙ ТОЧКЕ: по одинаковому
времени и по одинаковой температуре магнита.

⚠ УСЛОВИЯ ПРОГОНА — ещё НЕ исправленные (этап 2 плана): ток 40 А это ПИКОВЫЙ режим
(33,4 А/мм²), а не номинальный. Для абляции это осознанно — режим максимально активен по
размагничиванию, чувствительность к закону ν наибольшая. В отчёт идут только РАЗНОСТИ
между ветвями, не абсолютные числа.

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/ablation_nu_tensor.py [ambient] [i_peak]
         ... --compare   — только пересчитать сравнение по сохранённым историям
"""
from __future__ import annotations

import dataclasses
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

import numpy as np                                                        # noqa: E402

import scenario_paper1 as cfg                                             # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet      # noqa: E402
from magcore.fem2d.machines.magnet_loss import magnet_segment_width       # noqa: E402
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams     # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario              # noqa: E402
from magcore.fem2d.machines.thermal_scenario import run_machine_thermal_demag  # noqa: E402

# Физическая поперечная проницаемость. Поперечный отклик спечённого магнита — когерентный
# поворот моментов против поля анизотропии: chi_perp = Js^2/(2*mu0*K1).
#   NdFeB (Nd2Fe14B): mu_perp ~ 1.17   [Ibrayeva & Eriksson, IEEE Trans. Magn. 59(9) 2023,
#                                       DOI 10.1109/TMAG.2023.3296966 — исходные данные
#                                       анизотропной МКЭ-модели магнита; там же mu_par~1.04]
#   Sm2Co17:          mu_perp ~ 1.16   [оценка по chi_perp = Js^2/(2*mu0*K1), Js~1.15 Тл,
#                                       K1~3.3 МДж/м³ — ДОВЕРИЕ НИЖЕ, требует сверки]
MU_PERP = {"ndfeb": 1.17, "smco": 1.16}

AMBIENT = 130.0
I_PEAK = 40.0
GAMMA = 0.0
N_STEPS = 240                      # горизонт 120 с при dt = 0.5
OUT = HERE / "output_ablation_nu"
ARMS = (("a_scalar", "а) скаляр (было)", False, False),
        ("b_tensor_iso", "б) тензор, mu_perp=mu_rec", True, False),
        ("c_tensor_phys", "в) тензор, mu_perp физ.", True, True))


def run(material: str, *, anisotropic: bool, physical_perp: bool):
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.003,
                                 mesh_size_by_region=cfg.critical_mesh())
    mag = (n42sh_magnet if material == "ndfeb" else sm2co17_magnet)((1.0, 0.0, 0.0))
    if physical_perp:
        mag = dataclasses.replace(mag, mu_perp=MU_PERP[material])
    sc = machine_scenario(params, mag, cfg.STEEL_KINDS["steel10"]["curve"]())
    return run_machine_thermal_demag(
        sc, i_peak=I_PEAK, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=GAMMA,
        slot_fill=cfg.SLOT_FILL, h=cfg.H_OUT, T_amb=AMBIENT, h_in=cfg.H_IN, T_frame=AMBIENT,
        dt=cfg.DT, n_steps=N_STEPS,
        thermal=cfg.build_thermal_props(material, cfg.N_OPER, "steel10"),
        T0=AMBIENT, T_cap=cfg.T_CAP, steady_tol=cfg.STEADY_TOL, max_substeps=8,
        core_losses=True, speed_rpm=cfg.N_OPER,
        sigma_pm=cfg.magnet_conductivity(material, AMBIENT),
        magnet_seg_width=magnet_segment_width(params, cfg.N_SEG),
        loss_mech_span=2.0 * math.pi / 12, loss_n_positions=12,
        anisotropic_magnet=anisotropic,
        **cfg.steel_loss_kwargs("steel10"),
    )


def harvest(material: str = "ndfeb") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for key, label, aniso, phys in ARMS:
        t0 = time.time()
        res = run(material, anisotropic=aniso, physical_perp=phys)
        tr = res.transient
        np.savez(OUT / ("%s_%s.npz" % (material, key)),
                 times=tr.times, T_magnet=tr.T_magnet, ret_min=tr.retention_min,
                 ret_mean=tr.retention_mean, n_past_knee=tr.n_past_knee,
                 T_max=tr.T_max, retention=res.retention,
                 ke_drop=np.array([100.0 * res.torque_constant_drop]),
                 cascade=np.array([tr.magnet_cascade]), runaway=np.array([tr.runaway]),
                 stop=np.array([tr.stop_reason], dtype=object), wall=np.array([time.time() - t0]))
        print("%-28s %s  [%.0f с]" % (label, tr.stop_reason[:52], time.time() - t0), flush=True)


def _at_time(d, t: float, key: str) -> float:
    return float(np.interp(t, d["times"], d[key]))


def _at_temperature(d, T: float, key: str) -> float:
    """Значение в момент ПЕРВОГО достижения температуры магнита T (нагрев монотонен)."""
    Tm, y = np.asarray(d["T_magnet"], dtype=float), np.asarray(d[key], dtype=float)
    ok = np.isfinite(Tm)
    Tm, y = Tm[ok], y[ok]
    if Tm.max() < T:
        return float("nan")
    return float(np.interp(T, Tm, y))


def compare(material: str = "ndfeb") -> None:
    data = {}
    for key, label, *_ in ARMS:
        p = OUT / ("%s_%s.npz" % (material, key))
        if not p.exists():
            print("нет файла %s — сначала прогон без --compare" % p.name)
            return
        data[key] = dict(np.load(p, allow_pickle=True))

    print("=" * 100)
    print("АБЛЯЦИЯ А (%s): цена скалярной ν магнита" % material.upper())
    print("условия: сталь 10, %d об/мин, среда %.0f °C, ток %.0f A (ПИКОВЫЙ), "
          "сетка магнит+зазор 0.5 мм" % (cfg.N_OPER, AMBIENT, I_PEAK))
    print("=" * 100)

    print("\n1. ИСХОД РАСЧЁТА — качественный результат, сравнение корректно как есть")
    print("%-28s %9s %9s  %s" % ("ветвь", "T_магн", "время,с", "причина остановки"))
    for key, label, *_ in ARMS:
        d = data[key]
        print("%-28s %9.1f %9.1f  %s"
              % (label, np.nanmax(d["T_magnet"]), d["times"][-1], str(d["stop"][0])[:46]))

    t_common = min(float(data[k]["times"][-1]) for k, *_ in ARMS)
    T_common = min(float(np.nanmax(data[k]["T_magnet"])) for k, *_ in ARMS)
    T_probe = math.floor(T_common - 1.0)

    for title, getter, point in (
        ("2. В ОБЩИЙ МОМЕНТ ВРЕМЕНИ t = %.1f с" % t_common, _at_time, t_common),
        ("3. ПРИ ОДИНАКОВОЙ ТЕМПЕРАТУРЕ МАГНИТА T = %.0f °C" % T_probe, _at_temperature, T_probe),
    ):
        print("\n" + title)
        print("%-28s %10s %10s %10s" % ("ветвь", "ret_min", "ret_ср", "заколен"))
        base = None
        for key, label, *_ in ARMS:
            d = data[key]
            row = [getter(d, point, "ret_min"), getter(d, point, "ret_mean"),
                   getter(d, point, "n_past_knee")]
            print("%-28s %10.4f %10.5f %10.1f" % (label, *row))
            if base is None:
                base = row
            else:
                print("%-28s %+10.4f %+10.5f %+10.1f   <- сдвиг к (а)"
                      % ("", row[0] - base[0], row[1] - base[1], row[2] - base[2]))

    print("\n4. ЦЕНА ИТЕРАЦИИ (косвенный признак: крутая изотропная ν усложняет Пикар)")
    for key, label, *_ in ARMS:
        print("%-28s %8.0f с" % (label, data[key]["wall"][0]))


if __name__ == "__main__":
    mat = "ndfeb"
    if "--compare" not in sys.argv:
        harvest(mat)
    compare(mat)
