"""
Л-34: МАСШТАБ ОШИБКИ зубцового пролёта свипа ротора под нагрузкой — в рабочей точке карты B-full (сталь 10,
массивный ротор, NdFeB, среда 130 °C, ток 82 А = 24 А/мм², ось q, магнит и зазор 0,5 мм).

Одна волна поля в системе ротора за ПОЛНЫЙ оборот (144 положения = 12 на зубцовое деление) — эталон. Из неё:
потери ротора по каждому окну 60° (верный пролёт: период картины токов пазов у 12/14) и по каждому зубцовому
окну 30° (как считали прежние скрипты). Затем транзиент карты B-full (горизонт 45 с) с картой потерь ротора по
прежнему окну, по верному и по обороту; статор — один и тот же.

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/diag_rotor_span_l34.py [кэш_волны.npz]
Выход:   docs/papers/experiments/output_bfull/l34_span_scale.json; ≈20 мин на волну (кэш — по желанию).
Итог 2026-10-14 — в docs/pitfalls.md, Л-34.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
import scenario_paper1 as cfg  # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet  # noqa: E402
from magcore.fem2d.machines.iron_loss import stator_iron_loss_density  # noqa: E402
from magcore.fem2d.machines.magnet_loss import (  # noqa: E402
    magnet_segment_width, rotor_field_period, rotor_frame_B_series, rotor_side_cells,
    rotor_side_loss_density_from_waveform)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, build_outrunner_spm_pmsm  # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario  # noqa: E402
from magcore.fem2d.machines.thermal_scenario import run_machine_thermal_demag  # noqa: E402

OUT = os.path.join(HERE, "output_bfull")
STEEL = "steel10"
T_AMB = 130.0
I_PEAK = round(cfg.current_for_density(24.0), 1)          # точка карты B-full с расхождением Ньютона и Пикара
GAMMA = math.pi / 2.0
PPT = 12                                                   # положений на зубцовое деление, как в скриптах
params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=0.004, mesh_size_by_region=cfg.critical_mesh())
mag = n42sh_magnet((1.0, 0.0, 0.0))
steel = cfg.STEEL_KINDS[STEEL]["curve"]()
skw = cfg.steel_loss_kwargs(STEEL)
geo0 = build_outrunner_spm_pmsm(params)
mi, ri = rotor_side_cells(geo0)
area = np.array([geo0.mesh.cell_area(c) for c in range(geo0.mesh.n_cells)])
L = params.axial_length
print("ячеек %d, магнита %d, ярма ротора %d; ток %.1f А; период поля под нагрузкой %.0f°"
      % (geo0.mesh.n_cells, mi.size, ri.size, I_PEAK, math.degrees(rotor_field_period(params, loaded=True))), flush=True)

wave_file = sys.argv[1] if len(sys.argv) > 1 else None
if wave_file and os.path.exists(wave_file):
    d = np.load(wave_file)
    B, f_full = d["B"], float(d["f"])
else:
    t = time.time()
    B, f_full = rotor_frame_B_series(
        params, mag, steel, speed_rpm=cfg.N_OPER, cell_idx=np.concatenate([mi, ri]), i_peak=I_PEAK,
        gamma_elec=GAMMA, turns_per_slot=cfg.TURNS_PER_SLOT, T=T_AMB, mech_span=2 * math.pi,
        n_positions=PPT * params.n_slots, relaxation=0.1, max_iter=300)
    if wave_file:
        np.savez(wave_file, B=B, f=f_full)
    print("волна за оборот: %d положений, %.0f с" % (B.shape[0], time.time() - t), flush=True)

lkw = dict(sigma_pm=cfg.magnet_conductivity("ndfeb", T_AMB), magnet_seg_width=magnet_segment_width(params, cfg.N_SEG),
           steinmetz=skw.get("steinmetz"), rotor_solid=skw.get("rotor_solid", False),
           sigma_rotor=skw.get("sigma_rotor"), rotor_thickness=skw.get("rotor_thickness"),
           rotor_mu_r=skw.get("rotor_mu_r", 500.0))


def rotor_loss(Bw, f):
    q = rotor_side_loss_density_from_waveform(geo0, Bw, f, **lkw)
    return q, float((q[mi] * area[mi]).sum() * L), float((q[ri] * area[ri]).sum() * L)


N = B.shape[0]
q_full, m_full, y_full = rotor_loss(B, f_full)
print("\nПОТЕРИ РОТОРА [Вт]          магнит    ярмо   сумма")
print("  полный оборот (эталон)  %7.2f %7.2f %7.2f" % (m_full, y_full, m_full + y_full))
res = {"full": [m_full, y_full]}
for name, n_win in (("окно 60° (верный пролёт)", PPT * 2), ("зубцовое окно 30° (прежний)", PPT)):
    vals = np.array([rotor_loss(B[s0:s0 + n_win], f_full * N / n_win)[1:] for s0 in range(0, N, n_win)])
    res[name] = vals.tolist()
    print("  %-26s магнит %.2f…%.2f, ярмо %.2f…%.2f, сумма %.2f…%.2f (окон %d)"
          % (name, vals[:, 0].min(), vals[:, 0].max(), vals[:, 1].min(), vals[:, 1].max(),
             vals.sum(axis=1).min(), vals.sum(axis=1).max(), len(vals)))

# Что это даёт магниту: транзиент карты B-full с картой потерь ротора по ПРЕЖНЕМУ окну (0…30°, как в скриптах),
# по ВЕРНОМУ (0…60°) и по обороту; статор — один и тот же.
q_stator, _ = stator_iron_loss_density(
    params, mag, steel, speed_rpm=cfg.N_OPER, i_peak=I_PEAK, gamma_elec=GAMMA, turns_per_slot=cfg.TURNS_PER_SLOT,
    T=T_AMB, coeffs=skw.get("steinmetz"), n_positions=12, relaxation=0.1, max_iter=300)
q_old = rotor_loss(B[:PPT], f_full * N / PPT)[0]
q_new = rotor_loss(B[:2 * PPT], f_full * N / (2 * PPT))[0]
sc = machine_scenario(params, mag, steel)
res["transient"] = {}
for name, q_rot in (("прежний 30°", q_old), ("верный 60°", q_new), ("полный оборот", q_full)):
    q = q_stator + q_rot
    r = run_machine_thermal_demag(
        sc, i_peak=I_PEAK, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=GAMMA, slot_fill=cfg.SLOT_FILL,
        h=cfg.H_OUT, T_amb=T_AMB, h_in=cfg.H_IN, T_frame=T_AMB, dt=0.5, n_steps=90,
        thermal=cfg.build_thermal_props("ndfeb", cfg.N_OPER, STEEL), T0=T_AMB, T_cap=cfg.T_CAP,
        steady_tol=cfg.STEADY_TOL, max_substeps=8, extra_loss=lambda Tc, em, _q=q: _q)
    tr = r.transient
    res["transient"][name] = dict(T_magnet=float(r.T_magnet_max), T_max=float(tr.T_max.max()),
                                  ke_drop=float(r.torque_constant_drop), ret_min=float(r.retention.min()))
    print("  %-14s T_магн %.1f °C, T_max %.1f °C, падение K_e %.2f %%, r_min %.3f, t %.1f с | %s"
          % (name, r.T_magnet_max, float(tr.T_max.max()), 100 * r.torque_constant_drop, float(r.retention.min()),
             float(tr.times[-1]), tr.stop_reason[:40]), flush=True)
json.dump(res, open(os.path.join(OUT, "l34_span_scale.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
