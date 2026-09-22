"""
P-B7 — B-full харвест Статьи 1: собранный пред-регистрированный сценарий (scenario_paper1)
со ВСЕЙ физикой (дифф. ГУ статор→рама, Тейлор-зазор, core_losses = сталь+магнит+ротор),
worst-case d-ось, горячая среда. Режимы: MODE='smoke' (быстрая проверка сборки) / 'full' (свип).

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/harvest_bfull.py [smoke|full]
Выход:   docs/papers/experiments/output_bfull/
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))   # scenario_paper1 лежит рядом
import scenario_paper1 as cfg  # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import m270_35a_bh_curve
from magcore.fem2d.machines.magnet_loss import magnet_segment_width
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams
from magcore.fem2d.machines.scenario import machine_scenario
from magcore.fem2d.machines.thermal_scenario import run_machine_thermal_demag
from magcore.fem2d.runaway import runaway_threshold
from magcore.fem2d.spaces import LagrangeP1Space2D

MODE = sys.argv[1] if len(sys.argv) > 1 else "smoke"
STEEL_KIND = sys.argv[2] if len(sys.argv) > 2 else "steel10"   # магнитопровод: steel10|m270
OUT = Path(__file__).resolve().parent / "output_bfull"

# Оси карты (пред-регистрированы): среда — литературный диапазон (каркас на солнце 55-65°,
# агродрон держат <120°, eVTOL-пики >150° — см. sources_registry.md §4); ток — номинал 40 А …
# перегруз. Свипуются ОБА материала при ПОБИТОВО одинаковых условиях (§12 п.2).
AMBIENTS = [40.0, 70.0, 100.0, 130.0]
# ⚠ Токи ПЕРЕСЧИТАНЫ 2026-09-08 после правки обмотки (20 → 7 вит./паз, аудит 2026-09-03).
#   Прежний список [40, 70, 100, 140] А при 20 витках давал 33…117 А/мм² — свип начинался
#   НА ПИКЕ и уходил за предел разрушения. Теперь токи выводятся из ПЛОТНОСТИ ТОКА, для
#   которой есть нормы (см. `scenario_paper1.LOAD_REGIMES`).
MAP_CURRENTS = [round(cfg.current_for_density(j), 1) for j in (12.0, 18.0, 24.0, 30.0)]

CONF = {
    "smoke": dict(mesh=0.004, currents=[round(cfg.current_for_density(24.0), 1)],
                  ambients=[cfg.T_AMB_HOT], dt=0.5, n_steps=60,
                  loss_np=12, mech_span=2.0 * math.pi / 12, substeps=8),
    "map": dict(mesh=0.004, currents=MAP_CURRENTS, ambients=AMBIENTS, dt=0.5, n_steps=90,
                loss_np=12, mech_span=2.0 * math.pi / 12, substeps=8),
    "full": dict(mesh=0.003, currents=cfg.sweep_currents(), ambients=[cfg.T_AMB_HOT], dt=cfg.DT,
                 n_steps=cfg.N_STEPS, loss_np=48, mech_span=2.0 * math.pi, substeps=8),
}[MODE]


def _magnet(material):
    return (n42sh_magnet if material == "ndfeb" else sm2co17_magnet)((1.0, 0.0, 0.0))


def run_point(material, i_peak, T_amb, gamma_elec):
    # магнит и зазор — мельчайшая сетка (политика cfg.critical_mesh, уровень по сходимости)
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=CONF["mesh"],
                                 mesh_size_by_region=cfg.critical_mesh())
    sc = machine_scenario(params, _magnet(material), cfg.STEEL_KINDS[STEEL_KIND]["curve"]())
    return run_machine_thermal_demag(
        sc, i_peak=i_peak, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=gamma_elec,
        slot_fill=cfg.SLOT_FILL, h=cfg.H_OUT, T_amb=T_amb, h_in=cfg.H_IN, T_frame=T_amb,
        dt=CONF["dt"], n_steps=CONF["n_steps"],
        thermal=cfg.build_thermal_props(material, cfg.N_OPER, STEEL_KIND),
        T0=T_amb, T_cap=cfg.T_CAP, steady_tol=cfg.STEADY_TOL, max_substeps=CONF["substeps"],
        core_losses=True, speed_rpm=cfg.N_OPER, sigma_pm=cfg.magnet_conductivity(material, T_amb),
        magnet_seg_width=magnet_segment_width(params, cfg.N_SEG),
        loss_mech_span=CONF["mech_span"], loss_n_positions=CONF["loss_np"],
        **cfg.steel_loss_kwargs(STEEL_KIND),
    )


def energy_balance_err(tr):
    st = np.asarray(tr.stored_energy); lp = np.asarray(tr.loss_power); of = np.asarray(tr.outflow)
    if st.size < 3:
        return float("nan")
    d = np.diff(st) / CONF["dt"]
    return float(np.max(np.abs(d[1:] - (lp[2:] - of[2:])) / np.maximum(np.abs(lp[2:]), 1.0)))


def summarize(res):
    tr = res.transient
    return dict(
        T_magnet_max=float(res.T_magnet_max), ke_drop=float(res.torque_constant_drop),
        ret_min=float(res.retention.min()), survived=bool(res.survived),
        runaway=bool(tr.runaway), cascade=bool(tr.magnet_cascade),
        n_knee_end=int(tr.n_past_knee[-1]), loss0=float(tr.loss_power[0]),
        loss_end=float(tr.loss_power[-1]), t_end=float(tr.times[-1]), n_steps=int(len(tr.times)),
        eb_err=energy_balance_err(tr), stop=str(tr.stop_reason),
        times=[float(x) for x in tr.times], T_max=[float(x) for x in tr.T_max],
        T_magnet=[float(x) for x in tr.T_magnet], retention_mean=[float(x) for x in tr.retention_mean],
        n_past_knee=[int(x) for x in tr.n_past_knee],
    )


def log(*a):
    print(*a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("=" * 78)
    log("P-B7 B-full харвест [MODE=%s, магнитопровод=%s%s]: outrunner 12/14, рабочая точка q-оси"
        % (MODE, STEEL_KIND, ", ротор МАССИВНЫЙ" if cfg.STEEL_KINDS[STEEL_KIND]["rotor_solid"]
           else ", ротор шихтованный"))
    log("  сетка=%.4f (магнит/зазор %.4f/%.4f) dt=%.2f n_steps=%d loss_np=%d"
        % (CONF["mesh"], cfg.MESH_MAGNET, cfg.MESH_AIR_GAP, CONF["dt"],
           CONF["n_steps"], CONF["loss_np"]))
    log("  среды=%s °C   токи=%s А   (ОДИНАКОВЫ обоим материалам, §12)"
        % (CONF["ambients"], CONF["currents"]))
    log("  ⚠ ОДНА конфигурация = отработка физики; вывод о материале НЕ переносится на класс машин")
    log("=" * 78)
    report = {"mode": MODE, "steel_kind": STEEL_KIND,
              "conf": {k: (v if not isinstance(v, float) else round(v, 5))
                       for k, v in CONF.items()}, "ndfeb": [], "smco": []}

    # V — порог разгона s_crit (быстро, независимо от сценария)
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=CONF["mesh"],
                                 mesh_size_by_region=cfg.critical_mesh())
    sc0 = machine_scenario(params, _magnet("ndfeb"), m270_35a_bh_curve())
    space = LagrangeP1Space2D(sc0.geometry.mesh)
    k_cells, _ = cfg.build_thermal_props("ndfeb", cfg.N_OPER).cell_fields(sc0.geometry)
    s_crit = float(runaway_threshold(space, k_cells, cfg.H_OUT))
    report["s_crit"] = s_crit
    log("V. s_crit=%.4g" % s_crit)

    # РАБОЧАЯ ТОЧКА: демаг И момент считаем на ОДНОЙ q-оси (MTPA, γ=90°) — согласованно и без
    # подгонки. Демаг здесь ТЕМПЕРАТУРНЫЙ (реальная работа греет магнит), а не полевой. Прежние
    # варианты сняты: γ=0 был мислейбл (намагничивающий), «худший по мгновенному полю» γ грел
    # магнит МЕНЬШЕ (демаг температурный) — оба несогласованы с моментом (решение 2026-08-26).
    # cfg.worst_case_gamma оставлена дремать для будущего раздела «авария/field-weakening».
    op_gamma = math.pi / 2.0
    report["op_gamma_deg"] = math.degrees(op_gamma)
    log("V. рабочая точка γ = %.0f° (q-ось, MTPA; демаг и момент согласованы)" % math.degrees(op_gamma))

    # свип: среда × ток × материал (в smoke/full — одна среда)
    n_pt = len(CONF["ambients"]) * len(CONF["currents"]) * 2
    done = 0
    for Ta in CONF["ambients"]:
        for ip in CONF["currents"]:
            for tag, key in (("NdFeB", "ndfeb"), ("SmCo", "smco")):
                t = time.time()
                try:
                    s = summarize(run_point(key, ip, Ta, op_gamma))
                except Exception as e:      # перегрев за предел модели магнита = результат, не сбой
                    s = dict(error=str(e)[:160], T_magnet_max=float("nan"), ke_drop=float("nan"),
                             ret_min=float("nan"), survived=False, runaway=True, cascade=False,
                             loss0=float("nan"), loss_end=float("nan"), eb_err=float("nan"),
                             stop="исключение: %s" % str(e)[:80])
                s["i_peak"] = ip
                s["T_amb"] = Ta
                report[key].append(s)
                done += 1
                log("  [%2d/%d] %-5s T_ср=%3.0f i=%3.0f: T_mag=%6.1f  K_e-drop=%6.2f%%  ret_min=%.3f"
                    "  surv=%s run=%s casc=%s  eb=%.0e  [%.0fs]"
                    % (done, n_pt, tag, Ta, ip, s["T_magnet_max"], 100 * s["ke_drop"], s["ret_min"],
                       s["survived"], s["runaway"], s["cascade"], s["eb_err"], time.time() - t))
        (OUT / ("summary_%s_%s.json" % (MODE, STEEL_KIND))).write_text(   # промежуточное по строкам
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    tag_out = "summary_%s_%s.json" % (MODE, STEEL_KIND)
    (OUT / tag_out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n[%s сохранён; всего %.0f с]" % (tag_out, time.time() - t0))
    log("=" * 78)


if __name__ == "__main__":
    main()
