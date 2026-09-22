"""
АБСОЛЮТНЫЕ инженерные показатели (момент / КПД / тяга) по картам — вердикт выбора материала.

ПОЧЕМУ: `K_e-drop` — метрика ОТНОСИТЕЛЬНАЯ (потеря к собственному номиналу материала) и
материалы между собой НЕ сравнивает: SmCo может иметь 0 % потери и всё равно давать меньше
момента (у него Br 1.05 против 1.29 Тл). Вердикт «какой материал лучше на этом режиме»
решается АБСОЛЮТНЫМИ показателями (замечание Sergey, 2026-08-03).

ЧТО СЧИТАЕТСЯ (для каждой точки карты, при ОДИНАКОВЫХ токе и скорости — честная база):
  • момент [Н·м] в конечном тепловом/повреждённом состоянии,
  • K_t, K_e [абсолютные],
  • КПД = P_вых/(P_вых+ΣP) со ВСЕМИ потерями из связанного расчёта (медь+сталь+магнит+ротор),
  • мощность на валу [Вт] и ТЯГА винта (для фикс. винта T ∝ P^{2/3}).

КАК: состояние берётся из карты — T магнита и доля 1-й гармоники ремнантности
(= 1 − K_e-drop). Она и есть инвариантная мера необратимого повреждения, определяющая
K_e/K_t (см. `thermal_scenario.magnet_fundamental_ratio`), поэтому подставляется как
равномерная эквивалентная ремнантность. Обратимое падение входит через Br(T).

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/performance_paper1.py [steel10|m270]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_paper1 as cfg  # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet  # noqa: E402
from magcore.fem2d.machines.characteristics import (  # noqa: E402
    machine_characteristics,
    phase_resistance,
)
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region  # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario  # noqa: E402

KIND = sys.argv[1] if len(sys.argv) > 1 else "steel10"
OUT = Path(__file__).resolve().parent / "output_bfull"
MESH = 0.004
OMEGA = 2.0 * math.pi * cfg.N_OPER / 60.0        # рад/с при рабочей скорости
MAGNET_DENSITY = {"ndfeb": 7600.0, "smco": 8400.0}


def _scenario(material):
    # та же политика сетки, что в харвесте: магнит и зазор — мельчайшие (cfg.critical_mesh)
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH,
                                 mesh_size_by_region=cfg.critical_mesh())
    magnet = (n42sh_magnet if material == "ndfeb" else sm2co17_magnet)((1.0, 0.0, 0.0))
    return machine_scenario(params, magnet, cfg.STEEL_KINDS[KIND]["curve"]())


def perf_at_state(sc, material, *, T_magnet, ratio, i_peak, T_winding, loss_w_per_m):
    """Абсолютные показатели при заданном состоянии магнита (T + равномерная ремнантность)."""
    n_mag = int(np.count_nonzero(sc.geometry.mask(Region.MAGNET)))
    ch = machine_characteristics(
        sc, i_peak=i_peak, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=math.pi / 2,  # q-ось: момент
        T=T_magnet, retention=np.full(n_mag, float(ratio)),
        magnet_density=MAGNET_DENSITY[material], relaxation=0.1, max_iter=300,
    )
    torque = abs(ch.torque)
    p_out = torque * OMEGA
    p_loss = float(loss_w_per_m) * sc.geometry.params.axial_length      # [Вт/м]→[Вт]
    eff = p_out / (p_out + p_loss) if (p_out + p_loss) > 0 else 0.0
    R = phase_resistance(sc.geometry, turns_per_slot=cfg.TURNS_PER_SLOT,
                         slot_fill=cfg.SLOT_FILL, T=T_winding)
    return dict(torque=torque, K_t=ch.torque_constant, K_e=ch.emf_constant,
                flux=ch.flux_linkage, p_out=p_out, p_loss=p_loss, efficiency=eff,
                R_phase=R, magnet_mass=ch.magnet_mass, converged=ch.converged)


def main():
    path = OUT / ("summary_map_%s.json" % KIND)
    if not path.exists():                                   # первая карта сохранялась без суффикса
        alt = OUT / "summary_map.json"
        path = alt if alt.exists() else path
    R = json.loads(path.read_text(encoding="utf-8"))
    print("=" * 100)
    print("АБСОЛЮТНЫЕ ПОКАЗАТЕЛИ (магнитопровод=%s, ток=40 А, n=%.0f об/мин, ось q)" % (KIND, cfg.N_OPER))
    print("Вердикт материала решается МОМЕНТОМ/КПД/ТЯГОЙ, а не долей ячеек за коленом")
    print("=" * 100)

    scen = {m: _scenario(m) for m in ("ndfeb", "smco")}
    rows = []
    print("%-5s | %-32s | %-32s | вердикт по моменту" % ("T_ср", "NdFeB", "SmCo"))
    print("%-5s | %-32s | %-32s |" % ("", "M[Н·м] КПД[%] P[Вт] тяга", "M[Н·м] КПД[%] P[Вт] тяга"))
    print("-" * 100)
    for a, b in zip(R["ndfeb"], R["smco"]):
        if a["i_peak"] != 40.0:
            continue
        res = {}
        for tag, rec in (("ndfeb", a), ("smco", b)):
            res[tag] = perf_at_state(
                scen[tag], tag, T_magnet=rec["T_magnet_max"], ratio=1.0 - rec["ke_drop"],
                i_peak=rec["i_peak"], T_winding=max(rec["T_max"]), loss_w_per_m=rec["loss_end"])
        nd, sm = res["ndfeb"], res["smco"]
        # ТЯГА при ТОКОВОМ ограничении (полный газ БПЛА): момент задан τ=K_t·I, винт находит
        # равновесие τ=k·ω² ⇒ ω²∝τ, а тяга фикс. винта ∝ ω² ⇒ **тяга ∝ момент**.
        # (Скорость при этом тоже меняется: ω∝√τ — поэтому «мощность при 7000 об/мин» ниже
        #  приведена как справочная величина при ФИКСИРОВАННОЙ скорости, не как равновесие.)
        thrust_ratio = sm["torque"] / nd["torque"] if nd["torque"] > 0 else float("nan")
        win = "NdFeB" if nd["torque"] > sm["torque"] else "SmCo"
        adv = 100 * (max(nd["torque"], sm["torque"]) / min(nd["torque"], sm["torque"]) - 1)
        print("%-5.0f | %6.3f %5.1f %6.0f       | %6.3f %5.1f %6.0f       | %s +%.1f%% (тяга SmCo/NdFeB %.3f)"
              % (a["T_amb"], nd["torque"], 100 * nd["efficiency"], nd["p_out"],
                 sm["torque"], 100 * sm["efficiency"], sm["p_out"], win, adv, thrust_ratio))
        rows.append(dict(T_amb=a["T_amb"], ndfeb=nd, smco=sm, thrust_ratio_sm_nd=thrust_ratio,
                         winner_torque=win))

    print("-" * 100)
    # где меняется победитель по моменту и по КПД
    for metric, key in (("МОМЕНТУ", "torque"), ("КПД", "efficiency")):
        wins = [("NdFeB" if r["ndfeb"][key] > r["smco"][key] else "SmCo") for r in rows]
        flip = [i for i in range(1, len(wins)) if wins[i] != wins[i - 1]]
        if flip:
            i = flip[0]
            print("CROSSOVER по %s: между средой %.0f и %.0f °C (%s → %s)"
                  % (metric, rows[i - 1]["T_amb"], rows[i]["T_amb"], wins[i - 1], wins[i]))
        else:
            print("CROSSOVER по %s: нет в диапазоне (везде %s)" % (metric, wins[0]))

    (OUT / ("performance_%s.json" % KIND)).write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[performance_%s.json сохранён]" % KIND)


if __name__ == "__main__":
    main()
