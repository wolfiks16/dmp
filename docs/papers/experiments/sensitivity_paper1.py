"""
§12 п.7 — АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ вокруг критической точки решения.

Базовая точка = там, где Сталь 10 даёт демаг, а M270 не давал: среда 70 °C, ток 40 А
(номинал), NdFeB, магнитопровод Сталь 10 (статор шихт. 0.5 мм + ротор массивный).
Вопрос: УСТОЙЧИВ ли вывод «на Стали 10 NdFeB размагничивается уже при среде 70 °C»
на всём правдоподобном диапазоне плохо известных параметров?

Метод: по одному параметру за раз (OAT) вокруг базы. Для каждого — T магнита, падение K_e,
ret_min, исход. Плюс контроль SmCo в самом жёстком варианте (должен остаться цел).

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/sensitivity_paper1.py
Выход:   docs/papers/experiments/output_bfull/sensitivity.json
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scenario_paper1 as cfg  # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet  # noqa: E402
from magcore.fem2d.machines.iron_loss import (  # noqa: E402
    STEEL10_ALPHA_LIT,
    STEEL10_DENSITY,
    SteinmetzCoefficients,
)
from magcore.fem2d.machines.magnet_loss import magnet_segment_width  # noqa: E402
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams  # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario  # noqa: E402
from magcore.fem2d.machines.thermal_scenario import run_machine_thermal_demag  # noqa: E402

OUT = Path(__file__).resolve().parent / "output_bfull"

# База (критическая точка решения)
BASE = dict(T_amb=70.0, i_peak=40.0, material="ndfeb")
MESH, DT, N_STEPS, LOSS_NP, SUBSTEPS = 0.004, 0.5, 90, 12, 8
MECH_SPAN = 2.0 * math.pi / 12

# Базовые значения варьируемых параметров
B_KH, B_MUR, B_HOUT, B_RHO, B_SIGSC, B_NSEG = 0.10, 500.0, 20.0, 0.14e-6, 1.0, 1

# Диапазоны (обоснование — sources_registry.md, bfull_parameters.md §5)
VARIATIONS = [
    ("k_hyst Ст10 [ЛИТ., LOW]", "k_hyst", [0.06, 0.10, 0.15]),
    ("μ_r ярма ротора (скин)", "mu_r", [200.0, 500.0, 1000.0]),
    ("h_out обдув", "h_out", [10.0, 20.0, 50.0]),
    ("ρ стали 10", "rho", [0.13e-6, 0.14e-6, 0.16e-6]),
    ("σ магнита ×(ШИМ/разброс)", "sigma_scale", [0.85, 1.0, 1.15, 2.0]),
    ("сегментация магнита N", "n_seg", [1, 2, 4]),
]


def run_case(material="ndfeb", *, k_hyst=B_KH, mu_r=B_MUR, h_out=B_HOUT, rho=B_RHO,
             sigma_scale=B_SIGSC, n_seg=B_NSEG, T_amb=None, i_peak=None):
    T_amb = BASE["T_amb"] if T_amb is None else T_amb
    i_peak = BASE["i_peak"] if i_peak is None else i_peak
    params = OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH)
    magnet = (n42sh_magnet if material == "ndfeb" else sm2co17_magnet)((1.0, 0.0, 0.0))
    sc = machine_scenario(params, magnet, cfg.STEEL_KINDS["steel10"]["curve"]())
    # коэффициенты Штейнмеца: k_eddy ВЫВОДИТСЯ из ρ и толщины листа (физика), k_hyst — варьируем
    cf = SteinmetzCoefficients.from_lamination(
        k_hyst=k_hyst, alpha=STEEL10_ALPHA_LIT, thickness=cfg.LAMINATION_THICKNESS,
        resistivity=rho, density=STEEL10_DENSITY)
    res = run_machine_thermal_demag(
        sc, i_peak=i_peak, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=cfg.GAMMA_ELEC,
        slot_fill=cfg.SLOT_FILL, h=h_out, T_amb=T_amb, h_in=cfg.H_IN, T_frame=T_amb,
        dt=DT, n_steps=N_STEPS, thermal=cfg.build_thermal_props(material, cfg.N_OPER, "steel10"),
        T0=T_amb, T_cap=cfg.T_CAP, steady_tol=cfg.STEADY_TOL, max_substeps=SUBSTEPS,
        core_losses=True, speed_rpm=cfg.N_OPER,
        sigma_pm=cfg.magnet_conductivity(material, T_amb) * sigma_scale,
        magnet_seg_width=magnet_segment_width(params, n_seg),
        loss_mech_span=MECH_SPAN, loss_n_positions=LOSS_NP,
        steinmetz=cf, rotor_solid=True, sigma_rotor=1.0 / rho, rotor_mu_r=mu_r,
    )
    tr = res.transient
    return dict(T_magnet=float(res.T_magnet_max), ke_drop=float(res.torque_constant_drop),
                ret_min=float(res.retention.min()), survived=bool(res.survived),
                runaway=bool(tr.runaway), cascade=bool(tr.magnet_cascade))


def log(*a):
    print(*a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log("=" * 84)
    log("§12 АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ — база: Ст10, NdFeB, среда %.0f °C, ток %.0f А"
        % (BASE["T_amb"], BASE["i_peak"]))
    log("ВОПРОС: устойчив ли вывод «на Стали 10 NdFeB размагничивается при среде 70 °C»?")
    log("=" * 84)

    report = {"base": BASE, "runs": {}}
    t = time.time()
    base = run_case()
    report["base_result"] = base
    log("БАЗА: T_маг=%.1f  K_e-drop=%.2f%%  ret_min=%.3f  выжила=%s каскад=%s  [%.0fs]"
        % (base["T_magnet"], 100 * base["ke_drop"], base["ret_min"],
           base["survived"], base["cascade"], time.time() - t))

    for title, key, values in VARIATIONS:
        log("\n--- %s ---" % title)
        rows = []
        for v in values:
            t = time.time()
            r = run_case(**{key: v})
            r[key] = v
            rows.append(r)
            mark = " <- база" if abs(v - {"k_hyst": B_KH, "mu_r": B_MUR, "h_out": B_HOUT,
                                          "rho": B_RHO, "sigma_scale": B_SIGSC,
                                          "n_seg": B_NSEG}[key]) < 1e-12 else ""
            log("  %-12s=%-9.3g T_маг=%6.1f  K_e-drop=%6.2f%%  ret_min=%.3f  выж=%s кск=%s [%.0fs]%s"
                % (key, v, r["T_magnet"], 100 * r["ke_drop"], r["ret_min"],
                   r["survived"], r["cascade"], time.time() - t, mark))
        report["runs"][key] = rows

    # контроль SmCo в самом жёстком варианте (макс. потери: k_hyst высокий, слабый обдув, ШИМ)
    log("\n--- контроль SmCo в НАИХУДШЕМ варианте (k_hyst=0.15, h_out=10, σ×2) ---")
    t = time.time()
    sm = run_case("smco", k_hyst=0.15, h_out=10.0, sigma_scale=2.0)
    nd = run_case("ndfeb", k_hyst=0.15, h_out=10.0, sigma_scale=2.0)
    report["worst_case"] = {"smco": sm, "ndfeb": nd}
    log("  SmCo : T_маг=%.1f  K_e-drop=%.2f%%  ret_min=%.3f  каскад=%s"
        % (sm["T_magnet"], 100 * sm["ke_drop"], sm["ret_min"], sm["cascade"]))
    log("  NdFeB: T_маг=%.1f  K_e-drop=%.2f%%  ret_min=%.3f  каскад=%s  [%.0fs]"
        % (nd["T_magnet"], 100 * nd["ke_drop"], nd["ret_min"], nd["cascade"], time.time() - t))

    # ВЕРДИКТ устойчивости
    drops = [r["ke_drop"] for rows in report["runs"].values() for r in rows]
    n_demag = sum(1 for d in drops if d > 1e-6)
    report["verdict"] = dict(n_runs=len(drops), n_with_demag=n_demag,
                             ke_drop_min=float(min(drops)), ke_drop_max=float(max(drops)),
                             smco_intact_worst=bool(sm["ke_drop"] <= 1e-9))
    log("\n" + "=" * 84)
    log("ВЕРДИКТ: демаг NdFeB в %d из %d вариантов; падение K_e от %.2f%% до %.2f%%"
        % (n_demag, len(drops), 100 * min(drops), 100 * max(drops)))
    log("         SmCo в наихудшем варианте %s"
        % ("ЦЕЛ (вывод устойчив)" if sm["ke_drop"] <= 1e-9 else "ПОВРЕЖДЁН (вывод НЕ устойчив!)"))
    log("=" * 84)

    (OUT / "sensitivity.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("[sensitivity.json сохранён; всего %.0f с]" % (time.time() - t0))


if __name__ == "__main__":
    main()
