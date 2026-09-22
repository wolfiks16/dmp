"""
Results-harvest для Статьи 1 (ядро К6'): воспроизводимый прогон на outrunner 12/14.

Три блока:
  V — ВЕРИФИКАЦИЯ: энергобаланс транзиента, порог теплового разгона s_crit,
      сходимость по шагу времени dt.
  S — СВИП нагрузки (Рис. 7): K_e-drop vs i_peak, NdFeB vs SmCo → crossover.
  H — РАЗГОН/траектория (Рис. 4-6): один представительный отказной ток, оба материала.

Объект: дефолтный OutrunnerPMSMParams (12/14, Ø58/30 мм). Режим: worst-case d-ось (γ=0).
Материалы: N42SH (NdFeB) vs КС25ДЦ (представит. Sm2Co17, ориентир ГОСТ 21559-76).

Запуск:  PYTHONPATH=<repo> python docs/papers/experiments/harvest_paper1.py
Выход:   docs/papers/experiments/output/  (summary.json, trajectories.npz, *.png)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from magcore.domain.magnet_model import n42sh_magnet, sm2co17_magnet
from magcore.domain.steel_curves import m270_35a_cogent_bh_curve
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams
from magcore.fem2d.machines.scenario import machine_scenario
from magcore.fem2d.machines.thermal_scenario import (
    MachineThermalProperties,
    run_machine_thermal_demag,
)
from magcore.fem2d.runaway import couple_thermal_loss_picard, runaway_threshold
from magcore.fem2d.spaces import LagrangeP1Space2D

# ------------------------------------------------------------------ конфигурация
MESH = 0.0030
TURNS = 20.0
SLOT_FILL = 0.45
GAMMA = 0.0                      # worst-case d-ось (чисто размагничивающий ток)

T_AMB = 40.0                    # тёплый отсек БПЛА
H_CONV = 50.0                   # слабый обдув (зависание/набор)
DT = 5.0
N_STEPS = 24                    # 120 с горизонт
STEADY_TOL = 0.05              # °C/с — стоп по установившемуся режиму

SWEEP_CURRENTS = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0]
KE_DROP_THRESHOLD = 0.05       # 5 % падения K_e = «отказ по стойкости»
SUBSTEPS_SWEEP = 8
SUBSTEPS_HEAT = 32

OUT = Path(__file__).resolve().parent / "output"


def build(magnet):
    # Сталь — datasheet-точная кривая Cogent M270-35A (H≈1700 А/м при 1.5 Тл), а НЕ старая
    # «мягкая» представительная (700 А/м): та занижала насыщение в 2.4× и искажала бы результаты.
    return machine_scenario(
        OutrunnerPMSMParams(n_slots=12, n_poles=14, mesh_size=MESH),
        magnet, m270_35a_cogent_bh_curve(),
    )


def run(scenario, i_peak, *, max_substeps, n_steps=N_STEPS):
    return run_machine_thermal_demag(
        scenario, i_peak=i_peak, turns_per_slot=TURNS, gamma_elec=GAMMA,
        slot_fill=SLOT_FILL, h=H_CONV, T_amb=T_AMB, T0=T_AMB, dt=DT, n_steps=n_steps,
        max_substeps=max_substeps, steady_tol=STEADY_TOL,
    )


def summarize(res):
    tr = res.transient
    return dict(
        T_magnet_max=float(res.T_magnet_max),
        ke_drop=float(res.torque_constant_drop),
        ret_min=float(res.retention.min()),
        survived=bool(res.survived),
        runaway=bool(tr.runaway),
        cascade=bool(tr.magnet_cascade),
        n_knee_end=int(tr.n_past_knee[-1]),
        loss0=float(tr.loss_power[0]),
        loss_end=float(tr.loss_power[-1]),
        t_end=float(tr.times[-1]),
        n_steps=int(len(tr.times)),
        stop_reason=str(tr.stop_reason),
        times=[float(x) for x in tr.times],
        T_max=[float(x) for x in tr.T_max],
        T_magnet=[float(x) for x in tr.T_magnet],
        retention_mean=[float(x) for x in tr.retention_mean],
        retention_min=[float(x) for x in tr.retention_min],
        n_past_knee=[int(x) for x in tr.n_past_knee],
        stored_energy=[float(x) for x in tr.stored_energy],
        loss_power=[float(x) for x in tr.loss_power],
        outflow=[float(x) for x in tr.outflow],
    )


def log(*a):
    print(*a, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    t_start = time.time()

    log("=" * 78)
    log("HARVEST Статья 1 (К6'): outrunner 12/14, worst-case d-ось, T_amb=%.0f h=%.0f"
        % (T_AMB, H_CONV))
    log("=" * 78)

    t = time.time()
    nd = build(n42sh_magnet((1.0, 0.0, 0.0)))
    sm = build(sm2co17_magnet((1.0, 0.0, 0.0)))
    nc = nd.geometry.mesh.n_cells
    log("[геометрия построена за %.1f с; ячеек=%d]" % (time.time() - t, nc))

    # ---------------------------------------------------------------- V. ВЕРИФИКАЦИЯ
    log("\n### V. ВЕРИФИКАЦИЯ ###")
    ver = {}

    # V.b — порог теплового разгона s_crit (независимо от машины)
    space = LagrangeP1Space2D(nd.geometry.mesh)
    k_cells, _ = MachineThermalProperties.representative().cell_fields(nd.geometry)
    s_crit = runaway_threshold(space, k_cells, H_CONV)
    q0 = np.full(nc, 2.0e5, dtype=float)
    below = couple_thermal_loss_picard(space, k_cells, q0_cells=q0,
                                       loss_sensitivity=0.9 * s_crit, h=H_CONV, T_amb=T_AMB)
    above = couple_thermal_loss_picard(space, k_cells, q0_cells=q0,
                                       loss_sensitivity=1.1 * s_crit, h=H_CONV, T_amb=T_AMB)
    ver["s_crit"] = float(s_crit)
    ver["below_converged"] = bool(below.converged)   # ждём True
    ver["above_converged"] = bool(above.converged)   # ждём False (разгон)
    log("  V.b порог: s_crit=%.4g | 0.9·s_crit сходится=%s | 1.1·s_crit сходится=%s (ждём True/False)"
        % (s_crit, below.converged, above.converged))

    # V.a — энергобаланс: (ΔЗапас)/dt должно = Потери − Отвод (на шаге), оракул связки
    heat_nd_ver = run(nd, 100.0, max_substeps=SUBSTEPS_HEAT)
    S = summarize(heat_nd_ver)
    st = np.array(S["stored_energy"]); lp = np.array(S["loss_power"]); of = np.array(S["outflow"])
    d_store = np.diff(st) / DT
    balance = lp[1:] - of[1:]
    denom = np.maximum(np.abs(lp[1:]), 1.0)
    eb_rel = float(np.max(np.abs(d_store - balance) / denom))
    ver["energy_balance_max_rel_err"] = eb_rel
    log("  V.a энергобаланс: макс. относит. невязка (ΔЗапас/dt vs Потери−Отвод) = %.2e" % eb_rel)

    # V.c — сходимость по dt: тот же прогон с dt/2, сверка T_max в общих временах
    heat_nd_half = run_machine_thermal_demag(
        nd, i_peak=100.0, turns_per_slot=TURNS, gamma_elec=GAMMA, slot_fill=SLOT_FILL,
        h=H_CONV, T_amb=T_AMB, T0=T_AMB, dt=DT / 2, n_steps=N_STEPS * 2,
        max_substeps=SUBSTEPS_HEAT, steady_tol=STEADY_TOL,
    )
    Sh = summarize(heat_nd_half)
    tt = np.array(S["times"]); Tt = np.array(S["T_max"])
    th = np.array(Sh["times"]); Th = np.array(Sh["T_max"])
    tmax_common = min(tt[-1], th[-1])
    grid = tt[tt <= tmax_common + 1e-9]
    dt_diff = float(np.max(np.abs(np.interp(grid, tt, Tt) - np.interp(grid, th, Th)))) if grid.size else float("nan")
    ver["dt_convergence_max_T_diff"] = dt_diff
    log("  V.c сходимость по dt: макс |T_max(dt) − T_max(dt/2)| = %.3f °C на [0,%.0f]с" % (dt_diff, tmax_common))

    report["verification"] = ver

    # ---------------------------------------------------------------- S. СВИП (Рис. 7)
    log("\n### S. СВИП НАГРУЗКИ (crossover NdFeB vs SmCo) ###")
    sweep = {"i_peak": SWEEP_CURRENTS, "ndfeb": [], "smco": []}
    for ip in SWEEP_CURRENTS:
        for tag, scen, key in (("NdFeB", nd, "ndfeb"), ("SmCo", sm, "smco")):
            t = time.time()
            r = run(scen, ip, max_substeps=SUBSTEPS_SWEEP)
            s = summarize(r)
            s["i_peak"] = ip
            sweep[key].append(s)
            log("  %-5s i=%3.0f: T_mag=%.0f  K_e-drop=%5.2f%%  ret_min=%.3f  "
                "surv=%s run=%s casc=%s  [%.0fs]"
                % (tag, ip, s["T_magnet_max"], 100 * s["ke_drop"], s["ret_min"],
                   s["survived"], s["runaway"], s["cascade"], time.time() - t))
    report["sweep"] = sweep

    # crossover: наименьший ток, где NdFeB отказал (каскад/разгон/K_e>порог), а SmCo цел
    def nd_failed(s):
        return s["cascade"] or s["runaway"] or s["ke_drop"] > KE_DROP_THRESHOLD

    def sm_ok(s):
        return s["survived"] and s["ke_drop"] <= 1e-9

    crossover = None
    for i, ip in enumerate(SWEEP_CURRENTS):
        if nd_failed(sweep["ndfeb"][i]) and sm_ok(sweep["smco"][i]):
            crossover = ip
            break
    report["crossover_current"] = crossover
    log("  ⇒ crossover (NdFeB отказ, SmCo цел): %s A"
        % ("не достигнут в диапазоне" if crossover is None else "%.0f" % crossover))

    # ---------------------------------------------------------------- H. РАЗГОН (Рис. 4-6)
    # Представительный отказной ток = crossover (или первый ток отказа NdFeB), полные подшаги.
    i_rep = crossover
    if i_rep is None:
        for i, ip in enumerate(SWEEP_CURRENTS):
            if nd_failed(sweep["ndfeb"][i]):
                i_rep = ip; break
    if i_rep is None:
        i_rep = SWEEP_CURRENTS[-1]
    log("\n### H. ТРАЕКТОРИЯ при i=%.0f A (представительный отказ) ###" % i_rep)
    heat = {"i_peak": i_rep}
    for tag, scen, key in (("NdFeB", nd, "ndfeb"), ("SmCo", sm, "smco")):
        t = time.time()
        r = run(scen, i_rep, max_substeps=SUBSTEPS_HEAT)
        heat[key] = summarize(r)
        log("  %-5s: T_mag=%.0f  K_e-drop=%.2f%%  surv=%s  stop='%s'  [%.0fs]"
            % (tag, heat[key]["T_magnet_max"], 100 * heat[key]["ke_drop"],
               heat[key]["survived"], heat[key]["stop_reason"], time.time() - t))
    report["heat"] = heat

    # ---------------------------------------------------------------- сохранение
    (OUT / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("\n[summary.json сохранён; всего %.0f с]" % (time.time() - t_start))

    make_figures(report)
    log("=" * 78)
    log("HARVEST завершён. Выход: %s" % OUT)
    log("=" * 78)


def make_figures(report):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        log("[matplotlib недоступен (%s) — только числа в summary.json]" % e)
        return

    # Рис. 4 — T_max(t), NdFeB vs SmCo
    h = report["heat"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, c, lab in (("ndfeb", "tab:red", "NdFeB N42SH"), ("smco", "tab:blue", "SmCo КС25ДЦ")):
        s = h[key]
        ax.plot(s["times"], s["T_max"], marker="o", color=c, label=lab)
    ax.set_xlabel("время, с"); ax.set_ylabel("T_max, °C")
    ax.set_title("Рис.4 Нагрев при i=%.0f A (worst-case d-ось)" % h["i_peak"])
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "fig4_temperature.png", dpi=140); plt.close(fig)

    # Рис. 5 — сохранённая ремнантность (mean/min) во времени
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, c, lab in (("ndfeb", "tab:red", "NdFeB"), ("smco", "tab:blue", "SmCo")):
        s = h[key]
        ax.plot(s["times"], s["retention_mean"], color=c, marker="o", label="%s, средн." % lab)
        ax.plot(s["times"], s["retention_min"], color=c, ls="--", alpha=0.6, label="%s, мин." % lab)
    ax.set_xlabel("время, с"); ax.set_ylabel("сохранённая ремнантность r = B_r_eff/B_r(T)")
    ax.set_title("Рис.5 Необратимая потеря во времени (i=%.0f A)" % h["i_peak"])
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "fig5_retention.png", dpi=140); plt.close(fig)

    # Рис. 7 — crossover: K_e-drop vs i_peak
    sw = report["sweep"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, c, lab in (("ndfeb", "tab:red", "NdFeB N42SH"), ("smco", "tab:blue", "SmCo КС25ДЦ")):
        ip = [s["i_peak"] for s in sw[key]]
        drop = [100 * s["ke_drop"] for s in sw[key]]
        ax.plot(ip, drop, marker="o", color=c, label=lab)
        for s in sw[key]:
            if s["cascade"] or s["runaway"]:
                ax.scatter([s["i_peak"]], [100 * s["ke_drop"]], marker="x", s=120, color=c, zorder=5)
    ax.axhline(100 * KE_DROP_THRESHOLD, ls=":", color="gray", label="порог %.0f%%" % (100 * KE_DROP_THRESHOLD))
    if report.get("crossover_current"):
        ax.axvline(report["crossover_current"], ls="--", color="k", alpha=0.5)
        ax.annotate("crossover\n%.0f A" % report["crossover_current"],
                    (report["crossover_current"], ax.get_ylim()[1] * 0.6), fontsize=9)
    ax.set_xlabel("i_peak, A (worst-case d-ось)"); ax.set_ylabel("падение K_e, %")
    ax.set_title("Рис.7 Вердикт материала: crossover SmCo↔NdFeB\n(×=каскад/разгон)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(OUT / "fig7_crossover.png", dpi=140); plt.close(fig)

    # Рис. 2 — верификация (энергобаланс + dt-сходимость текстом)
    ver = report["verification"]
    fig, ax = plt.subplots(figsize=(6, 4)); ax.axis("off")
    txt = (
        "Рис.2 ВЕРИФИКАЦИЯ (оракулы)\n\n"
        "• Энергобаланс транзиента:\n    макс. относит. невязка = %.2e\n\n"
        "• Порог теплового разгона s_crit = %.4g\n"
        "    0.9·s_crit сходится: %s (ждём True)\n"
        "    1.1·s_crit сходится: %s (ждём False)\n\n"
        "• Сходимость по шагу dt:\n    макс |T_max(dt)−T_max(dt/2)| = %.3f °C"
        % (ver["energy_balance_max_rel_err"], ver["s_crit"],
           ver["below_converged"], ver["above_converged"], ver["dt_convergence_max_T_diff"])
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=11, family="monospace")
    fig.tight_layout(); fig.savefig(OUT / "fig2_verification.png", dpi=140); plt.close(fig)

    log("[рисунки сохранены: fig2, fig4, fig5, fig7]")


if __name__ == "__main__":
    main()
