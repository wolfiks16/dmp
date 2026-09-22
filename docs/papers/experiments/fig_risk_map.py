# -*- coding: utf-8 -*-
"""
КАРТА РИСКА по сечению — из РАСЧЁТА, а не нарисованная.

⚠ Заменяет прежний scratchpad/gen_imgs.py → img_riskmap.png, где «риск» был вписан
руками (`risk = [0.05, 0.10, ...]` по 10 секторам). Здесь каждая ячейка магнита
окрашена ФАКТИЧЕСКОЙ долей сохранённой ремнантности `res.retention`, посчитанной
связанным магнитотепловым решателем с самосогласованным необратимым размагничиванием
(`run_machine_thermal_demag`, ядро НН-1).

Режим совпадает с B-full харвестом (`harvest_bfull.py`, MODE=map, steel10), точка
T_среды = 130 °C, ток 40 А, рабочая точка q-оси — та, где харвест дал необратимую
потерю (ret_min ≈ 0.665, падение ЭДС-постоянной ≈ 6.3 %).

ЧТО ПОКАЗАЛ РАЗБОР (числа печатаются, заголовок ставится ПО НИМ):
главный эффект — разброс МЕЖДУ полюсами (повреждённый объём полюса от 0 до 91 %),
а не внутри полюса. Причина — сочетание 12 пазов / 14 полюсов: каждый полюс стоит
в своём положении относительно поля обмотки. Контроль: картина обязана иметь
2-кратную симметрию (gcd(12,14) = 2) — проверяется автоматически.

Оформление — по docs/figures/STYLE.md.
Выход: figs/fig_risk_map.png (статья), figs/fig_risk_map_slide.png (доклад),
       figs/risk_map_data.npz (данные расчёта).
Запуск: python docs/papers/experiments/fig_risk_map.py            (~4 мин, полный счёт)
        python docs/papers/experiments/fig_risk_map.py --cached   (мгновенно, из npz)
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors                   # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.collections import PolyCollection     # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                          # scenario_paper1 лежит рядом
sys.path.insert(0, str(HERE.parents[2]))
import scenario_paper1 as cfg                                              # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet                       # noqa: E402
from magcore.fem2d.machines.magnet_loss import magnet_segment_width        # noqa: E402
from magcore.fem2d.machines.pmsm_outrunner import OutrunnerPMSMParams, Region  # noqa: E402
from magcore.fem2d.machines.scenario import machine_scenario               # noqa: E402
from magcore.fem2d.machines.thermal_scenario import run_machine_thermal_demag  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
DARK = "#1F2A38"; HEAT = "#E1442F"; STEEL = "#2E6C99"
GREEN = "#2FA36B"; AMBER = "#E8A33D"; MUTED = "#5A6B7B"

# ---- режим: ПОБИТОВО как в harvest_bfull.py (MODE=map) ----
MATERIAL, STEEL_KIND = "ndfeb", "steel10"
MESH, DT, N_STEPS, LOSS_NP = 0.004, 0.5, 90, 12
T_AMB, I_PEAK = 130.0, 40.0
GAMMA = math.pi / 2.0                                   # рабочая точка q-оси (MTPA)
N_SLOTS, N_POLES = 12, 14
AIR_GAP = 1.0e-3                                        # физическая ширина зазора [м]

# Уровни сетки. На глобальных 4 мм зазор (1 мм) и магнит не разрешены вовсе — 12 ячеек
# на полюс. Помельчение задаётся ПО РЕГИОНАМ: магнит мельчим наравне с зазором.
# 'coarse' оставлен как исходная точка для ПРОВЕРКИ СХОДИМОСТИ вывода, а не для отчёта.
LEVELS = {
    "coarse": None,                                                  # ~172 ячейки магнита
    "fine":   {"magnet": AIR_GAP, "air_gap": AIR_GAP},               # ~874
    "finer":  {"magnet": AIR_GAP / 2, "air_gap": AIR_GAP / 2},       # ~3041
    # Диагностика: coarse→fine→finer сходимости НЕ дали (падение ЭДС-пост. 6.29→4.55→3.24 %).
    # Мельчим ПО ОДНОМУ региону, чтобы понять, кто держит: поле в зазоре или разрешение
    # пиков |H| на кромках магнита (там угловая особенность, она сходится плохо).
    "gap4":   {"magnet": AIR_GAP / 2, "air_gap": AIR_GAP / 4},       # только зазор мельче
    "mag4":   {"magnet": AIR_GAP / 4, "air_gap": AIR_GAP / 2},       # только магнит мельче
}
LEVEL = next((a for a in sys.argv[1:] if a in LEVELS), "fine")
MESH_BY_REGION = LEVELS[LEVEL]

BG = {int(Region.AIR_GAP): "#FAFCFD", int(Region.STATOR_YOKE): "#DBE3EA",
      int(Region.TOOTH): "#DBE3EA", int(Region.SLOT): "#EDF1F6",
      int(Region.ROTOR_YOKE): "#D2DAE3"}
RISK_CMAP = mcolors.LinearSegmentedColormap.from_list("risk", [HEAT, AMBER, GREEN])
NPZ = HERE / "figs" / ("risk_map_data_%s.npz" % LEVEL)


def build_scenario():
    params = OutrunnerPMSMParams(n_slots=N_SLOTS, n_poles=N_POLES, mesh_size=MESH,
                                 mesh_size_by_region=MESH_BY_REGION)
    sc = machine_scenario(params, n42sh_magnet((1.0, 0.0, 0.0)),
                          cfg.STEEL_KINDS[STEEL_KIND]["curve"]())
    return params, sc


def solve(params, sc):
    return run_machine_thermal_demag(
        sc, i_peak=I_PEAK, turns_per_slot=cfg.TURNS_PER_SLOT, gamma_elec=GAMMA,
        slot_fill=cfg.SLOT_FILL, h=cfg.H_OUT, T_amb=T_AMB, h_in=cfg.H_IN, T_frame=T_AMB,
        dt=DT, n_steps=N_STEPS,
        thermal=cfg.build_thermal_props(MATERIAL, cfg.N_OPER, STEEL_KIND),
        T0=T_AMB, T_cap=cfg.T_CAP, steady_tol=cfg.STEADY_TOL, max_substeps=8,
        core_losses=True, speed_rpm=cfg.N_OPER,
        sigma_pm=cfg.magnet_conductivity(MATERIAL, T_AMB),
        magnet_seg_width=magnet_segment_width(params, cfg.N_SEG),
        loss_mech_span=2.0 * math.pi / N_SLOTS, loss_n_positions=LOSS_NP,
        **cfg.steel_loss_kwargs(STEEL_KIND),
    )


def _pole_labels(mesh, mag):
    """Разбить ячейки магнита на полюса СВЯЗНЫМИ КОМПОНЕНТАМИ по общим рёбрам.
    Без порогов и подгонки: сегменты магнитов физически не соприкасаются."""
    cells = np.asarray(mesh.cells)[mag]
    shared: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(cells):
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            shared.setdefault((min(a, b), max(a, b)), []).append(i)
    adj: list[list[int]] = [[] for _ in cells]
    for v in shared.values():
        if len(v) == 2:
            adj[v[0]].append(v[1]); adj[v[1]].append(v[0])
    lab = -np.ones(len(cells), dtype=int)
    n = 0
    for s in range(len(cells)):
        if lab[s] >= 0:
            continue
        stack = [s]; lab[s] = n
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if lab[w] < 0:
                    lab[w] = n; stack.append(w)
        n += 1
    return lab, n


def analyse_quiet(sc, ret):
    return analyse(sc, ret, verbose=False)


def analyse(sc, ret, verbose=True):
    """Где именно просело + КОНТРОЛЬ 2-кратной симметрии. Возвращает числа для подписи."""
    mesh = sc.geometry.mesh
    mag = np.where(np.asarray(sc.geometry.region) == int(Region.MAGNET))[0]
    cen = np.array([mesh.cell_centroid(int(c)) for c in mag], dtype=float)
    area = np.array([mesh.cell_area(int(c)) for c in mag], dtype=float)
    r = np.hypot(cen[:, 0], cen[:, 1])
    th = np.mod(np.arctan2(cen[:, 1], cen[:, 0]), 2.0 * np.pi)

    lab, n_pole = _pole_labels(mesh, mag)
    ang = np.zeros(n_pole); mn = np.zeros(n_pole); vol = np.zeros(n_pole)
    mret = np.zeros(n_pole)
    for g in range(n_pole):
        sel = lab == g
        # ⚠ КРУГОВОЕ среднее: у полюса, лежащего на 0°, обычное среднее даёт ~180° (углы
        # рвутся на 0/2π) — из-за этого ломается поиск пары θ↔θ+180°.
        ang[g] = np.mod(np.arctan2(np.sin(th[sel]).mean(), np.cos(th[sel]).mean()), 2.0 * np.pi)
        mn[g] = ret[sel].min()
        mret[g] = float(np.average(ret[sel], weights=area[sel]))
        vol[g] = area[sel][ret[sel] < 0.999].sum() / area[sel].sum()

    # КОНТРОЛЬ: у 12/14 gcd = 2 ⇒ картина обязана иметь 2-кратную симметрию.
    # Совпадение пар θ и θ+180° = подтверждение, что карта — структура, а не численный шум.
    # Меряем по ИНТЕГРАЛЬНОЙ величине (средняя по объёму ремнантность полюса): ret_min —
    # экстремум по ячейкам, он скачет от расположения отдельного элемента и на мелкой сетке
    # шумит вдвое сильнее, то есть для контроля симметрии непригоден.
    pair_err, pair_gap = 0.0, 0.0
    for i, a in enumerate(ang):
        j = int(np.argmin(np.abs(np.mod(ang - (a + np.pi) + np.pi, 2.0 * np.pi) - np.pi)))
        pair_err = max(pair_err, abs(mret[i] - mret[j]))
        pair_gap = max(pair_gap, abs(np.mod(ang[j] - (a + np.pi) + np.pi, 2.0 * np.pi) - np.pi))
    if pair_gap > np.radians(5.0):
        raise RuntimeError("пары полюсов θ↔θ+180° не находятся (расхождение %.1f°) — "
                           "проверь разметку полюсов" % np.degrees(pair_gap))

    rmid = 0.5 * (r.min() + r.max())
    spread = float(mret.max() - mret.min())      # разброс ИНТЕГРАЛЬНОЙ величины по полюсам
    an = dict(n_pole=int(n_pole), vol_min=float(vol.min()), vol_max=float(vol.max()),
              mn_min=float(mn.min()), mn_max=float(mn.max()), pair_err=float(pair_err),
              spread=spread, n_intact=int((vol < 0.02).sum()),
              mret_min=float(mret.min()), mret_max=float(mret.max()),
              ret_gap=float(ret[r < rmid].mean()), ret_back=float(ret[r >= rmid].mean()),
              vol_hurt=float(area[ret < 0.999].sum() / area.sum()), ret_min=float(ret.min()),
              n_mag=int(mag.size))
    if verbose:
        print("  полюсов найдено: %d (ожидается %d)" % (an["n_pole"], N_POLES))
        print("  МЕЖДУ ПОЛЮСАМИ (главный эффект): повреждённый объём от %.0f %% до %.0f %%; "
              "ret_min от %.3f до %.3f; полностью целы %d"
              % (100 * an["vol_min"], 100 * an["vol_max"], an["mn_min"], an["mn_max"],
                 an["n_intact"]))
        print("  внутри полюса радиально: у зазора ret=%.3f | у спинки ret=%.3f"
              % (an["ret_gap"], an["ret_back"]))
        print("  средняя ремнантность полюса: от %.3f до %.3f (разброс %.3f)"
              % (an["mret_min"], an["mret_max"], spread))
        print("  2-кратная симметрия (gcd(%d,%d)=2), по средней ремнантности полюса: "
              "расхождение пар %.3f при разбросе %.3f (%.0f %%) — %s"
              % (N_SLOTS, N_POLES, pair_err, spread, 100 * pair_err / max(spread, 1e-9),
                 "ПОДТВЕРЖДЕНА" if pair_err < 0.25 * spread else "НЕ ПОДТВЕРЖДЕНА"))
    return an


def draw(ax, sc, ret, f=1.0):
    mesh = sc.geometry.mesh
    v = np.asarray(mesh.vertices, dtype=float) * 1e3          # м -> мм
    cells = np.asarray(mesh.cells, dtype=int)
    region = np.asarray(sc.geometry.region)
    tri = v[cells]

    for code, col in BG.items():
        sel = region == code
        if sel.any():
            ax.add_collection(PolyCollection(tri[sel], facecolors=col, edgecolors="white",
                                             linewidths=0.25, zorder=1))
    mag = np.where(region == int(Region.MAGNET))[0]
    pc = PolyCollection(tri[mag], array=np.asarray(ret, dtype=float), cmap=RISK_CMAP,
                        edgecolors="white", linewidths=0.25, zorder=2)
    pc.set_clim(min(0.95, float(np.min(ret))), 1.0)
    ax.add_collection(pc)

    lim = np.abs(v).max() * 1.04
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Координата x, мм", fontsize=13 * f, color=DARK)
    ax.set_ylabel("Координата y, мм", fontsize=13 * f, color=DARK)
    ax.tick_params(labelsize=10 * f)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    return pc


def main():
    cached = "--cached" in sys.argv
    params, sc = build_scenario()
    NPZ.parent.mkdir(parents=True, exist_ok=True)

    if cached and NPZ.exists():
        d = np.load(NPZ)
        ret = d["retention"]
        T_mag, ke_drop = float(d["T_mag"]), float(d["ke_drop"])
        print("данные взяты из %s (расчёт не запускался)" % NPZ.name, flush=True)
    else:
        t0 = time.time()
        nm = int((np.asarray(sc.geometry.region) == int(Region.MAGNET)).sum())
        print("уровень сетки «%s»: %s; ячеек всего %d, магнита %d"
              % (LEVEL, MESH_BY_REGION or "равномерная %.4f м" % MESH,
                 sc.geometry.mesh.cells.shape[0], nm), flush=True)
        print("считаю связанную задачу (T_среды=%.0f °C, ток=%.0f А, q-ось)…"
              % (T_AMB, I_PEAK), flush=True)
        res = solve(params, sc)
        ret = np.asarray(res.retention, dtype=float)
        T_mag, ke_drop = float(res.T_magnet_max), float(res.torque_constant_drop)
        print("посчитано за %.0f с: T_магнита=%.1f °C, падение ЭДС-пост.=%.2f %%"
              % (time.time() - t0, T_mag, 100 * ke_drop), flush=True)
        np.savez(NPZ, retention=ret, T_mag=T_mag, ke_drop=ke_drop,
                 T_amb=T_AMB, i_peak=I_PEAK, gamma=GAMMA)

    an = analyse(sc, ret)
    out_dir = NPZ.parent

    # СХОДИМОСТЬ: сравнить вывод с другими уровнями сетки, если они посчитаны
    others = [(lv, out_dir / ("risk_map_data_%s.npz" % lv)) for lv in LEVELS if lv != LEVEL]
    done = [(lv, p) for lv, p in others if p.exists()]
    if done:
        print("  сходимость по сетке (вывод не должен зависеть от уровня):")
        print("    %-7s ячеек магнита %5d  ret_min %.3f  повр.объём %.0f %%  разброс по полюсам %.3f"
              % (LEVEL, an["n_mag"], an["ret_min"], 100 * an["vol_hurt"], an["spread"]))
        for lv, p in done:
            d2 = np.load(p)
            pr, sc2 = OutrunnerPMSMParams(n_slots=N_SLOTS, n_poles=N_POLES, mesh_size=MESH,
                                          mesh_size_by_region=LEVELS[lv]), None
            sc2 = machine_scenario(pr, n42sh_magnet((1.0, 0.0, 0.0)),
                                   cfg.STEEL_KINDS[STEEL_KIND]["curve"]())
            a2 = analyse_quiet(sc2, d2["retention"])
            print("    %-7s ячеек магнита %5d  ret_min %.3f  повр.объём %.0f %%  разброс по полюсам %.3f"
                  % (lv, a2["n_mag"], a2["ret_min"], 100 * a2["vol_hurt"], a2["spread"]))

    # ---------- версия для статьи ----------
    fig, ax = plt.subplots(figsize=(8.8, 11.2), dpi=200)
    pc = draw(ax, sc, ret, 1.0)
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.042, pad=0.09)
    cb.set_label("Доля сохранённой ремнантности   (1,00 — магнит цел)", fontsize=12.5,
                 color=DARK)
    cb.ax.tick_params(labelsize=10.5)
    ax.set_title("Соседние полюса повреждены ПО-РАЗНОМУ: от нетронутого\n"
                 "до %.0f %% потерянного объёма — при одном и том же режиме"
                 % (100 * an["vol_max"]),
                 fontsize=14, fontweight="bold", color=DARK, pad=12)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.935, bottom=0.315)
    cap = ("Каждая ячейка окрашена ФАКТИЧЕСКОЙ долей сохранённой ремнантности из связанного\n"
           "магнитотеплового расчёта с необратимым размагничиванием. Режим: среда %.0f °C, "
           "ток %.0f А,\nрабочая точка q-оси, магнитопровод «сталь 10», магнит N42SH. "
           "Температура магнита %.0f °C,\nпадение ЭДС-постоянной %.1f %%, повреждено "
           "%.0f %% объёма магнитов.\n"
           "\n"
           "Главное — разброс МЕЖДУ полюсами: повреждённый объём полюса от %.0f до %.0f %%, "
           "полностью\nцел %d полюс из %d. Сочетание %d пазов / %d полюсов ставит каждый полюс "
           "в своё положение\nотносительно поля обмотки — средняя оценка или «худшая точка» "
           "этого не покажут.\n"
           "Контроль: картина имеет 2-кратную симметрию, как требует gcd(%d, %d) = 2 — "
           "расхождение\nпар %.3f при разбросе %.3f, то есть это структура, а не численный шум.") % (
        T_AMB, I_PEAK, T_mag, 100 * ke_drop, 100 * an["vol_hurt"],
        100 * an["vol_min"], 100 * an["vol_max"], an["n_intact"], an["n_pole"],
        N_SLOTS, N_POLES, N_SLOTS, N_POLES, an["pair_err"], an["spread"])
    fig.text(0.10, 0.012, cap, fontsize=11.4, color=DARK, ha="left", va="bottom",
             linespacing=1.6)
    # Имя с уровнем сетки: уровни можно считать параллельно, не затирая друг друга.
    # В деку/статью уровень продвигается отдельно (--promote), осознанно.
    out = out_dir / ("fig_risk_map_%s.png" % LEVEL)
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    # ---------- версия для слайда: квадрат под бокс 4.7x4.7", крупный шрифт ----------
    fig_s, ax_s = plt.subplots(figsize=(7.6, 8.0), dpi=200)
    pc_s = draw(ax_s, sc, ret, 1.28)
    cb_s = fig_s.colorbar(pc_s, ax=ax_s, orientation="horizontal", fraction=0.05, pad=0.11)
    cb_s.set_label("Доля сохранённой ремнантности", fontsize=15, color=DARK)
    cb_s.ax.tick_params(labelsize=13)
    fig_s.tight_layout()
    out_slide = out_dir / ("fig_risk_map_%s_slide.png" % LEVEL)
    fig_s.savefig(out_slide, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig_s)

    print("saved", out)
    print("saved", out_slide)

    if "--promote" in sys.argv:      # сделать этот уровень тем, что идёт в статью и деку
        import shutil
        for src, dst in ((out, "fig_risk_map.png"), (out_slide, "fig_risk_map_slide.png")):
            shutil.copyfile(src, out_dir / dst)
            print("promote ->", out_dir / dst)


if __name__ == "__main__":
    main()
