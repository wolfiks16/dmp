# -*- coding: utf-8 -*-
"""
ПОЛЕ И ЗАПАС ДО КОЛЕНА магнитов outrunner 12N14P в худшем режиме нагрузки при 140 °C — из расчёта.

Режим: магнит N42SH, сталь M270-35A, ток 30 А по оси d (γ = 180°, размагничивает), 40 витков на паз.
Поле — статический связанный расчёт (`solve_machine_static`: магнит законом ветви в касательной, сталь
с насыщением); потеря — `evaluate_demag_impact` (после нагрузки магнит возвращается по линии возврата,
падение ЭДС = падение потокосцепления холостого хода при той же температуре).

Сетка по правилу проекта: магнит и зазор — мельчайшая (H_FINE), остальная машина — H_BASE. Для контроля
тот же расчёт на сетке вдвое грубее ВСЮДУ: числа в подписи — с мелкой сетки, сходимость печатается и
входит в подпись. На равномерной сетке 2,8 мм (прежняя версия рисунка) падение ЭДС выходило 1,20 %
против устоявшихся 0,42–0,44 % — завышение почти втрое; сгустить один лишь магнит мало — сетка стали сдвигает
падение ЭДС ещё на 5 % (4 → 2 мм).

Где магнит за коленом, тоже считается, а не пишется словами: магниты разбиваются на отдельные тела
(связные компоненты по общим рёбрам) и для каждого берётся доля площади за коленом.

Оформление — по docs/figures/STYLE.md. Выход по умолчанию — docs/figures/pmsm_riskmap_T140.png.
Запуск: python docs/figures/make_pmsm_riskmap.py [выход.png]            (≈ 15 мин: две сетки, 27 и 102 тыс. ячеек)
        python docs/figures/make_pmsm_riskmap.py --cached [выход.png]   (мгновенно: только перерисовать по
        последнему расчёту, который сохраняется во временную папку системы)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors                   # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.collections import PolyCollection     # noqa: E402
from matplotlib.ticker import FuncFormatter            # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))              # корень репозитория: magcore
from magcore.domain.magnet_model import n42sh_magnet                        # noqa: E402
from magcore.domain.steel_curves import m270_35a_bh_curve                   # noqa: E402
from magcore.fem2d.machines import (                                        # noqa: E402
    OutrunnerPMSMParams, Region, build_outrunner_spm_pmsm, evaluate_demag_impact,
    solve_machine_static, star_of_slots_layout,
)
from magcore.post.palette import field_rainbow_cmap                         # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

plt.rcParams["font.family"] = "DejaVu Sans"
DARK = "#1F2A38"; HEAT = "#E1442F"; STEEL = "#2E6C99"
GREEN = "#2FA36B"; AMBER = "#E8A33D"; MUTED = "#5A6B7B"


def _split_cmap(below, above_lo, above_hi):
    """Шкала с разрывом в середине: ниже нуля — сплошной цвет, выше — переход от above_lo к above_hi."""
    b, lo, hi = (mcolors.to_rgb(c) for c in (below, above_lo, above_hi))
    seg = {ch: [(0.0, b[i], b[i]), (0.5, b[i], lo[i]), (1.0, hi[i], hi[i])]
           for i, ch in enumerate(("red", "green", "blue"))}
    return mcolors.LinearSegmentedColormap("margin", seg)


# Запас до колена: за коленом (ΔH < 0) — сплошной «опасный» цвет, как бы мало ни было заглубление;
# безопасные ячейки — от «пограничного» к «безопасному» по величине запаса (палитра STYLE.md).
MARGIN_CMAP = _split_cmap(HEAT, AMBER, GREEN)

BG = {int(Region.AIR_GAP): "#FAFCFD", int(Region.STATOR_YOKE): "#DBE3EA",
      int(Region.TOOTH): "#DBE3EA", int(Region.SLOT): "#EDF1F6",
      int(Region.ROTOR_YOKE): "#D2DAE3"}

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
OUT = Path(ARGS[0]) if ARGS else HERE / "pmsm_riskmap_T140.png"
CACHE = Path(tempfile.gettempdir()) / "magfield_pmsm_riskmap_T140.npz"     # последний расчёт — для перерисовки
T, I_PEAK, GAMMA, TURNS = 140.0, 30.0, np.pi, 40.0
H_BASE, H_FINE = 1.0e-3, 0.125e-3                     # остальная машина / магнит и зазор [м]


def ru(x: float, nd: int) -> str:
    """Число по-русски: десятичная запятая, знак минус (не дефис)."""
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def case(h_base: float, h_fine: float):
    params = OutrunnerPMSMParams(mesh_size=h_base,
                                 mesh_size_by_region={"magnet": h_fine, "air_gap": h_fine})
    g = build_outrunner_spm_pmsm(params)
    layout = star_of_slots_layout(g.params.n_slots, g.params.n_poles)
    magnet, steel = n42sh_magnet(easy_axis=(1, 0, 0)), m270_35a_bh_curve()
    kw = dict(i_peak=I_PEAK, gamma_elec=GAMMA, turns_per_slot=TURNS)
    field = solve_machine_static(g, magnet, steel, T=T, layout=layout, max_iter=60, **kw)
    impact = evaluate_demag_impact(g, magnet, steel, layout, T=T, **kw)
    if not (field.converged and impact.load_converged):
        raise RuntimeError("расчёт под нагрузкой не сошёлся — рисунок не строится")
    return g, field, impact


def magnet_bodies(mesh, cells_idx):
    """Разбить ячейки магнитов на отдельные магниты — связные компоненты по общим рёбрам (без порогов)."""
    cells = np.asarray(mesh.cells)[cells_idx]
    shared: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(cells):
        for a, b in ((c[0], c[1]), (c[1], c[2]), (c[2], c[0])):
            shared.setdefault((min(a, b), max(a, b)), []).append(i)
    adj: list[list[int]] = [[] for _ in cells]
    for v in shared.values():
        if len(v) == 2:
            adj[v[0]].append(v[1])
            adj[v[1]].append(v[0])
    lab = -np.ones(len(cells), dtype=int)
    n = 0
    for s0 in range(len(cells)):
        if lab[s0] >= 0:
            continue
        stack = [s0]
        lab[s0] = n
        while stack:
            u = stack.pop()
            for w in adj[u]:
                if lab[w] < 0:
                    lab[w] = n
                    stack.append(w)
        n += 1
    return lab, n


def damaged_magnets(g, risk):
    """Доля площади за коленом по магнитам: [(доля своей площади, угол центра в градусах)] по убыванию;
    число магнитов; доля всей площади за коленом, приходящаяся на два самых повреждённых магнита."""
    idx = risk.cell_indices
    cen = np.array([g.mesh.cell_centroid(int(c)) for c in idx], dtype=float)
    area = np.array([g.mesh.cell_area(int(c)) for c in idx], dtype=float)
    th = np.arctan2(cen[:, 1], cen[:, 0])
    lab, n = magnet_bodies(g.mesh, idx)
    past = risk.demagnetized
    per = []
    for k in range(n):
        sel = lab == k
        ang = np.degrees(np.arctan2(np.sin(th[sel]).mean(), np.cos(th[sel]).mean()))     # круговое среднее
        per.append((float(area[sel & past].sum() / area[sel].sum()), float(ang), k))
    per.sort(reverse=True)
    top2 = np.isin(lab, [per[0][2], per[1][2]]) & past
    share2 = float(area[top2].sum() / max(area[past].sum(), 1e-30))
    return [(f, a) for f, a, _ in per], n, share2


def style_axes(ax):
    ax.set_aspect("equal")
    ax.set_xlabel("Координата x, мм", fontsize=13, color=DARK)
    ax.set_ylabel("Координата y, мм", fontsize=13, color=DARK)
    ax.tick_params(labelsize=10.5, colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def compute() -> dict:
    """Расчёт на двух сетках; всё, что нужно рисунку и подписи, — числами и массивами."""
    g_c, _, imp_c = case(2.0 * H_BASE, 2.0 * H_FINE)        # контроль: вся сетка вдвое грубее
    g, field, imp = case(H_BASE, H_FINE)
    a, a_c = imp.aggregate, imp_c.aggregate
    per, n_bodies, share2 = damaged_magnets(g, imp.risk)
    (f1, ang1), (f2, ang2) = per[0], per[1]
    for gg, ii, hb, hf in ((g_c, imp_c, 2.0 * H_BASE, 2.0 * H_FINE), (g, imp, H_BASE, H_FINE)):
        aa = ii.aggregate
        print(f"сетка {ru(1e3 * hb, 2)} мм, в магнитах и зазоре {ru(1e3 * hf, 3)} мм: ячеек {gg.mesh.n_cells}, "
              f"за коленом {ru(100 * aa.demag_area_fraction, 2)} %, ср. потеря Br {ru(100 * aa.mean_loss_frac, 3)} %, "
              f"падение ЭДС {ru(100 * ii.flux_linkage_drop_frac, 3)} %, худший запас {ru(ii.risk.worst_margin / 1e3, 1)} кА/м")
    return dict(
        vertices=np.asarray(g.mesh.vertices, dtype=float), cells=np.asarray(g.mesh.cells, dtype=int),
        region=np.asarray(g.region), bmag=np.hypot(field.B_cells[:, 0], field.B_cells[:, 1]),
        mag_idx=np.asarray(imp.risk.cell_indices), margin=np.asarray(imp.risk.margin, dtype=float),
        n_slots=g.params.n_slots, n_poles=g.params.n_poles, n_bodies=n_bodies,
        f1=f1, f2=f2, ang1=ang1, ang2=ang2, share2=share2,
        drop=imp.flux_linkage_drop_frac, drop_c=imp_c.flux_linkage_drop_frac,
        area=a.demag_area_fraction, area_c=a_c.demag_area_fraction, mean_loss=a.mean_loss_frac,
        h_base=H_BASE, h_fine=H_FINE)


def render(d: dict):
    opposite = abs(abs(((d["ang1"] - d["ang2"]) + 180.0) % 360.0 - 180.0) - 180.0) < 10.0
    print(f"два самых повреждённых магнита: {ru(100 * d['f1'], 1)} % и {ru(100 * d['f2'], 1)} % площади (углы "
          f"{ru(d['ang1'], 0)}° и {ru(d['ang2'], 0)}°, противоположны: {opposite}); на них "
          f"{ru(100 * d['share2'], 2)} % всей площади за коленом")
    v = d["vertices"] * 1e3                                     # м → мм
    tri = v[d["cells"]]
    region = d["region"]
    lim = np.abs(v).max() * 1.04

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15.2, 9.4), dpi=200)

    # (а) модуль индукции по всей машине — палитра поля проекта (радуга)
    pa = PolyCollection(tri, array=d["bmag"], cmap=field_rainbow_cmap(), edgecolors="face", linewidths=0.15)
    axA.add_collection(pa)
    axA.set_xlim(-lim, lim)
    axA.set_ylim(-lim, lim)
    style_axes(axA)
    axA.set_title("(а)  Поле в худшем режиме нагрузки", fontsize=13.5, fontweight="bold", color=DARK, pad=10)
    cbA = fig.colorbar(pa, ax=axA, orientation="horizontal", fraction=0.045, pad=0.11)
    cbA.set_label("Модуль индукции  |B|, Тл", fontsize=12.5, color=DARK)
    cbA.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: ru(x, 1)))
    cbA.ax.tick_params(labelsize=10.5)

    # (б) запас до колена в магнитах: меньше нуля — магнит за коленом, потеря необратима
    for code, col in BG.items():
        sel = region == code
        if sel.any():
            axB.add_collection(PolyCollection(tri[sel], facecolors=col, edgecolors="face", linewidths=0.15))
    margin = d["margin"] / 1e3                                   # кА/м
    lo, hi = min(float(margin.min()), -1e-6), max(float(margin.max()), 1e-6)
    norm = mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
    pb = PolyCollection(tri[d["mag_idx"]], array=margin, cmap=MARGIN_CMAP, norm=norm, edgecolors="face", linewidths=0.15)
    axB.add_collection(pb)
    axB.set_xlim(-lim, lim)
    axB.set_ylim(-lim, lim)
    style_axes(axB)
    axB.set_title("(б)  Запас до колена в магнитах", fontsize=13.5, fontweight="bold", color=DARK, pad=10)
    cbB = fig.colorbar(pb, ax=axB, orientation="horizontal", fraction=0.045, pad=0.11)
    cbB.set_label("Запас поля до колена  ΔH, кА/м", fontsize=12.5, color=DARK)
    # Деления — строго внутри шкалы: крайнее левое — сам минимум (подпись округлена), иначе шкала пустеет.
    step = 100.0 if hi > 250.0 else 50.0
    ticks = [lo, 0.0] + [float(x) for x in np.arange(step, hi, step)]
    cbB.set_ticks(ticks, labels=[ru(t, 0) for t in ticks])
    cbB.ax.tick_params(labelsize=10.5)

    drop, drop_c = 100 * d["drop"], 100 * d["drop_c"]
    area, area_c = 100 * d["area"], 100 * d["area_c"]
    fig.suptitle(f"Ток якоря при {ru(T, 0)} °C уводит за колено {ru(area, 0)} % площади магнитов, "
                 f"а ЭДС падает лишь на {ru(drop, 1)} %", fontsize=15, fontweight="bold", color=DARK, y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.905, bottom=0.235, wspace=0.16)

    pair = "двух противоположных магнитах" if opposite else "двух магнитах"
    where = (f"{ru(100 * d['share2'], 0)} % этой площади — в {pair} из {int(d['n_bodies'])}, "
             f"у них за коленом {ru(100 * d['f1'], 0)} и {ru(100 * d['f2'], 0)} % площади")
    cap = "\n".join([
        f"Двигатель outrunner: {int(d['n_slots'])} пазов, {int(d['n_poles'])} полюсов, магниты N42SH, сталь M270-35A. "
        f"Режим: {ru(T, 0)} °C, ток {ru(I_PEAK, 0)} А по оси d (размагничивает), {ru(TURNS, 0)} витков на паз.",
        f"Красное на (б) — магнит за коленом (ΔH < 0), там потеря необратима: это {where}.",
        f"За коленом {ru(area, 1)} % площади магнитов, но средняя потеря остаточной индукции "
        f"{ru(100 * d['mean_loss'], 2)} %, а падение ЭДС — {ru(drop, 2)} %.",
        f"Сетка {ru(1e3 * float(d['h_base']), 0)} мм, в магнитах и зазоре {ru(1e3 * float(d['h_fine']), 3)} мм. "
        f"На сетке вдвое грубее всюду: падение ЭДС {ru(drop_c, 2)} %, за коленом {ru(area_c, 1)} % площади.",
    ])
    fig.text(0.055, 0.015, cap, fontsize=12.0, color=DARK, ha="left", va="bottom", linespacing=1.7)
    fig.savefig(OUT, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT)


def main():
    if "--cached" in sys.argv and CACHE.exists():
        z = np.load(CACHE)
        d = {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}
        print("данные — из", CACHE, "(расчёт не запускался)")
    else:
        d = compute()
        np.savez(CACHE, **d)
    render(d)


if __name__ == "__main__":
    main()
