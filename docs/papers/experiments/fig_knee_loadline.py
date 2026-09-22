# -*- coding: utf-8 -*-
"""
Механизм теплового размагничивания на РЕАЛЬНЫХ кривых модели, два кадра:
  (а) ПОЛНЫЕ кривые целиком в РАБОЧЕМ квадранте (B >= 0, как в даташите) —
      кривая по НАМАГНИЧЕННОСТИ J(H) и по ИНДУКЦИИ B(H) («нормальная») при 20 и
      150 °C, оба колена; кривая по индукции обрывается на H_cB, по намагниченности
      — на H_cJ;
  (б) увеличение рабочей зоны — нагрузочная прямая и путь рабочей точки ①→②→③.

Кривые НЕ рисуются от руки: берутся из magnet.curve_at(T) — та самая методика
(magcore/domain/magnet_curves.py, docs/math/nonlinear_materials.md §4):
кривая по намагниченности J(H) = прямая выше колена + КАСАТЕЛЬНАЯ ПАРАБОЛА ниже
(стык C¹), кривая по индукции («нормальная») B(H) = J(H) + mu0*H. Параметры при T — из двух температурных
коэффициентов (alpha_Br для Br/HcB, gamma_Hc для колена H_k и HcJ).

Ключевое (⚠ было исправлено): рабочая точка лежит на НАГРУЗОЧНОЙ ПРЯМОЙ
B = mu0*Pc*|H| (из начала координат, наклон = геометрия), а НЕ на вертикали
«H фиксировано». Поэтому при нагреве точка съезжает вниз И К НАЧАЛУ КООРДИНАТ —
|H| УМЕНЬШАЕТСЯ. За колено магнит уходит потому, что КОЛЕНО ДВИЖЕТСЯ К НУЛЮ
БЫСТРЕЕ точки (gamma_Hc = 0.55 против alpha_Br = 0.115 %/°C), плюс реакция якоря
сдвигает нагрузочную прямую по оси H.

Запуск: python docs/papers/experiments/fig_knee_loadline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import ConnectionPatch, Rectangle   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from magcore.constants import MU0                          # noqa: E402
from magcore.domain.magnet_model import n42sh_magnet       # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"
DARK = "#1F2A38"; HEAT = "#E1442F"; STEEL = "#2E6C99"
GREEN = "#2FA36B"; AMBER = "#E8A33D"; MUTED = "#5A6B7B"

T_COLD, T_HOT = 20.0, 150.0
PC = 4.0            # наклон нагрузочной прямой: задан ГЕОМЕТРИЕЙ (толщина магнита / зазор)
HA = 330.0e3        # сдвиг от реакции якоря (ток обмотки), перегруз [А/м]

mag = n42sh_magnet((1.0, 0.0, 0.0))
cur_c, cur_h = mag.curve_at(T_COLD), mag.curve_at(T_HOT)


def curve_B(cur, h_abs):
    """B нормальной кривой при |H| [А/м] (кривая задана в 2-м квадранте, H<0)."""
    return np.interp(-np.asarray(h_abs, float), cur.H_values, cur.B_values)


def load_line(h_abs, shift=0.0):
    return MU0 * PC * (np.asarray(h_abs, float) - shift)


def intersect(cur, shift=0.0):
    """|H| пересечения нагрузочной прямой с кривой (кривая падает, прямая растёт)."""
    h = np.linspace(shift + 1.0, float(-cur.H_values[0]) * 0.999, 400_000)
    g = curve_B(cur, h) - load_line(h, shift)
    i = int(np.argmax(g <= 0.0))
    if i == 0:
        raise RuntimeError("нет пересечения нагрузочной прямой с кривой")
    t = g[i - 1] / (g[i - 1] - g[i])
    return float(h[i - 1] + t * (h[i] - h[i - 1]))


h1 = intersect(cur_c, 0.0);  b1 = float(load_line(h1))          # ① холодный, х.х.
h2 = intersect(cur_h, 0.0);  b2 = float(load_line(h2))          # ② горячий, та же прямая
h3 = intersect(cur_h, HA);   b3 = float(load_line(h3, HA))      # ③ горячий + реакция якоря
hk_c, hk_h = mag.Hk(T_COLD), mag.Hk(T_HOT)
bk_c, bk_h = float(curve_B(cur_c, hk_c)), float(curve_B(cur_h, hk_h))

k = 1e-3
WBOX = dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.8)
ZOOM = (-545.0, 95.0, 0.0, 1.48)      # окно кадра (б) в координатах кадра (а)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(16.0, 9.0), dpi=200,
                               gridspec_kw=dict(width_ratios=[1.06, 1.0]))

# ============================ (а) КРИВЫЕ ЦЕЛИКОМ ============================
for cur, col in ((cur_c, STEEL), (cur_h, HEAT)):
    h_abs = -cur.H_values[::-1]                        # |H| по возрастанию
    B = cur.B_values[::-1]
    axA.plot(-h_abs * k, B - MU0 * (-h_abs), color=col, lw=1.9, ls=(0, (5, 3)), alpha=0.85)
    axA.plot(-h_abs * k, B, color=col, lw=3.6)

for cur, hk, col in ((cur_c, hk_c, STEEL), (cur_h, hk_h, HEAT)):
    axA.plot(-hk * k, float(curve_B(cur, hk)) + MU0 * hk, "o", color=col, ms=12,
             zorder=6, mec="white", mew=1.5)
axA.annotate("колено", xy=(-hk_c * k, bk_c + MU0 * hk_c), xytext=(-hk_c * k - 40, 1.34),
             color=STEEL, fontsize=12.5, fontweight="bold", ha="center", bbox=WBOX, zorder=9,
             arrowprops=dict(arrowstyle="->", color=STEEL, lw=1.8))
axA.annotate("колено", xy=(-hk_h * k, bk_h + MU0 * hk_h), xytext=(-hk_h * k - 300, 1.34),
             color=HEAT, fontsize=12.5, fontweight="bold", ha="center", bbox=WBOX, zorder=9,
             arrowprops=dict(arrowstyle="->", color=HEAT, lw=1.8))
axA.annotate("", xy=(-hk_h * k, 0.72), xytext=(-hk_c * k, 0.72),
             arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.8))
axA.text((-hk_h - hk_c) * k / 2, 0.79, "колено:  %.0f → %.0f кА/м" % (hk_c * k, hk_h * k),
         color=DARK, fontsize=13, fontweight="bold", ha="center", bbox=WBOX, zorder=9)

axA.text(20, mag.Br(T_COLD), "20 °C", color=STEEL, fontsize=13, fontweight="bold", va="center")
axA.text(20, mag.Br(T_HOT), "150 °C", color=HEAT, fontsize=13, fontweight="bold", va="center")
axA.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0], ZOOM[3] - ZOOM[2],
                        fill=False, ec=DARK, lw=1.6, ls=":", zorder=7))

axA.set_xlim(-1730.0, 190.0); axA.set_ylim(0.0, 1.62)
axA.set_title("(а)  Кривые целиком", fontsize=13.5, fontweight="bold", color=DARK, pad=10)
axA.set_xlabel("Напряжённость магнитного поля  H, кА/м", fontsize=13, color=DARK)
axA.set_ylabel("Индукция B  и  намагниченность J,  Тл", fontsize=13, color=DARK)
axA.legend(handles=[Line2D([], [], color=MUTED, lw=3.4,
                           label="B(H) — по индукции"),
                    Line2D([], [], color=MUTED, lw=1.9, ls=(0, (5, 3)),
                           label="J(H) — по намагниченности")],
           loc="lower left", fontsize=11, frameon=True, facecolor="white",
           edgecolor="none", framealpha=0.92)

# ============================ (б) РАБОЧАЯ ЗОНА ============================
def draw_zone(ax, f=1.0):
    """Рабочая зона: кривые, нагрузочные прямые, путь точки ①→②→③.
    f — масштаб шрифтов/маркеров (1.0 — кадр (б) статьи, ~1.3 — отдельный слайд)."""
    ax.axvspan(ZOOM[0], -hk_h * k, color=HEAT, alpha=0.07, zorder=0)
    ax.text(ZOOM[0] + 14, 0.70, "за коленом", color=HEAT, fontsize=11.5 * f,
            fontweight="bold", ha="center", va="center", rotation=90, alpha=0.9)
    h = np.linspace(0.0, -ZOOM[0] / k, 1200)
    ax.plot(-h * k, curve_B(cur_c, h), color=STEEL, lw=3.6 * f)
    ax.plot(-h * k, curve_B(cur_h, h), color=HEAT, lw=3.6 * f)
    ax.plot(-hk_h * k, bk_h, "o", color=HEAT, ms=12 * f, zorder=6)
    ax.annotate("колено", xy=(-hk_h * k, bk_h), xytext=(-hk_h * k - 42, bk_h + 0.26),
                color=HEAT, fontsize=12.5 * f, fontweight="bold", ha="center",
                bbox=WBOX, zorder=9, arrowprops=dict(arrowstyle="->", color=HEAT, lw=1.8))

    ax.plot(-np.linspace(0.0, h1 * 1.08, 60) * k, load_line(np.linspace(0.0, h1 * 1.08, 60)),
            color=MUTED, lw=2.4 * f, label="нагрузочная кривая")
    hl2 = np.linspace(HA, h3 * 1.06, 60)
    ax.plot(-hl2 * k, load_line(hl2, HA), color=MUTED, lw=2.4 * f, ls="--",
            label="нагрузочная кривая + реакция якоря")
    ax.annotate("", xy=(-HA * k, 0.02), xytext=(0, 0.02),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.8))
    ax.text(-HA * k / 2, 0.062, "$H_a$", color=MUTED, fontsize=13 * f, ha="center", bbox=WBOX)

    ax.text(7, mag.Br(T_COLD), "20 °C", color=STEEL, fontsize=13 * f, fontweight="bold",
            va="center")
    ax.text(7, mag.Br(T_HOT), "150 °C", color=HEAT, fontsize=13 * f, fontweight="bold",
            va="center")

    for hh, b, c, lab in ((h1, b1, GREEN, "1"), (h2, b2, AMBER, "2"), (h3, b3, HEAT, "3")):
        ax.plot(-hh * k, b, "o", color=c, ms=18 * f, zorder=8, mec="white", mew=1.9)
        ax.text(-hh * k, b, lab, color="white", fontsize=11.5 * f, fontweight="bold",
                ha="center", va="center", zorder=9)
    ax.annotate("", xy=(-h2 * k, b2 + 0.016), xytext=(-h1 * k, b1 - 0.016),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.6 * f), zorder=7)
    ax.annotate("", xy=(-h3 * k + 9, b3 + 0.03), xytext=(-h2 * k - 6, b2 - 0.02),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=2.6 * f, ls=":",
                                connectionstyle="arc3,rad=0.18"), zorder=7)
    ax.text(-h1 * k + 12, (b1 + b2) / 2, "нагрев", color=DARK, fontsize=12.5 * f,
            fontweight="bold", ha="left", va="center", bbox=WBOX, zorder=10)
    ax.text(-300, 0.27, "реакция якоря", color=DARK, fontsize=12.5 * f, fontweight="bold",
            ha="center", bbox=WBOX, zorder=10)
    ax.annotate("", xy=(-312, 0.50), xytext=(-303, 0.33),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.4 * f), zorder=10)

    ax.set_xlim(ZOOM[0], ZOOM[1]); ax.set_ylim(ZOOM[2], ZOOM[3])
    ax.set_xlabel("Напряжённость магнитного поля  H, кА/м", fontsize=13 * f, color=DARK)
    ax.set_ylabel("Индукция  B,  Тл", fontsize=13 * f, color=DARK)
    ax.tick_params(labelsize=10 * f)
    ax.legend(loc="upper left", fontsize=11 * f, frameon=True, facecolor="white",
              edgecolor="none", framealpha=0.92)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


draw_zone(axB, 1.0)
axB.set_title("(б)  Рабочая зона (увеличено)", fontsize=13.5, fontweight="bold",
              color=DARK, pad=10)

for s in ("top", "right"):
    axA.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axA.spines[s].set_color(MUTED)

fig.suptitle("За колено магнит уходит не потому, что поле выросло, — а потому что КОЛЕНО ДОГНАЛО точку",
             fontsize=15, fontweight="bold", color=DARK, y=0.975)
fig.subplots_adjust(left=0.058, right=0.988, top=0.885, bottom=0.20, wspace=0.20)

for y in (ZOOM[3], ZOOM[2]):     # правые углы рамки в (а) → левые углы кадра (б)
    fig.add_artist(ConnectionPatch(xyA=(ZOOM[1], y), coordsA=axA.transData,
                                   xyB=(ZOOM[0], y), coordsB=axB.transData,
                                   color=DARK, lw=1.1, ls=":", alpha=0.75))

cap = (
    "①  20 °C: рабочее поле %.0f кА/м, до колена ещё %.0f кА/м.\n"
    "②  нагрев до 150 °C: точка съехала по ТОЙ ЖЕ прямой, поле даже уменьшилось до %.0f кА/м, "
    "но колено ушло к нулю — запас упал до %.0f кА/м.\n"
    "③  реакция якоря сдвинула прямую на %.0f кА/м → точка за коленом на %.0f кА/м: необратимая потеря."
) % (h1 * k, (hk_c - h1) * k, h2 * k, (hk_h - h2) * k, HA * k, (h3 - hk_h) * k)
fig.text(0.058, 0.018, cap, fontsize=12.4, color=DARK, ha="left", va="bottom", linespacing=1.75)

out_dir = Path(__file__).resolve().parent / "figs"
out_dir.mkdir(parents=True, exist_ok=True)
out = out_dir / "fig_knee_loadline.png"
plt.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
plt.close()

# ---- версия ДЛЯ СЛАЙДА: только рабочая зона, крупным шрифтом, под бокс 5.9x4.6" (4:3) ----
# Расшифровка ①②③ на слайде идёт отдельной колонкой справа, поэтому подписи под рисунком нет.
fig_s, ax_s = plt.subplots(figsize=(9.0, 7.0), dpi=200)
draw_zone(ax_s, 1.32)
fig_s.tight_layout()
out_slide = out_dir / "fig_knee_loadline_slide.png"
fig_s.savefig(out_slide, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig_s)

print("saved", out)
print("saved", out_slide)
print("Pc = %.1f, H_a = %.0f кА/м" % (PC, HA * k))
print("(1) 20 C, x.x.:    |H| = %6.1f  B = %.3f   запас до колена = %7.1f кА/м" % (h1 * k, b1, (hk_c - h1) * k))
print("(2) 150 C, x.x.:   |H| = %6.1f  B = %.3f   запас до колена = %7.1f кА/м" % (h2 * k, b2, (hk_h - h2) * k))
print("(3) 150 C + якорь: |H| = %6.1f  B = %.3f   ЗА КОЛЕНОМ на %7.1f кА/м" % (h3 * k, b3, (h3 - hk_h) * k))
print("колено: %.0f -> %.0f кА/м (-%.0f %%);  Br: %.3f -> %.3f Тл" %
      (hk_c * k, hk_h * k, 100 * (1 - hk_h / hk_c), mag.Br(T_COLD), mag.Br(T_HOT)))
for T, cur in ((T_COLD, cur_c), (T_HOT, cur_h)):
    print("T=%3.0f C  HcB(факт с кривой) = %6.1f кА/м   [параметр Hcb(T) = %6.1f]" %
          (T, cur.Hcb_actual() * k, mag.Hcb(T) * k))
